#!/usr/bin/env python3
"""读取 2_attribute/gold.json（及 4_augmented/attribute 训练划分），微调 PP-HGNet_small。"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
os.environ.setdefault("FLAGS_enable_pir_api", "0")
sys.path.insert(0, str(HERE))

import config  # noqa: E402

_DLL_HANDLES = config.configure_runtime_dlls(config.TRAINING_VENV_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402

from attribute_model import build_from_deployment  # noqa: E402
from dataset_schema import (  # noqa: E402
    BODY_ATTRIBUTES,
    flatten_annotations,
    load_dataset,
    resolved_path,
    split_name,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调现有 PP-HGNet_small 行人属性模型")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--checkpoint", type=Path, help="继续微调此前保存的 .pdparams")
    return parser.parse_args()


class AttributeDataset(paddle.io.Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        # 裁剪小图总量约百 MB 级，一次性解码进内存，避免每个 epoch 反复读盘解码。
        # 不做任何在线增强，增强只来自离线尘土化副本。
        self.cache: dict[int, np.ndarray] = {}
        total_bytes = 0
        for index, record in enumerate(records):
            path = resolved_path(record["image"]["file_name"], config.PROJECT_ROOT)
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取训练图片: {path}")
            self.cache[index] = image
            total_bytes += image.nbytes
        print(
            f"属性数据集预载 {len(self.cache)} 张到内存（{total_bytes / 2**20:.0f} MB）",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        image = self.cache[index]
        image = cv2.resize(image, (192, 256), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        target = np.asarray([record["attributes"][name] for name in BODY_ATTRIBUTES], dtype=np.float32)
        return (image - mean) / std, target


def f1_scores(logits: paddle.Tensor, targets: paddle.Tensor) -> tuple[float, np.ndarray]:
    predicted = (paddle.nn.functional.sigmoid(logits).numpy() >= 0.5)
    expected = targets.numpy() >= 0.5
    true_positive = np.logical_and(predicted, expected).sum(axis=0)
    false_positive = np.logical_and(predicted, ~expected).sum(axis=0)
    false_negative = np.logical_and(~predicted, expected).sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    scores = np.divide(2 * true_positive, denominator, out=np.zeros(26), where=denominator > 0)
    return float(scores.mean()), scores


def evaluate(model: paddle.nn.Layer, loader: paddle.io.DataLoader) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    logits: list[paddle.Tensor] = []
    targets: list[paddle.Tensor] = []
    with paddle.no_grad():
        for images, labels in loader:
            output = model(images)
            losses.append(float(paddle.nn.functional.binary_cross_entropy_with_logits(output, labels)))
            logits.append(output.cpu())
            targets.append(labels.cpu())
    macro_f1, _ = f1_scores(paddle.concat(logits), paddle.concat(targets))
    return float(np.mean(losses)), macro_f1


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: paddle.nn.Layer,
    optimizer: paddle.optimizer.Optimizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paddle.save(model.state_dict(), str(output_dir / f"{name}.pdparams"))
    paddle.save(optimizer.state_dict(), str(output_dir / f"{name}.pdopt"))


def export_model(model: paddle.nn.Layer, output_dir: Path) -> None:
    """在独立 PIR 进程导出 Paddle 3 部署模型，避免混用旧静态图运行时。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ".export.pdparams"
    paddle.save(model.state_dict(), str(checkpoint))
    environment = os.environ.copy()
    environment.pop("FLAGS_enable_pir_api", None)
    try:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "export_attribute.py"),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--paddleclas-dir",
                str(config.PADDLECLAS_DIR),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    finally:
        checkpoint.unlink(missing_ok=True)


def main() -> int:
    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 和 learning-rate 必须大于零")
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gold = load_dataset(config.ATTRIBUTE_GOLD_PATH, "attribute", "human", gold=True)
    records = flatten_annotations(gold)
    augmented_path = config.AUGMENTED_DATA_DIR / "attribute/annotations.json"
    if augmented_path.is_file():
        augmented = load_dataset(augmented_path, "attribute", "augmented", gold=True)
        records += [
            row
            for row in flatten_annotations(augmented)
            if split_name(row["image"]["source_image_id"]) == "train"
        ]
    train_records = [row for row in records if split_name(row["image"]["source_image_id"]) == "train"]
    val_records = [row for row in records if split_name(row["image"]["source_image_id"]) == "val"]
    if not train_records or not val_records:
        raise ValueError(f"训练/验证样本不足: train={len(train_records)}, val={len(val_records)}")

    device = args.device.lower()
    # 微调起点固定用官方部署权重；输出仍到 models/finetuned/person_attribute。
    base_model_dir = config.MODEL_DIR / "human/PPHGNet_small_person_attribute_954_infer"
    model = build_from_deployment(base_model_dir, config.PADDLECLAS_DIR, device)
    if args.checkpoint:
        model.set_state_dict(paddle.load(str(args.checkpoint)))

    train_loader = paddle.io.DataLoader(
        AttributeDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = paddle.io.DataLoader(
        AttributeDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=5e-4,
    )
    output_dir = config.TRAINING_OUTPUT_DIR / "person_attribute"
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets in train_loader:
            logits = model(images)
            loss = paddle.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_losses.append(float(loss))
        val_loss, val_f1 = evaluate(model, val_loader)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={np.mean(train_losses):.6f} "
            f"val_loss={val_loss:.6f} val_macro_f1={val_f1:.6f}",
            flush=True,
        )
        save_checkpoint(output_dir, "last", model, optimizer)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(output_dir, "best", model, optimizer)
            export_model(model, output_dir)
    print(f"完成: {output_dir}，最佳 macro-F1={best_f1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
