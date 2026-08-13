#!/usr/bin/env python3
"""只读取人工 gold_labels.json，基于当前 PP-HGNet_small 权重微调属性模型。"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
os.environ.setdefault("FLAGS_enable_pir_api", "0")
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402

_DLL_HANDLES = config.configure_runtime_dlls(config.TRAINING_VENV_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402

from attribute_model import build_from_deployment  # noqa: E402
from business_degradation import business_degradation  # noqa: E402
from dataset_schema import BODY_ATTRIBUTES, flatten_gold, load_gold, resolved_path  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调现有 PP-HGNet_small 行人属性模型")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--checkpoint", type=Path, help="继续微调此前保存的 .pdparams")
    return parser.parse_args()


def split_name(full_image: str) -> str:
    bucket = int(hashlib.sha1(full_image.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


class AttributeDataset(paddle.io.Dataset):
    def __init__(self, records: list[dict], training: bool):
        self.records = records
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        path = resolved_path(record["图片"], config.PROJECT_ROOT)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取训练图片: {path}")
        if self.training:
            image = business_degradation(image)
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
        image = cv2.resize(image, (192, 256), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        target = np.asarray([record["标签"][name] for name in BODY_ATTRIBUTES], dtype=np.float32)
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
    gold = load_gold(config.GOLD_LABELS_PATH)
    records = flatten_gold(gold, "属性")
    train_records = [row for row in records if split_name(row["全图"]) == "train"]
    val_records = [row for row in records if split_name(row["全图"]) == "val"]
    if not train_records or not val_records:
        raise ValueError(f"训练/验证样本不足: train={len(train_records)}, val={len(val_records)}")

    device = args.device.lower()
    model = build_from_deployment(config.PERSON_ATTRIBUTE_DIR, config.PADDLECLAS_DIR, device)
    if args.checkpoint:
        model.set_state_dict(paddle.load(str(args.checkpoint)))

    train_loader = paddle.io.DataLoader(
        AttributeDataset(train_records, True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = paddle.io.DataLoader(
        AttributeDataset(val_records, False),
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
