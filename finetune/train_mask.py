#!/usr/bin/env python3
"""只读取人工 gold_labels.json，基于当前口罩 ONNX 权重微调 YOLOv5s。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402

if config.TORCH_PACKAGE_DIR.is_dir():
    sys.path.insert(0, str(config.TORCH_PACKAGE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

_DLL_HANDLES = config.configure_runtime_dlls(config.TRAINING_VENV_DIR)
from business_degradation import business_degradation  # noqa: E402
from dataset_schema import flatten_gold, load_gold, resolved_path  # noqa: E402
from mask_model import build_from_onnx  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调当前 ONNX 对应的 YOLOv5s 口罩模型")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--checkpoint", type=Path, help="继续微调此前保存的 best.pt/last.pt")
    return parser.parse_args()


def split_name(full_image: str) -> str:
    bucket = int(hashlib.sha1(full_image.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def letterbox(image: np.ndarray, box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(640.0 / width, 640.0 / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_left = (640 - new_width) // 2
    pad_top = (640 - new_height) // 2
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    left, top, right, bottom = box
    left = left * scale + pad_left
    right = right * scale + pad_left
    top = top * scale + pad_top
    bottom = bottom * scale + pad_top
    yolo_box = np.asarray(
        [
            (left + right) / 2.0 / 640.0,
            (top + bottom) / 2.0 / 640.0,
            (right - left) / 640.0,
            (bottom - top) / 640.0,
        ],
        dtype=np.float32,
    )
    return canvas, yolo_box


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], training: bool):
        self.records = records
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = resolved_path(record["图片"], config.PROJECT_ROOT)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取训练图片: {path}")
        box = list(map(float, record["训练框"]))
        if self.training:
            image = business_degradation(image)
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                width = image.shape[1]
                box[0], box[2] = width - box[2], width - box[0]
        image, yolo_box = letterbox(image, box)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        # 保持 YOLOv5 DataLoader 的 uint8 协议；训练循环和官方 val.run 各自只归一化一次。
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        class_id = 1.0 if record["标签"] == "w/ mask" else 0.0
        target = torch.tensor([[0.0, class_id, *yolo_box.tolist()]], dtype=torch.float32)
        shapes = ((640, 640), ((1.0, 1.0), (0.0, 0.0)))
        return image_tensor, target, str(path), shapes


def collate(batch):
    images, targets, paths, shapes = zip(*batch)
    adjusted = []
    for index, target in enumerate(targets):
        target = target.clone()
        target[:, 0] = index
        adjusted.append(target)
    return torch.stack(images), torch.cat(adjusted), list(paths), list(shapes)


def save_checkpoint(path: Path, epoch: int, best_fitness: float, model, optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "model": copy.deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def export_onnx(model, output_dir: Path, device: torch.device) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).float().eval().to(device)
    detect = model.model[-1]
    detect.inplace = True
    detect.dynamic = False
    detect.export = True
    output_path = output_dir / "face_mask_detection.onnx"
    probe = torch.zeros(1, 3, 640, 640, device=device)
    torch.onnx.export(
        model,
        probe,
        str(output_path),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
        dynamo=False,
    )
    (output_dir / "synset.txt").write_text("w/o mask\nw/ mask\n", encoding="utf-8")
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    if session.get_outputs()[0].shape != [1, 25200, 7]:
        raise RuntimeError(f"导出的 ONNX 输出协议错误: {session.get_outputs()[0].shape}")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (output_dir / "SHA256.txt").write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
    return output_path


def main() -> int:
    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 和 learning-rate 必须大于零")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = args.device == "GPU" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if args.device == "GPU" and not use_cuda:
        raise RuntimeError("配置要求 GPU，但 PyTorch CUDA 当前不可用")

    gold = load_gold(config.GOLD_LABELS_PATH)
    records = flatten_gold(gold, "口罩")
    train_records = [row for row in records if split_name(row["全图"]) == "train"]
    val_records = [row for row in records if split_name(row["全图"]) == "val"]
    if not train_records or not val_records:
        raise ValueError(f"训练/验证样本不足: train={len(train_records)}, val={len(val_records)}")

    sys.path.insert(0, str(config.YOLOV5_DIR.resolve()))
    from utils.loss import ComputeLoss
    from val import run as validate

    model = build_from_onnx(
        config.FACE_MASK_DIR / "face_mask_detection.onnx",
        config.YOLOV5_DIR,
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    start_epoch = 0
    best_fitness = -1.0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"].float().state_dict())
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_fitness = float(checkpoint.get("best_fitness", -1.0))

    hyp = yaml.safe_load((config.YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml").read_text())
    hyp["lr0"] = args.learning_rate
    hyp["cls"] *= 2 / 80 * 3
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = torch.ones(2, device=device)
    model.names = ["w/o mask", "w/ mask"]
    compute_loss = ComputeLoss(model)

    train_loader = torch.utils.data.DataLoader(
        MaskDataset(train_records, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        MaskDataset(val_records, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    output_dir = config.TRAINING_OUTPUT_DIR / "face_mask"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for images, targets, _, _ in train_loader:
            images = images.to(device).float() / 255.0
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                prediction = model(images)
                loss, _ = compute_loss(prediction, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        results, _, _ = validate(
            data={"nc": 2, "names": model.names},
            batch_size=args.batch_size,
            imgsz=640,
            model=model,
            dataloader=val_loader,
            compute_loss=compute_loss,
            plots=False,
            save_dir=output_dir / "validation",
        )
        precision, recall, map50, map5095 = map(float, results[:4])
        fitness = map50
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={np.mean(losses):.6f} "
            f"precision={precision:.6f} recall={recall:.6f} mAP50={map50:.6f} "
            f"mAP50-95={map5095:.6f}",
            flush=True,
        )
        save_checkpoint(output_dir / "last.pt", epoch, max(best_fitness, fitness), model, optimizer)
        if fitness > best_fitness:
            best_fitness = fitness
            save_checkpoint(output_dir / "best.pt", epoch, best_fitness, model, optimizer)
            export_onnx(model, output_dir, device)
    print(f"完成: {output_dir}，最佳 mAP@0.5={best_fitness:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
