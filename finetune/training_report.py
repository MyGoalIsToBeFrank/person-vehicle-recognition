# -*- coding: utf-8 -*-
"""汇总三个模型的训练日志，绘制 loss/指标曲线并生成训练报告。

用法（全部训练完成后执行）:
    .venv-train/Scripts/python.exe finetune/training_report.py \
        --det-log logs/training/detector.log \
        --attr-log logs/training/attribute.log \
        --mask-log logs/training/mask.log

输出:
    finetune/TRAINING_REPORT.md   报告正文
    finetune/report/*.png         曲线图
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REPORT_DIR = HERE / "report"
PROCESSED = PROJECT_ROOT / "dataset/processed"
EXPORT_DIR = PROCESSED / "5_export/person_detection_coco/annotations"

# ---------------- 日志解析 ----------------

DET_ITER = re.compile(
    r"Epoch: \[(\d+)\] \[\s*\d+/\d+\] learning_rate: ([\d.]+) loss: ([\d.]+) "
    r"loss_cls: ([\d.]+) loss_iou: ([\d.]+) loss_dfl: ([\d.]+) loss_l1: ([\d.]+)"
)
DET_AP5095 = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([\d.]+)"
)
DET_AP50 = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([\d.]+)"
)
ATTR_LINE = re.compile(
    r"epoch=(\d+)/\d+ train_loss=([\d.]+) val_loss=([\d.]+) val_macro_f1=([\d.]+)"
)
MASK_LINE = re.compile(
    r"epoch=(\d+)/\d+ train_loss=([\d.]+) precision=([\d.]+) recall=([\d.]+) "
    r"mAP50=([\d.]+) mAP50-95=([\d.]+)"
)


def parse_detection_log(path: Path) -> dict:
    per_epoch: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ap5095: list[float] = []
    ap50: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DET_ITER.search(line)
        if match:
            epoch = int(match.group(1))
            for name, value in zip(
                ("lr", "loss", "loss_cls", "loss_iou", "loss_dfl", "loss_l1"),
                match.groups()[1:],
            ):
                per_epoch[epoch][name].append(float(value))
            continue
        match = DET_AP5095.search(line)
        if match:
            ap5095.append(float(match.group(1)))
            continue
        match = DET_AP50.search(line)
        if match:
            ap50.append(float(match.group(1)))
    epochs = sorted(per_epoch)
    return {
        "epochs": [e + 1 for e in epochs],
        "loss": [sum(per_epoch[e]["loss"]) / max(1, len(per_epoch[e]["loss"])) for e in epochs],
        "loss_cls": [sum(per_epoch[e]["loss_cls"]) / max(1, len(per_epoch[e]["loss_cls"])) for e in epochs],
        "loss_iou": [sum(per_epoch[e]["loss_iou"]) / max(1, len(per_epoch[e]["loss_iou"])) for e in epochs],
        "loss_dfl": [sum(per_epoch[e]["loss_dfl"]) / max(1, len(per_epoch[e]["loss_dfl"])) for e in epochs],
        "loss_l1": [sum(per_epoch[e]["loss_l1"]) / max(1, len(per_epoch[e]["loss_l1"])) for e in epochs],
        "ap5095": ap5095,
        "ap50": ap50,
    }


def parse_simple_log(path: Path, pattern: re.Pattern, names: tuple[str, ...]) -> dict:
    rows: dict[str, list[float]] = {name: [] for name in names}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            for name, value in zip(names, match.groups()):
                rows[name].append(float(value))
    return rows


# ---------------- 数据统计 ----------------


def count_coco(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return (0, 0)
    data = json.loads(path.read_text(encoding="utf-8"))
    return (len(data["images"]), len(data["annotations"]))


def dataset_stats() -> dict:
    stats: dict = {}
    stats["detection"] = {
        split: count_coco(EXPORT_DIR / f"instances_{split}.json")
        for split in ("train", "val", "test")
    }
    stats["attribute"] = {
        "gold": count_coco(PROCESSED / "2_attribute/gold.json"),
    }
    stats["mask"] = {
        "gold": count_coco(PROCESSED / "3_mask/gold.json"),
    }
    stats["augmented"] = {
        stage: count_coco(PROCESSED / f"4_augmented/{stage}/annotations.json")
        for stage in ("detection", "attribute", "mask")
    }
    return stats


# ---------------- 绘图 ----------------


def plot_detection(det: dict, out: Path) -> list[str]:
    images: list[str] = []
    if det["epochs"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(det["epochs"], det["loss"], marker="o", label="loss（总）")
        for key, label in (
            ("loss_cls", "loss_cls"),
            ("loss_iou", "loss_iou"),
            ("loss_dfl", "loss_dfl"),
            ("loss_l1", "loss_l1"),
        ):
            if any(det[key]):
                ax.plot(det["epochs"], det[key], marker=".", alpha=0.8, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("人物检测：训练 loss 下降曲线（epoch 均值）")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "detection_loss.png", dpi=130)
        plt.close(fig)
        images.append("detection_loss.png")
    if det["ap5095"] or det["ap50"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        epochs = list(range(1, len(det["ap5095"]) + 1))
        if det["ap5095"]:
            ax.plot(epochs, det["ap5095"], marker="o", label="mAP@0.50:0.95")
        if det["ap50"]:
            ax.plot(list(range(1, len(det["ap50"]) + 1)), det["ap50"], marker="s", label="mAP@0.50")
        ax.set_xlabel("eval 序号")
        ax.set_ylabel("mAP")
        ax.set_title("人物检测：验证集 mAP")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "detection_map.png", dpi=130)
        plt.close(fig)
        images.append("detection_map.png")
    return images


def plot_attribute(attr: dict, out: Path) -> list[str]:
    if not attr["epoch"]:
        return []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(attr["epoch"], attr["train_loss"], marker="o", label="train_loss")
    ax.plot(attr["epoch"], attr["val_loss"], marker="s", label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(attr["epoch"], attr["val_macro_f1"], marker="^", color="green", label="val macro-F1")
    ax2.set_ylabel("macro-F1", color="green")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right")
    ax.set_title("人物属性：loss 与验证 macro-F1")
    fig.tight_layout()
    fig.savefig(out / "attribute_curves.png", dpi=130)
    plt.close(fig)
    return ["attribute_curves.png"]


def plot_mask(mask: dict, out: Path) -> list[str]:
    if not mask["epoch"]:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(mask["epoch"], mask["train_loss"], marker="o", color="tab:red")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train_loss")
    axes[0].set_title("口罩识别：训练 loss")
    axes[0].grid(alpha=0.3)
    for key, label in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("mAP50", "mAP@0.5"),
        ("mAP50-95", "mAP@0.5:0.95"),
    ):
        axes[1].plot(mask["epoch"], mask[key], marker=".", label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("口罩识别：验证指标")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "mask_curves.png", dpi=130)
    plt.close(fig)
    return ["mask_curves.png"]


# ---------------- 报告 ----------------


def fmt_pair(pair: tuple[int, int]) -> str:
    return f"{pair[0]} 图 / {pair[1]} 框"


def build_report(
    stats: dict,
    det: dict,
    attr: dict,
    mask: dict,
    det_images: list[str],
    attr_images: list[str],
    mask_images: list[str],
) -> str:
    det_best = max(det["ap5095"]) if det["ap5095"] else None
    det_best50 = max(det["ap50"]) if det["ap50"] else None
    attr_best = max(attr["val_macro_f1"]) if attr.get("val_macro_f1") else None
    mask_best = max(mask["mAP50"]) if mask.get("mAP50") else None

    det_train, det_val, det_test = (
        stats["detection"]["train"],
        stats["detection"]["val"],
        stats["detection"]["test"],
    )
    det_total = (
        sum(v[0] for v in stats["detection"].values()),
        sum(v[1] for v in stats["detection"].values()),
    )
    lines = [
        "# 微调训练报告",
        "",
        "## 数据概况（人工金标准）",
        "",
        "| 环节 | 金标准 | train | val | test | 尘土化增强副本 |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| ① 人物检测 | {fmt_pair(det_total)} | {fmt_pair(det_train)} | {fmt_pair(det_val)} "
        f"| {fmt_pair(det_test)} | {fmt_pair(stats['augmented']['detection'])} |",
        f"| ② 人物属性 | {fmt_pair(stats['attribute']['gold'])} | 按源图哈希划分 | - | - "
        f"| {fmt_pair(stats['augmented']['attribute'])} |",
        f"| ③ 口罩识别 | {fmt_pair(stats['mask']['gold'])}（含 AIZOO 原标签批量批注 800） "
        f"| 按源图哈希划分 | - | - | {fmt_pair(stats['augmented']['mask'])} |",
        "",
        "## 训练配置",
        "",
        "| 环节 | 模型 | 起点权重 | epochs | 输出 |",
        "| --- | --- | --- | --- | --- |",
        f"| ① 人物检测 | PP-YOLOE-S（PaddleDetection） | 官方 `mot_ppyoloe_s_36e_pipeline.pdparams` "
        f"| {len(det['epochs']) or '-'} | `models/finetuned/person_detector` |",
        f"| ② 人物属性 | PP-HGNet small 26 属性（PaddleClas） | 官方 `PPHGNet_small_person_attribute_954_infer` "
        f"| {len(attr.get('epoch', [])) or '-'} | `models/finetuned/person_attribute` |",
        f"| ③ 口罩识别 | YOLOv5s 2 类（vendor/yolov5） | 官方 `face_mask_detection.onnx` 反建权重 "
        f"| {len(mask.get('epoch', [])) or '-'} | `models/finetuned/face_mask` |",
        "",
        "## 训练曲线",
        "",
    ]
    for name in det_images + attr_images + mask_images:
        title = Path(name).stem
        lines.append(f"![{title}](report/{name})")
        lines.append("")
    lines += [
        "## 最终指标",
        "",
        "| 环节 | 指标 | 最佳值 |",
        "| --- | --- | --- |",
        f"| ① 人物检测 | mAP@0.50:0.95 | {f'{det_best:.4f}' if det_best is not None else '-'} |",
        f"| ① 人物检测 | mAP@0.50 | {f'{det_best50:.4f}' if det_best50 is not None else '-'} |",
        f"| ② 人物属性 | val macro-F1 | {f'{attr_best:.4f}' if attr_best is not None else '-'} |",
        f"| ③ 口罩识别 | mAP@0.50 | {f'{mask_best:.4f}' if mask_best is not None else '-'} |",
        "",
        "尘土化增强只放大训练划分，标签与源一致；验证/测试集不参与增强。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成训练曲线与报告")
    parser.add_argument("--det-log", type=Path)
    parser.add_argument("--attr-log", type=Path)
    parser.add_argument("--mask-log", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "TRAINING_REPORT.md")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    det = parse_detection_log(args.det_log) if args.det_log else {"epochs": [], "ap5095": [], "ap50": []}
    attr = (
        parse_simple_log(args.attr_log, ATTR_LINE, ("epoch", "train_loss", "val_loss", "val_macro_f1"))
        if args.attr_log
        else {}
    )
    mask = (
        parse_simple_log(
            args.mask_log, MASK_LINE, ("epoch", "train_loss", "precision", "recall", "mAP50", "mAP50-95")
        )
        if args.mask_log
        else {}
    )
    stats = dataset_stats()
    det_images = plot_detection(det, REPORT_DIR) if det["epochs"] or det["ap5095"] else []
    attr_images = plot_attribute(attr, REPORT_DIR) if attr.get("epoch") else []
    mask_images = plot_mask(mask, REPORT_DIR) if mask.get("epoch") else []
    report = build_report(stats, det, attr, mask, det_images, attr_images, mask_images)
    args.output.write_text(report, encoding="utf-8")
    print(f"报告已写入: {args.output}")
    print(f"曲线图: {[str(REPORT_DIR / name) for name in det_images + attr_images + mask_images]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
