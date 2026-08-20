#!/usr/bin/env python3
"""把旧版 pdmodel 检测图的 multiclass_nms3 尾巴切掉，让 paddle2onnx 能转换。

背景：paddle2onnx 2.x 的 PIR 解析器处理不了 PP-YOLOE 部署图里 NMS 区域的
某个 elementwise 算子（报 "elementwise MUST be ir::FloatAttribute"）。
NMS 之前的图（backbone + neck + head + 坐标解码）是标准算子，可以正常转换，
所以把 NMS 子图裁掉、把解码后的框和分数直接作为输出，NMS 改到推理侧用
numpy 实现（阈值等参数原样抄自被裁算子的属性，见输出目录的 nms_config.json）。

用法：

    .venv-train/Scripts/python.exe deploy/strip_detector_nms.py \
        --model-dir models/vehicle/mot_ppyoloe_s_36e_ppvehicle \
        --output-dir logs/onnx_stage/vehicle_detector_nonms
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from paddle.base.proto import framework_pb2


def strip_nms(model_dir: Path, output_dir: Path, cut_before_op: int | None = None,
              fetch_tensors: tuple[str, ...] | None = None) -> None:
    program = framework_pb2.ProgramDesc()
    program.ParseFromString((model_dir / "model.pdmodel").read_bytes())
    block = program.blocks[0]

    if cut_before_op is not None:
        # 深裁模式：按算子索引截断，直接 fetch 指定的中间张量（如三级 concat 的
        # 原始分数/回归量），解码和 NMS 全部挪到推理侧 numpy。
        if not fetch_tensors:
            raise ValueError("深裁模式必须显式给出 --fetch 张量名")
        nms_index = cut_before_op
        boxes_name, scores_name = fetch_tensors
        nms_config = {}
    else:
        nms_index = next(
            (i for i, op in enumerate(block.ops) if op.type == "multiclass_nms3"), None
        )
        if nms_index is None:
            raise RuntimeError("图里没有 multiclass_nms3，这个脚本不适用")
        nms = block.ops[nms_index]
        boxes_name = next(v for v in nms.inputs if v.parameter == "BBoxes").arguments[0]
        scores_name = next(v for v in nms.inputs if v.parameter == "Scores").arguments[0]

    # NMS 参数原样带走，推理侧的 numpy NMS 要和它对齐。
    nms_op = next((op for op in block.ops if op.type == "multiclass_nms3"), None)
    if not nms_config and nms_op is not None:
        for attr in nms_op.attrs:
            if attr.HasField("f"):
                nms_config[attr.name] = attr.f
            elif attr.HasField("i"):
                nms_config[attr.name] = attr.i
            elif attr.HasField("b"):
                nms_config[attr.name] = attr.b

    # 裁掉切点及其后的所有算子，换成指向目标张量的两个 fetch。
    del block.ops[nms_index:]
    for column, tensor_name in enumerate((boxes_name, scores_name)):
        fetch = block.ops.add()
        fetch.type = "fetch"
        fetch_input = fetch.inputs.add()
        fetch_input.parameter = "X"
        fetch_input.arguments.append(tensor_name)
        fetch_output = fetch.outputs.add()
        fetch_output.parameter = "Out"
        fetch_output.arguments.append("fetch")
        attr = fetch.attrs.add()
        attr.name = "col"
        attr.type = framework_pb2.AttrType.INT
        attr.i = column

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.pdmodel").write_bytes(program.SerializeToString())
    shutil.copy2(model_dir / "model.pdiparams", output_dir / "model.pdiparams")
    (output_dir / "nms_config.json").write_text(
        json.dumps(nms_config, indent=2), encoding="utf-8"
    )
    print(f"切除点: 第 {nms_index} 个算子")
    print(f"输出: boxes={boxes_name}, scores={scores_name}")
    if nms_config:
        print(f"NMS 参数: {nms_config}")
    print(f"完成: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="切除 pdmodel 检测图的 NMS 尾巴")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cut-before-op", type=int, help="深裁：从该算子索引处截断")
    parser.add_argument("--fetch", nargs=2, metavar=("BOXES", "SCORES"),
                        help="深裁模式下要 fetch 的两个张量名（框、分数）")
    args = parser.parse_args()
    strip_nms(
        args.model_dir,
        args.output_dir,
        cut_before_op=args.cut_before_op,
        fetch_tensors=tuple(args.fetch) if args.fetch else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
