#!/usr/bin/env python3
"""在 Paddle 3 PIR 进程中把属性训练权重导出为部署模型。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


if "--legacy" in sys.argv:
    # 旧 IR（inference.pdmodel）：paddle2onnx 转 ONNX 用这个格式最稳。
    # 注意不能用 FLAGS_enable_pir_api=0（会让 eager 的 sigmoid 收到旧 IR Variable 崩溃），
    # 3.3 的正路是 paddle.pir_utils.OldIrGuard（见下方导出段）。
    pass
else:
    os.environ.pop("FLAGS_enable_pir_api", None)

import paddle  # noqa: E402


class InferenceModel(paddle.nn.Layer):
    def __init__(self, model: paddle.nn.Layer):
        super().__init__()
        self.model = model

    def forward(self, image: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.sigmoid(self.model(image))


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 PP-HGNet_small 属性部署模型")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paddleclas-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="导出旧 IR（inference.pdmodel），供 paddle2onnx 转 ONNX 用",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.paddleclas_dir.resolve()))
    from ppcls.arch.backbone.legendary_models.pp_hgnet import PPHGNet_small

    model = PPHGNet_small(class_num=26)
    model.set_state_dict(paddle.load(str(args.checkpoint)))
    inference = InferenceModel(model)
    inference.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.legacy:
        with paddle.pir_utils.OldIrGuard():
            static = paddle.jit.to_static(
                inference,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, 3, 256, 192], dtype="float32", name="x")
                ],
            )
            paddle.jit.save(static, str(args.output_dir / "inference"))
    else:
        static = paddle.jit.to_static(
            inference,
            input_spec=[
                paddle.static.InputSpec(shape=[None, 3, 256, 192], dtype="float32", name="x")
            ],
        )
        paddle.jit.save(static, str(args.output_dir / "inference"))
    model_file = args.output_dir / ("inference.pdmodel" if args.legacy else "inference.json")
    params_file = args.output_dir / "inference.pdiparams"
    if not model_file.is_file() or not params_file.is_file():
        raise RuntimeError("Paddle 部署模型导出不完整")
    print(f"属性部署模型: {model_file}, {params_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
