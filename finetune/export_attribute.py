#!/usr/bin/env python3
"""在 Paddle 3 PIR 进程中把属性训练权重导出为部署模型。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
    args = parser.parse_args()

    sys.path.insert(0, str(args.paddleclas_dir.resolve()))
    from ppcls.arch.backbone.legendary_models.pp_hgnet import PPHGNet_small

    model = PPHGNet_small(class_num=26)
    model.set_state_dict(paddle.load(str(args.checkpoint)))
    inference = InferenceModel(model)
    inference.eval()
    static = paddle.jit.to_static(
        inference,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, 256, 192], dtype="float32", name="x")
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paddle.jit.save(static, str(args.output_dir / "inference"))
    model_file = args.output_dir / "inference.json"
    params_file = args.output_dir / "inference.pdiparams"
    if not model_file.is_file() or not params_file.is_file():
        raise RuntimeError("Paddle 部署模型导出不完整")
    print(f"属性部署模型: {model_file}, {params_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
