"""从当前 PP-HGNet_small 部署权重恢复同架构可训练模型。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("FLAGS_enable_pir_api", "0")
import paddle  # noqa: E402


def build_from_deployment(
    deployment_dir: Path,
    paddleclas_dir: Path,
    device: str,
) -> paddle.nn.Layer:
    """完整映射 218 个参数，并证明零训练输出与部署模型一致。"""
    paddle.set_device(device.lower())
    source = paddle.jit.load(
        str(deployment_dir),
        model_filename="inference.pdmodel",
        params_filename="inference.pdiparams",
    )
    source.eval()
    source_parameters = {parameter.name: parameter for parameter in source.parameters()}

    sys.path.insert(0, str(paddleclas_dir.resolve()))
    from ppcls.arch.backbone.legendary_models.pp_hgnet import PPHGNet_small

    model = PPHGNet_small(class_num=26)
    recovered = {}
    for state_name, parameter in model.state_dict().items():
        generic_name = _deployment_parameter_name(state_name, parameter.name)
        if generic_name not in source_parameters:
            raise RuntimeError(f"部署权重缺少参数: {generic_name} ({state_name})")
        source_parameter = source_parameters[generic_name]
        if list(parameter.shape) != list(source_parameter.shape):
            raise RuntimeError(
                f"参数形状不匹配: {state_name}, {list(parameter.shape)} != "
                f"{list(source_parameter.shape)}"
            )
        recovered[state_name] = source_parameter
    if len(recovered) != 218 or len(source_parameters) != 218:
        raise RuntimeError(
            f"参数数量不匹配: recovered={len(recovered)}, source={len(source_parameters)}"
        )
    model.set_state_dict(recovered)
    model.eval()

    paddle.seed(20260813)
    probe = paddle.randn([2, 3, 256, 192], dtype="float32")
    with paddle.no_grad():
        expected = source(probe)
        if isinstance(expected, (list, tuple)):
            expected = expected[0]
        actual = paddle.nn.functional.sigmoid(model(probe))
        maximum_error = float(paddle.max(paddle.abs(expected - actual)))
    if maximum_error > 1e-6:
        raise RuntimeError(f"恢复模型没有复现部署输出，最大误差={maximum_error}")
    print(f"属性权重恢复: 218/218 参数，零训练最大误差={maximum_error:.3g}")
    return model


def _deployment_parameter_name(state_name: str, runtime_name: str) -> str:
    layer_name = runtime_name.rsplit(".", 1)[0]
    if layer_name.startswith("batch_norm2d_"):
        if state_name.endswith("._mean"):
            suffix = "w_1"
        elif state_name.endswith("._variance"):
            suffix = "w_2"
        elif state_name.endswith(".weight"):
            suffix = "w_0"
        elif state_name.endswith(".bias"):
            suffix = "b_0"
        else:
            raise RuntimeError(f"未知 BN 参数: {state_name}")
    elif state_name.endswith(".weight"):
        suffix = "w_0"
    elif state_name.endswith(".bias"):
        suffix = "b_0"
    else:
        raise RuntimeError(f"未知参数: {state_name}")
    return f"{layer_name}.{suffix}"


class AttributeInferenceModel(paddle.nn.Layer):
    """为部署接口补回原模型自带的 sigmoid。"""

    def __init__(self, model: paddle.nn.Layer):
        super().__init__()
        self.model = model

    def forward(self, image: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.sigmoid(self.model(image))
