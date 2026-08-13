"""从当前口罩 ONNX 恢复融合的、可求梯度的 YOLOv5s。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


def build_from_onnx(onnx_path: Path, yolov5_dir: Path, device: torch.device):
    sys.path.insert(0, str(yolov5_dir.resolve()))
    from models.yolo import Model

    model = Model(str(yolov5_dir / "models/yolov5s.yaml"), nc=2).fuse()
    graph = onnx.load(str(onnx_path))
    onnx_weights = {
        initializer.name: torch.from_numpy(numpy_helper.to_array(initializer).copy())
        for initializer in graph.graph.initializer
    }
    state = model.state_dict()
    missing = set(onnx_weights) - set(state)
    extras = set(state) - set(onnx_weights)
    if missing or extras != {"model.24.anchors"}:
        raise RuntimeError(f"ONNX/YOLOv5 参数结构不匹配: missing={missing}, extras={extras}")
    for name, value in onnx_weights.items():
        if list(value.shape) != list(state[name].shape):
            raise RuntimeError(f"ONNX 参数形状不匹配: {name}, {value.shape} != {state[name].shape}")
        state[name] = value
    model.load_state_dict(state)
    # YOLOv5 v6.2 的 fuse() 在新 PyTorch 下会留下非叶参数；非叶参数不会收到梯度。
    # 复制成普通 Parameter 后，模型数值不变，但所有 120 个权重都可继续训练。
    for module in model.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is not None:
                module._parameters[name] = torch.nn.Parameter(parameter.detach().clone())
    model.names = ["w/o mask", "w/ mask"]
    model.nc = 2
    model.cpu().eval()

    torch.manual_seed(20260813)
    probe = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        pytorch_output = model(probe)[0].numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {session.get_inputs()[0].name: probe.numpy()})[0]
    maximum_error = float(np.max(np.abs(pytorch_output - onnx_output)))
    mean_error = float(np.mean(np.abs(pytorch_output - onnx_output)))
    if maximum_error > 0.005 or mean_error > 1e-4:
        raise RuntimeError(
            f"恢复模型没有复现 ONNX 输出: max={maximum_error}, mean={mean_error}"
        )
    print(
        "口罩权重恢复: 120/120 参数，"
        f"零训练误差 max={maximum_error:.3g}, mean={mean_error:.3g}"
    )
    return model.to(device)
