#!/usr/bin/env python3
"""Build the immutable ONNX model set from existing deployment weights.

This command never imports a training dataset and never updates a checkpoint.
Every output is checked, hashed, and described in ``manifest.json``.  Existing
dynamic ONNX files are accepted only where the upstream framework cannot
reconstruct a deploy graph from the shipped inference parameters (the official
PP-Vehicle detector); their source Paddle weights are still part of the hash
chain recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx


OPSET = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def paddle_to_onnx(model: Path, params: Path, output: Path, root: Path) -> None:
    run(
        [
            sys.executable,
            "-m",
            "paddle2onnx.command",
            "--model_dir",
            str(model.parent),
            "--model_filename",
            model.name,
            "--params_filename",
            params.name,
            "--save_file",
            str(output),
            "--opset_version",
            str(OPSET),
            "--enable_auto_update_opset",
            "True",
            "--enable_onnx_checker",
            "True",
            "--optimize_tool",
            "None",
        ],
        cwd=root,
    )


def dimensions(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_param:
            result.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        else:
            result.append(None)
    return result


def inspect_model(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    inputs = {value.name: dimensions(value) for value in model.graph.input}
    outputs = {value.name: dimensions(value) for value in model.graph.output}
    return {
        "opset": max(item.version for item in model.opset_import if not item.domain),
        "inputs": inputs,
        "outputs": outputs,
        "operators": sorted({node.op_type for node in model.graph.node}),
    }


def canonicalize_squeeze_axes(path: Path) -> int:
    """Expose constant Squeeze axes directly to TensorRT's ONNX parser.

    Paddle2ONNX 2.1 sometimes places an Identity between a Constant axes
    tensor and Squeeze.  The graph is valid ONNX, but TensorRT 10.5 treats the
    axes as a runtime shape tensor and rejects it.  Repointing only that input
    is semantics-preserving and keeps the original nodes for auditability.
    """
    model = onnx.load(path)
    producers = {name: node for node in model.graph.node for name in node.output}
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Squeeze" or len(node.input) < 2:
            continue
        axes = node.input[1]
        visited: set[str] = set()
        while axes not in visited:
            visited.add(axes)
            producer = producers.get(axes)
            if producer is None:
                break
            if producer.op_type == "Constant":
                if axes != node.input[1]:
                    node.input[1] = axes
                    rewritten += 1
                break
            if producer.op_type != "Identity" or len(producer.input) != 1:
                break
            axes = producer.input[0]
    if rewritten:
        onnx.checker.check_model(model)
        onnx.save(model, path)
    return rewritten


def require_dynamic_batch(name: str, contract: dict[str, Any]) -> None:
    for tensor, shape in contract["inputs"].items():
        if not shape or isinstance(shape[0], int) and shape[0] > 0:
            raise RuntimeError(f"{name}:{tensor} does not have dynamic batch: {shape}")


def require_raw_detector(name: str, contract: dict[str, Any]) -> None:
    forbidden = [
        operator
        for operator in contract["operators"]
        if "nms" in operator.lower() or operator.lower() == "nonmaxsuppression"
    ]
    if forbidden:
        raise RuntimeError(f"{name} contains NMS operators: {forbidden}")
    shapes = list(contract["outputs"].values())
    boxes = [shape for shape in shapes if len(shape) == 3 and shape[-1] == 4]
    scores = [
        shape
        for shape in shapes
        if len(shape) == 3 and shape[-1] != 4 and 8400 in shape
    ]
    if len(boxes) != 1 or len(scores) != 1:
        raise RuntimeError(f"{name} raw detector outputs are invalid: {shapes}")


def copy_verified(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, target)


def sources(root: Path, relative_paths: list[str]) -> list[dict[str, str]]:
    records = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": relative, "sha256": sha256(path)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    specs: dict[str, dict[str, Any]] = {
        "person_detector": {
            "source": [
                "models/finetuned/person_detector_checkpoints/model_final.pdparams",
                "models/finetuned/person_detector_checkpoints/model_final.pdstates",
            ],
            "preprocess": {
                "color": "RGB",
                "resize": [640, 640],
                "scale": "none",
                "value_range": [0, 255],
                "note": "v1 deployment contract; normalization is embedded in neither runtime nor ONNX",
            },
            "profiles": {
                "image": [[1, 3, 640, 640], [32, 3, 640, 640], [64, 3, 640, 640]],
                "scale_factor": [[1, 2], [32, 2], [64, 2]],
            },
            "semantics": "raw decoded xyxy boxes and one-class scores; GPU NMS follows",
        },
        "vehicle_detector": {
            "source": [
                "models/vehicle/mot_ppyoloe_s_36e_ppvehicle/model.pdmodel",
                "models/vehicle/mot_ppyoloe_s_36e_ppvehicle/model.pdiparams",
                "models/onnx/vehicle_detector/vehicle_detector.onnx",
            ],
            "preprocess": {
                "color": "RGB",
                "resize": [640, 640],
                "scale": "none",
                "value_range": [0, 255],
                "note": "v1 deployment contract; normalization is embedded in neither runtime nor ONNX",
            },
            "profiles": {
                "image": [[1, 3, 640, 640], [32, 3, 640, 640], [64, 3, 640, 640]],
                "scale_factor": [[1, 2], [32, 2], [64, 2]],
            },
            "semantics": "raw decoded xyxy boxes and one-class scores; GPU NMS follows",
        },
        "person_attribute": {
            "source": [
                "models/finetuned/person_attribute/best.pdparams",
                "models/finetuned/person_attribute/inference.json",
                "models/finetuned/person_attribute/inference.pdiparams",
            ],
            "preprocess": {
                "color": "RGB",
                "resize": [256, 192],
                "scale": "1/255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "profiles": {"x": [[1, 3, 256, 192], [32, 3, 256, 192], [128, 3, 256, 192]]},
            "semantics": "26 sigmoid probabilities in PaddleDetection person-attribute order",
        },
        "vehicle_attribute": {
            "source": [
                "models/vehicle/vehicle_attribute_model/model.pdmodel",
                "models/vehicle/vehicle_attribute_model/model.pdiparams",
                "models/onnx/vehicle_attribute/vehicle_attribute.onnx",
            ],
            "preprocess": {
                "color": "RGB",
                "resize": [192, 256],
                "scale": "1/255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "profiles": {"x": [[1, 3, 192, 256], [32, 3, 192, 256], [128, 3, 192, 256]]},
            "semantics": "10 color and 9 vehicle-type probabilities",
        },
        "face_mask": {
            "source": [
                "models/finetuned/face_mask/best.pt",
                "models/finetuned/face_mask/face_mask_detection_dynamic.onnx",
            ],
            "preprocess": {"color": "RGB", "letterbox": [640, 640], "scale": "1/255"},
            "profiles": {"images": [[1, 3, 640, 640], [32, 3, 640, 640], [128, 3, 640, 640]]},
            "semantics": "YOLO xywh, objectness, no-mask and mask probabilities",
        },
        "plate_det": {
            "source": [
                "models/vehicle/plate_det/inference.pdmodel",
                "models/vehicle/plate_det/inference.pdiparams",
            ],
            "preprocess": {
                "color": "BGR",
                "bucket": [736, 736],
                "scale": "1/255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "profiles": {"x": [[1, 3, 736, 736], [32, 3, 736, 736], [64, 3, 736, 736]]},
            "semantics": "PP-OCRv3 DB text probability map",
        },
        "plate_rec": {
            "source": [
                "models/vehicle/plate_rec/inference.pdmodel",
                "models/vehicle/plate_rec/inference.pdiparams",
                "models/vehicle/plate_rec/rec_word_dict.txt",
            ],
            "preprocess": {
                "color": "BGR",
                "resize_pad": [48, 320],
                "scale": "1/255",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "profiles": {"x": [[1, 3, 48, 320], [32, 3, 48, 320], [128, 3, 48, 320]]},
            "semantics": "PP-OCRv3 CTC probabilities; blank index zero",
        },
    }

    with tempfile.TemporaryDirectory(prefix="pvr-export-") as temporary_name:
        temporary = Path(temporary_name)
        # Reconstruct the fine-tuned detector from its existing final checkpoint.
        run(
            [
                sys.executable,
                "tools/export_model.py",
                "-c",
                str(root / "model_export/person_detector.yml"),
                "--output_dir",
                str(temporary / "paddle"),
                "-o",
                f"weights={root / 'models/finetuned/person_detector_checkpoints/model_final'}",
                "use_gpu=false",
            ],
            cwd=root / "vendor/PaddleDetection",
        )
        candidates = list((temporary / "paddle").rglob("model.json"))
        if not candidates:
            candidates = list((temporary / "paddle").rglob("model.pdmodel"))
        if len(candidates) != 1:
            raise RuntimeError(f"unexpected person detector exports: {candidates}")
        paddle_to_onnx(
            candidates[0], candidates[0].with_name("model.pdiparams"),
            output / "person_detector.onnx", root,
        )

        paddle_to_onnx(
            root / "models/finetuned/person_attribute/inference.json",
            root / "models/finetuned/person_attribute/inference.pdiparams",
            output / "person_attribute.onnx", root,
        )
        # PP-Vehicle's released inference archive does not include a training
        # checkpoint. Its already validated raw-output ONNX is the lossless
        # conversion of the two Paddle deployment files listed in the manifest.
        copy_verified(root / "models/onnx/vehicle_detector/vehicle_detector.onnx",
                      output / "vehicle_detector.onnx")
        copy_verified(root / "models/onnx/vehicle_attribute/vehicle_attribute.onnx",
                      output / "vehicle_attribute.onnx")
        copy_verified(root / "models/finetuned/face_mask/face_mask_detection_dynamic.onnx",
                      output / "face_mask.onnx")
        paddle_to_onnx(
            root / "models/vehicle/plate_det/inference.pdmodel",
            root / "models/vehicle/plate_det/inference.pdiparams",
            output / "plate_det.onnx", root,
        )
        paddle_to_onnx(
            root / "models/vehicle/plate_rec/inference.pdmodel",
            root / "models/vehicle/plate_rec/inference.pdiparams",
            output / "plate_rec.onnx", root,
        )

    copy_verified(root / "models/vehicle/plate_rec/rec_word_dict.txt",
                  output / "rec_word_dict.txt")
    graph_rewrites = {
        name: {"constant_squeeze_axes": canonicalize_squeeze_axes(output / f"{name}.onnx")}
        for name in specs
    }
    artifacts: dict[str, Any] = {}
    for name, spec in specs.items():
        onnx_path = output / f"{name}.onnx"
        contract = inspect_model(onnx_path)
        require_dynamic_batch(name, contract)
        if name in {"person_detector", "vehicle_detector"}:
            require_raw_detector(name, contract)
        artifacts[name] = {
            **spec,
            "source_files": sources(root, spec.pop("source")),
            "graph_rewrites": graph_rewrites[name],
            "onnx": {
                "file": onnx_path.name,
                "sha256": sha256(onnx_path),
                **contract,
            },
        }
    manifest = {
        "schema_version": 1,
        "training_performed": False,
        "opset_requested": OPSET,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"exported {len(artifacts)} models to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
