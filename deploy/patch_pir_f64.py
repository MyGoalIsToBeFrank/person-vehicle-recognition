"""把 PIR model.json 里 nearest_interp 的 scale 属性从 a_f64 改成 a_f32。

paddle2onnx 的 pir_parser 用 GetOpAttr<vector<float>> 读 scale，要求元素是
ir::FloatAttribute；Paddle 3.3 导出时写成了 DoubleAttribute (a_f64)，导致
"(Unimplemented) the 0th elementwise MUST be ir::FloatAttribute"。
nearest_interp 的 scale 都是 2.0 这类可精确表示的值，f64->f32 无损。
参考: https://github.com/PaddlePaddle/Paddle/issues/77757
"""
import json, sys
from pathlib import Path

def patch(path: Path):
    m = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for region in m["program"]["regions"]:
        for block in region["blocks"]:
            for op in block["ops"]:
                if op.get("#") != "1.nearest_interp":
                    continue
                for attr in op.get("A", []):
                    at = attr.get("AT", {})
                    if at.get("#") == "0.a_array":
                        for el in at.get("D", []):
                            if el.get("#") == "0.a_f64":
                                el["#"] = "0.a_f32"
                                n += 1
    path.write_text(json.dumps(m, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{path}: patched {n} elements")

for p in sys.argv[1:]:
    patch(Path(p))
