# 模型来源与完整性索引

本目录只保存 `independent` 运行所需的模型。运行时不会自动下载或替换权重。

来源核验日期为 2026-08-13。核验方法是重新下载上游归档，计算归档 SHA-256，并逐个比较归档内模型文件与本地文件的 SHA-256；所有当前模型均匹配。口罩模型最初于 2026-08-11 下载并完成同样的归档与文件校验。

| 用途 | 本地目录 | 上游归档 SHA-256 | 详细记录 |
|---|---|---|---|
| 行人检测 | `human/mot_ppyoloe_s_36e_pipeline` | `33221117050792b9fe36e13736f7c7d7880ff3ec0f6f72c872bde4b830a2bef7` | [SOURCE.md](human/mot_ppyoloe_s_36e_pipeline/SOURCE.md) |
| 行人属性 | `human/PPHGNet_small_person_attribute_954_infer` | `c3fa8647118f24697fddc2be1bbe9749d2c5669c7ea1e79f699af1dee89d406a` | [SOURCE.md](human/PPHGNet_small_person_attribute_954_infer/SOURCE.md) |
| 车辆检测 | `vehicle/mot_ppyoloe_s_36e_ppvehicle` | `63ccc458856e30574b8f87f104485100502ffaa651de97ee577190e903ea115d` | [SOURCE.md](vehicle/mot_ppyoloe_s_36e_ppvehicle/SOURCE.md) |
| 车辆属性 | `vehicle/vehicle_attribute_model` | `3e687207dfac519e76351cd8e08716e15a075f61e4b2ba0f3d4d2abbe9627192` | [SOURCE.md](vehicle/vehicle_attribute_model/SOURCE.md) |
| 车牌检测、分类、识别 | `vehicle/.hyperlpr3/20230229/onnx` | `ce1cb895dc754a1bf6b50f99f4d745c0a8e1bcd5fecd02ca9b66ad7c24dae15e` | [SOURCE.md](vehicle/.hyperlpr3/20230229/onnx/SOURCE.md) |
| 口罩检测 | `face_mask_yolov5` | `550a3098d4d8b793172eaf78ac0c942a017ec4577c3ad523218ce63c68233b9c` | [SOURCE.md](face_mask_yolov5/SOURCE.md) |

## 文件格式

- `.pdmodel`：Paddle 旧格式静态推理网络结构。
- `.pdiparams`：与对应 `.pdmodel` 配套的参数。
- `.onnx`：ONNX 推理模型。
- `synset.txt`：口罩模型的原始类别顺序。

`.pdmodel` 与 `.pdiparams` 必须成对使用，不能跨模型目录混配。SHA-256 不一致时应停止使用，不要尝试“兼容”未知权重。

## 授权边界

- PaddleDetection 源码仓库使用 Apache License 2.0；这四个官方模型归档没有附带单独的权重许可证文件。
- HyperLPR GitHub 仓库和 PyPI `hyperlpr3==0.1.3` 元数据标明 Apache License 2.0；`20230229.zip` 没有附带单独的逐权重许可证文件。
- DJL 口罩模型包没有附带许可证文件，不能仅凭 DJL 项目本身的许可证推定该权重可商业分发。

以上是来源事实记录，不是法律意见，也不替代对训练数据权利、商用范围或再分发要求的单独审查。

## 本地复核

在项目根目录运行：

```powershell
Get-ChildItem models -File -Recurse |
  Where-Object Extension -In '.pdmodel', '.pdiparams', '.onnx', '.txt' |
  Get-FileHash -Algorithm SHA256
```

输出应与各模型目录中的 `SOURCE.md` 一致。
