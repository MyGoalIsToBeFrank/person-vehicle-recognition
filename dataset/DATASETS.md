# 数据源清单

本项目只在内部研发范围内使用这些数据。原始压缩包和解压图片位于 `raw/`，
程序不会修改它们；模型候选写入 `processed/candidates.json`，人工保存的最终结果只写入
`processed/gold_labels.json` 和 `processed/gold_images/`。只有后两者参与训练。

| 数据集 | 本项目用途 | 官方来源 | 当前状态 |
|---|---|---|---|
| AIZOO FaceMaskDataset | 口罩/未戴口罩检测 | https://github.com/AIZOOTech/FaceMaskDetection | 已下载官方 Google Drive 包；归档 SHA-256 `01680ec050d3562759e6ff28c366f116bdb358a76304226ed8733a7307a75baf` |
| PRW v2016.04.20 | 室外固定摄像头全景和人员候选 | https://zheng-lab-anu.github.io/Project/project_prw.html | 已从 Kaggle/OpenDataLab 镜像下载并解压；11,816 张全景及 11,816 个对应标注文件 |
| MSP60K | 室外场景属性补充 | https://github.com/Event-AHU/OpenPAR/tree/main/MSP60K_Benchmark_Dataset | 已锁定官方 Google Drive 包；下载后仅使用经人工确认的 Outdoors 固定摄像头样本 |
| PA-100K | 26 属性原能力保持 | https://github.com/xh-liu/HydraPlus-Net | 作为保持集，不把它当作业务域改善证据 |

AIZOO 上游页面称 7,971 张图片，而下载包内 `readme.md` 写明 train 6,120、val
1,839，共 7,959 张；本项目以实际下载包内容为准，并保留该差异记录。
