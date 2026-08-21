# 项目源码

- GitHub：<https://github.com/MyGoalIsToBeFrank/person-vehicle-recognition>
- HTTPS：`https://github.com/MyGoalIsToBeFrank/person-vehicle-recognition.git`
- SSH：`git@github.com:MyGoalIsToBeFrank/person-vehicle-recognition.git`

仓库公开后，接收方无需申请权限即可浏览或克隆：

```bash
git clone https://github.com/MyGoalIsToBeFrank/person-vehicle-recognition.git
cd person-vehicle-recognition
git rev-parse HEAD
```

GitHub 只包含代码、配置、依赖锁和文档，不包含 Docker 镜像、原始数据集、模型权重、
TensorRT engine 或第三方源码树。直接部署请使用交接包内的 Docker 镜像；重新构建或微调
还需要另外获得有权使用的 `models/`、`vendor/` 和数据集。

交接包内的 `BUILD_INFO.txt` 记录打包时的 Git commit。首次接手请依次阅读：

1. `README.md`：功能、架构、数据标注、训练、导出和推理优化总览；
2. `DEPLOY.md`：镜像加载、API、视频输入、多卡部署、验收和排障；
3. `finetune/README.md`：二次标注、离线增强和模型微调；
4. `models/MODEL_SOURCES.md`：模型来源、哈希及再分发边界。

公开仓库并不改变第三方数据、模型权重和样例图片各自的许可证或授权范围。不要把密码、
PAT、SSH 私钥、生产主机地址或其他凭据提交到仓库。
