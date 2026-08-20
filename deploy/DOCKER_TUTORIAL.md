# A30 / Ampere Docker 部署手册

本镜像运行单实例原生 C++/CUDA/TensorRT worker，FastAPI 只提供异步 HTTP 边界。构建过程不会训练模型，也不会修改 `finetune/`。Paddle 只存在于 ONNX 导出阶段；最终镜像不含 Paddle、ONNX Runtime、DALI 或 Python 多进程模型实例。

## 1. 主机要求

- Linux x86_64、Docker 24+、BuildKit、NVIDIA Container Toolkit。
- NVIDIA 驱动能够运行 CUDA 12.6 容器；先确认 `nvidia-smi` 正常。
- 目标生产卡为 A30 24GB。RTX 3080 Ti 仅用于功能、稳定性和相对优化预验。
- 首次构建建议预留 25GB 空间；先用 `docker system df` 判断，不要无差别删除服务器上的镜像。

验证 GPU 透传：

```bash
docker run --rm --gpus all \
  m.daocloud.io/docker.io/nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi
```

## 2. 构建镜像

代码仓库不保存模型权重和 `vendor/PaddleDetection`：二者体积较大，并且模型权重在外部分发前需要单独确认许可。服务器推荐使用固定工作区 `/home/ubuntu/pvr-src`，Git 管代码、忽略文件保存本机资产。首次建立：

仓库为私有仓库时，先把服务器公钥作为只读 Deploy key 添加到 GitHub 仓库的 `Settings → Deploy keys`，不要把个人访问令牌写入服务器文件。然后执行：

```bash
GIT_SSH_COMMAND='ssh -i /home/ubuntu/.ssh/pvr_github_deploy_ed25519 -o IdentitiesOnly=yes' \
  git clone git@github.com:MyGoalIsToBeFrank/person-vehicle-recognition.git \
  /home/ubuntu/pvr-src

git -C /home/ubuntu/pvr-src config core.sshCommand \
  'ssh -i /home/ubuntu/.ssh/pvr_github_deploy_ed25519 -o IdentitiesOnly=yes'

mkdir -p /home/ubuntu/pvr-src/models /home/ubuntu/pvr-src/vendor
cp -a /home/ubuntu/docker-ctx/models/. /home/ubuntu/pvr-src/models/
cp -a /home/ubuntu/docker-ctx/vendor/. /home/ubuntu/pvr-src/vendor/
```

以后直接在服务器更新和构建：

```bash
cd /home/ubuntu/pvr-src
git pull --ff-only origin main

DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f deploy/Dockerfile \
  -t person-vehicle-recognition:v2.0.1 .
```

`models/` 中的构建权重和 `vendor/` 是被 Git 忽略的本机资产，普通 `git pull` 不会删除它们。不要在这个工作区执行 `git clean -fdx`，否则会清除这些被忽略的构建输入。

在仓库根目录执行：

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f deploy/Dockerfile \
  -t person-vehicle-recognition:v2.0.0 .
```

唯一的 `deploy/Dockerfile` 分三段：

1. Paddle 3.2.1 从现有 checkpoint/部署权重导出并检查 7 个动态 ONNX；
2. CUDA 12.6 / TensorRT 10.5 / cuDNN 9.5 builder 编译 C++20/CUDA 扩展；
3. CUDA 12.6.3 runtime 安装锁定的 HTTP/客户端环境并复制必要动态库。

Python 包只由固定版本 `uv 0.11.2` 处理，默认使用清华 PyPI；apt 使用中科大镜像。Dockerfile 不调用 `pip`。

## 3. 首次启动与 engine cache

```bash
docker volume create pvr-engine-cache

docker run -d --name pvr-v2 --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0
```

TensorRT engine 不烘焙进镜像。首次启动会先逐个构建 7 个 engine，再一次性创建常驻执行 context，避免前面的 context 挤占显存、迫使后面的模型跳过可用 tactic。builder 固定最高优化级别 5；12GB 卡允许 6GiB workspace，24GB A30 允许 12GiB。

稳定输入/输出地址上的批次 `1/8/16/32/64/128` 会在首次执行后捕获 CUDA Graph。若某个 TensorRT tactic 不支持捕获，该批次自动回退普通 `enqueueV3`；这项运行时优化不改变 engine 文件，也不影响跨机器缓存隔离。

运行时不会让 7 个 context 各自静态占满最大 profile 激活内存：两个并发检测器各使用独立池，严格串行的属性、口罩和 PP-OCRv3 det/rec 共享一个按最大需求分配的池。实际池大小通过 `/v1/health` 的 `trt_context_memory_pool_bytes` 暴露。

首次构建可能持续几十分钟。此时服务进程存活，但：

```bash
curl -i http://127.0.0.1:8000/v1/health
# HTTP 503，ready=false

docker logs -f pvr-v2
```

完成后 `/v1/health` 返回 200 且 `ready=true`。缓存键包含 GPU 名称、SM、TensorRT 精确补丁版本、精度、builder 优化级别、workspace、ONNX SHA-256 和输入 profile；3080 Ti 的 engine 不会在 A30 上复用。更新模型、profile 或运行时后会生成新的键。

重启验证：

```bash
docker restart pvr-v2
curl -fsS http://127.0.0.1:8000/v1/health
```

命中缓存时应快速 ready。不要把 `/var/cache/pvr` 放进共享镜像；每台机器挂载自己的持久卷。

## 4. 异步调用

单图字段名固定为 `file`。提交和取结果是两个独立请求：

```bash
curl -sS -F 'file=@sample.jpg;type=image/jpeg' \
  http://127.0.0.1:8000/v1/tasks
# HTTP 202: {"session_id":"...","status":"pending"}

curl -sS http://127.0.0.1:8000/v1/tasks/<session_id>
# pending / running / done / error
```

高吞吐调用必须使用 `POST /v1/task-batches`，Content-Type 为 `application/vnd.pvr.tasks-v1`，每批最多 64 张；使用 `POST /v1/results:batch` 一次查询最多 512 个 ID。协议详见 `service/README.md`。

这不是“异步接口但后台无限堆积”：原生队列同时限制 8192 图和 1GiB，结果限制 60 秒、1GiB 和 262144 条；队列满立即返回 429 和 `Retry-After`。HTTP 请求体摄入也有独立并发闸门，最坏 64MiB 批请求不能无界并发占用内存。

常见状态码：

- 413：单图、批请求、图片数或像素数超限；
- 415：格式或批协议错误；
- 429：队列满，调用方应退避或丢弃低价值帧；
- 503：engine 尚未 ready；
- 410：结果已经过期。

## 5. 视频抽帧客户端

推理容器默认只运行服务。采集进程使用同一镜像独立运行，便于按摄像头横向分片：

```bash
docker run --rm --network host \
  -v "$PWD/output:/output" \
  person-vehicle-recognition:v2.0.0 \
  video-client \
  --camera-id gate-1 \
  --source 'rtsp://user:password@camera/stream' \
  --sample-fps 2 \
  --server http://127.0.0.1:8000 \
  --output /output/gate-1.jsonl
```

客户端按批提交并批量轮询；429 会计为 dropped，不会把帧无限留在进程里。

## 6. 性能验收

挂载真实 JPEG 目录：

```bash
docker run --rm --network host \
  -v /data/pvr-jpegs:/images:ro \
  person-vehicle-recognition:v2.0.0 \
  benchmark \
  --server http://127.0.0.1:8000 \
  --input-dir /images \
  --batch-size 64 \
  --duration 600 \
  --rate 0 \
  --drain-timeout 120
```

分别扫描 batch `1/8/16/32/64`。正式可发布吞吐是连续 10 分钟内同时满足以下条件的最高完成速率：

- 有效图片推理错误为 0；
- 完成/接受至少 99.9%；
- p95 总延迟不超过 1 秒；
- 无 OOM；
- 稳定负载没有 429。

再用两倍输入负载运行 10 分钟，确认快速 429、内存有界以及降载后恢复。必须区分 accepted throughput 与 completed throughput，不能用 HTTP 提交速度冒充完整识别速度。

A30 上要重新构建 engine 并重新跑全部测试。实例数采用：

```text
ceil(目标图片/s / (A30 单实例实测完成图片/s × 0.7))
```

1000/2000/3000 图片/s 都应代入真实 A30 结果计算，不能预先宣称单卡达到数千张/s。

现有 43 图端到端回归（旧车牌文本不参与比较，新 PP-OCRv3 结果完整写入报告）：

```bash
docker run --rm --network host \
  -v /data/easy_test:/images:ro \
  -v /data/result_v1_baseline.json:/baseline.json:ro \
  -v "$PWD/output:/output" \
  person-vehicle-recognition:v2.0.0 \
  regression --input-dir /images --baseline /baseline.json \
  --output /output/regression.json
```

### 本次 RTX 3080 Ti 预验记录（2026-08-20）

以下数字只证明镜像、接口、边界和相对性能，不是 A30 发布吞吐，也没有满足正式的连续 10 分钟口径：

- 正式镜像 ID：`sha256:9aebf7312a1607e07d1c184ec8bebc288488a254dd382d5c485f770b3adc7439`；原生扩展 SHA-256：`6b800a5f978e08b3c96a4407d085b6d1ff42fd818512ed6083c9855001048ede`。
- 7 个 RTX 3080 Ti 专用 FP16 engine 均命中缓存；三个常驻 TensorRT context 池合计 `4,810,759,168` 字节，服务空闲时 GPU 占用约 5.0GiB。
- 单图提交在 3.23ms 返回 `202 + pending`，紧接着查询为 `running`，最终约 241ms 为 `done`，证明提交和完成没有同步耦合。
- 混合提交中的坏 JPEG 只把自身标成 `error`；同批有效图为 `done`。413、415、批查询和 Prometheus 指标均通过。
- 43 图回归全部完成、推理错误为 0；严格去掉车牌文本后仍有 7 图不完全一致，其中 4 图是车辆数量变化（PP-OCRv3 替换 HyperLPR 后的同牌去重和 FP16 边界共同影响），3 图是阈值附近属性变化。因此这不是精度闸门通过证明，扩大代表集和 A30 上的逐模型 FP16/FP32 对照仍是发布前必做项。
- 5 秒饱和短扫的完成速率随 batch `1/8/16/32/64` 约为 `136.2/142.4/144.0/147.2/153.6` 图/s，全部最终完成且无错误；饱和排队延迟不满足 SLA。
- batch 64 固定速率短扫中，160 图/s 输入的 1600 张最终全部完成、p95 服务总延迟约 580ms；180 图/s 输入时完成能力约 154 图/s、p95 升至约 1.71s。当前短测稳定工作点只能记为“不高于 160 图/s 输入”。
- 将同一镜像临时以 128 图队列过载 5 秒：896 张接受并全部完成，70,848 张快速返回 429，HTTP/推理错误为 0；恢复默认容器后 cache hit、ready、重启计数 0。

所有 A30 数字必须在 A30 24GB 上首启重建 engine 后重新测量。若精度闸门不通过，应仅把相应模型退回 FP32/FP16，不能修改业务阈值来“凑”旧基线。

## 7. 监控和容量参数

```bash
curl -sS http://127.0.0.1:8000/v1/health
curl -sS http://127.0.0.1:8000/metrics
```

关键环境变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `PVR_MAX_IMAGE_BYTES` | 8388608 | 单图字节上限 |
| `PVR_MAX_IMAGE_PIXELS` | 20000000 | 解码像素上限 |
| `PVR_MAX_BATCH_IMAGES` | 64 | 单批图片上限 |
| `PVR_MAX_BATCH_BYTES` | 67108864 | 单批字节上限 |
| `PVR_MAX_QUEUE_IMAGES` | 8192 | 队列图片上限 |
| `PVR_MAX_QUEUE_BYTES` | 1073741824 | 队列字节上限 |
| `PVR_BATCH_WAIT_US` | 2000 | 跨请求聚批窗口 |
| `PVR_RESULT_TTL_SECONDS` | 60 | 完成结果保留时间 |
| `PVR_MAX_RESULT_BYTES` | 1073741824 | 结果字节上限 |
| `PVR_INGEST_CONCURRENCY` | 2 | 同时摄入大请求数 |
| `PVR_SUBMIT_CONCURRENCY` | 8 | Python 到原生队列的并发提交数 |

不要只提高队列上限来掩盖算力不足；那只会增大延迟和内存占用。

## 8. 离线分发与镜像仓库

当前已经验证的离线产物位于构建服务器：

```text
/home/ubuntu/pvr-v2.0.0-ampere.tar.zst
SHA-256: e8d4823e3967ae40e29e26b33628c201c150854b3b68558a13ad9550a7cb1f25
```

发送方可以使用 `scp`、移动硬盘或内网文件服务传输压缩包和校验文件：

```bash
scp ubuntu@SERVER:/home/ubuntu/pvr-v2.0.0-ampere.tar.zst .
scp ubuntu@SERVER:/home/ubuntu/pvr-v2.0.0-ampere.tar.zst.sha256 .
```

离线文件：

```bash
docker save person-vehicle-recognition:v2.0.0 | \
  zstd -T0 -19 -o pvr-v2.0.0-ampere.tar.zst
sha256sum pvr-v2.0.0-ampere.tar.zst > pvr-v2.0.0-ampere.tar.zst.sha256
```

接收方：

```bash
sha256sum -c pvr-v2.0.0-ampere.tar.zst.sha256
zstd -dc pvr-v2.0.0-ampere.tar.zst | docker load
docker volume create pvr-engine-cache
```

镜像仓库路线：

```bash
docker tag person-vehicle-recognition:v2.0.0 registry.example.com/pvr:v2.0.0
docker push registry.example.com/pvr:v2.0.0
# 接收方：docker pull registry.example.com/pvr:v2.0.0
```

外部分发前必须单独确认所有权重的再分发权利，尤其口罩模型。源码许可证不能自动证明第三方模型权重可再分发。

## 9. 关机、恢复和继续开发

当前正式容器使用 `--restart unless-stopped`，Docker 服务应设置为开机启动，engine cache 位于命名卷 `pvr-engine-cache`。确认：

```bash
systemctl is-enabled docker
docker inspect pvr-v2 \
  --format 'status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}'
docker volume inspect pvr-engine-cache
```

队列、正在推理的任务和 60 秒结果缓存都只存在于内存；关机后旧 `session_id` 不保证可查询，调用方必须重新提交未确认完成的图片。镜像、容器配置、Git 工作区和 engine cache 卷不会因为正常关机丢失。

确认队列为空后可以直接正常关机，不必先手动停止容器：

```bash
curl -fsS http://127.0.0.1:8000/v1/health
sudo shutdown -h now
```

开机后 Docker 会按重启策略启动容器。若关机前曾人工执行 `docker stop pvr-v2`，则手动恢复：

```bash
docker start pvr-v2
curl -fsS http://127.0.0.1:8000/v1/health
```

继续开发时只修改 `/home/ubuntu/pvr-src` 或本地 Git 工作区，构建新镜像标签并用不同端口预验；不要进入运行中的容器直接改文件。

## 10. 回滚和清理

本次重构唯一回滚点是 `v1.0` tag。不要删除仍需回滚的 `person-vehicle-recognition:v1.0/slim`。

只清理明确属于本项目且可重建的对象：

```bash
docker system df
docker buildx prune -f
docker rm <已停止的旧容器名>
docker rmi <确认不再使用的旧镜像名>
```

不要在共享服务器上使用无目标的 `docker system prune -a`。engine 文件可重建，但删除 `/var/cache/pvr` 会让下次启动重新经历长时间的本机优化构建。
