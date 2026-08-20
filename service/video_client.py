"""视频抽帧客户端：从视频文件或 RTSP 流抽帧，异步提交给 FastAPI 识别服务。

用法：
    python video_client.py --source video.mp4 --fps 2 --output results.jsonl
    python video_client.py --source rtsp://... --fps 1 --server http://127.0.0.1:8000

流程：
    1. 按 --fps（每秒抽帧数）从视频源抽帧，编码为 JPEG 后 POST /v1/tasks；
    2. 服务端返回 session_id 后立即继续抽下一帧（输入异步，不阻塞等待结果）；
    3. 若服务端队列已满（HTTP 429），丢弃该帧并计数，保护服务端不积压；
    4. 抽帧结束（或 Ctrl-C）后，统一轮询所有 session_id，结果写入 --output（JSONL）。
"""
import argparse
import json
import time

import cv2
import requests


def extract_and_submit(source, sample_fps, server, session):
    """抽帧并提交，返回 [(frame_index, timestamp_ms, session_id)] 和丢弃数。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, round(src_fps / sample_fps))

    submitted = []
    dropped = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % interval == 0:
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                jpg = cv2.imencode(".jpg", frame)[1].tobytes()
                try:
                    resp = session.post(
                        f"{server}/v1/tasks",
                        files={"file": (f"frame_{frame_index}.jpg", jpg, "image/jpeg")},
                        timeout=10,
                    )
                except requests.RequestException as exc:
                    print(f"[frame {frame_index}] 提交失败: {exc}")
                    dropped += 1
                    frame_index += 1
                    continue
                if resp.status_code == 202:
                    sid = resp.json()["session_id"]
                    submitted.append((frame_index, timestamp_ms, sid))
                elif resp.status_code == 429:
                    dropped += 1
                else:
                    print(f"[frame {frame_index}] 意外响应 {resp.status_code}: {resp.text}")
                    dropped += 1
                if len(submitted) % 100 == 0 and submitted:
                    print(f"已提交 {len(submitted)} 帧，丢弃 {dropped} 帧")
            frame_index += 1
    except KeyboardInterrupt:
        print("收到中断，停止抽帧")
    finally:
        cap.release()
    return submitted, dropped


def collect_results(submitted, server, session, poll_interval=0.5, timeout=600):
    """轮询所有 session_id，返回 {session_id: result_dict}。"""
    pending = {sid: (fi, ts) for fi, ts, sid in submitted}
    results = {}
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for sid in list(pending):
            resp = session.get(f"{server}/v1/tasks/{sid}", timeout=10)
            body = resp.json()
            if body.get("status") in ("done", "error"):
                fi, ts = pending.pop(sid)
                results[sid] = {
                    "frame_index": fi,
                    "timestamp_ms": ts,
                    "session_id": sid,
                    **body,
                }
        if pending:
            time.sleep(poll_interval)
    for sid, (fi, ts) in pending.items():
        results[sid] = {
            "frame_index": fi,
            "timestamp_ms": ts,
            "session_id": sid,
            "status": "timeout",
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="视频抽帧 → FastAPI 异步识别客户端")
    parser.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    parser.add_argument("--fps", type=float, default=1.0, help="每秒抽帧数，默认 1")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="识别服务地址")
    parser.add_argument("--output", default="results.jsonl", help="结果输出文件（JSONL）")
    parser.add_argument("--timeout", type=float, default=600, help="结果轮询总超时（秒）")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    session = requests.Session()
    health = session.get(f"{args.server}/v1/health", timeout=10).json()
    print(f"服务状态: {health}")

    t0 = time.time()
    submitted, dropped = extract_and_submit(source, args.fps, args.server, session)
    print(f"抽帧完成：提交 {len(submitted)} 帧，丢弃 {dropped} 帧，耗时 {time.time() - t0:.1f}s")

    results = collect_results(submitted, args.server, session, timeout=args.timeout)
    ordered = sorted(results.values(), key=lambda r: r["frame_index"])
    with open(args.output, "w", encoding="utf-8") as f:
        for record in ordered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已写入 {len(ordered)} 条结果 → {args.output}")


if __name__ == "__main__":
    main()
