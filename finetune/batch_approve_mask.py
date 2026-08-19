# -*- coding: utf-8 -*-
"""批量批注口罩候选：按 AIZOO 预填类别直接确认进金标准。

复用 review_server.ReviewStore 的保存逻辑（裁剪、落盘、gold/candidates 同步），
出图顺序与 WebUI 一致（随机，--seed 可固定）。用于候选类别基本正确的场景，
批注后仍可在 WebUI「回改已保存」里逐个修正。

用法:
    python finetune/batch_approve_mask.py --count 800 --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_server import ReviewStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="批量批注口罩候选")
    parser.add_argument("--count", type=int, default=800, help="批注数量（默认 800）")
    parser.add_argument("--seed", type=int, default=None, help="随机顺序种子")
    args = parser.parse_args()

    store = ReviewStore(prelabel=False, device="cpu", shuffle_seed=args.seed)
    done = 0
    tally = {1: 0, 2: 0}
    while done < args.count:
        try:
            record = store.mask_record(0)
        except IndexError:
            print(f"候选不足，实际批注 {done} 个", flush=True)
            break
        annotation = record["annotation"]
        category = annotation.get("category_id")
        if category not in (1, 2):
            # 候选缺类别时跳过不如直接报错——AIZOO 导入必然带类别
            raise ValueError(f"{annotation['id']} 缺少预填类别，无法批量批注")
        store.mask_save(annotation["id"], category)
        tally[category] += 1
        done += 1
        if done % 100 == 0:
            print(f"已批注 {done}/{args.count}", flush=True)

    summary = store.summary()["mask"]
    print(
        f"完成：新批注 {done} 个（未佩戴 {tally[1]} / 佩戴 {tally[2]}）；"
        f"口罩金标准累计 {summary['已保存']}，候选剩余 {summary['待核对']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
