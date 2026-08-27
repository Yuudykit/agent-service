# -*- coding: utf-8 -*-
"""
JDDC 数据解析与统计分析脚本。

输入: data/raw/jddc_chat.txt（京东客服真实脱敏对话，tab 分隔）
列:   session_id, user_id, waiter_send, is_transfer, is_repeat, content
      waiter_send: 0=用户发言, 1=客服发言

输出:
    1. 按 session_id 分组，统计会话数量、轮数分布、脱敏标记种类
    2. 抽样打印 3 个完整会话（观察对话质量与主题）
    3. 高频关键词统计（帮助后续提炼 FAQ 知识库）

用法:
    python scripts/analyze_jddc.py [--samples 3] [--limit 100000]   # --limit 按会话数截断
"""
import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import RAW_PATH, parse_raw  # noqa: E402  共享解析（口径唯一）


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sessions = parse_raw(RAW_PATH, limit=args.limit, min_messages=2)
    print(f"原始文件: {RAW_PATH}")
    print(f"会话总数: {len(sessions)}")
    print(f"总消息数: {sum(len(s['messages']) for s in sessions)}")

    # 轮数分布
    turn_counts = [len(s["messages"]) for s in sessions]
    print(f"\n轮数分布: min={min(turn_counts)} 中位数={sorted(turn_counts)[len(turn_counts)//2]} "
          f"max={max(turn_counts)} 平均={sum(turn_counts)/len(turn_counts):.1f}")
    hist = collections.Counter(min(t, 30) for t in turn_counts)
    print("轮数直方图(截断到30):", dict(sorted(hist.items())))

    # 脱敏标记种类
    mask_pat = re.compile(r"\[([^\]]+)\]")
    masks = collections.Counter()
    for s in sessions:
        for m in s["messages"]:
            masks.update(mask_pat.findall(m["text"]))
    print(f"\n脱敏标记种类({len(masks)}):", dict(masks.most_common(15)))

    # 高频词（客服场景特征词）
    keywords = ["发货", "退货", "退款", "发票", "物流", "快递", "保修", "维修",
                "改地址", "优惠", "优惠券", "客服", "投诉", "人工", "订单", "多久",
                "多少", "怎么", "吗", "什么时候", "售后", "赠品", "换货", "差价", "保价"]
    word_hits = {w: 0 for w in keywords}
    for s in sessions:
        text = "".join(m["text"] for m in s["messages"])
        for w in keywords:
            if w in text:
                word_hits[w] += 1
    print("\n主题关键词命中会话数:")
    for w, c in sorted(word_hits.items(), key=lambda x: -x[1]):
        print(f"  {w}: {c} ({c/len(sessions)*100:.1f}%)")

    # 抽样会话
    print(f"\n===== 抽样 {args.samples} 个会话 =====")
    for s in sessions[: args.samples]:
        print(f"\n--- session: {s['session_id']} ({len(s['messages'])} 条消息) ---")
        for m in s["messages"][: 12]:
            who = "用户" if m["role"] == "user" else "客服"
            print(f"  [{who}] {m['text'][:80]}")
        if len(s["messages"]) > 12:
            print(f"  ... 共 {len(s['messages'])} 条")


if __name__ == "__main__":
    main()
