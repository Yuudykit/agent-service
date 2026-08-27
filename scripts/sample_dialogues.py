# -*- coding: utf-8 -*-
"""
从 JDDC 采样高质量完整会话 → data/dialogues.jsonl（管线B）

筛选标准（够用即可）：
    1. 轮数在 8-30 之间（太短没有多轮价值，太长有噪声）
    2. 客服有实质回复（排除全流程话的会话）
    3. 含至少 1 个问句（用户真正在咨询）

输出格式（每行一个 JSON，schema 稳定，供多轮评测用例生成消费）:
    {
        "id": "d_0001",
        "session_id": "...",
        "topic": "物流发货",              # classify() 主题归类
        "turn_count": 13,                 # 消息条数
        "user_turns": 7, "agent_turns": 6,
        "has_placeholder": true,          # 是否残留脱敏占位符（[数字x] 等）
        "messages": [{"role": "user"|"agent", "text": "..."}]
    }
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import QUESTION_PAT, classify, has_placeholder  # noqa: E402
from data_utils import RAW_PATH, parse_raw  # noqa: E402  共享解析（口径唯一）

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dialogues.jsonl"

FLOW_WORDS = ["稍等", "核实", "为您处理", "久等", "抱歉", "在的", "在呢", "您好"]


def is_good(session: dict) -> bool:
    msgs = session["messages"]
    if not (8 <= len(msgs) <= 30):
        return False
    agent_texts = [m["text"] for m in msgs if m["role"] == "agent"]
    user_texts = [m["text"] for m in msgs if m["role"] == "user"]
    if not agent_texts or not user_texts:
        return False
    # 客服至少有一条实质回复（长度 > 20 且不是纯流程话）
    solid = [t for t in agent_texts if len(t) > 20 and not any(w in t for w in FLOW_WORDS)]
    if not solid:
        return False
    # 用户至少有一个问句
    if not any(QUESTION_PAT.search(t) for t in user_texts):
        return False
    return True


def enrich(session: dict, idx: int) -> dict:
    """在原始会话上补元数据（topic/轮数/占位符），供下游消费方直接使用。"""
    msgs = session["messages"]
    user_texts = [m["text"] for m in msgs if m["role"] == "user"]
    agent_texts = [m["text"] for m in msgs if m["role"] == "agent"]
    # 主题从用户问句里判（客服话多为流程语，噪声大）
    topic = classify(" ".join(user_texts)) if user_texts else "其他"
    return {
        "id": f"d_{idx:04d}",
        "session_id": session["session_id"],
        "topic": topic,
        "turn_count": len(msgs),
        "user_turns": len(user_texts),
        "agent_turns": len(agent_texts),
        "has_placeholder": any(has_placeholder(t) for t in user_texts + agent_texts),
        "messages": msgs,
    }


def main() -> None:
    # 纯解析不过滤：合格口径由 is_good 定义（8~30 轮等），parse 层不掺和
    sessions = parse_raw(RAW_PATH)
    good = [s for s in sessions if is_good(s)]
    print(f"总会话: {len(sessions)}，合格会话: {len(good)}")

    # 按 session_id 稳定排序后间隔采样，保证来源分散
    good.sort(key=lambda s: s["session_id"])
    sampled = good[:: max(1, len(good) // 120)][:120]

    with open(OUT, "w", encoding="utf-8") as f:
        for i, s in enumerate(sampled):
            f.write(json.dumps(enrich(s, i + 1), ensure_ascii=False) + "\n")
    print(f"已输出: {OUT}（{len(sampled)} 个会话）")

    # 主题分布
    from collections import Counter

    topics = Counter(json.loads(l)["topic"] for l in open(OUT, encoding="utf-8"))
    print("主题分布:", dict(topics.most_common()))

    # 抽样打印 1 个
    first = json.loads(open(OUT, encoding="utf-8").readline())
    print(f"\n--- {first['id']} ({first['session_id']}) topic={first['topic']} turns={first['turn_count']} ---")
    for m in first["messages"][:8]:
        who = "用户" if m["role"] == "user" else "客服"
        print(f"  [{who}] {m['text'][:60]}")
    print(f"  ... 共 {first['turn_count']} 条")


if __name__ == "__main__":
    main()
