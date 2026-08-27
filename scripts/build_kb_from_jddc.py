# -*- coding: utf-8 -*-
"""
JDDC → FAQ 候选问答对提取脚本。

思路：
    真实客服对话里藏着大量"用户问 → 客服答"的标准问答。
    本脚本把高频主题的候选问答对提取出来，作为人工整理知识库的原料。

规则（够用即可，不求精确）：
    1. 解析 chat.txt → 会话列表（复用 analyze_jddc 的解析）
    2. 对每个会话：找到"用户问句"（含 ? ？ 吗 怎么 多少 多久 什么时候 能否 可以...吗 等问句信号）
    3. 取该问句之后最近的客服回复（1-2 条）作为候选答案
    4. 按主题关键词归类，输出候选 FAQ（带 session 溯源）

输出: data/intermediate/faq_candidates.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import QUESTION_PAT, classify  # noqa: E402  共享分类口径
from data_utils import RAW_PATH, parse_raw  # noqa: E402  共享解析（口径唯一）

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "intermediate"


def extract_candidates(session: dict, max_per_session: int = 3) -> list[dict]:
    """提取候选问答对：用户问句 → 后续最近客服回复。"""
    cands = []
    msgs = session["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        if not QUESTION_PAT.search(m["text"]):
            continue
        # 找后续最近的客服回复（跳过用户追问）
        answers = []
        for nxt in msgs[i + 1 :]:
            if nxt["role"] == "agent":
                answers.append(nxt["text"])
                if len(answers) >= 2:
                    break
        if not answers:
            continue
        cands.append({
            "session_id": session["session_id"],
            "question": m["text"][:120],
            "answers": [a[:200] for a in answers],
            "topic": classify(m["text"] + " " + " ".join(answers)),
        })
        if len(cands) >= max_per_session:
            break
    return cands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="最多处理的会话数（None=全量，默认全量以复现 18,341 条候选）")
    args = parser.parse_args()

    # 复用 analyze 的基础会话口径（至少一问一答），候选质量更干净
    sessions = parse_raw(RAW_PATH, limit=args.max_sessions, min_messages=2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cands: list[dict] = []
    for s in sessions:
        all_cands.extend(extract_candidates(s))

    # 按主题统计
    from collections import Counter
    topic_count = Counter(c["topic"] for c in all_cands)
    print(f"处理会话: {len(sessions)}，提取候选问答对: {len(all_cands)}")
    print("\n主题分布:")
    for t, c in topic_count.most_common():
        print(f"  {t}: {c}")

    # 输出
    out_path = OUT_DIR / "faq_candidates.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n已输出: {out_path}")

    # 抽样打印每个主题 2 个候选
    print("\n===== 抽样候选（每主题 2 条）=====")
    shown: set[str] = set()
    for c in all_cands:
        if c["topic"] in shown:
            continue
        shown.add(c["topic"])
        print(f"\n[{c['topic']}] session={c['session_id']}")
        print(f"  Q: {c['question'][:60]}")
        print(f"  A: {c['answers'][0][:80]}")
        if len(shown) >= 9:
            break


if __name__ == "__main__":
    main()
