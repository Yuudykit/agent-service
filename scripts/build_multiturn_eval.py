# -*- coding: utf-8 -*-
"""
把 120 条真实多轮对话 → 多轮评测用例（dialogues 闭环的核心一步）

背景：
    此前评测集 28 条用例里多轮只有 2 条手工构造，而手边 120 条真实多轮
    会话闲置——数据和评测脱节。本脚本把真实会话改造成评测用例：
        - history：会话里最后一条用户消息之前的最近 3 个回合
        - query：最后一条用户消息（真实用户的真实问法）
        - expect.tool：按意图规则推断期望工具
    这样评测多轮覆盖从 2 条扩到几十条，且每个 query 都是真实语料，
    不是人工编的"标准问法"——正好检验 Agent 对口语化多轮提问的路由。

评测语义：
    - mock 模式（离线回归）：路由只看 query 本身，验证"真实问法 → 正确工具"
    - 真实模式（DeepSeek）：history 在场，验证"多轮记忆 → 正确承接话题"
    同一个用例两种模式都有价值，这是设计要点。

用法：
    python scripts/build_multiturn_eval.py [--max-cases 30] [--dry-run]
输出：data/eval_multiturn.json（独立文件，不污染人工评测集 eval_set.json）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import QUESTION_PAT, classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIALOGUES = ROOT / "data" / "dialogues.jsonl"
OUT = ROOT / "data" / "eval_multiturn.json"

HISTORY_TURNS = 3   # 保留 query 前最近几个"用户-客服"回合
MAX_HISTORY_MSGS = HISTORY_TURNS * 2

# 意图推断规则（与 core/agent_loop._mock_judge 对齐，保证 mock 下路由一致）
_ORDER_PAT = re.compile(r"JD\d{8,}|\d{10,}")
_HUMAN_WORDS = ["人工", "客服热线", "打电话", "950618", "投诉", "专员"]
_BUSINESS_WORDS = [
    "发货", "物流", "快递", "配送", "到货", "多久",
    "退款", "退货", "退钱", "换货", "维修", "保修", "售后",
    "发票", "税号", "差价", "保价", "优惠", "券", "订单", "取消",
    "缺货", "补货", "先发", "有货", "付款", "支付", "签收",
    "退",
]
_GREETING_WORDS = ["在吗", "你好", "您好", "在不在", "谢谢", "辛苦了", "嗯", "再见", "没了", "好的", "拜拜"]

# topic → 期望回答里应出现的关键词（search_kb 类用例用；kw_mode=any，
# 所以每个 topic 给多个候选词，覆盖 kb answer 的实际用词差异，避免误杀）
_TOPIC_KWS = {
    "物流发货": ["发货", "发出", "到货", "物流", "快递"],
    "退货退款": ["退款", "退货"],
    "发票": ["发票"],
    "换货维修": ["换货", "维修"],
    "保价差价": ["差价"],
    "订单修改": ["订单"],
    "优惠活动": ["优惠券"],
    "售后政策": ["售后"],
    "人工客服": ["人工"],
}


def infer_expect(query: str) -> dict | None:
    """推断期望工具 + 关键词。无法可靠判定的返回 None（跳过该会话）。"""
    q = query.strip()
    # 0) 订单号语境
    m = _ORDER_PAT.search(q)
    if m:
        if any(w in q for w in ("钱", "多少", "算")):
            return {"tool": "calculate_refund", "keywords": [], "note": "真实多轮：退款金额计算"}
        return {"tool": "query_info", "keywords": [], "note": "真实多轮：订单状态查询"}
    # 1) 转人工（最高优先级）
    if any(w in q for w in _HUMAN_WORDS):
        return {"tool": "transfer_human", "keywords": ["人工"], "note": "真实多轮：转人工"}
    # 1.5) "转接"意图边界：转快递/转专员等口语，mock 走 search_kb 且返回
    #      转人工指引 → 期望关键词对准"人工/客服"而不是业务主题词
    if "转" in q and any(w in q for w in ("客服", "快递", "专员", "电话")):
        return {"tool": "search_kb", "keywords": ["人工", "客服"], "note": "真实多轮：转接客服"}
    # 2) 业务问题 → 检索知识库
    if any(w in q for w in _BUSINESS_WORDS):
        topic = classify(q)
        kws = _TOPIC_KWS.get(topic, [])
        if not kws:
            return None  # 主题无法可靠归类 → 不生成，保证用例质量
        return {"tool": "search_kb", "keywords": kws, "note": f"真实多轮：{topic}"}
    # 3) 寒暄 → 不调工具
    if any(w in q for w in _GREETING_WORDS) or len(q) <= 2:
        return {"tool": "none", "keywords": [], "note": "真实多轮：寒暄收尾"}
    return None


def build_case(dlg: dict, idx: int) -> dict | None:
    """从一条会话构造评测用例；构造失败（无可路由意图/无足够历史）返回 None。

    策略：真实客服会话几乎都以"没了/谢谢"寒暄收尾，不能取最后一条用户消息。
    改为从后往前找"第一条可路由意图的用户消息"（即会话里最后一个真正
    的业务提问），它之前的对话自然构成多轮历史。
    """
    msgs = dlg["messages"]
    # 从后往前找第一条能推断出非寒暄意图的用户消息
    target = None
    expect = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] != "user":
            continue
        q = msgs[i]["text"].strip()
        if not q:
            continue
        e = infer_expect(q)
        if e is None:
            continue
        if e["tool"] == "none":
            continue  # 寒暄收尾不是业务提问，继续往前找
        target, expect = i, e
        break
    if target is None:
        return None
    query = msgs[target]["text"].strip()

    # history = query 之前的消息，最多取最近 N 条
    history = msgs[max(0, target - MAX_HISTORY_MSGS):target]
    # 过滤纯噪音消息（超长商品快照等），避免污染评测上下文
    history = [m for m in history if len(m["text"]) <= 200]
    # 至少要有一个 用户→客服 回合，才算"多轮"
    roles = [m["role"] for m in history]
    if "user" not in roles or "agent" not in roles:
        return None

    return {
        "id": f"mt_{idx:03d}",
        "source_session": dlg["session_id"],
        "source_dialogue": dlg["id"],
        "history": history,
        "query": query,
        "expect": {**expect, "kw_mode": "any"},  # 自动用例：命中任一关键词即过（宽松，防 AND 误杀）
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=30, help="生成用例数上限")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = parser.parse_args()

    dialogues = [json.loads(l) for l in open(DIALOGUES, encoding="utf-8")]
    cases: list[dict] = []
    skipped_reasons: dict[str, int] = {}
    idx = 0
    for dlg in dialogues:
        case = build_case(dlg, idx)
        if case is None:
            continue
        cases.append(case)
        idx += 1
        if len(cases) >= args.max_cases:
            break

    from collections import Counter

    tools = Counter(c["expect"]["tool"] for c in cases)
    topics = Counter((c["expect"].get("note") or "").replace("真实多轮：", "") for c in cases)
    print(f"dialogues: {len(dialogues)} 条 → 多轮用例: {len(cases)} 条")
    print("工具分布:", dict(tools))
    print("主题分布:", dict(topics))

    if args.dry_run:
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"已输出: {OUT}")

    # 抽样展示 3 条
    for c in cases[:3]:
        print(f"\n[{c['id']}] 来源 {c['source_dialogue']} 期望工具={c['expect']['tool']}")
        for m in c["history"][-4:]:
            who = "用户" if m["role"] == "user" else "客服"
            print(f"  [历史-{who}] {m['text'][:50]}")
        print(f"  [query] {c['query'][:60]}")


if __name__ == "__main__":
    main()
