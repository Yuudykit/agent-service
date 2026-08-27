# -*- coding: utf-8 -*-
"""
知识库结构增强 —— 从原始语料提炼"问法变体"写入 kb.json

背景：
    kb.json 的 47 条 FAQ 是"人工校准的标准问法"，但真实用户不会按标准问法问。
    例：标准问"退款多久能到账？"，真实问法有"钱什么时候退回来""退款几天到账"
    "什么时候给我退钱"……纯向量检索对"标准问法"拟合好，对"变体问法"召回弱。

    本脚本从管线A的中间产物 faq_candidates.jsonl（18,341 条真实问答候选）里，
    按主题对齐 + 关键词重合度，为每条 FAQ 抽取高频问法变体，写入新字段
    "variants"。向量化时把 variants 拼进检索文本 → 检索语义覆盖从"1 个标准
    问法"扩到"1+N 个真实问法"。

    这正是数据工程的价值：同样的 KB 条目，不靠换模型，靠数据增强提升召回。

字段设计（向后兼容）：
    kb.json 每条新增:
        "variants": ["钱什么时候退回来", "退款几天到账", ...]   # 最多 6 条

用法：
    python scripts/enrich_kb.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import PLACEHOLDER_PAT, classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "kb.json"
CANDIDATES_PATH = ROOT / "data" / "intermediate" / "faq_candidates.jsonl"

MAX_VARIANTS = 6          # 每条 FAQ 最多保留的变体数
MIN_LEN, MAX_LEN = 4, 40  # 变体长度范围（太短无语义，太长是长句非问法）
MIN_HITS = 1              # 与 FAQ keywords 至少重合的关键词数（口语问法常只命中 1 词）


def load_candidates(path: Path) -> dict[str, list[str]]:
    """按主题分组收集候选问法。返回 topic -> list[question]（去重、过滤占位符）。"""
    pool: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        q = (r.get("question") or "").strip()
        if not (MIN_LEN <= len(q) <= MAX_LEN):
            continue
        if PLACEHOLDER_PAT.search(q):
            continue  # 残留 [数字x] 等脱敏占位符的问法不可用
        topic = r.get("topic") or classify(q)
        key = (topic, q)
        if key in seen:
            continue
        seen.add(key)
        pool[topic].append(q)
    return pool


def pick_variants(entry: dict, pool: dict[str, list[str]]) -> list[str]:
    """从对应主题的问法池里挑与 FAQ 最相关的变体。

    打分 = 问法里命中的 FAQ keywords 数；同分时问法更短者优先（更像口语问法）。
    主题池为空时退回关键词全局模糊匹配。
    """
    keywords = entry.get("keywords") or []
    category = entry.get("category") or ""
    if not keywords:
        return []

    candidates = pool.get(category, [])
    if not candidates:
        # 兜底：kb 的 category 不在 9 大 topic（如"订单状态/商品咨询"）时，
        # 用 keywords 子串在全部候选里粗筛
        candidates = [q for qs in pool.values() for q in qs
                      if any(kw in q for kw in keywords)]

    scored = []
    for q in candidates:
        hits = sum(1 for kw in keywords if kw in q)
        if hits < MIN_HITS:
            continue
        if q in (entry.get("question") or ""):
            continue
        scored.append((hits, -len(q), q))
    scored.sort(reverse=True)
    return [q for _, _, q in scored[:MAX_VARIANTS]]


def main() -> None:
    pool = load_candidates(CANDIDATES_PATH)
    pool_size = {t: len(qs) for t, qs in pool.items()}
    print("主题问法池:", dict(sorted(pool_size.items(), key=lambda x: -x[1])))

    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    filled = 0
    for entry in kb:
        variants = pick_variants(entry, pool)
        entry["variants"] = variants
        if variants:
            filled += 1

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"kb.json 已更新：{len(kb)} 条，其中 {filled} 条获得变体")

    # 抽样展示 3 条
    shown = 0
    for entry in kb:
        if entry.get("variants") and shown < 3:
            print(f"\n[{entry['id']}] {entry['question']}")
            for v in entry["variants"]:
                print(f"  · {v}")
            shown += 1


if __name__ == "__main__":
    main()
