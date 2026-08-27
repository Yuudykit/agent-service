# -*- coding: utf-8 -*-
"""
重排序 —— RAG 管线第 4 步

职责：对检索结果二次排序，把最相关的顶到最前面。

为什么需要重排:
    - 检索（retrieve）追求"召回"：宁可多捞，别漏掉 → 双通道合并打分
    - 重排（rerank）追求"精排"：从捞回来的里面挑最准的
    - 简单版用规则（精确命中 > 混合命中 > 纯向量），工业版用 rerank 模型
      （如 bge-reranker），或让 LLM 自己对候选排序

本次实现（规则重排）:
    1. 加分项：
       - match_type=both（关键词+向量都命中）→ 最可信，+0.08
       - 答案/问题里含 query 的业务词 → +0.05（实体/主题强相关）
    2. 减分项：
       - match_type=weak（两通道都没过阈值，纯靠分数硬捞）→ -0.10
    3. 按调整后分数降序，截断到 top_k
"""
from __future__ import annotations

import re

_BONUS_BOTH = 0.08
_BONUS_QUERY_TERM = 0.05
_PENALTY_WEAK = 0.10


def _query_terms(query: str) -> list[str]:
    """抽取 query 里的可命中片段：2-4 字滑动窗口（简单做法，零依赖）。"""
    q = re.sub(r"\s+", "", query)
    terms = set()
    for n in (2, 3, 4):
        for i in range(len(q) - n + 1):
            terms.add(q[i : i + n])
    return sorted(terms)


def rerank(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """对 retrieve() 的结果重排，返回最终 top_k（含调整后的 final_score）。"""
    terms = _query_terms(query)

    for r in results:
        score = r["score"]
        if r.get("match_type") == "both":
            score += _BONUS_BOTH
        if r.get("match_type") == "weak":
            score -= _PENALTY_WEAK
        # query 片段在问题或答案里出现 → 主题强相关
        hay = (r.get("question", "") + r.get("answer", "")).replace(" ", "")
        if any(t in hay for t in terms):
            score += _BONUS_QUERY_TERM
        r["final_score"] = round(score, 4)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]
