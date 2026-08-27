# -*- coding: utf-8 -*-
"""
混合检索 —— RAG 管线第 3 步

职责：给定 query，从知识库 chunk 里找出最相关的 top-k。

为什么混合:
    - 纯向量：语义好，但"精确业务词"（如"增值税专用发票"）可能被语义漂移
    - 纯关键词：精确名词命中好，但"换个说法"（"退钱" vs "退款"）就抓瞎
    - 混合 = 关键词打分 + 向量打分加权，两个通道互为兜底

双通道设计:
    1. 关键词通道：query 是否包含 chunk.keywords 里的业务词（精确子串命中）
       → 可解释、可调试，是 P0 关键词检索的延续
    2. 向量通道：query 向量 vs chunk 向量余弦相似度（特征哈希 bigram 近似语义）
       → 兜住"换说法但用词相近"的查询
    加权：score = 0.6 * 向量分 + 0.4 * 关键词分（权重可调，见 _WEIGHT_VEC）

延迟构建：索引文件不存在时自动跑 build_index()，拿到项目直接能用。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.build import build_index  # noqa: E402
from rag.embedder import cosine, current_backend, encode, get_embedder  # noqa: E402

_WEIGHT_VEC = 0.6  # 向量通道权重
_WEIGHT_KW = 0.4   # 关键词通道权重
_VEC_HIT_TH = 0.55  # 判定"向量命中"的余弦阈值（特征哈希下经验值）

# 口语 → 业务标准词 归一化表（只用于关键词通道，不污染向量通道！
# 教训：若在检索前整体替换 query，会破坏向量通道的原始 bigram 语义，
# 例如"家电坏了"→"家电质量问题"，导致正确的 kb_026 掉出 top-k）
_SYNONYMS = [
    ("退钱", "退款"),       # 退钱 → 退款（kb_010 等）
    ("钱什么时候退", "退款到账"),
    ("发货没有", "发货"),
    ("怎么还没有到", "物流"),
    ("运单", "物流"),
    ("坏了", "质量问题"),
    ("报销凭证", "发票"),
    ("纸质的", "纸质发票"),
    ("邮费", "运费"),       # 口语"邮费" → 业务词"运费"（kb_012/kb_022，评测反哺）
    ("没有收到", "没收到"),  # "没有收到货" 子串不含"没收到"（中间隔了"有"），归一化后命中 kb_007
    ("转快递", "转人工"),    # 用户想找人工/快递客服
]


def _normalize_keywords(query: str) -> str:
    """归一化口语词（仅关键词通道用），如"坏了"→"质量问题"。"""
    for src, dst in _SYNONYMS:
        query = query.replace(src, dst)
    return query

_index_cache: dict | None = None


def _load_index() -> dict:
    """加载索引（带缓存）；不存在则自动构建。

    索引路径按当前 embedding 后端动态解析（与 build.py 同一逻辑），
    保证同一进程内切换 EMBEDDER_BACKEND 后加载到的是对应后端的索引；
    且先触发 embedder 解析——后端降级时路径跟着实际后端走，不会错配。
    """
    global _index_cache
    if _index_cache is None:
        # 先触发 embedder 解析（可能降级），current_backend() 才报实际后端。
        # 否则降级进程会拿着 hash 查询向量去加载 bge 的索引文件——
        # 两者都是 512 维，余弦不报错但分数全是噪声（静默劣化）。
        get_embedder()
        index_path = f"data/kb_index_{current_backend()}.json"
        if not os.path.exists(index_path):
            build_index()
        with open(index_path, encoding="utf-8") as f:
            _index_cache = json.load(f)
    return _index_cache


def reset_index_cache() -> None:
    """重建索引后清缓存（测试用）。"""
    global _index_cache
    _index_cache = None


def _keyword_score(query: str, chunk: dict) -> float:
    """关键词通道：归一化后的 query 命中 chunk.keywords 的比例，命中 2 词封顶。"""
    norm_q = _normalize_keywords(query)
    kws = chunk.get("keywords") or []
    if not kws:
        return 0.0
    hits = sum(1 for kw in kws if kw and (kw in query or kw in norm_q))
    return min(1.0, hits / 2.0)


def retrieve(query: str, top_k: int = 3, use_vector: bool = True) -> list[dict]:
    """混合检索。

    Returns:
        [{"id","category","question","answer","source","score",
          "vec_score","kw_score","match_type"}, ...] 按加权分降序
    """
    index = _load_index()
    q_vec = encode([query])[0]

    scored = []
    for chunk in index["chunks"]:
        vec_score = cosine(q_vec, chunk.get("vector") or []) if use_vector else 0.0
        kw_score = _keyword_score(query, chunk)
        score = _WEIGHT_VEC * vec_score + _WEIGHT_KW * kw_score

        kw_hit = kw_score > 0
        vec_hit = vec_score >= _VEC_HIT_TH
        if kw_hit and vec_hit:
            match_type = "both"
        elif vec_hit:
            match_type = "vector"
        elif kw_hit:
            match_type = "keyword"
        else:
            match_type = "weak"

        scored.append(
            {
                "id": chunk.get("id", ""),
                "category": chunk.get("category", ""),
                "question": chunk.get("question", ""),
                "answer": chunk.get("answer", ""),
                "source": "JDDC 知识库",
                "score": round(score, 4),
                "vec_score": round(vec_score, 4),
                "kw_score": round(kw_score, 2),
                "match_type": match_type,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":  # 快速自测
    for q in ["退款多久到账", "发票税号怎么改", "商品缺货要等多久", "今天天气怎么样"]:
        print(f"\nQ: {q}")
        for r in retrieve(q, top_k=2):
            print(f"  {r['score']:.3f} [{r['match_type']}] {r['id']} {r['question']}")
