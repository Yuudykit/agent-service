# -*- coding: utf-8 -*-
"""
索引构建 —— RAG 管线的"离线"一步

职责：把知识库（data/kb.json）转成带向量的索引文件（data/kb_index.json），
     检索时直接加载，不用每次重算向量。

用法：
    python -m rag.build

流程：
    kb.json（47 条 FAQ）→ 每条目一个 chunk → encode → 落盘 kb_index.json
    chunk 结构：
    {
        "id": "kb_013", "category": "缺货退货",
        "question": "...", "answer": "...", "keywords": [...],
        "text": "question。answer",   # 检索对象
        "vector": [0.01, ...]          # 512 维
    }

为什么这样建索引:
    - 每条 FAQ 本身已经是"精炼过的块"（人工校准过），无需再切分；
      若换成原始对话语料（万级），才需要 chunker.py 按段落/窗口切
    - 向量预计算：检索时只算 query 一次向量 + 与全库点积，毫秒级
    - 索引文件与源数据分离：改 kb.json 后重跑一次本脚本即可
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import current_backend, encode  # noqa: E402

KB_PATH = "data/kb.json"


def _index_path() -> str:
    """索引按 embedding 后端分文件：换后端（hash/bge/bge-m3）不会互相覆盖。

    必须在调用时动态求值（不能做成模块级常量）：compare_embedders.py
    会在同一进程内切换 EMBEDDER_BACKEND，模块级常量只在 import 时求值一次，
    会导致第二次 build_index 写错文件（教训）。
    """
    return f"data/kb_index_{current_backend()}.json"


def build_index(kb_path: str = KB_PATH, index_path: str | None = None) -> dict:
    """读 kb.json → 生成带向量的索引 → 落盘。返回索引 dict。

    index_path 缺省时按当前 embedding 后端动态取路径。
    """
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"知识库不存在：{kb_path}")

    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    chunks = []
    for entry in kb:
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        variants = entry.get("variants") or []
        # 检索文本 = 标准问法 + 真实问法变体 + 答案
        # variants 是数据增强：让向量空间覆盖"换说法"，纯靠 embedding 也召得回
        # keywords 不拼进来 —— 它走关键词通道，拼进向量会稀释问法语义
        v_text = " ".join(variants)
        text = " ".join(part for part in (q, v_text, a) if part)
        chunks.append(
            {
                "id": entry.get("id", ""),
                "category": entry.get("category", ""),
                "question": q,
                "answer": a,
                "keywords": entry.get("keywords", []),
                "variants": variants,
                "text": text,
            }
        )

    # 批量向量化（单次调用，方便后端缓存/批处理）
    vectors = encode([c["text"] for c in chunks])
    for c, vec in zip(chunks, vectors):
        c["vector"] = [round(v, 6) for v in vec]

    index = {"version": 1, "dim": len(vectors[0]) if vectors else 0, "chunks": chunks}
    index_path = index_path or _index_path()
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    print(f"[build] {len(chunks)} 条 → {index_path}（向量维度 {index['dim']}）")
    return index


if __name__ == "__main__":
    build_index()
