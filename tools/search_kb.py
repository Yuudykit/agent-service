# -*- coding: utf-8 -*-
"""
知识库检索工具 —— P1 RAG 版（升级自 P0 关键词版）

对标 JD：RAG（Retrieval-Augmented Generation）

职责：用户问一句客服问题 → 混合检索知识库 → 重排 → 返回最相关的 1-3 条
     给 LLM 作为回答依据，避免模型凭空编造（防幻觉）。

P1 升级（vs P0）：
    P0 只做关键词 + 2-gram 字面重合，换说法就失效；
    P1 走 rag 包完整管线：
        retriever（关键词通道 + 向量通道加权）→ reranker（规则精排）
    向量通道用 rag.embedder 的特征哈希向量（零依赖），
    真实项目可设 EMBEDDER_BACKEND=ml 切到 m3e-small，接口不变。

输出格式约定（与 core/agent_loop._mock_chat 的解析耦合，勿改）：
    - 首行以 [检索结果] 开头
    - 每条答案一行，以 "  答：" 开头（两个空格），mock 模式据此组织最终回答
"""
from __future__ import annotations

import json
import re

from rag.retriever import retrieve
from rag.reranker import rerank
from tools.registry import register_tool


def _load_kb(path: str | None = None) -> list[dict]:
    """加载知识库（保留给 fallback 等模块复用）。path 默认取 config 配置。"""
    if path is None:
        with open("config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        path = cfg["knowledge_base_path"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@register_tool
def search_kb(query: str, top_k: int = 3) -> str:
    """在客服知识库中检索最相关的问题与答案。

    当用户询问发货、物流、退款、退货、换货、发票、保修、优惠等业务问题时使用；
    检索结果供回答引用，不要编造知识库里没有的信息。

    注：口语归一化（退钱→退款 等）在 rag.retriever 的关键词通道内完成，
    不污染向量通道；这里只处理"退退退"这类极端输入。
    """
    # 输入归一化：极端口语（"退退退退退！"）映射到业务词，提高召回
    query = re.sub(r"^退+[！!]?$", "退货退款", query.strip()) or query

    # RAG 管线：混合检索 → 重排 → top_k
    results = rerank(query, retrieve(query, top_k=top_k * 2), top_k=top_k)
    if not results:
        return "[检索结果] 知识库中未找到相关问题，请如实告知用户并提供转人工建议。"

    lines = [f"[检索结果] 找到 {len(results)} 条相关条目："]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}) [{r['id']}] {r['category']}：{r['question']}"
            f"（匹配:{r['match_type']} 分:{r['final_score']}）"
        )
        lines.append(f"   答：{r['answer']}")
        lines.append(f"   源：{r['source']}")
    return "\n".join(lines)
