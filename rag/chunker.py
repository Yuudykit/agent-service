# -*- coding: utf-8 -*-
"""
文档切分 —— RAG 管线第 1 步

职责：把长文档切成适合检索的片段（chunk）。

为什么必须切（面试必答）:
    - 模型上下文有限：整篇文档塞不进一次检索
    - 检索粒度：片段太短缺上下文，太长命中噪音大
    - 行业常见：500-800 字/块 + 50-100 字重叠（重叠防止关键句恰好被切在边界）

当前知识库的说明：
    data/kb.json 的每条 FAQ 已经是"人工校准过的精炼块"（question + answer +
    keywords），检索粒度天然合适，所以 build.py 不调用本模块。
    本模块是为"换成原始对话语料/长文档"准备的通用切分器——
    先按段落拆，段内按句子边界累积到接近 chunk_size 断块，
    块间保留 overlap 字符的重叠（从句子边界回退，避免切碎句子）。
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")  # 在句子边界后切（零宽断言，保留标点）
_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    """按中文/英文句子标点切句，滤掉空句。"""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "",
) -> list[dict]:
    """把一段文本切成多个 chunk（段落优先，句内累积，块间重叠）。

    Args:
        text: 原始文本
        chunk_size: 每块目标字数（中文按字符算）
        overlap: 相邻块重叠字数（从句子边界回退，尽量不切碎句子）
        source: 来源标识（文件名/标题），存入每块元信息

    Returns:
        [{"text": "...", "source": "...", "idx": 0}, ...]
    """
    chunks: list[dict] = []
    idx = 0

    for para in _PARA_SPLIT.split(text.strip()):
        para = para.strip()
        if not para:
            continue

        sentences = _split_sentences(para)
        if not sentences:
            continue

        buf = ""
        for sent in sentences:
            # 句子本身超长（无标点长串）：按固定窗口硬切
            if len(sent) > chunk_size:
                if buf:
                    chunks.append({"text": buf, "source": source, "idx": idx})
                    idx += 1
                    buf = ""
                for i in range(0, len(sent), chunk_size):
                    chunks.append(
                        {"text": sent[i : i + chunk_size], "source": source, "idx": idx}
                    )
                    idx += 1
                continue

            if len(buf) + len(sent) > chunk_size and buf:
                chunks.append({"text": buf, "source": source, "idx": idx})
                idx += 1
                # 重叠：从上一块尾部找句子边界回退到 overlap 字符附近
                tail = buf[-overlap:]
                cut = tail.find("。") if "。" in tail else -1
                buf = tail[cut + 1 :] if cut >= 0 else tail
            buf += sent

        if buf:
            chunks.append({"text": buf, "source": source, "idx": idx})
            idx += 1

    return chunks
