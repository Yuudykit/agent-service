# -*- coding: utf-8 -*-
"""
JDDC 原始数据解析共享模块（唯一实现，三个脚本共用）。

背景：
    analyze_jddc / build_kb_from_jddc / sample_dialogues 曾各自实现 parse_raw，
    过滤口径互相不一致——有的 >=2 条消息、有的不过滤；limit 一个按行号截断、
    一个按会话数切片。统一到这里，消除口径分裂：以后改解析逻辑只动这一处。

口径约定：
    - parse_raw 只做「解析 + 可选基础过滤」，业务级过滤（如 sample 的 is_good）
      留在各脚本，互不掺和。
    - min_messages=2 是「基础会话」口径（至少一问一答），与 analyze 的统计口径
      一致，build_kb 显式传 2 复用。注意它不同于 sample 的「合格会话」
      （is_good：8~30 轮 + 实质客服回复 + 用户问句），两者层级不同，勿混用。
    - limit 语义统一为「最多保留的会话数」（解析后按出现顺序取前 N），
      修复了旧 analyze 按行号截断会把一个会话截成半截的怪癖。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "jddc_chat.txt"


def parse_raw(path=RAW_PATH, limit=None, min_messages=0) -> list[dict]:
    """解析 chat.txt，返回按 session 分组的对话列表。

    每个会话: {"session_id": str, "messages": [{"role": "user"|"agent", "text": str}]}

    Args:
        path: chat.txt 路径（默认 data/raw/jddc_chat.txt）
        limit: 最多保留的会话数（None=全量，按出现顺序取前 N）
        min_messages: 会话至少包含的消息条数；>=2 即一问一答的合格会话
    """
    sessions: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # 跳过表头
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            # 列结构: session_id, user_id, waiter_send, is_transfer, is_repeat, (空列), content
            sid, _, waiter_send, _, _, _, content = parts[:7]
            content = content.strip()
            if not content:
                continue
            role = "agent" if waiter_send == "1" else "user"
            sessions.setdefault(sid, {"session_id": sid, "messages": []})
            sessions[sid]["messages"].append({"role": role, "text": content})
    result = list(sessions.values())
    if min_messages > 0:
        result = [s for s in result if len(s["messages"]) >= min_messages]
    if limit is not None:
        result = result[:limit]
    return result
