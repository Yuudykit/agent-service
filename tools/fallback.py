# -*- coding: utf-8 -*-
"""
兜底与转人工 —— 对标 JD：异常处理 / 边界控制

职责：Agent 回答不了的场景，给出"有边界的"反应，而不是硬编造。

三条兜底路径（面试可讲）：
    1. 检索无结果：知识库没有 → 明确说"没查到"，建议转人工
    2. 超步数上限：ReAct 循环跑满 max_steps 还没出结果 → 兜底话术 + 转人工
    3. 用户明确要人工：直接给出人工客服入口（电话/在线）

P2 升级：转人工不再只是话术，而是"交接协议"——
    transfer_human 被调用时落盘一份交接包 data/handoffs/{时间戳}-{sid}.json：
    会话主要诉求、关键槽位（订单号/金额/问题类型/情绪）、最近对话，
    并给用户返回一个工单号。人工客服接手时对会话不再一无所知。

会话绑定：工具是纯函数、拿不到会话上下文，靠线程局部存储传递——
webapp 每个请求一个线程，请求开始时 bind_session(sid, context)，
transfer_human 执行时在同一线程内读取。CLI 模式没绑定时降级为
"只有 reason 的最小交接包"。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Optional

from tools.registry import register_tool

# 人工客服入口（真实客服场景脱敏版）
HUMAN_CONTACTS = "热线 950618（京东客服），或 App 内「我的-客户服务」在线咨询"

_HANDOFF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "handoffs")

# 当前请求的会话绑定（线程局部：webapp 每请求一线程，run/execute_tool 同线程）
_CURRENT = threading.local()


def bind_session(session_id: str, context: Any) -> None:
    """请求开始时由 server 调用：把会话 id 和 ContextManager 绑到当前线程。"""
    _CURRENT.sid = session_id
    _CURRENT.context = context


def unbind_session() -> None:
    """请求结束时解绑（防止线程复用时读到旧会话）。"""
    _CURRENT.sid = None
    _CURRENT.context = None


# ----------------------------------------------------------------------
# 槽位抽取（规则版：不调 LLM，交接包生成零成本零延迟）
# ----------------------------------------------------------------------
_ORDER_RE = re.compile(r"JD\d{8,}|\d{10,}")
_AMOUNT_RE = re.compile(r"[¥￥]\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*[元块]")

# 问题类型 → 关键词（按优先级排：越靠前越具体）
_TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("退款退货", ("退款", "退货", "退钱", "退单", "取消订单")),
    ("质量问题", ("质量", "坏了", "破损", "故障", "不能用")),
    ("物流时效", ("发货", "物流", "快递", "到哪", "多久", "送达", "揽收")),
    ("发票税务", ("发票", "税号", "开票", "报销")),
    ("换货维修", ("换货", "换一个", "维修", "保修")),
    ("价保优惠", ("差价", "保价", "降价", "优惠券")),
]
# 情绪标记词
_NEGATIVE_WORDS = ("投诉", "骗子", "垃圾", "过分", "恶心", "退款都没", "等着", "急")
_ANGRY_WORDS = ("投诉", "骗子", "垃圾", "过分", "恶心")


def _extract_slots(messages: list[dict]) -> dict[str, Any]:
    """从会话消息里抽关键槽位：订单号、金额、问题类型、情绪。"""
    user_text = "\n".join(
        m.get("content") or "" for m in messages if m["role"] == "user")
    all_text = "\n".join(str(m.get("content") or "") for m in messages)

    order_ids = sorted(set(_ORDER_RE.findall(user_text)))
    amounts = []
    for m in _AMOUNT_RE.finditer(user_text):
        amounts.append(m.group(1) or m.group(2))

    topic = "其他"
    for name, words in _TOPICS:
        if any(w in user_text for w in words):
            topic = name
            break

    if any(w in all_text for w in _ANGRY_WORDS):
        emotion = "激动（含投诉倾向）"
    elif any(w in all_text for w in _NEGATIVE_WORDS):
        emotion = "着急"
    else:
        emotion = "平稳"

    return {
        "order_ids": order_ids,
        "amounts_mentioned": amounts,
        "topic": topic,
        "emotion": emotion,
    }


def _condense(messages: list[dict], keep: int = 8, per_len: int = 200) -> list[dict]:
    """最近 N 条消息的精简版（每条截断），给人工客服快速过目用。"""
    out = []
    for m in messages[-keep:]:
        content = str(m.get("content") or "")
        if m.get("tool_calls"):
            names = ",".join(tc["function"]["name"] for tc in m["tool_calls"])
            content = f"[调用工具: {names}] {content}"
        out.append({"role": m["role"], "content": content[:per_len]})
    return out


def _write_handoff(reason: str) -> Optional[str]:
    """生成交接包，返回工单号。落盘失败不致命（话术仍可返回）。"""
    sid = getattr(_CURRENT, "sid", None) or "cli"
    context = getattr(_CURRENT, "context", None)
    messages = list(context.messages) if context is not None else []

    # 主要诉求：用户的第一句 + 最后一句（通常一句就够，长会话时最后一句更重要）
    user_msgs = [m.get("content") or "" for m in messages if m["role"] == "user"]
    first_ask = user_msgs[0] if user_msgs else ""
    last_ask = user_msgs[-1] if user_msgs else ""

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    ticket = "GD" + time.strftime("%Y%m%d%H%M%S") + "-" + sid[-4:]

    package = {
        "ticket": ticket,
        "created_at": ts,
        "session_id": sid,
        "reason": reason or "（模型未填写原因）",
        "summary": {
            "first_question": first_ask[:200],
            "last_question": last_ask[:200],
        },
        "slots": _extract_slots(messages),
        "recent_turns": _condense(messages),
    }

    try:
        os.makedirs(_HANDOFF_DIR, exist_ok=True)
        path = os.path.join(_HANDOFF_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}-{sid}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(package, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return ticket
    except OSError as e:
        print(f"[warn] 交接包落盘失败: {e}")
        return None


@register_tool
def transfer_human(reason: str = "") -> str:
    """生成本地人工交接记录并提供人工客服渠道。

    用户明确要求人工、情绪激动或问题超出知识库范围时使用。reason 记录具体诉求。
    """
    ticket = _write_handoff(reason)
    cause = f"（原因：{reason}）" if reason else ""
    if ticket:
        return (
            f"[转人工] 已生成本地人工交接记录，编号 {ticket}。"
            f"如需联系人工客服，可通过{HUMAN_CONTACTS}。{cause}"
        )
    return (
        f"[转人工] 本地人工交接记录未能保存。如需联系人工客服，可通过{HUMAN_CONTACTS}。"
        f"{cause}"
    )


def fallback_message(reason: str = "超出当前知识范围") -> str:
    """通用兜底话术：不编造 + 给退路。"""
    return (
        f"抱歉，这个问题我暂时没有查到可靠答案{reason}。"
        f"为避免误导您，建议通过{HUMAN_CONTACTS}咨询人工客服，"
        f"我们会尽快为您处理。"
    )
