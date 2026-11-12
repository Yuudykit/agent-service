# -*- coding: utf-8 -*-
"""
对话记忆 —— 对标 JD：Agent Memory

职责：让 Agent 记住「这轮之前发生了什么」，不只靠 messages 原文。

三层记忆（对应行业里的 short-term / long-term / episodic）：
    1. 短期记忆：就是 context.messages —— 最近几轮的原文（Day 4 已有）
    2. 长期记忆：会话级摘要 —— 把早期对话压成一段事实摘要，跨会话可复用
    3. 用户画像：记住用户的偏好/背景（如"用户要开增值税专用发票"），长期有效

本次实现：
    - summarize()：双通道
        真实模式：调 LLM 生成事实型摘要（延迟导入 openai，只记事实不记寒暄）
        mock 模式：规则抽取（抓订单号/业务意图/关键实体），零依赖离线可用
    - update_profile()：正则抽取用户事实 → KV 画像
    - recall()：生成可注入 system prompt 的记忆回顾文本
    - 持久化：data/memory.json（_load/_save）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from core.config import load_config

# 业务意图词 → 画像标签（客服场景）
_INTENT_RULES: list[tuple[str, str]] = [
    ("发票|开票|税号|报销", "发票"),
    ("退钱|退款|退货款", "退款"),
    ("退货|退回|寄回", "退货"),
    ("换货|换新|换一个", "换货"),
    ("发货|物流|快递|配送|到货|签收", "物流"),
    ("保修|维修|质保|售后", "售后"),
    ("优惠券|满减|优惠|活动", "优惠"),
    ("差价|价保|保价", "价保"),
    ("人工|投诉|专员|客服热线", "转人工"),
]

# 订单号抽取：脱敏格式 [订单号:x] / 纯数字 6-24 位
_ORDER_RE = re.compile(r"订单号[:：]?\s*\[?([0-9]{6,24})\]?|[0-9]{12,20}")


class Memory:
    """对话记忆：会话摘要 + 用户画像。持久化到 data/memory.json。"""

    def __init__(self, path: str = "data/memory.json"):
        self.path = path
        self.summaries: list[str] = []     # 按时间排列的会话摘要
        self.profile: dict[str, Any] = {}  # 用户画像 KV
        self._load()

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------
    def summarize(self, messages: list[dict]) -> str:
        """把一段对话压成摘要。

        真实模式：调 LLM 生成"事实型摘要"（延迟导入，避免循环依赖）；
        mock 模式：规则抽取——保留用户说过的业务事实，丢掉寒暄。
        """
        if os.environ.get("MEMORY_BACKEND") == "llm" and self._has_api_key():
            return self._summarize_llm(messages)
        return self._summarize_rules(messages)

    def _has_api_key(self) -> bool:
        try:
            return bool(load_config().get("api_key"))
        except (OSError, json.JSONDecodeError):
            return False

    def _summarize_llm(self, messages: list[dict]) -> str:
        """调 LLM 压缩摘要：只保留事实，可被后续对话直接使用。"""
        from core.llm_client import get_client

        cfg = load_config()
        client = get_client(cfg["api_key"], cfg["base_url"])
        transcript = "\n".join(
            f"{m['role']}: {m.get('content') or ''}" for m in messages if m.get("content")
        )
        prompt = (
            "把下面的客服对话压缩成一段事实摘要（中文，100 字内）。"
            "只保留：用户咨询的业务、订单号、用户诉求、客服给的关键结论。"
            "不要寒暄。\n\n对话：\n" + transcript
        )
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    def _summarize_rules(self, messages: list[dict]) -> str:
        """规则抽取摘要：抓用户消息里的业务意图与实体，拼成事实句。"""
        facts: list[str] = []
        for m in messages:
            if m.get("role") != "user":
                continue
            text = m.get("content") or ""
            intents = [tag for pat, tag in _INTENT_RULES if re.search(pat, text)]
            orders = re.findall(r"[0-9]{12,20}", text)
            if intents:
                fact = "用户咨询：" + "、".join(dict.fromkeys(intents))
                if fact not in facts:
                    facts.append(fact)
            if orders and ("订单 " + orders[0]) not in facts:
                facts.append("涉及订单：" + orders[0])
        return "；".join(facts[:5]) if facts else "（无实质业务内容）"

    def store_summary(self, summary: str) -> None:
        """追加一条会话摘要并落盘（最多保留 10 条）。"""
        self.summaries.append(summary)
        self.summaries = self.summaries[-10:]
        self._save()

    # ------------------------------------------------------------------
    # 用户画像
    # ------------------------------------------------------------------
    def update_profile(self, message: str) -> None:
        """从用户发言里抽取事实（客服场景：业务意图/订单号/发票诉求）。"""
        changed = False

        for pat, tag in _INTENT_RULES:
            if re.search(pat, message):
                intents = self.profile.setdefault("intents", [])
                if tag not in intents:
                    intents.append(tag)
                    changed = True

        m = re.search(r"(企业|公司|单位).{0,6}(抬头|税号)", message)
        if m or re.search(r"增值税专用发票|专票|企业发票", message):
            key = "enterprise_invoice"
            if self.profile.get(key) != "需要企业发票":
                self.profile[key] = "需要企业发票"
                changed = True

        orders = re.findall(r"[0-9]{12,20}", message)
        if orders:
            known = self.profile.setdefault("order_ids", [])
            for o in orders:
                if o not in known:
                    known.append(o)
                    changed = True

        if changed:
            self._save()

    # ------------------------------------------------------------------
    # 回顾与持久化
    # ------------------------------------------------------------------
    def recall(self) -> str:
        """生成记忆回顾文本，注入 system prompt。无记忆时返回空串。"""
        parts: list[str] = []

        if self.profile.get("intents"):
            parts.append(
                "用户历史关注：" + "、".join(self.profile["intents"])
            )
        if self.profile.get("enterprise_invoice"):
            parts.append("用户需要开具企业发票（增值税专用/企业抬头），涉及发票问题优先走企业开票流程")
        if self.profile.get("order_ids"):
            parts.append("用户提到的订单号：" + "、".join(self.profile["order_ids"]))

        if self.summaries:
            parts.append("历史会话摘要：" + " | ".join(self.summaries[-2:]))

        return "\n".join(parts) if parts else ""

    def _load(self) -> None:
        """从磁盘读回记忆。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                self.summaries = data.get("summaries", [])
                self.profile = data.get("profile", {})
            except (json.JSONDecodeError, OSError):
                self.summaries, self.profile = [], {}

    def _save(self) -> None:
        """记忆落盘（JSON）。"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"summaries": self.summaries, "profile": self.profile},
                f, ensure_ascii=False, indent=1,
            )
