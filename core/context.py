# -*- coding: utf-8 -*-
"""
上下文管理 —— 对标 JD：上下文管理

职责：管理 messages 列表的生命周期，让 Agent 在有限的上下文窗口里长期工作。

核心问题：
    对话越长，messages 越长，最终会撑爆模型上下文窗口（如 4096 token）。
    必须决定"哪些历史保留、哪些丢弃、哪些压缩"。

方案（三层递进）：
    1. 窗口滑动：只保留最近 N 条消息（粗暴但有效）
    2. Token 截断：总 token 超限时，从最旧的开始丢
    3. 摘要压缩：对话很长时，把早期对话压成一段摘要，替代原文（Day 6 memory 配合）

设计要点：
    - 裁剪以"完整回合"为单位：一个 user 消息 + 其后直到下一个 user 之前的
      所有消息（assistant 的 tool_calls 与 tool 结果）必须同生共死，
      否则会出现"tool 结果还在、调它的 assistant 消息却没了"的孤立配对，
      传 API 会直接报错。
    - 摘要占位用 role="system" 并固定在 messages[0]，保证 system 始终在前，
      兼容 DeepSeek / OpenAI 的消息校验。
"""
from __future__ import annotations

from typing import Any, Optional


class ContextManager:
    """管理 messages 列表：增删、窗口、截断、摘要占位。"""

    def __init__(self, max_tokens: int = 4096, max_messages: int = 20):
        self.max_tokens = max_tokens      # 上下文窗口上限（token）
        self.max_messages = max_messages  # 保留消息条数上限
        self.messages: list[dict] = []    # 完整消息历史（含可能被摘要压缩的部分）

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_user(self, text: str) -> None:
        """追加一条 user 消息。"""
        self.messages.append({"role": "user", "content": text})
        self._after_append()

    def add_assistant(self, text: str, tool_calls: Optional[list[dict]] = None) -> None:
        """追加一条 assistant 消息。

        tool_calls: function calling 的标准结构（[{id, type, function}]）。
        模型调用工具时，这字段必须原样保留在消息里，tool 结果靠
        tool_call_id 与它配对。
        """
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._after_append()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """追加一条 role=tool 的消息（function calling 回填）。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._after_append()

    # ------------------------------------------------------------------
    # 内部：每次追加后的维护（窗口 → token → 摘要）
    # ------------------------------------------------------------------
    def _after_append(self) -> None:
        self._trim_window()
        self._summarize_if_needed()

    # ------------------------------------------------------------------
    # 第 1 层：窗口滑动
    # ------------------------------------------------------------------
    def _trim_window(self) -> None:
        """超出 max_messages 时，按"完整回合"丢弃最旧消息。

        回合定义：一条 user 消息 + 其后直到下一条 user 消息之前的所有消息。
        以 user 消息为切点整段删除，保证 tool_calls 与 tool 结果永不分离。
        """
        while len(self.messages) > self.max_messages:
            # 找第一条 user 消息的位置
            first_user = self._find_first_user()
            if first_user is None:
                break  # 没有 user 消息（只有摘要占位），不再裁剪
            # 找下一条 user 消息的位置作为回合结束边界
            next_user = None
            for i in range(first_user + 1, len(self.messages)):
                if self.messages[i]["role"] == "user":
                    next_user = i
                    break
            end = next_user if next_user is not None else len(self.messages)
            del self.messages[first_user:end]

    def _find_first_user(self) -> Optional[int]:
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user":
                return i
        return None

    # ------------------------------------------------------------------
    # 第 2 层：Token 估算（无分词器的近似）
    # ------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数。

        近似规则（够用且面试好讲）：
            - 中文：1 个字 ≈ 1 token
            - 英文：1 个单词 ≈ 1.3 token（4 字符 ≈ 1 token）
        实测：BPE 分词对中文密集文本的偏差通常 < 20%，对窗口裁剪足够。
        """
        if not text:
            return 0
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other_chars = len(text) - cjk
        # 非中文字符按 4 字符 ≈ 1 token；中文按 1 字 ≈ 1 token
        return cjk + other_chars // 4 + 1

    def _total_tokens(self) -> int:
        """当前 messages 的总 token（每条消息的 role 也算一点开销）。"""
        total = 0
        for msg in self.messages:
            content = msg.get("content") or ""
            total += self._estimate_tokens(content) + 4  # 每消息固定开销
            # assistant 消息里的 tool_calls 参数也要算
            for tc in msg.get("tool_calls") or []:
                total += self._estimate_tokens(
                    json_dumps(tc.get("function", {}).get("arguments", ""))
                )
        return total

    # ------------------------------------------------------------------
    # 第 3 层：摘要压缩（Day 4 只做触发判定 + 占位，Day 6 接 memory）
    # ------------------------------------------------------------------
    def _summarize_if_needed(self) -> None:
        """总 token 超限时，把最早一个完整回合压缩成摘要占位。

        递归执行：压完一轮仍超限就继续压下一轮，直到达标或无可压缩。
        Day 4：占位文本写死，标记 [TODO Day6: memory.summarize]；
        Day 6：替换成调 LLM 生成的真实事实摘要（core/memory.py）。
        """
        while self._total_tokens() > self.max_tokens:
            # 找到第一个非摘要占位的完整回合
            first_user = self._find_first_user()
            if first_user is None:
                break  # 全是摘要，压无可压
            next_user = None
            for i in range(first_user + 1, len(self.messages)):
                if self.messages[i]["role"] == "user":
                    next_user = i
                    break
            end = next_user if next_user is not None else len(self.messages)

            # 把这一回合的原文拼成待压缩文本
            raw = "\n".join(
                f"{m['role']}: {m.get('content') or m.get('tool_calls') or ''}"
                for m in self.messages[first_user:end]
            )
            # TODO(Day 6): summary = memory.summarize(raw)
            summary = f"[历史对话摘要] {raw[:120]}……" if raw else "[历史对话摘要]"

            # 整段替换成一条摘要占位，并固定在最前面（保证 system 在前）
            del self.messages[first_user:end]
            self.messages.insert(0, {"role": "system", "content": summary})

    # ------------------------------------------------------------------
    # 读取 / 重置
    # ------------------------------------------------------------------
    def get_messages(self) -> list[dict]:
        """返回当前应发给 LLM 的消息列表（不含系统提示词，由调用方拼装）。"""
        return list(self.messages)

    def reset(self) -> None:
        """清空上下文（新会话）。"""
        self.messages = []


def json_dumps(obj: Any) -> str:
    """小工具：dict/str 安全转 JSON 字符串（避免顶层 import json 的循环依赖感）。"""
    import json
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)
