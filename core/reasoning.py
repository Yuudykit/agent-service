# -*- coding: utf-8 -*-
"""
规划与推理 —— 对标 JD：规划与推理

职责：让 Agent 不只是"单次 tool 调用"，而是能"先想后做"。

方案：ReAct（Reasoning + Acting）循环

    每轮让 LLM 输出：
      Thought:  我先想一下用户要什么、需要哪个工具、参数怎么填
      Action:   调用哪个工具 + 参数
      Observation: 工具返回了什么（这是真实结果，不是模型猜的）
    然后重复，直到 LLM 判断可以给出 Final Answer。

    和 function calling 循环的关系：
      - function calling 是"模型直接输出 tool_calls"（隐式推理）
      - ReAct 是"模型先输出 Thought 再输出 Action"（显式推理）
      - 工程上两者可以结合：agent_loop 主循环用 function calling
        （DeepSeek 原生支持、业界主流），ReActPlanner 提供
        ① 步数控制与终止判定 ② 文本协议解析（Action/Final Answer 格式）
        ③ mock 模式下的显式推理轨迹展示（verbose 演示用）

核心组件：
    should_continue(step, last_observation)  —— 是否继续循环
    extract_action(llm_output)               —— 解析 Action → (工具名, 参数)
    extract_final_answer(llm_output)         —— 判断/提取 Final Answer
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Action 行格式：Action: search_kb({"query": "..."})
_ACTION_RE = re.compile(
    r"Action\s*:\s*([a-zA-Z_]\w*)\s*\(\s*(\{.*?\}|.*?)\s*\)\s*$",
    re.DOTALL,
)
# Final Answer 行格式：Final Answer: xxx
_FINAL_RE = re.compile(r"Final\s+Answer\s*:\s*(.+)$", re.DOTALL)
# Thought 行格式：Thought: xxx
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+)$", re.DOTALL)


class ReActPlanner:
    """ReAct 循环的步数控制与结果判定。"""

    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps

    def should_continue(self, step: int, last_observation: Optional[str]) -> bool:
        """判断是否继续循环。

        规则:
            - step 超过 max_steps → False（转兜底）
            - last_observation 为空/异常（含"未找到"/"错误"/"失败"）→ False（转兜底）
            - 否则 → True
        """
        if step > self.max_steps:
            return False
        if last_observation is None:
            return True  # 还没有任何观察，先跑第一步
        text = str(last_observation)
        if not text.strip():
            return False
        if any(kw in text for kw in ("未找到", "错误", "失败", "异常", "NOT_FOUND")):
            return False
        return True

    def extract_action(self, llm_output: str) -> tuple[str, dict] | None:
        """从 LLM 输出里解析 Action（工具名 + 参数）。

        兼容三种写法：
            Action: search_kb({"query": "退款"})          # JSON 参数
            Action: search_kb(query="退款", top_k=3)      # kwargs 写法
            Action: transfer_human(reason="投诉")          # 无空格

        解析失败返回 None（调用方应走兜底）。
        """
        m = _ACTION_RE.search(llm_output)
        if not m:
            return None
        name, raw_args = m.group(1), m.group(2).strip()

        # 写法 1：JSON
        try:
            args = json.loads(raw_args)
            if isinstance(args, dict):
                return name, args
        except json.JSONDecodeError:
            pass

        # 写法 2：kwargs（key=value，值带引号或数字）
        kwargs: dict[str, Any] = {}
        pat = re.compile(r"([a-zA-Z_]\w*)\s*=\s*(\"[^\"]*\"|'[^']*'|\d+(?:\.\d+)?)")
        for k, v in pat.findall(raw_args):
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                kwargs[k] = v[1:-1]
            else:
                try:
                    kwargs[k] = int(v) if "." not in v else float(v)
                except ValueError:
                    kwargs[k] = v
        if kwargs:
            return name, kwargs

        # 写法 3：裸参数字符串 → 当作 query
        if raw_args and not raw_args.startswith("{"):
            return name, {"query": raw_args}
        return name, {}

    def extract_final_answer(self, llm_output: str) -> Optional[str]:
        """判断 LLM 输出是否为 Final Answer，是则提取。"""
        m = _FINAL_RE.search(llm_output)
        return m.group(1).strip() if m else None

    def extract_thought(self, llm_output: str) -> Optional[str]:
        """提取 Thought 行（verbose 展示用，非必须）。"""
        m = _THOUGHT_RE.search(llm_output)
        return m.group(1).strip() if m else None

    def build_react_prompt(self, user_input: str, tool_names: list[str]) -> str:
        """构造 ReAct 文本协议提示词（给不支持 function calling 的模型备用）。"""
        return (
            "你是一个电商客服助手。请严格按以下格式输出：\n"
            "Thought: 分析用户意图，决定是否调用工具、调哪个、参数是什么\n"
            "Action: 工具名(参数 JSON)\n"
            "（收到工具结果后继续 Thought/Action 循环）\n"
            "Final Answer: 最终回复用户的话\n\n"
            f"可用工具：{', '.join(tool_names)}\n\n"
            f"用户问题：{user_input}"
        )
