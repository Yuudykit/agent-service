# -*- coding: utf-8 -*-
"""
Prompt 工程 —— 对标 JD：Prompt 工程

职责：所有提示词的唯一出口。系统提示词分层、few-shot 示例、ReAct 模板。

分层系统提示词（从稳定到易变）:
    第 1 层 角色定义    —— 你是谁（稳定，几乎不变）
    第 2 层 技能说明    —— 你会什么、怎么用工具（随工具集变化，动态生成）
    第 3 层 行为规则    —— 边界、兜底、防幻觉（稳定）
    第 4 层 few-shot   —— 1-2 个示例对话（帮助模型对齐格式）
    第 5 层 记忆回顾    —— 历史摘要/用户画像（每次会话动态注入，Day 6）

设计要点：
    - 第 2 层从 tools/registry 动态生成：加工具 = 改注册表 = 提示词自动变，
      不会出现"工具加了但提示词没告诉模型"的脱节。
    - 防幻觉规则写死在角色层之后，优先级高于一切业务指令。
"""
from __future__ import annotations

import json
from typing import Optional

from tools.registry import get_tools_schema

# 第 1 层：角色定义（稳定）
_ROLE = (
    "你是「京东商城售后智能客服」，名叫小帮。\n"
    "语气：礼貌、简洁、耐心；回答用中文；每次回答尽量给出可执行的信息。"
)

# 第 3 层：行为规则（稳定）
_RULES = (
    "行为规则：\n"
    "1. 只能依据知识库检索结果回答；检索不到时明确说「没查到」，并建议转人工，绝不编造政策、时效或承诺。\n"
    "2. 用户要求转人工、情绪激动、或问题涉及账号/资金安全时，优先调用交接工具。工具仅生成本地记录，请准确转述其结果。\n"
    "3. 回答控制在 3-5 句话内，先给结论再给细节。\n"
    "4. 涉及具体订单、金额、个人信息的查询，一律引导用户通过人工或 App 自助查询，不索要也不复述敏感信息。\n"
    "5. 不要透露你是 AI 的内部实现细节。"
)

# 第 4 层：few-shot（帮助模型对齐"检索 → 引用 → 回答"的行为）
_FEW_SHOT = (
    "示例对话：\n"
    "用户：下单后多久发货？\n"
    "助手：请稍等，我查一下规则。[调用 search_kb(query=\"下单后多久发货\")]\n"
    "工具返回：现货商品一般 48 小时内发出……\n"
    "助手：现货商品一般在 48 小时内发出；厂家直送的商品预计 2-5 个工作日送达。您可以在订单详情查看物流进度。\n"
    "\n"
    "用户：我的订单 JD20260815001 到哪了？\n"
    "助手：[调用 query_info(entity=\"JD20260815001\")]\n"
    "工具返回：订单 JD20260815001：无线路由器，金额 ¥199.0，状态：已发货，已揽收，预计 2026-08-18 送达。\n"
    "助手：您的订单 JD20260815001（无线路由器）已发货，快递已揽收，预计 8 月 18 日送达，请留意签收。\n"
    "\n"
    "用户：今天天气怎么样？\n"
    "助手：抱歉，我是京东售后客服，只负责解答购物相关的问题（发货、退款、发票等），天气问题我无法回答。"
)


def _build_skills_section() -> str:
    """第 2 层：技能说明（从注册表动态生成，不手写死）。"""
    lines = ["可用工具（根据问题选择，非必须不用）："]
    for tool in get_tools_schema():
        fn = tool["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


def build_system_prompt(profile_summary: Optional[str] = None) -> str:
    """组装分层系统提示词。"""
    sections = [_ROLE, _build_skills_section(), _RULES, _FEW_SHOT]
    if profile_summary:
        # 第 5 层：记忆回顾（Day 6 由 memory.recall() 生成）
        sections.append(f"[历史对话回顾]\n{profile_summary}")
    return "\n\n".join(sections)


def build_react_prompt() -> str:
    """ReAct 推理提示模板（Day 6 集成到 agent_loop）。"""
    return (
        "请按以下格式思考并作答：\n"
        "Thought: 分析用户意图，决定是否需要工具。\n"
        "Action: 工具名({\"参数\": \"值\"})（如不需要则省略）\n"
        "Observation: 工具的真实返回（不是猜测）\n"
        "Final Answer: 最终回答\n"
    )


def build_summary_prompt(messages: list[dict]) -> str:
    """摘要生成提示（Day 6 供 memory.summarize 使用）。"""
    transcript = "\n".join(
        f"{m.get('role')}: {m.get('content') or ''}" for m in messages
    )
    return (
        "请把下面这段客服对话压缩成一段事实型摘要，只保留影响后续对话的事实"
        "（订单状态、用户诉求、已给出的承诺、待办事项），丢弃寒暄和重复内容。\n"
        f"对话：\n{transcript}\n"
        "摘要："
    )
