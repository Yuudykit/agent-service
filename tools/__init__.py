# -*- coding: utf-8 -*-
"""tools 包：工具注册表 + 各业务工具（查询类/操作类/兜底类）。

导入本包时自动注册所有工具（@register_tool 装饰器生效），
因此 agent_loop 只需 `from tools.registry import ...` 即可拿到完整工具表。
新增工具 = 新建一个模块 + 在下方加一行导入。
"""
from tools import calculate_refund  # noqa: F401  退款金额计算
from tools import fallback  # noqa: F401  转人工 / 兜底话术
from tools import query_info  # noqa: F401  订单结构化查询
from tools import search_kb  # noqa: F401  知识库检索（RAG 混合检索）
from tools.registry import execute_tool, get_tools_schema, list_tools, register_tool

__all__ = [
    "register_tool",
    "execute_tool",
    "get_tools_schema",
    "list_tools",
]
