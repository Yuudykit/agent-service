# -*- coding: utf-8 -*-
"""
工具注册表 —— 对标 JD：Function Calling / 工具调用

职责：所有工具的"登记 + 分发 + 异常兜底"唯一出口。

三个核心能力：
    1. register_tool()  把任意 Python 函数登记为 Agent 工具
    2. get_tools_schema()  生成 OpenAI 格式的 tools 列表，喂给 LLM
    3. execute_tool()  按工具名分发执行，异常统一捕获

设计要点：
    - 用装饰器 + 函数 __doc__ 做零配置注册：函数文档字符串就是工具描述，
      参数签名（含类型注解和默认值）就是 parameters schema，天然不重不漏。
    - execute_tool 返回 str 而不是抛异常：tool 结果最终要回填进 messages，
      LLM 看到的是文本，异常信息也要以文本形式让它理解（Observation 语义）。
"""
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Optional

# 全局注册表：name -> {fn, description, parameters}
_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """装饰器：把函数注册进工具表。

    用法：
        @register_tool
        def search_kb(query: str, top_k: int = 3) -> str:
            \"\"\"搜索知识库。\"\"\"
            ...

    参数 schema 从函数签名自动推导：
        - 参数名 + 类型注解 → type（str/int/float/bool）
        - 无默认值的参数 → required
    """
    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        tool_desc = description or (inspect.getdoc(f) or f.__name__)
        parameters = _schema_from_signature(f)
        _REGISTRY[tool_name] = {
            "fn": f,
            "description": tool_desc,
            "parameters": parameters,
        }
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def _schema_from_signature(f: Callable) -> dict[str, Any]:
    """从函数签名推导 JSON Schema（OpenAI function calling 格式）。"""
    sig = inspect.signature(f)
    properties: dict[str, Any] = {}
    required: list[str] = []
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls") or pname.startswith("_"):
            continue
        # 类型注解 → JSON type；没注解默认 string（客服场景参数基本是文本）
        ptype = type_map.get(param.annotation, "string")
        prop: dict[str, Any] = {"type": ptype}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop
    return {"type": "object", "properties": properties, "required": required}


def get_tools_schema() -> list[dict]:
    """返回 OpenAI 风格 tools 列表（喂给 chat() 的 tools=tools_schema）。"""
    tools = []
    for name, meta in _REGISTRY.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": meta["parameters"],
            },
        })
    return tools


def list_tools() -> list[str]:
    """已注册的工具名列表（调试/展示用）。"""
    return list(_REGISTRY.keys())


def execute_tool(name: str, arguments: dict[str, Any] | str) -> str:
    """执行工具，返回字符串结果。

    - arguments 兼容 dict 或 JSON 字符串（LLM 有时返回字符串）
    - 工具内部异常 → 返回以 "[工具异常]" 开头的文本，
      让 LLM 感知到失败而不是默默崩掉
    - 参数缺失 → 用 schema 里的 default 补位
    """
    meta = _REGISTRY.get(name)
    if meta is None:
        return f"[工具异常] 未知工具: {name}。可用工具: {', '.join(list_tools())}"

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return f"[工具异常] 参数不是合法 JSON: {e}"
    if not isinstance(arguments, dict):
        return "[工具异常] 参数必须是 JSON 对象"

    # 缺失参数用 schema 默认值补位
    filled: dict[str, Any] = {}
    for pname, prop in meta["parameters"]["properties"].items():
        if pname in arguments:
            filled[pname] = arguments[pname]
        elif "default" in prop:
            filled[pname] = prop["default"]

    try:
        result = meta["fn"](**filled)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        # 缺必填参数
        return f"[工具异常] 参数不完整: {e}"
    except Exception as e:  # noqa: BLE001 —— 工具异常必须全兜住
        return f"[工具异常] {type(e).__name__}: {e}"
