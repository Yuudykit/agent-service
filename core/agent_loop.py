# -*- coding: utf-8 -*-
"""
Agent 主循环 —— 项目的灵魂模块。

对标 JD：Function Calling + 工具调用

职责：把「LLM 判断 + 工具执行 + 结果回填」串成一个可多轮运行的循环。

数据流（Day 1 六步循环的升级版）:
    1. 用户输入 → 追加到 messages
    2. 调 LLM → 得到 assistant message
    3. 判断有没有 tool_calls
       ├─ 没有 → 这就是最终回答，返回
       └─ 有   → 逐个执行工具（tools/registry 分发）
    4. 工具结果以 role=tool 回填 messages
    5. 回到 2（有 max_steps 上限，防死循环）
    6. 多轮对话：本轮结束后 messages 保留在 ContextManager，下轮接着用

两种模式：
    - mock=True：关键词规则模拟"模型判断"，不依赖 API，可离线演示/测试
    - mock=False：真调 DeepSeek（OpenAI 兼容协议），tools=tools_schema
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional

from core.config import load_config
from core.context import ContextManager
from core.llm_client import get_client
from core.memory import Memory
from core.prompts import build_system_prompt
from tools.fallback import fallback_message
from tools.registry import execute_tool, get_tools_schema, list_tools

# ----------------------------------------------------------------------
# mock 模式：关键词 → 模拟模型的工具判断
# ----------------------------------------------------------------------
# 业务词 → 触发 search_kb（客服问题）
_BUSINESS_WORDS = [
    "发货", "物流", "快递", "配送", "到货", "多久",
    "退款", "退货", "退钱", "换货", "维修", "保修", "售后",
    "发票", "税号", "差价", "保价", "优惠", "券", "订单", "取消",
    "缺货", "补货", "先发", "有货", "付款", "支付", "签收",
    "退",  # 简短意图兜底：单字"退"（退退退/退钱）视为退货退款
]
# 转人工词
_HUMAN_WORDS = ["人工", "客服热线", "打电话", "950618", "投诉", "专员"]
# 寒暄词 → 直接答（不调工具）
_GREETING_WORDS = ["在吗", "你好", "您好", "在不在", "谢谢", "辛苦了", "嗯"]


def _mock_judge(user_text: str) -> dict[str, Any]:
    """mock 版模型判断：返回 {"reply": str} 或 {"tool": name, "arguments": dict}。"""
    text = user_text.strip()
    import re as _re

    # 0) 转人工（最高优先级）
    if any(w in text for w in _HUMAN_WORDS):
        reason = text.partition("诉求：")[2].strip() or "用户要求转人工"
        return {"tool": "transfer_human", "arguments": {"reason": reason}}

    # 1) 订单号（JD 前缀 + 数字，或 10 位以上纯数字）→ 精确查询 / 退款计算
    #    语境分流：含"钱/多少/算"（金额语境）→ calculate_refund；否则查状态/物流 → query_info
    m = _re.search(r"JD\d{8,}|\d{10,}", text)
    if m:
        oid = m.group()
        if any(w in text for w in ("钱", "多少", "算")):
            return {
                "tool": "calculate_refund",
                "arguments": {"order_id": oid, "quality_issue": "质量" in text or "坏" in text},
            }
        return {"tool": "query_info", "arguments": {"entity": oid}}

    # 2) 业务问题 → 检索知识库
    if any(w in text for w in _BUSINESS_WORDS):
        return {"tool": "search_kb", "arguments": {"query": text}}
    # 3) 寒暄
    if any(w in text for w in _GREETING_WORDS):
        return {"reply": "您好，我是智能客服小帮手，可以帮您解答发货、退款、发票等售后问题，请问有什么需要帮助？"}
    # 4) 无关话题 → 明确边界（防幻觉）
    return {
        "reply": "这个问题超出了我的服务范围。我只负责解答京东商城购物相关的售后问题"
                 "（发货、退款、发票、换货等），其他问题建议您咨询相应渠道。"
    }


def _mock_chat(messages: list[dict]) -> dict[str, Any]:
    """用最后一条 user 消息做关键词判断，返回 OpenAI 风格的 assistant message。

    关键状态机：如果最后一条是 role=tool（工具刚执行完、正在回填阶段），
    模拟真实 LLM 的"阅读工具结果 → 总结成最终回答"，否则会死循环调工具。
    """
    # 状态 1：刚拿到工具结果 → 总结回答
    if messages and messages[-1]["role"] == "tool":
        content = messages[-1]["content"] or ""
        if content.startswith("[检索结果]"):
            # 综合前 2 条命中答案（模拟 LLM 阅读检索结果后组织回答，
            # 而不是只复述第一条——更接近真实 RAG 行为）
            # 注意：用 lstrip 匹配"答："行，兼容不同缩进（P1 search_kb 用 4 空格）
            lines = [ln for ln in content.splitlines() if ln.lstrip().startswith("答：")]
            if lines:
                parts = [ln.lstrip()[len("答："):].strip() for ln in lines[:2]]
                answer = parts[0]
                for extra in parts[1:]:
                    answer += "；另外，" + extra
                return {"role": "assistant", "content": f"根据平台规则：{answer}"}
            if "未找到" in content:
                return {
                    "role": "assistant",
                    "content": "抱歉，这个问题我暂时没有查到准确信息。为避免给您错误答复，建议联系人工客服（950618）进一步核实。",
                }
        if content.startswith("[转人工]"):
            return {
                "role": "assistant",
                "content": content.removeprefix("[转人工]").lstrip(),
            }
        if content.startswith("[NOT_FOUND]"):
            return {
                "role": "assistant",
                "content": "没有查询到相关信息，可能是订单号有误。请您核对订单号后重新提供，或通过「我的-订单」自助查询。",
            }
        if content.startswith(("[查询结果]", "[计算结果]")):
            # 结构化查询/计算的结果本身就是最终答案
            return {"role": "assistant", "content": content}
        # 工具返回了其他内容：原样交给用户（mock 的兜底行为）
        return {"role": "assistant", "content": content}

    # 状态 2：新用户消息 → 关键词路由
    user_text = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_text = m["content"] or ""
            break
    judge = _mock_judge(user_text)

    if "tool" in judge:
        # 模拟 LLM 输出的 tool_calls 结构
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_mock_1",
                "type": "function",
                "function": {
                    "name": judge["tool"],
                    "arguments": json.dumps(judge["arguments"], ensure_ascii=False),
                },
            }],
        }
    return {"role": "assistant", "content": judge["reply"]}


# ----------------------------------------------------------------------
# chat()：单次 LLM 调用
# ----------------------------------------------------------------------
def chat(
    messages: list[dict],
    system_prompt: str,
    tools_schema: list[dict],
    verbose: bool = False,
    mock: bool = False,
    on_delta: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """调 LLM 一次，返回完整 assistant message（含可能的 tool_calls）。

    on_delta: 可选回调。传入后真实模式改用流式请求，每收到一段
    文本增量就回调一次（前端 SSE 逐字渲染的数据源）。
    mock 模式无流可流，不回调。
    """
    if mock:
        msg = _mock_chat(messages)
        if verbose:
            tools = (msg.get("tool_calls") or [])
            if tools:
                fn = tools[0]["function"]
                print(f"  [react] Thought: 分析用户意图 → 调用工具 {fn['name']}")
                print(f"  [react] Action: {fn['name']}({fn['arguments']})")
            else:
                print(f"  [react] Final Answer: {msg['content'][:50]}…")
        return msg

    # ---- 真实调用：DeepSeek（OpenAI 兼容）----
    config = load_config()
    if not config.get("api_key"):
        raise RuntimeError(
            "请将 config.example.json 复制为 config.json 并填写 API Key，"
            "或使用 mock 模式离线演示。"
        )
    client = get_client(config["api_key"], config["base_url"])
    payload = {
        "model": config["model"],
        "messages": [{"role": "system", "content": system_prompt}, *messages],
    }
    if tools_schema:
        payload["tools"] = tools_schema

    if verbose:
        print(f"  [api] {config['model']} 请求 {len(payload['messages'])} 条消息, "
              f"{len(tools_schema)} 个工具"
              f"{'（流式）' if on_delta else ''}")

    # 健壮性三件套之一二：60s 超时 + 429/5xx/连接错误指数退避重试（共 3 次尝试）
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            if on_delta is None:
                return _chat_once(client, payload)
            return _chat_stream(client, payload, on_delta)
        except Exception as e:  # noqa: BLE001 —— 先判断是否可重试
            if not _RETRYABLE(e):
                raise  # 不可重试的错误（鉴权失败/参数错误等）直接上抛
            last_err = e
            if attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                if verbose:
                    print(f"  [api] {type(e).__name__}，{wait}s 后重试 "
                          f"({attempt + 1}/2)")
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _RETRYABLE(exc: Exception) -> bool:
    """判断异常是否值得重试：429 限流、5xx 服务端错误、连接/超时类网络错误。"""
    import openai
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                        openai.InternalServerError)):
        return True
    # APITimeoutError 是 APIConnectionError 的子类，上面已覆盖
    return False


def _chat_once(client, payload: dict) -> dict[str, Any]:
    """非流式调用（原逻辑）：完整响应一次性返回。"""
    resp = client.chat.completions.create(**payload)
    msg = resp.choices[0].message
    result = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    if getattr(resp, "usage", None) is not None:
        result["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return result


def _chat_stream(client, payload: dict, on_delta: Callable[[str], None]) -> dict[str, Any]:
    """流式调用：逐块下发文本增量，同时聚合出与非流式一致的结果结构。

    聚合两件事：
    - content：增量拼接，每块回调 on_delta
    - tool_calls：按 index 累加（id/name 首块到达，arguments 分片追加）
    - usage：靠 stream_options.include_usage，最后一块带统计
    """
    stream_payload = {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    usage: dict | None = None

    for chunk in client.chat.completions.create(**stream_payload):
        # usage 专用块：choices 为空
        if getattr(chunk, "usage", None):
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
            continue
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        if delta is None:
            continue
        # 文本增量
        if delta.content:
            content_parts.append(delta.content)
            on_delta(delta.content)
        # 工具调用增量（按 index 聚合）
        for tc in (getattr(delta, "tool_calls", None) or []):
            slot = tool_calls_acc.setdefault(tc.index, {
                "id": "", "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["function"]["name"] += tc.function.name
                if tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments

    result = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls_acc:
        result["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    if usage:
        result["usage"] = usage
    return result


# ----------------------------------------------------------------------
# run()：完整一轮对话
# ----------------------------------------------------------------------
def run(
    user_input: str,
    context: ContextManager,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mock: bool = False,
    max_steps: int = 3,
    memory: Optional[Memory] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """完整的一轮对话：用户输入 → 可能多步工具调用 → 最终回答。

    核心循环（六步）：
        1. 用户输入追加进 context
        2. chat() 调 LLM
        3. 无 tool_calls → 返回 content（最终回答）
        4. 有 tool_calls → execute_tool 逐个执行
        5. 工具结果 add_tool_result 回填
        6. 回到 2，直到无 tool_calls 或超过 max_steps

    可选 memory：传入 Memory 实例后，历史记忆会注入 system prompt，
    本轮用户发言也会喂给用户画像（update_profile）。

    可选 on_event：进程内事件流（webapp SSE 的数据源）：
        {"type": "tool",  "name": "query_info"}        每次工具执行前
        {"type": "delta", "text": "..."}               最终回答的文本增量
    mock / 真实模式通用（mock 无 delta，只有 tool 事件）。
    """
    def _emit(evt: dict) -> None:
        if on_event is not None:
            try:
                on_event(evt)
            except Exception:  # noqa: BLE001 —— 事件回调绝不能打断主流程
                pass

    if system_prompt is None:
        system_prompt = build_system_prompt()
    # 记忆注入：历史摘要 + 用户画像 → 系统提示词尾部
    if memory is not None:
        recall_text = memory.recall()
        if recall_text:
            system_prompt = system_prompt + "\n\n[历史记忆]\n" + recall_text
    tools_schema = get_tools_schema()

    # 步骤 1
    context.add_user(user_input)
    if verbose:
        print(f"  [ctx] user: {user_input}")

    # 步骤 2-6：ReAct 循环
    for step in range(1, max_steps + 1):
        if verbose:
            print(f"  [step] {step}/{max_steps}")

        assistant = chat(
            context.get_messages(),
            system_prompt=system_prompt,
            tools_schema=tools_schema,
            verbose=verbose,
            mock=mock,
            on_delta=(lambda d: _emit({"type": "delta", "text": d}))
            if on_event is not None else None,
        )
        # token 用量事件：真实模式每次调用上报，供 server 做
        # 日预算熔断 / 结构化日志（mock 模式无 usage，不上报）
        if assistant.get("usage"):
            _emit({"type": "usage", "usage": assistant["usage"]})

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            # 步骤 3：最终回答（同时更新用户画像）
            # 回答本身也要回填 messages——否则多轮上下文里永远缺
            # assistant 的回复，历史是"断"的（会话落盘后尤其明显）
            context.add_assistant(text=assistant["content"])
            if memory is not None:
                memory.update_profile(user_input)
            return assistant["content"]

        # 步骤 4-5：先回填 assistant 消息（带全部 tool_calls，仅一次），
        # 再逐个执行工具并把结果按 tool_call_id 回填。
        # 消息顺序：assistant(tool_calls) → tool → tool → …（OpenAI 配对规则）
        context.add_assistant(text=assistant["content"], tool_calls=tool_calls)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            _emit({"type": "tool", "name": fn_name})
            result = execute_tool(fn_name, fn_args)
            if verbose:
                print(f"  [react] Observation: {result[:70].replace(chr(10), ' ')}…")
            context.add_tool_result(tool_call_id=tc["id"], content=result)

        # 步骤 6：回到 2 —— 循环继续，把工具结果喂给模型再判断

    # 超步数 → 兜底（防死循环）
    if verbose:
        print("  [fallback] 超过 max_steps，转兜底")
    return fallback_message("（已超过最大处理步数）")
