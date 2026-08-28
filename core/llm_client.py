# -*- coding: utf-8 -*-
"""
OpenAI 兼容 client 复用（v6.2）。

为什么复用：每次调用新建 openai.OpenAI 会重建 HTTP 连接池，
真实模式一轮对话要调 2-3 次模型，逐次新建纯浪费（且高并发下
TIME_WAIT 堆积）。按 (api_key, base_url, timeout) 缓存实例，
配置没变就复用同一个连接池。

agent_loop.chat 和 memory._summarize_llm 共用这里，
避免各自 new client（也避免 memory 反向 import agent_loop 的循环依赖）。
"""
from __future__ import annotations

from typing import Any

_CLIENTS: dict[tuple, Any] = {}


def get_client(api_key: str, base_url: str, timeout: int = 60) -> Any:
    """按 (api_key, base_url, timeout) 缓存 OpenAI client 实例。"""
    import openai  # 延迟导入，mock 模式不依赖

    key = (api_key, base_url, timeout)
    if key not in _CLIENTS:
        _CLIENTS[key] = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout)
    return _CLIENTS[key]


def reset_clients() -> None:
    """清空缓存（测试用 / 运行期换 key）。"""
    _CLIENTS.clear()
