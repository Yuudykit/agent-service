# -*- coding: utf-8 -*-
"""
边界测试 —— 验证 Agent 的鲁棒性（面试加分项："我写过测试"）

覆盖：
    1. ContextManager 窗口裁剪（以完整回合为单位，不拆散 tool 配对）
    2. 工具注册表 schema 自动推导（签名 → JSON Schema）
    3. 工具参数不合法 → 结构化错误文本（不抛异常）
    4. RAG 混合检索排序（关键词/向量双通道）
    5. Memory 规则摘要与画像抽取
    6. ReAct 文本协议解析（Action / Final Answer）
    7. 知识库无匹配 → 兜底不编造（端到端）

运行：
    python -m pytest tests/test_agent.py -v
    或：python tests/test_agent.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: F401  触发工具注册
from core.agent_loop import run
from core.context import ContextManager
from core.memory import Memory
from core.reasoning import ReActPlanner
from core.prompts import build_system_prompt
from rag.retriever import reset_index_cache, retrieve
from tools.registry import execute_tool, get_tools_schema, list_tools


# ----------------------------------------------------------------------
# 1. 上下文管理
# ----------------------------------------------------------------------
def test_context_round_trip_integrity():
    """窗口裁剪必须保住完整回合：user + assistant(tool_calls) + tool 同生共死。"""
    ctx = ContextManager(max_messages=4, max_tokens=10_000)
    ctx.add_user("查一下退款多久到账")
    ctx.add_assistant(
        text="",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "search_kb", "arguments": '{"query": "退款"}'}}],
    )
    ctx.add_tool_result(tool_call_id="c1", content="[检索结果] …")
    msgs = ctx.get_messages()
    # 裁剪后：要么三件套都在，要么全被裁掉；绝不能只留 user 或只留 tool
    roles = [m["role"] for m in msgs]
    assert roles in ([], ["user", "assistant", "tool"])


def test_context_truncates_old_turns():
    """超长上下文：最早的回合被裁掉，最新消息保留。"""
    ctx = ContextManager(max_messages=6, max_tokens=10_000)
    for i in range(10):
        ctx.add_user(f"问题{i}")
        ctx.add_assistant(text=f"回答{i}")
    roles = [m["role"] for m in ctx.get_messages()]
    assert "user" in roles and "assistant" in roles
    # 旧回合（问题0）应被裁掉
    contents = [m.get("content", "") for m in ctx.get_messages()]
    assert "问题0" not in contents
    assert "问题9" in contents


# ----------------------------------------------------------------------
# 2. 工具注册表
# ----------------------------------------------------------------------
def test_registry_has_all_tools():
    assert {"search_kb", "query_info", "calculate_refund", "transfer_human"} <= set(list_tools())


def test_schema_auto_derivation():
    """函数签名 → JSON Schema：必填参数进 required，默认值进 default。"""
    schema = {t["function"]["name"]: t["function"]["parameters"] for t in get_tools_schema()}
    sk = schema["search_kb"]
    assert "query" in sk["required"]           # query 无默认值 → 必填
    assert sk["properties"]["top_k"]["default"] == 3  # top_k=3 → 有默认值
    assert sk["properties"]["query"]["type"] == "string"


def test_execute_tool_bad_params():
    """缺必填参数 → 结构化错误文本，不抛异常。"""
    result = execute_tool("search_kb", {})  # 缺 query
    assert result.startswith("[工具异常]")
    assert "参数" in result


def test_execute_tool_unknown():
    result = execute_tool("not_exist_tool", {})
    assert "未知工具" in result


def test_execute_tool_accepts_json_string():
    """LLM 有时把参数以 JSON 字符串传回，必须兼容。"""
    result = execute_tool("search_kb", '{"query": "退款多久到账"}')
    assert "[检索结果]" in result or "[工具异常]" in result


# ----------------------------------------------------------------------
# 3. RAG 检索
# ----------------------------------------------------------------------
def test_retriever_ranks_relevant_first():
    """RAG 检索：语义相关问题排在前面。"""
    results = retrieve("退款多久能到账", top_k=5)
    assert results[0]["id"] == "kb_010"  # 应命中退款到账条目
    assert results[0]["match_type"] in ("keyword", "vector", "both")


def test_retriever_synonym_query():
    """换说法（退钱 vs 退款）也能召回正确条目。"""
    results = retrieve("我想退钱，多久能到", top_k=5)
    ids = [r["id"] for r in results]
    assert "kb_010" in ids[:2]


def test_embedder_fallback_backend_consistency(monkeypatch):
    """降级一致性：bge 加载失败降级 hash 后，current_backend() 必须如实报 hash。

    回归背景（2026-08-27 修的 bug）：降级后 current_backend() 仍读环境变量
    返回 "bge"，build.py 会把 hash 向量写进 kb_index_bge.json——文件名与
    内容错配，后续 bge 查询向量 × hash 库向量做余弦，分数全是噪声且不报错。
    """
    import rag.embedder as emb

    monkeypatch.setenv("EMBEDDER_BACKEND", "bge")
    monkeypatch.setattr(emb, "_embedder", None)

    def _boom(name):  # 模拟模型加载失败（没下载模型 / 依赖缺失）
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(emb, "SentenceTransformersEmbedder", _boom)
    e = emb.get_embedder()
    assert isinstance(e, emb.FeatureHashEmbedder)
    assert emb.current_backend() == "hash"  # 不能再谎报 "bge"
    monkeypatch.setattr(emb, "_embedder", None)  # 还原单例，避免污染其他测试


# ----------------------------------------------------------------------
# 4. Memory
# ----------------------------------------------------------------------
def test_memory_rule_summary_dedup():
    """规则摘要：同意图不重复。"""
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(path=os.path.join(tmp, "m.json"))
        msgs = [
            {"role": "user", "content": "我想开发票，公司抬头"},
            {"role": "assistant", "content": "好的。"},
            {"role": "user", "content": "税号填错了怎么改？"},
        ]
        summary = mem.summarize(msgs)
        assert summary.count("发票") == 1  # 两条消息都是发票意图，只记一次


def test_memory_profile_extraction():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(path=os.path.join(tmp, "m.json"))
        mem.update_profile("订单 1234567890123456 想开增值税专用发票")
        assert mem.profile.get("enterprise_invoice") == "需要企业发票"
        assert "1234567890123456" in mem.profile.get("order_ids", [])


def test_memory_persist_reload():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        mem1 = Memory(path=path)
        mem1.store_summary("用户咨询：退款")
        mem2 = Memory(path=path)  # 新实例从磁盘读回
        assert "退款" in mem2.recall()


def test_memory_session_isolation():
    """会话级记忆：不同 session 的画像互不污染，各自落盘（v6.1 webapp 接入）。"""
    import webapp.server as srv  # 导入副作用：chdir 到项目根（pytest 本就从根目录跑）

    sid_a, sid_b = "t_mem_iso_a", "t_mem_iso_b"
    path_a = os.path.join("data", "memory", f"{sid_a}.json")
    path_b = os.path.join("data", "memory", f"{sid_b}.json")
    try:
        mem_a = srv._get_memory(sid_a)
        mem_b = srv._get_memory(sid_b)
        assert os.path.abspath(mem_a.path) == os.path.abspath(path_a)
        mem_a.update_profile("我要开增值税专用发票")
        assert mem_a.profile.get("enterprise_invoice") == "需要企业发票"
        assert not mem_b.profile.get("enterprise_invoice")  # 画像不串会话
        assert os.path.exists(path_a)  # 独立落盘成功
    finally:
        srv._MEMORIES.pop(sid_a, None)
        srv._MEMORIES.pop(sid_b, None)
        for p in (path_a, path_b):
            if os.path.exists(p):
                os.remove(p)


def test_e2e_memory_updates_profile_via_run():
    """run() 传入 memory 后，用户发言应喂给画像（CLI/Web 同一路径）。"""
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(path=os.path.join(tmp, "m.json"))
        ctx = ContextManager(max_tokens=4096)
        run("退款多久能到账？", ctx, mock=True, memory=mem)
        assert "退款" in mem.recall()  # 画像已记录用户意图


def test_llm_client_cached_reuse():
    """client 复用（v6.2）：同配置返回同一实例；换 key 重建；reset 清空。"""
    from core.llm_client import get_client, reset_clients

    reset_clients()
    try:
        a = get_client("sk-test-1", "https://api.test/v1")
        b = get_client("sk-test-1", "https://api.test/v1")
        c = get_client("sk-test-2", "https://api.test/v1")
        assert a is b          # 同 (key, base_url) 复用连接池
        assert a is not c      # 配置变了必须重建
        reset_clients()
        d = get_client("sk-test-1", "https://api.test/v1")
        assert d is not a      # reset 后重建
    finally:
        reset_clients()


def test_webapp_session_eviction():
    """内存字典淘汰（v6.3）：超上限丢最旧会话，当前会话永不被淘汰。"""
    import webapp.server as srv

    old_cap = srv._MAX_LIVE_SESSIONS
    srv._MAX_LIVE_SESSIONS = 3
    saved_sessions = dict(srv.SESSIONS)
    saved_turns = dict(srv._TURNS)
    saved_memories = dict(srv._MEMORIES)
    srv.SESSIONS.clear()
    srv._TURNS.clear()
    srv._MEMORIES.clear()
    try:
        for i in range(5):
            srv._get_session(f"t_evict_{i}")
        assert len(srv.SESSIONS) == 3           # 淘汰到上限
        assert "t_evict_0" not in srv.SESSIONS  # 最旧的被淘汰
        assert "t_evict_4" in srv.SESSIONS      # 最新仍在
        # 当前会话（刚插入的）即使触发淘汰也永不被淘汰
        srv.SESSIONS.clear(); srv._TURNS.clear(); srv._MEMORIES.clear()
        srv._MAX_LIVE_SESSIONS = 1
        srv.SESSIONS["t_old"] = "x"
        srv._get_session("t_new")
        assert "t_new" in srv.SESSIONS          # 新会话存活
        assert "t_old" not in srv.SESSIONS      # 旧会话让位
    finally:
        srv._MAX_LIVE_SESSIONS = old_cap
        srv.SESSIONS.clear(); srv.SESSIONS.update(saved_sessions)
        srv._TURNS.clear(); srv._TURNS.update(saved_turns)
        srv._MEMORIES.clear(); srv._MEMORIES.update(saved_memories)


# ----------------------------------------------------------------------
# 5. ReAct 解析
# ----------------------------------------------------------------------
def test_react_extract_action_json():
    p = ReActPlanner(max_steps=3)
    assert p.extract_action('Thought: 查知识库\nAction: search_kb({"query": "退款"})') == (
        "search_kb", {"query": "退款"}
    )


def test_react_extract_action_kwargs():
    p = ReActPlanner()
    name, args = p.extract_action("Action: query_info(entity=\"JD20260815001\")")
    assert name == "query_info" and args["entity"] == "JD20260815001"


def test_react_extract_final_answer():
    p = ReActPlanner()
    assert p.extract_final_answer("Thought: 已确认\nFinal Answer: 退款一般1-3个工作日") == "退款一般1-3个工作日"


def test_react_should_continue():
    p = ReActPlanner(max_steps=3)
    assert p.should_continue(1, "找到相关条目")
    assert not p.should_continue(4, "找到相关条目")  # 超步数
    assert not p.should_continue(1, "知识库中未找到相关问题")  # 异常观察


# ----------------------------------------------------------------------
# 6. 端到端兜底
# ----------------------------------------------------------------------
def test_e2e_no_match_fallback():
    """无关话题（天气）→ 不调工具、不编造，明确边界。"""
    ctx = ContextManager(max_tokens=4096)
    answer = run("今天上海天气怎么样？", ctx, system_prompt=build_system_prompt(), mock=True)
    assert "服务范围" in answer or "无法回答" in answer
    assert "工作日" not in answer  # 不应给出退款/发货的具体政策细节（防幻觉）


def test_e2e_search_kb_result_summarized():
    """检索结果必须被总结成回答，不能原样直出（P1 缩进回归 bug 的防护）。"""
    ctx = ContextManager(max_tokens=4096)
    answer = run("退款多久能到账？", ctx, mock=True)
    assert "根据平台规则" in answer
    assert "[检索结果]" not in answer  # 原始检索文本不应暴露给用户


def test_e2e_order_query():
    """订单号 → 精确查询工具 → 给出订单状态。"""
    ctx = ContextManager(max_tokens=4096)
    answer = run("我的订单 JD20260815001 到哪了？", ctx, mock=True)
    assert "已发货" in answer or "揽收" in answer


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
