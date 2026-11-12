# -*- coding: utf-8 -*-
"""
webapp server：把 main.py 的 CLI 包成 Web 接口，零第三方依赖。

用法（在项目根目录执行）:
    python webapp/server.py [--port 8765]

接口:
    GET  /              → 聊天页 index.html
    GET  /api/health    → 服务健康检查（是否已配置 API key）
    GET  /monitor       → 运行监控页
    GET  /api/monitor   → 本机请求与反馈汇总
    POST /api/feedback  → 按回复编号保存用户评价
    POST /api/chat      → {"message": "...", "session_id": "...", "mock": bool}
                          → SSE 流（text/event-stream），事件依次为:
                              {"type":"session","session_id":...}   会话 id（新会话时分配）
                              {"type":"tool","name":"query_info"}   工具执行状态
                              {"type":"delta","text":"..."}         回答文本增量（真实模式流式）
                              {"type":"done","answer":...}          本轮结束（degraded=true 表示降级）
                          异常返回兜底话术与 degraded 标记，原始错误保存在本地日志。

会话管理：
    - 每个 session_id 一个 ContextManager；重启不丢——每轮结束落盘
      data/sessions/{sid}.json（消息历史 + 轮次计数），启动懒加载。
    - "新会话"按钮 = 换一个新 session_id。
    - 会话级记忆（v6.1）：每个会话一个 Memory 实例，画像/摘要落盘
      data/memory/{sid}.json，跨重启随会话一起恢复。
    - 内存缓存淘汰（v6.2）：SESSIONS/_TURNS/_MEMORIES 超过
      _MAX_LIVE_SESSIONS（默认 200）按 FIFO 丢最旧；磁盘数据都在，
      下次访问自动懒加载，长期运行不再缓慢泄漏。

用量护栏（P2）：
    - 日预算：真实模式 token 累计到 data/usage/{日期}.json，超过
      config.daily_token_budget 后本轮直接降级（不调 API，零成本）。
    - 会话轮次：超过 config.max_turns_per_session 后降级（防单会话无限刷）。

结构化日志（P2）：
    - logs/events.jsonl，一行一事件（每轮对话一条）：
      {ts, session_id, mode, message, tools, tokens, latency_ms, degraded, detail}
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

# 切到项目根目录（config.json / data/ 都是相对路径），并把项目根加进 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from core.agent_loop import run            # noqa: E402
from core.config import load_config        # noqa: E402
from core.context import ContextManager    # noqa: E402
from core.memory import Memory             # noqa: E402
from core.prompts import build_system_prompt  # noqa: E402
from tools.fallback import bind_session, fallback_message, unbind_session  # noqa: E402
from webapp.monitoring import MonitorStore, SERVICES, TOOL_SERVICES  # noqa: E402

CONFIG = load_config()

_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
_PAGE_ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/monitor": ("monitor.html", "text/html"),
    "/monitor.css": ("monitor.css", "text/css"),
    "/monitor.js": ("monitor.js", "text/javascript"),
}
SESSIONS: dict[str, ContextManager] = {}
_SESSIONS_DIR = os.path.join(_ROOT, "data", "sessions")
_USAGE_DIR = os.path.join(_ROOT, "data", "usage")
_LOGS_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_SESSIONS_DIR, exist_ok=True)
os.makedirs(_USAGE_DIR, exist_ok=True)
os.makedirs(_LOGS_DIR, exist_ok=True)
_MONITOR = MonitorStore(_LOGS_DIR)
# run() 不是线程安全（共享 data/、RAG 索引），加全局锁串行化请求
_LOCK = threading.Lock()

_DAILY_BUDGET = int(CONFIG.get("daily_token_budget") or 0)      # 0 = 不设限
_MAX_TURNS = int(CONFIG.get("max_turns_per_session") or 0)      # 0 = 不设限


# ----------------------------------------------------------------------
# 会话持久化：data/sessions/{sid}.json（单文件 JSON，够用且好进 git 演示）
# ----------------------------------------------------------------------
def _save_session(sid: str, ctx: ContextManager, turns: int) -> None:
    path = os.path.join(_SESSIONS_DIR, f"{sid}.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"messages": ctx.messages, "turns": turns},
                      f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)  # 原子替换：写一半断电也不会留下坏文件
    except OSError as e:  # 落盘失败不致命：内存里还有，只打日志
        print(f"[warn] 会话 {sid} 落盘失败: {e}", file=sys.stderr)


def _get_session(sid: str) -> tuple[str, ContextManager, int]:
    """取会话：内存 → 磁盘懒加载 → 新建。返回 (sid, ctx, turns)。"""
    if sid in SESSIONS:
        return sid, SESSIONS[sid], _TURNS.get(sid, 0)
    path = os.path.join(_SESSIONS_DIR, f"{sid}.json")
    ctx = ContextManager(max_tokens=CONFIG["max_context_tokens"])
    turns = 0
    if sid and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ctx.messages = data.get("messages", [])
            turns = int(data.get("turns", 0))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] 会话 {sid} 加载失败，按新会话处理: {e}", file=sys.stderr)
    else:
        sid = sid or uuid.uuid4().hex[:12]
    SESSIONS[sid] = ctx
    _TURNS[sid] = turns
    _evict_stale(sid)  # v6.2：超上限丢最旧会话的内存缓存（磁盘还在，可懒加载）
    return sid, ctx, turns


_TURNS: dict[str, int] = {}

# 会话级记忆：每个 session_id 一个独立 Memory（v6.1 接入）。
# 无用户认证的演示场景里 session_id 就是身份的最小单位——
# 共用全局 data/memory.json 会让不同访客的画像互相污染。
_MEMORY_DIR = os.path.join(_ROOT, "data", "memory")
_MEMORIES: dict[str, Memory] = {}

# 内存字典淘汰（v6.2）：SESSIONS/_TURNS/_MEMORIES 常驻内存无上限，
# 长期运行会缓慢泄漏。会话历史/轮次/画像每轮都已落盘，淘汰只丢
# 内存缓存，下次访问自动懒加载，所以按 FIFO 超上限丢最旧即可。
_MAX_LIVE_SESSIONS = 200


def _evict_stale(keep_sid: str) -> None:
    """超过 _MAX_LIVE_SESSIONS 时，按插入顺序淘汰最旧会话的内存缓存。"""
    while len(SESSIONS) > _MAX_LIVE_SESSIONS:
        oldest = next((s for s in SESSIONS if s != keep_sid), None)
        if oldest is None:
            break
        SESSIONS.pop(oldest, None)
        _TURNS.pop(oldest, None)
        _MEMORIES.pop(oldest, None)


def _get_memory(sid: str) -> Memory:
    """取会话记忆：内存 → 按路径懒加载（data/memory/{sid}.json）。"""
    mem = _MEMORIES.get(sid)
    if mem is None:
        mem = Memory(path=os.path.join(_MEMORY_DIR, f"{sid}.json"))
        _MEMORIES[sid] = mem
    return mem


# ----------------------------------------------------------------------
# 用量记账：data/usage/{YYYY-MM-DD}.json（真实模式才记账）
# ----------------------------------------------------------------------
def _today_usage() -> dict:
    path = os.path.join(_USAGE_DIR, time.strftime("%Y-%m-%d") + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"total_tokens": 0, "requests": 0}


def _add_usage(tokens: int) -> None:
    usage = _today_usage()
    usage["total_tokens"] = usage.get("total_tokens", 0) + tokens
    usage["requests"] = usage.get("requests", 0) + 1
    path = os.path.join(_USAGE_DIR, time.strftime("%Y-%m-%d") + ".json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(usage, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        print(f"[warn] 用量记账失败: {e}", file=sys.stderr)


# ----------------------------------------------------------------------
# 结构化日志：logs/events.jsonl（一行一事件）
# ----------------------------------------------------------------------
def _log_event(evt: dict) -> bool:
    try:
        _MONITOR.log_event(evt)
    except OSError as e:
        print(f"[warn] 事件日志写入失败: {e}", file=sys.stderr)
        return False
    return True


# ----------------------------------------------------------------------
# bad case 收集：系统自己记录自己"答不上/需要人/降级"的时刻，
# 作为下一轮知识库迭代的原料（证明价值层的持续化）
# ----------------------------------------------------------------------
_BAD_ANSWER_MARKS = ("没有查到", "没有查询到", "服务范围")


def _log_bad_case(evt: dict) -> None:
    try:
        with open(os.path.join(_ROOT, "data", "bad_cases.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[warn] bad case 写入失败: {e}", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, obj):
        frame = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        url = urlsplit(self.path)
        asset = _PAGE_ASSETS.get(url.path)
        if asset:
            filename, media_type = asset
            with open(os.path.join(_WEBAPP_DIR, filename), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{media_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/health":
            self._send_json({
                "ok": True,
                "model": CONFIG.get("model"),
                "has_api_key": bool(CONFIG.get("api_key")),
                "today_tokens": _today_usage().get("total_tokens", 0),
                "daily_budget": _DAILY_BUDGET,
            })
        elif url.path == "/api/monitor":
            query = parse_qs(url.query)
            mode = query.get("mode", ["all"])[0]
            service = query.get("service", ["all"])[0]
            if mode not in ("all", "mock", "real") or service not in ("all", *SERVICES):
                self._send_json({"error": "筛选条件无效"}, 400)
                return
            try:
                snapshot = _MONITOR.snapshot(mode, service, query.get("request", [None])[0])
            except OSError:
                self._send_json({"error": "暂时无法读取监控记录"}, 503)
                return
            self._send_json(snapshot)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/feedback"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("invalid request body length")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
        except (OSError, ValueError):
            self._send_json({"error": "请求体格式无效"}, 400)
            return

        if self.path == "/api/feedback":
            self._save_feedback(data)
            return

        message = data.get("message")
        sid = data.get("session_id") or ""
        if not isinstance(message, str) or not message.strip():
            self._send_json({"error": "message 不能为空"}, 400)
            return
        if not isinstance(sid, str) or (sid and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid)):
            self._send_json({"error": "会话标识无效"}, 400)
            return
        self._chat(message.strip(), sid, bool(data.get("mock")), data.get("service"))

    def _save_feedback(self, data):
        request_id = data.get("request_id")
        sid = data.get("session_id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id) or not isinstance(sid, str):
            self._send_json({"error": "回复编号无效"}, 400)
            return
        try:
            saved = _MONITOR.save_feedback(request_id, sid, data.get("rating"), data.get("reason", ""))
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
            return
        except OSError:
            self._send_json({"error": "暂时无法保存反馈"}, 503)
            return
        self._send_json({"ok": True} if saved else {"error": "未找到可评价的回复"}, 200 if saved else 404)

    def _chat(self, message, sid, mock, service):
        started = time.perf_counter()
        sid, context, turns = _get_session(sid)
        request_id = uuid.uuid4().hex
        stats = {"tools": [], "tokens": 0, "queue_ms": None, "first_delta_ms": None}
        log_evt = {"request_id": request_id, "session_id": sid, "mode": "mock" if mock else "real",
                   "message": message[:80]}
        disconnected = False

        def push(event):
            nonlocal disconnected
            if not disconnected:
                try:
                    self._send_sse(event)
                except OSError:
                    disconnected = True

        def finish(answer, status="completed", outcome=None, detail=None):
            tool = next((name for name in stats["tools"] if name in TOOL_SERVICES), None)
            event = {
                **log_evt, **stats,
                "service": service if service in SERVICES else TOOL_SERVICES.get(tool, "other"),
                "tools": [name if name in TOOL_SERVICES else "unknown_tool" for name in stats["tools"]],
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "status": "disconnected" if disconnected else status,
                "outcome": outcome,
                "degraded": status in ("degraded", "limited"),
                "detail": detail,
            }
            recorded = _log_event(event)
            if disconnected:
                return
            push({"type": "done", "answer": answer, "session_id": sid, "request_id": request_id,
                  "degraded": event["degraded"], "detail": outcome if event["degraded"] else None,
                  "tokens": stats["tokens"] or None, "latency_ms": event["latency_ms"],
                  "feedback_available": recorded})
            if disconnected and recorded:
                _log_event({**event, "status": "disconnected"})

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        except OSError:
            disconnected = True
        push({"type": "session", "session_id": sid, "request_id": request_id})
        if disconnected:
            finish("", "disconnected")
            return

        if not mock and _DAILY_BUDGET and _today_usage().get("total_tokens", 0) >= _DAILY_BUDGET:
            answer = ("抱歉，今日服务用量已达上限，暂时无法继续智能应答。"
                      "您可以明天再来，或直接拨打热线 950618 联系人工客服。")
            finish(answer, "limited", "daily_token_budget_exceeded")
            return
        if _MAX_TURNS and turns >= _MAX_TURNS:
            answer = (f"本次会话已达最大轮次（{_MAX_TURNS} 轮）。"
                      "建议点击「新会话」重新开始，或通过热线 950618 联系人工客服。")
            finish(answer, "limited", "max_turns_per_session_exceeded")
            return

        def on_event(evt):
            if evt["type"] == "usage":
                stats["tokens"] += evt["usage"].get("total_tokens", 0)
                return
            if evt["type"] == "tool":
                stats["tools"].append(evt["name"])
            if evt["type"] == "delta" and stats["first_delta_ms"] is None:
                stats["first_delta_ms"] = round((time.perf_counter() - started) * 1000)
            push(evt)

        detail = None
        waiting = time.perf_counter()
        try:
            with _LOCK:
                stats["queue_ms"] = round((time.perf_counter() - waiting) * 1000)
                bind_session(sid, context)
                try:
                    answer = run(
                        message, context=context, system_prompt=build_system_prompt(), mock=mock,
                        max_steps=CONFIG["max_react_steps"], memory=_get_memory(sid), on_event=on_event,
                    )
                finally:
                    unbind_session()
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            print(f"[error] session={sid} mock={mock} {detail}", file=sys.stderr)
            answer = fallback_message("（服务暂时不可用，请稍后重试）")
        else:
            _TURNS[sid] = turns + 1
            _save_session(sid, context, turns + 1)
            if not mock and stats["tokens"]:
                _add_usage(stats["tokens"])

        handoff = "transfer_human" in stats["tools"]
        fallback = any(mark in answer for mark in _BAD_ANSWER_MARKS)
        bad_reasons = [reason for reason, triggered in (
            (detail, bool(detail)), ("transfer_human", handoff), ("fallback_answer", fallback)
        ) if triggered]
        if bad_reasons:
            _log_bad_case({**log_evt, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "reasons": bad_reasons, "tools": stats["tools"]})

        outcome = None
        if detail:
            outcome = "service_error"
        elif handoff:
            outcome = "handoff_requested"
        elif fallback:
            outcome = "fallback_reply"
        finish(answer, "degraded" if detail else "completed", outcome, detail)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="智能客服 Web 版")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"智能客服 Web 版已启动: http://127.0.0.1:{args.port}")
    print(f"  模型: {CONFIG.get('model')} | API key 已配置: {bool(CONFIG.get('api_key'))}")
    print(f"  会话持久化目录: {_SESSIONS_DIR}")
    print(f"  日预算: {_DAILY_BUDGET or '不设限'} tokens | 会话轮次上限: {_MAX_TURNS or '不设限'}")
    print(f"  事件日志: {_LOGS_DIR}/events.jsonl")
    server.serve_forever()


if __name__ == "__main__":
    main()
