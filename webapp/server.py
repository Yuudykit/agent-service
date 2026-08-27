# -*- coding: utf-8 -*-
"""
webapp server：把 main.py 的 CLI 包成 Web 接口，零第三方依赖。

用法（在 D:\\agent-service 下）:
    C:\\Users\\Administrator\\.conda\\envs\\agent-service\\python.exe webapp\\server.py [--port 8765]

接口:
    GET  /              → 聊天页 index.html
    GET  /api/health    → 服务健康检查（是否已配置 API key）
    POST /api/chat      → {"message": "...", "session_id": "...", "mock": bool}
                          → SSE 流（text/event-stream），事件依次为:
                              {"type":"session","session_id":...}   会话 id（新会话时分配）
                              {"type":"tool","name":"query_info"}   工具执行状态
                              {"type":"delta","text":"..."}         回答文本增量（真实模式流式）
                              {"type":"done","answer":...}          本轮结束（degraded=true 表示降级）
                          出错时不再甩裸异常：返回兜底话术 + degraded 标记，
                          原始错误写进 detail 字段和日志。

会话管理：
    - 每个 session_id 一个 ContextManager；重启不丢——每轮结束落盘
      data/sessions/{sid}.json（消息历史 + 轮次计数），启动懒加载。
    - "新会话"按钮 = 换一个新 session_id。

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
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 切到项目根目录（config.json / data/ 都是相对路径），并把项目根加进 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from core.agent_loop import run            # noqa: E402
from core.context import ContextManager    # noqa: E402
from core.prompts import build_system_prompt  # noqa: E402
from tools.fallback import bind_session, fallback_message, unbind_session  # noqa: E402

with open("config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS: dict[str, ContextManager] = {}
_SESSIONS_DIR = os.path.join(_ROOT, "data", "sessions")
_USAGE_DIR = os.path.join(_ROOT, "data", "usage")
_LOGS_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_SESSIONS_DIR, exist_ok=True)
os.makedirs(_USAGE_DIR, exist_ok=True)
os.makedirs(_LOGS_DIR, exist_ok=True)
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
    return sid, ctx, turns


_TURNS: dict[str, int] = {}


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
def _log_event(evt: dict) -> None:
    evt = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **evt}
    try:
        with open(os.path.join(_LOGS_DIR, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[warn] 事件日志写入失败: {e}", file=sys.stderr)


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

    # ---------- 工具方法 ----------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, obj):
        """下发一个 SSE 事件帧：data: {...}\\n\\n"""
        frame = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    # ---------- GET ----------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(_WEBAPP_DIR, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            self._send_json({
                "ok": True,
                "model": CONFIG.get("model"),
                "has_api_key": bool(CONFIG.get("api_key")),
                "today_tokens": _today_usage().get("total_tokens", 0),
                "daily_budget": _DAILY_BUDGET,
            })
        else:
            self.send_error(404)

    # ---------- POST ----------
    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send_json({"error": "请求体不是合法 JSON"}, 400)
            return

        message = (data.get("message") or "").strip()
        sid = data.get("session_id") or ""
        mock = bool(data.get("mock"))

        if not message:
            self._send_json({"error": "message 不能为空"}, 400)
            return

        sid, context, turns = _get_session(sid)

        # SSE 响应头（此后只能走事件流，不能再发普通 JSON）
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self._send_sse({"type": "session", "session_id": sid})

        t0 = time.time()
        stats = {"tools": [], "tokens": 0}
        log_evt = {"session_id": sid, "mode": "mock" if mock else "real",
                   "message": message[:80]}

        # ---- 护栏 1：日 token 预算（真实模式才检查；超限不调 API，零成本）----
        if not mock and _DAILY_BUDGET and \
                _today_usage().get("total_tokens", 0) >= _DAILY_BUDGET:
            answer = ("抱歉，今日服务用量已达上限，暂时无法继续智能应答。"
                      "您可以明天再来，或直接拨打热线 950618 联系人工客服。")
            self._send_sse({"type": "done", "answer": answer, "degraded": True,
                            "detail": "daily_token_budget_exceeded",
                            "session_id": sid})
            _log_event({**log_evt, "degraded": True,
                        "detail": "daily_token_budget_exceeded"})
            return

        # ---- 护栏 2：单会话轮次上限（mock/真实都限，防无限刷）----
        if _MAX_TURNS and turns >= _MAX_TURNS:
            answer = (f"本次会话已达最大轮次（{_MAX_TURNS} 轮），上下文即将超限。"
                      f"建议点击「新会话」重新开始，或通过热线 950618 联系人工客服。")
            self._send_sse({"type": "done", "answer": answer, "degraded": True,
                            "detail": "max_turns_per_session_exceeded",
                            "session_id": sid})
            _log_event({**log_evt, "degraded": True,
                        "detail": "max_turns_per_session_exceeded"})
            return

        def on_event(evt: dict) -> None:
            """run() 的事件流 → 拦截 usage 做记账，其余转发 SSE。"""
            if evt.get("type") == "usage":
                stats["tokens"] += evt["usage"].get("total_tokens", 0)
                return  # usage 不下发前端（内部记账用）
            if evt.get("type") == "tool":
                stats["tools"].append(evt.get("name"))
            self._send_sse(evt)

        degraded = False
        try:
            with _LOCK:
                # 绑定会话 → transfer_human 可生成交接包（转人工协议）
                bind_session(sid, context)
                try:
                    answer = run(
                        message,
                        context=context,
                        system_prompt=build_system_prompt(),
                        mock=mock,
                        max_steps=CONFIG["max_react_steps"],
                        on_event=on_event,
                    )
                finally:
                    unbind_session()
        except Exception as e:  # API 超时/限流重试仍失败等 → 降级话术，不甩裸异常
            detail = f"{type(e).__name__}: {e}"
            print(f"[error] session={sid} mock={mock} {detail}", file=sys.stderr)
            answer = fallback_message("（服务暂时不可用，请稍后重试）")
            degraded = True
            log_evt["detail"] = detail
        else:
            _TURNS[sid] = turns + 1
            _save_session(sid, context, turns + 1)
            if not mock and stats["tokens"]:
                _add_usage(stats["tokens"])

        log_evt.update({
            "tools": stats["tools"],
            "tokens": stats["tokens"],
            "latency_ms": int((time.time() - t0) * 1000),
            "degraded": degraded,
        })
        _log_event(log_evt)

        # bad case 判定：降级 / 触发转人工 / 回答是兜底话术
        bad_reasons = []
        if degraded:
            bad_reasons.append(log_evt.get("detail") or "degraded")
        if "transfer_human" in stats["tools"]:
            bad_reasons.append("transfer_human")
        if any(mark in answer for mark in _BAD_ANSWER_MARKS):
            bad_reasons.append("fallback_answer")
        if bad_reasons:
            _log_bad_case({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": sid,
                "mode": log_evt["mode"],
                "message": message[:80],
                "reasons": bad_reasons,
                "tools": stats["tools"],
            })

        self._send_sse({
            "type": "done",
            "answer": answer,
            "session_id": sid,
            "degraded": degraded or None,
            "tokens": stats["tokens"] or None,
            "latency_ms": int((time.time() - t0) * 1000),
        })

    def log_message(self, fmt, *args):
        # 安静模式：不刷屏。出错时 BaseHTTPRequestHandler 自己会打 sys.stderr
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
