"""Local request metrics and per-reply feedback for the Web workbench."""

import json
import math
import threading
import time
from collections import Counter, deque
from pathlib import Path


SERVICES = ("faq", "order", "refund", "handoff", "other")
TOOL_SERVICES = {
    "search_kb": "faq",
    "query_info": "order",
    "calculate_refund": "refund",
    "transfer_human": "handoff",
}
FEEDBACK_REASONS = ("irrelevant", "incorrect", "unresolved", "slow")
_WINDOW = 2000
_PUBLIC_FIELDS = (
    "request_id", "ts", "mode", "service", "status", "tools", "tokens",
    "latency_ms", "queue_ms", "first_delta_ms", "outcome",
)


def _read_rows(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentile(values, fraction):
    return values[math.ceil(len(values) * fraction) - 1] if values else None


class MonitorStore:
    def __init__(self, directory):
        self.events = Path(directory) / "events.jsonl"
        self.feedback = Path(directory) / "feedback.jsonl"
        self._lock = threading.Lock()

    def log_event(self, event):
        with self._lock:
            _append(self.events, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **event})

    def _recent(self):
        rows = deque((row for row in _read_rows(self.events) if row.get("request_id")), maxlen=_WINDOW)
        requests = {}
        for row in reversed(rows):
            requests.setdefault(row["request_id"], row)
        return requests

    def save_feedback(self, request_id, session_id, rating, reason):
        if rating not in ("helpful", "unhelpful") or reason not in ("", *FEEDBACK_REASONS):
            raise ValueError("请选择有效的评价和原因")
        if rating == "helpful" and reason:
            raise ValueError("仅「没帮助」可填写原因")
        with self._lock:
            request = self._recent().get(request_id)
            if not request or request["session_id"] != session_id or request["status"] == "disconnected":
                return False
            _append(self.feedback, {"request_id": request_id, "rating": rating, "reason": reason})
        return True

    def snapshot(self, mode, service, selected_id=None):
        with self._lock:
            requests = self._recent()
            feedback = {row["request_id"]: row for row in _read_rows(self.feedback)
                        if row.get("request_id") in requests}
        rows = []
        for request_id, request in requests.items():
            if mode != "all" and request["mode"] != mode:
                continue
            if service != "all" and request["service"] != service:
                continue
            row = {key: request.get(key) for key in _PUBLIC_FIELDS}
            row["feedback"] = feedback.get(request_id) if row["status"] != "disconnected" else None
            rows.append(row)

        total = len(rows)
        statuses = Counter(row["status"] for row in rows)
        completed = total - statuses["disconnected"]
        latency = sorted(row["latency_ms"] for row in rows if row["status"] != "disconnected")
        ratings = Counter(row["feedback"]["rating"] for row in rows if row["feedback"])
        reasons = Counter(row["feedback"]["reason"] for row in rows
                          if row["feedback"] and row["feedback"]["reason"])
        return {
            "total": total,
            "returned": completed,
            "statuses": dict(statuses),
            "outcomes": dict(Counter(row["outcome"] for row in rows if row["outcome"])),
            "latency_p95_ms": _percentile(latency, 0.95),
            "tokens": sum(row["tokens"] for row in rows),
            "services": dict(Counter(row["service"] for row in rows)),
            "ratings": dict(ratings),
            "reasons": dict(reasons),
            "requests": rows[:80],
            "selected": next((row for row in rows if row["request_id"] == selected_id), None),
            "window_limit": _WINDOW,
        }
