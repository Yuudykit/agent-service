# -*- coding: utf-8 -*-
"""
评测脚本 —— 对标 JD 隐含要求：Agent 评估/监控

职责：用标准测试对话，量化 Agent 的表现。开发期验收脚本是
      scripts/run_eval_p0.py（28 用例快速回归）；本文件是正式评测入口，
      输出四指标报告，供面试展示与持续监控。

指标（面试必答）:
    - 工具路由准确率：期望调用工具 X 的用例里，实际正确路由到 X 的比例
    - 回答准确率：最终回答命中期望关键词的比例（宽松判分）
    - 兜底正确率：期望不调工具（寒暄/无关话题）时，确实没调工具的比例
    - 多轮通过率：多轮用例（turns>1）中，所有轮次都达标的比例

用法：
    python -m eval.eval_script [--verbose]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, ".")

import tools  # noqa: F401  触发工具注册
from core.agent_loop import run
from core.context import ContextManager
from core.prompts import build_system_prompt
from tools import registry


class _ToolRecorder:
    """monkey-patch execute_tool：记录每次调用而不改生产代码。

    注意：core.agent_loop 里是 `from tools.registry import execute_tool`，
    绑定的模块级名字，所以必须 patch core.agent_loop.execute_tool。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        import core.agent_loop as loop_mod
        self._orig = loop_mod.execute_tool
        self._module = loop_mod

    def __enter__(self) -> "_ToolRecorder":
        def wrapper(name, arguments):
            self.calls.append((name, arguments if isinstance(arguments, dict) else {}))
            return self._orig(name, arguments)
        self._module.execute_tool = wrapper
        return self

    def __exit__(self, *args) -> None:
        self._module.execute_tool = self._orig


def load_test_cases(path: str = "data/eval_set.json",
                    extra: str = "data/eval_multiturn.json") -> list[dict]:
    """加载评测用例。

    两份数据：
        eval_set.json        —— 28 条人工用例（P0 回归基准，含 2 条手工多轮）
        eval_multiturn.json  —— 30 条真实多轮用例（由 120 条真实客服会话自动生成）
    两种格式：
        旧格式 {"turns": [...], "expect": {...}}        —— 逐轮 run
        新格式 {"history": [...], "query": "...", "expect": {...}} —— 先注入历史再跑 query
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        cases.extend(json.load(f))
    if os.path.exists(extra):
        with open(extra, encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases


def _tool_hits(recorder: "_ToolRecorder") -> set[str]:
    return {name for name, _ in recorder.calls}


def run_eval(mock: bool = True, verbose: bool = False) -> dict:
    """跑评测，返回指标汇总 + 逐条明细。"""
    cases = load_test_cases()
    system_prompt = build_system_prompt()
    detail: list[dict[str, Any]] = []

    for case in cases:
        ctx = ContextManager(max_tokens=4096)  # 每个用例独立上下文（互不干扰）
        expected_tool = (case.get("expect") or {}).get("tool", "none")
        expected_kws = (case.get("expect") or {}).get("keywords", [])

        # 新格式：history 注入上下文（多轮记忆的真实载体），再跑最后一条 query
        if "history" in case and "query" in case:
            for h in case["history"]:
                if h["role"] == "user":
                    ctx.add_user(h["text"])
                else:
                    ctx.add_assistant(text=h["text"])
            turns = [case["query"]]
        else:
            turns = case["turns"]

        with _ToolRecorder() as rec:
            for t in turns:
                answer = run(
                    t, ctx,
                    system_prompt=system_prompt,
                    verbose=False, mock=mock,
                    max_steps=3,
                )

        used_tools = _tool_hits(rec)
        expects_no_tool = expected_tool in ("none", "fallback")
        tool_ok = (expects_no_tool and not used_tools) or (
            expected_tool in used_tools
        )
        # 关键词判分：人工用例默认 all（严格），自动生成的多轮用例标
        # kw_mode=any（宽松）——它验证"回答了该主题"，不要求同时出现多个词
        kw_mode = (case.get("expect") or {}).get("kw_mode", "all")
        if not expected_kws:
            kw_ok = True
        elif kw_mode == "any":
            kw_ok = any(k in answer for k in expected_kws)
        else:
            kw_ok = all(k in answer for k in expected_kws)
        multi = len(turns) > 1 or bool(case.get("history"))

        passed = tool_ok and kw_ok
        detail.append(
            {
                "id": case["id"],
                "tool_ok": tool_ok,
                "kw_ok": kw_ok,
                "passed": passed,
                "multi": multi,
                "used": sorted(used_tools) or "none",
                "answer": answer[:60],
            }
        )
        if verbose:
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {case['id']} 期望工具={expected_tool} 实际={detail[-1]['used']} 关键词={'OK' if kw_ok else 'MISS'}")

    total = len(detail)
    return {
        "total": total,
        "passed": sum(1 for d in detail if d["passed"]),
        "tool_ok": sum(1 for d in detail if d["tool_ok"]),
        "kw_ok": sum(1 for d in detail if d["kw_ok"]),
        "multi_total": sum(1 for d in detail if d["multi"]),
        "multi_passed": sum(1 for d in detail if d["multi"] and d["passed"]),
        "detail": detail,
    }


def print_report(metrics: dict) -> None:
    """打印评测报告：逐条 pass/fail + 四指标汇总。"""
    total = metrics["total"]
    print("=" * 50)
    print(f"用例总数: {total}   通过: {metrics['passed']} ({metrics['passed'] / total:.0%})")
    print(f"工具路由正确: {metrics['tool_ok']}/{total} ({metrics['tool_ok'] / total:.0%})")
    print(f"回答关键词命中: {metrics['kw_ok']}/{total} ({metrics['kw_ok'] / total:.0%})")
    if metrics["multi_total"]:
        print(f"多轮用例: {metrics['multi_passed']}/{metrics['multi_total']}")
    print("-" * 50)
    for d in metrics["detail"]:
        if not d["passed"]:
            why = []
            if not d["tool_ok"]:
                why.append(f"工具期望不符(实际 {d['used']})")
            if not d["kw_ok"]:
                why.append(f"回答缺关键词: {d['answer']}…")
            print(f"  FAIL {d['id']}: {'; '.join(why)}")
    print("=" * 50)
    if metrics["passed"] == total:
        print("全部通过 ✅")
    else:
        print(f"{total - metrics['passed']} 条未通过，见上。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print_report(run_eval(mock=True, verbose=args.verbose))
