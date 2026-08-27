# -*- coding: utf-8 -*-
"""
P0 验收脚本 —— 基于 data/eval_set.json 跑 25 问，量化核心闭环表现。

指标：
    1. 工具路由正确率：该调 search_kb/transfer_human 的问题，是否正确路由
       （fallback/none 用例 = 没有错误调用工具且没有编造）
    2. 关键词命中率：最终回答是否覆盖 expect.keywords（宽松判分）
    3. 多轮通过率：turns 数 > 1 的用例是否全部通过

用法：python scripts/run_eval_p0.py [--verbose]
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import tools  # noqa: F401  触发工具注册
import core.agent_loop as agent_loop  # patch 模块级名字（from-import 绑定的是值）
from core.agent_loop import run
from core.context import ContextManager
from tools import registry


class _ToolRecorder:
    """monkey-patch execute_tool：记录每次调用而不改生产代码。

    注意：agent_loop 里是 `from tools.registry import execute_tool`，
    名字已绑定在 core.agent_loop 模块命名空间，必须 patch 那里，
    patch tools.registry.execute_tool 属性不会生效。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._orig = registry.execute_tool

    def __enter__(self):
        def wrapper(name, arguments):
            self.calls.append((name, arguments if isinstance(arguments, dict) else {}))
            return self._orig(name, arguments)
        agent_loop.execute_tool = wrapper
        return self

    def __exit__(self, *args):
        agent_loop.execute_tool = self._orig


def run_case(turns: list[str]) -> tuple[list[tuple[str, dict]], str]:
    """跑一个用例的所有轮次，返回 (工具调用记录, 最终回答)。"""
    ctx = ContextManager(max_tokens=4096)
    with _ToolRecorder() as rec:
        final = ""
        for t in turns:
            final = run(t, context=ctx, mock=True, max_steps=3)
    return rec.calls, final


def check(case: dict, calls: list[tuple[str, dict]], answer: str) -> tuple[bool, str]:
    """判分：工具路由 + 关键词。返回 (pass, 失败原因)。"""
    expect = case["expect"]
    exp_tool = expect.get("tool")          # search_kb / transfer_human / fallback / none
    called = {name for name, _ in calls}

    # 1) 工具路由
    if exp_tool in ("search_kb", "transfer_human"):
        if exp_tool not in called:
            return False, f"期望调用 {exp_tool}，实际 {sorted(called) or '未调工具'}"
        if exp_tool == "transfer_human" and "search_kb" in called:
            return False, f"转人工场景不应同时检索知识库，实际 {sorted(called)}"
    elif exp_tool == "fallback":
        if called:
            return False, f"兜底场景不应调工具，实际 {sorted(called)}"
        if not any(w in answer for w in ("超出", "无法回答", "不负责", "服务范围")):
            return False, f"兜底话术未识别到边界声明: {answer[:50]}"
    elif exp_tool == "none":
        if called:
            return False, f"寒暄场景不应调工具，实际 {sorted(called)}"

    # 2) 关键词
    missing = [kw for kw in expect.get("keywords", []) if kw not in answer]
    if missing:
        return False, f"回答缺关键词 {missing}: {answer[:60]}"
    return True, "ok"


def main() -> None:
    verbose = "--verbose" in sys.argv
    cases = json.load(open("data/eval_set.json", encoding="utf-8"))

    total = route_ok = kw_ok = multi_ok = multi_total = 0
    failures: list[tuple[str, str]] = []

    for case in cases:
        calls, answer = run_case(case["turns"])
        passed, reason = check(case, calls, answer)
        total += 1
        if passed:
            route_ok += 1
        else:
            failures.append((case["id"], reason))
        if len(case["turns"]) > 1:
            multi_total += 1
            multi_ok += passed
        if verbose:
            tools_used = {n for n, _ in calls}
            print(f"[{'PASS' if passed else 'FAIL'}] {case['id']} "
                  f"{case['expect']['note']} | 工具={tools_used or '无'} | "
                  f"回复={answer[:45]}")

    print("\n" + "=" * 50)
    print(f"用例总数: {total}")
    print(f"路由/兜底正确: {route_ok}/{total} ({route_ok / total:.0%})")
    print(f"多轮用例通过: {multi_ok}/{multi_total}")
    if failures:
        print("\n失败明细:")
        for cid, reason in failures:
            print(f"  {cid}: {reason}")
    else:
        print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
