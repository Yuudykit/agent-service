# -*- coding: utf-8 -*-
"""
真实模式 5 问验证（README 第七节验收清单）+ token 消耗统计。

用法：
    conda run -n agent-service python scripts/run_real_eval.py
    conda run -n agent-service python scripts/run_real_eval.py --verbose

设计：
    - patch core.agent_loop.chat，拦截每次真实 API 调用的 usage 字段做汇总
    - 5 个验证用例：退款时效 / 多轮发票税号 / 天气兜底 / 防幻觉 / 订单查询
    - 多轮用例共享 ContextManager（验证窗口内上下文承接），其余用例独立会话，
      避免上下文无限累积导致 token 膨胀（面试可讲：评测成本控制）
"""
import json
import sys

sys.path.insert(0, ".")

import core.agent_loop as al  # noqa: E402  需在 patch 前导入
import tools  # noqa: E402,F401  触发工具注册
from core.agent_loop import run  # noqa: E402
from core.context import ContextManager  # noqa: E402
from core.prompts import build_system_prompt  # noqa: E402

# ----------------------------------------------------------------------
# token 统计：patch chat，抓每次真实调用的 usage
# ----------------------------------------------------------------------
USAGE = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
_orig_chat = al.chat


def _wrapped_chat(messages, system_prompt, tools_schema, verbose=False, mock=False):
    msg = _orig_chat(messages, system_prompt, tools_schema, verbose, mock)
    u = msg.get("usage")
    if u:
        USAGE["calls"] += 1
        USAGE["prompt"] += u["prompt_tokens"]
        USAGE["completion"] += u["completion_tokens"]
        USAGE["total"] += u["total_tokens"]
        print(
            f"    [token] 调用#{USAGE['calls']} "
            f"prompt={u['prompt_tokens']} + completion={u['completion_tokens']} "
            f"= {u['total_tokens']}"
        )
    return msg


al.chat = _wrapped_chat

# ----------------------------------------------------------------------
# 5 问验证清单
# ----------------------------------------------------------------------
CASES = [
    ("C1 退款时效", "退款多久能到账？", "share:false", ["1-3", "工作日", "到账"]),
    ("C2 多轮-发票", "我买了个路由器想开发票，能开吗？", "share:multi", ["发票"]),
    ("C2 多轮-税号", "刚才那个订单税号填错了，怎么改？", "share:multi", ["税号", "修改", "发票"]),
    ("C3 天气兜底", "今天上海天气怎么样？", "share:false", None),  # 只要求不编造、明确边界
    ("C4 防幻觉", "你们老板姓什么？", "share:false", None),  # 只要求拒绝回答/边界
    ("C5 订单查询", "我的订单 JD20260815001 现在到哪了？", "share:false", ["已发货", "揽收"]),
]

# 共享会话：多轮用例用同一个 context
shared_ctx = None
shared_sp = None


def main() -> None:
    verbose = "--verbose" in sys.argv
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]

    # 展示 system prompt 体量（每次请求都会带上，是 token 消耗大头）
    sp = build_system_prompt()
    print(f"[env] system_prompt 约 {len(sp)} 字符（中文近似 1 字≈1 token）")
    print(f"[env] tools_schema 工具数: {len(__import__('tools.registry', fromlist=['get_tools_schema']).get_tools_schema())}")
    print()

    results = []
    for name, q, share, keywords in CASES:
        if only and only not in name:
            continue
        global shared_ctx, shared_sp
        if share == "share:multi":
            if shared_ctx is None:
                shared_ctx = ContextManager(max_tokens=4096)
                shared_sp = sp
            ctx, sp_used = shared_ctx, shared_sp
            tag = "共享会话"
        else:
            ctx = ContextManager(max_tokens=4096)
            sp_used = sp
            tag = "独立会话"

        print(f"=== {name}（{tag}）===")
        print(f"  Q: {q}")
        answer = run(q, ctx, system_prompt=sp_used, verbose=verbose, mock=False)
        print(f"  A: {answer}")
        print()

        ok = True
        note = ""
        if keywords is None:
            # 兜底/防幻觉：要求不编造业务细节（有"服务范围/无法回答/抱歉"等边界信号即可）
            ok = any(w in answer for w in ("服务范围", "无法", "抱歉", "不负责", "超出", "不能", "没有"))
            note = "（兜底边界）" if ok else "（未识别边界信号）"
        else:
            miss = [k for k in keywords if k not in answer]
            ok = not miss
            note = f"（缺关键词: {miss}）" if miss else ""
        results.append((name, ok, note))
        print(f"  ✅ {name}: {'通过' if ok else '失败'} {note}")

    print()
    print("=" * 46)
    print("汇总：")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"  用例: {passed}/{len(results)} 通过")
    print(f"  API 调用: {USAGE['calls']} 次")
    print(f"  prompt tokens: {USAGE['prompt']}")
    print(f"  completion tokens: {USAGE['completion']}")
    print(f"  总消耗: {USAGE['total']} tokens")
    if USAGE["calls"]:
        print(f"  平均每次: {USAGE['total'] // USAGE['calls']} tokens")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
