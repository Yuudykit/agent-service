# -*- coding: utf-8 -*-
"""
agent-service 入口：命令行交互循环。

用法:
    python main.py [--mock] [--verbose]

    --mock      不调用真实 LLM，用关键词匹配模拟（Day 1 同款，API key 过期时用）
    --verbose   打印每一轮的数据流细节

Day 4 完成 agent_loop.py 后，这里能跑通"多轮对话 + 工具调用"。
"""
import argparse
import json
import sys

from core.agent_loop import run
from core.context import ContextManager
from core.memory import Memory
from core.prompts import build_system_prompt


def load_config(path: str = "config.json") -> dict:
    """读取配置文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="智能体客服")
    parser.add_argument("--mock", action="store_true", help="mock 模式（不调 API）")
    parser.add_argument("--verbose", action="store_true", help="打印详细过程")
    parser.add_argument("--memory", action="store_true", help="启用记忆持久化（data/memory.json）")
    args = parser.parse_args()

    config = load_config()
    context = ContextManager(max_tokens=config["max_context_tokens"])
    memory = Memory() if args.memory else None

    # 分层系统提示词（角色/技能/规则/few-shot，技能部分随工具注册表动态生成）
    system_prompt = build_system_prompt()

    print("智能体客服已启动（mock=%s, memory=%s）。输入 exit 退出。\n"
          % (args.mock, bool(memory)))
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见。")
            break
        if not user_input:
            continue

        # 调 agent_loop.run()：用户输入 + 上下文管理器 → 最终回答
        try:
            answer = run(
                user_input,
                context=context,
                system_prompt=system_prompt,
                verbose=args.verbose,
                mock=args.mock,
                max_steps=config["max_react_steps"],
                memory=memory,
            )
        except RuntimeError as e:
            print(f"客服> [错误] {e}")
            continue
        print("客服>", answer)


if __name__ == "__main__":
    sys.exit(main())
