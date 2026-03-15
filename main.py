from __future__ import annotations

import os

from agent import (
    Planner,
    ReActExecutor,
    PlanAndReActAgent,
    ReplanPolicy,
    build_default_tools,
)


def build_agent(use_deepseek: bool | None = None) -> PlanAndReActAgent:
    """
    构建一个带默认工具的 Plan-and-Execute + ReAct Agent。
    use_deepseek=True 或环境变量 LLM_PROVIDER=deepseek 时使用 DeepSeek（需配置 DEEPSEEK_API_KEY）。
    """
    tools = build_default_tools()
    if use_deepseek or (use_deepseek is None and os.environ.get("LLM_PROVIDER", "").strip().lower() == "deepseek"):
        planner = Planner(provider="deepseek", model="deepseek-chat")
        executor = ReActExecutor(tools=tools, max_react_rounds=3, provider="deepseek", model="deepseek-chat")
    else:
        planner = Planner()
        executor = ReActExecutor(tools=tools, max_react_rounds=3)
    # 启用 Replan：子任务失败时按策略重规划（规则：含错误/失败等关键词即触发）
    replan_policy = ReplanPolicy(use_llm=False)
    agent = PlanAndReActAgent(planner=planner, executor=executor, replan_policy=replan_policy)
    return agent


def main() -> None:
    import sys
    agent = build_agent()

    # 支持单次 demo：python main.py --demo "你的问题"
    if len(sys.argv) >= 3 and sys.argv[1] == "--demo":
        query = " ".join(sys.argv[2:]).strip()
        if not query:
            print("用法: python main.py --demo \"你的问题\"")
            return
        print("=== Plan-and-Execute + ReAct Agent Demo（单次）===\n")
        print(f"问题: {query}\n")
        result = agent.run(query)
        print(result)
        return

    print("=== Plan-and-Execute + ReAct Agent Demo ===")
    print("输入自然语言问题，Agent 会先生成计划，再按 ReAct 模式逐步执行。")
    print("输入 q 或 quit 退出。\n")

    while True:
        try:
            query = input("请输入你的问题（输入 q 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if query.lower() in {"q", "quit", "exit"}:
            print("再见！")
            break

        if not query:
            continue

        print("\n========== Agent 输出 ==========\n")
        result = agent.run(query)
        print(result)
        print("\n================================\n")


if __name__ == "__main__":
    main()

