"""
真实 Case 集成测试（无 Mock）：真实调用 Planner、ReActExecutor 与工具。

用例说明：
1. 不需要 Replan：帮我规划5天深圳到厦门的行程 —— 全程顺利，不触发重规划。
2. 需要全量 Replan：先调用「测试失败」工具（会报错）再用计算器算 10+20 —— 第一步失败触发 full replan，整份重跑。
3. 需要增量 Replan：先查深圳天气 → 调用「测试失败」→ 算 1+1 —— 第二步失败触发 incremental replan，保留第一步结果。

需配置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY；运行较慢。
可用 unittest 或 pytest 运行：python -m unittest tests.test_plan_react_agent_real -v
"""
from __future__ import annotations

import os
import unittest

from agent import PlanAndReActAgent, Planner, ReActExecutor, ReplanPolicy
from agent.tools import Tool, build_default_tools


def _has_llm_key() -> bool:
    return bool(
        os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )


class TestFailTool(Tool):
    """测试用工具：始终返回失败文案，用于触发 Replan 的真实用例。"""
    name = "测试失败"
    description = "仅用于测试：调用后返回错误信息，用于触发重规划"

    def __call__(self, *args, **kwargs) -> str:
        return "错误：测试用失败"


def _build_agent_with_fail_tool(replan_mode: str = "incremental") -> PlanAndReActAgent:
    tools = build_default_tools()
    tools[TestFailTool.name] = TestFailTool()
    planner = Planner(provider="deepseek", model="deepseek-chat") if os.environ.get("DEEPSEEK_API_KEY") else Planner()
    executor = ReActExecutor(tools=tools, max_react_rounds=4)
    if os.environ.get("DEEPSEEK_API_KEY"):
        executor.provider = "deepseek"
        executor.model = "deepseek-chat"
    policy = ReplanPolicy(use_llm=False, mode=replan_mode)
    return PlanAndReActAgent(planner=planner, executor=executor, replan_policy=policy)


def _build_agent_default() -> PlanAndReActAgent:
    tools = build_default_tools()
    planner = Planner(provider="deepseek", model="deepseek-chat") if os.environ.get("DEEPSEEK_API_KEY") else Planner()
    executor = ReActExecutor(tools=tools, max_react_rounds=4)
    if os.environ.get("DEEPSEEK_API_KEY"):
        executor.provider = "deepseek"
        executor.model = "deepseek-chat"
    policy = ReplanPolicy(use_llm=False, mode="incremental")
    return PlanAndReActAgent(planner=planner, executor=executor, replan_policy=policy)


class TestRealNoReplan(unittest.TestCase):
    """真实用例 1：不需要 Replan —— 如「帮我规划5天深圳到厦门的行程」全程顺利执行。"""

    @unittest.skipUnless(_has_llm_key(), "需要 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")
    def test_travel_plan_shenzhen_to_xiamen(self):
        agent = _build_agent_default()
        query = "帮我规划5天深圳到厦门的行程"
        result = agent.run(query)
        self.assertIn("用户问题", result)
        self.assertIn(query, result)
        self.assertIn("高层计划", result)
        self.assertIn("执行过程", result)
        self.assertIn("最终总结", result)
        self.assertIn("深圳", result)
        self.assertIn("厦门", result)


class TestRealFullReplan(unittest.TestCase):
    """真实用例 2：需要全量 Replan —— 第一步就失败，触发整份重规划后从步骤 0 重跑。"""

    @unittest.skipUnless(_has_llm_key(), "需要 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")
    def test_first_step_fail_then_full_replan(self):
        agent = _build_agent_with_fail_tool(replan_mode="full")
        query = (
            "请按顺序做两件事："
            "第一步先调用「测试失败」工具（会报错），"
            "第二步用计算器算 10+20。"
        )
        result = agent.run(query)
        self.assertIn("用户问题", result)
        self.assertIn("高层计划", result)
        self.assertIn("重规划", result)
        self.assertIn("10+20", result)
        self.assertIn("30", result)


class TestRealIncrementalReplan(unittest.TestCase):
    """真实用例 3：需要增量 Replan —— 第一步成功、第二步失败，保留第一步结果，仅重规划剩余。"""

    @unittest.skipUnless(_has_llm_key(), "需要 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")
    def test_second_step_fail_then_incremental_replan(self):
        agent = _build_agent_with_fail_tool(replan_mode="incremental")
        query = (
            "请按顺序做三件事："
            "1) 用搜索查一下深圳天气；"
            "2) 调用「测试失败」工具（会报错）；"
            "3) 用计算器算 1+1。"
        )
        result = agent.run(query)
        self.assertIn("用户问题", result)
        self.assertIn("高层计划", result)
        self.assertIn("重规划", result)
        self.assertIn("深圳", result)
        self.assertIn("1+1", result)
        self.assertIn("2", result)


if __name__ == "__main__":
    unittest.main()
