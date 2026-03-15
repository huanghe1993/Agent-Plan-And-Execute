"""
Plan-and-Execute + ReAct Agent 测试：
1. 不需要 Replan：全部步骤成功，不触发重规划
2. 需要全量 Replan：第一步失败，mode=full，整份重规划后重跑
3. 需要增量 Replan：第二步失败，mode=incremental，保留第一步结果，仅重规划剩余
"""
from __future__ import annotations

import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from agent import PlanAndReActAgent, Planner, ReActExecutor, ReplanPolicy


def _make_agent(
    planner: Planner,
    executor: ReActExecutor,
    replan_policy: Optional[ReplanPolicy] = None,
) -> PlanAndReActAgent:
    return PlanAndReActAgent(planner=planner, executor=executor, replan_policy=replan_policy)


class TestNoReplan(unittest.TestCase):
    """用例 1：不需要 Replan，所有步骤一次成功。"""

    @patch("builtins.print")  # 抑制 run() 内 print
    def test_all_steps_succeed_no_replan(self, mock_print: MagicMock):
        planner = MagicMock(spec=Planner)
        planner.make_plan.return_value = ["步骤A", "步骤B", "步骤C"]

        executor = MagicMock(spec=ReActExecutor)
        # 每一步都返回成功内容（不含「错误」「失败」等关键词）
        executor.execute_step.side_effect = [
            "Thought: 执行A\nFinal Answer: A完成",
            "Thought: 执行B\nFinal Answer: B完成",
            "Thought: 执行C\nFinal Answer: C完成",
        ]

        policy = ReplanPolicy(use_llm=False, mode="incremental")
        agent = _make_agent(planner, executor, policy)

        result = agent.run("测试任务：完成A、B、C")

        self.assertEqual(planner.make_plan.call_count, 1)
        self.assertEqual(executor.execute_step.call_count, 3)
        self.assertIn("步骤A", result)
        self.assertIn("步骤B", result)
        self.assertIn("步骤C", result)
        self.assertIn("A完成", result)
        self.assertIn("B完成", result)
        self.assertIn("C完成", result)
        self.assertNotIn("重规划", result)
        self.assertNotIn("触发 Replan", result)


class TestFullReplan(unittest.TestCase):
    """用例 2：需要全量 Replan。第一步失败，触发整份重规划后从步骤 0 重跑。"""

    @patch("builtins.print")
    def test_first_step_fails_then_full_replan(self, mock_print: MagicMock):
        planner = MagicMock(spec=Planner)
        # 第一次规划；第二次（重规划）给新计划
        planner.make_plan.side_effect = [
            ["步骤1", "步骤2"],
            ["步骤1替代", "步骤2"],
        ]

        executor = MagicMock(spec=ReActExecutor)
        # 第一次跑：步骤1 返回失败（含「错误」）→ 触发 full replan；重规划后两步都成功
        executor.execute_step.side_effect = [
            "错误：工具调用失败",           # 步骤1 失败 → 触发 full replan
            "Thought: 执行1\nFinal Answer: 1完成",   # 重规划后新计划第 0 步
            "Thought: 执行2\nFinal Answer: 2完成",   # 新计划第 1 步
        ]

        policy = ReplanPolicy(use_llm=False, mode="full")
        agent = _make_agent(planner, executor, policy)

        result = agent.run("测试任务")

        self.assertEqual(planner.make_plan.call_count, 2)
        self.assertEqual(executor.execute_step.call_count, 3)
        self.assertIn("重规划", result)
        self.assertIn("1完成", result)
        self.assertIn("2完成", result)


class TestIncrementalReplan(unittest.TestCase):
    """用例 3：需要增量 Replan。第一步成功、第二步失败，仅重规划剩余路径并保留第一步结果。"""

    @patch("builtins.print")
    def test_second_step_fails_then_incremental_replan(self, mock_print: MagicMock):
        planner = MagicMock(spec=Planner)
        # 第一次：完整计划三步骤；第二次：仅「剩余任务」的两步（替代原步骤2、3）
        planner.make_plan.side_effect = [
            ["步骤一", "步骤二", "步骤三"],
            ["步骤二重试", "步骤三"],
        ]

        executor = MagicMock(spec=ReActExecutor)
        # 步骤一成功 → 步骤二失败（含「失败」）→ 触发 incremental replan → 执行 步骤二重试、步骤三 成功
        executor.execute_step.side_effect = [
            "Thought: 做一\nFinal Answer: 第一步完成",
            "失败：未找到工具",
            "Thought: 重试二\nFinal Answer: 第二步完成",
            "Thought: 做三\nFinal Answer: 第三步完成",
        ]

        policy = ReplanPolicy(use_llm=False, mode="incremental")
        agent = _make_agent(planner, executor, policy)

        result = agent.run("测试任务：一二三")

        self.assertEqual(planner.make_plan.call_count, 2)
        self.assertEqual(executor.execute_step.call_count, 4)
        # 第一步结果应保留在最终输出中（增量重规划不丢弃）
        self.assertIn("第一步完成", result)
        self.assertIn("第二步完成", result)
        self.assertIn("第三步完成", result)
        self.assertIn("重规划", result)


if __name__ == "__main__":
    unittest.main()
