from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .planner import Planner
from .react_executor import ReActExecutor
from .state import AgentState, ReplanPolicy


@dataclass
class PlanAndReActAgent:
    """
    顶层 Agent：Plan -> Execute（按步 ReAct）-> 可选 Replan（子任务失败时重规划）。
    参考 LangGraph：将流程建模为状态机，Replan 为专用节点，通过 ReplanPolicy 决定是否从 execute 转入 replan。
    """

    planner: Planner
    executor: ReActExecutor
    replan_policy: Optional[ReplanPolicy] = field(default=None, repr=False)

    def _build_remaining_task_prompt(self, state: AgentState) -> str:
        """为增量式重规划构造「剩余任务」描述，供 Planner 仅规划从失败步起的新路径。"""
        completed_count = state.current_step_index - 1
        # 第一步就失败时，尚无已完成步骤，仅基于失败信息重规划
        if completed_count <= 0:
            return (
                f"原始任务：{state.user_query}\n\n"
                "（第一步即失败，尚无已完成步骤。）\n\n"
                f"当前步骤「{state.last_step}」执行失败，结果如下：\n{state.last_step_result}\n\n"
                "请针对「剩余任务」给出新的执行步骤（从本步起，可包含重试或替代方案），每行一步，带编号。"
            )
        # 不是第一步就失败时，基于已完成步骤和失败信息重规划
        # 构造已完成步骤及结果的描述
        completed_lines = []
        for i in range(completed_count):
            step_text = state.plan[i]
            result_preview = (state.step_results[i] or "")[:300].replace("\n", " ") # 结果预览
            completed_lines.append(f"  {i + 1}. {step_text} → {result_preview}…")
        return (
            f"原始任务：{state.user_query}\n\n"
            "已完成步骤及结果：\n" + "\n".join(completed_lines) + "\n\n" # 已完成步骤及结果
            f"当前步骤「{state.last_step}」执行失败，结果如下：\n{state.last_step_result}\n\n" # 失败步骤及结果
            "请仅针对「剩余任务」给出新的执行步骤（从当前失败步骤起，可包含重试或替代方案），每行一步，带编号。" # 剩余任务描述
        )

    def run(self, user_query: str) -> str:
        # 初始化AgentState,用户问题
        state = AgentState(user_query=user_query)

        # 状态机：plan -> execute（单步）-> [replan? -> 新 plan] -> 下一步 …
        while True:
            # 1. plan 节点：无计划时生成
            if not state.plan:
                # 生成计划存储到state里面的plan列表中
                state.plan = self.planner.make_plan(user_query)
                state.current_step_index = 0
            # 如果计划为空，则跳出循环
            if not state.plan:
                break
            # 打印计划
            print(f"计划：{state.plan}")
            print("-"*50)
            # 2. execute 节点：执行当前一步
            step = state.plan[state.current_step_index]
            step_header = f"=== 执行步骤 {state.current_step_index + 1}: {step} ==="
            print(step_header)
            # 执行步骤存储到state里面的step_results列表中
            step_detail = self.executor.execute_step(step, user_query)
            # 执行完当前步骤后：记录结果并指向下一步。
            state.advance_to_next_step(step_header + "\n" + step_detail)

            # 3. 是否进入 replan 节点（LangGraph 风格的条件边）
            if self.replan_policy and self.replan_policy.should_replan(state):
                print("\n⚠️ 触发 Replan：上一步执行不理想，重新规划中…\n")
                if self.replan_policy.mode == "incremental":
                    print("\n⚠️  触发增量 Replan\n")
                    # 为增量式重规划构造「剩余任务」描述，供 Planner 仅规划从失败步起的新路径。
                    remaining_prompt = self._build_remaining_task_prompt(state)
                    # 生成剩余任务计划存储到new_plan_remaining列表中
                    new_plan_remaining = self.planner.make_plan(remaining_prompt)
                    if new_plan_remaining:
                        # 触发增量式重规划，保留已完成步骤，仅重塑剩余路径
                        state.trigger_incremental_replan(new_plan_remaining)
                        print("（增量式重规划：保留已完成步骤，仅重塑剩余路径）\n")
                    else:
                        # 如果生成剩余任务计划失败，则进行整份重规划
                        fallback_plan = self.planner.make_plan(user_query) or state.plan
                        state.trigger_replan(fallback_plan, keep_previous_results=False)
                else:
                    # 整份重规划, 重新规划全部步骤
                    hint = f"（上一轮步骤「{state.last_step}」执行不理想，请重新规划）" if state.last_step else ""
                    new_plan = self.planner.make_plan(user_query + hint)
                    state.trigger_replan(new_plan, keep_previous_results=False)
                continue

            if not state.has_next_step:
                break

        # 5. 汇总
        plan_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state.plan))
        final_answer = (
            f"用户问题：{user_query}\n\n"
            f"高层计划：\n{plan_str}\n\n"
            f"执行过程与中间推理（ReAct）"
            + (f"（含 {state.replan_count} 次重规划）" if state.replan_count else "")
            + "：\n\n"
            + "\n\n".join(state.step_results)
            + "\n\n最终总结：\n"
            "以上是基于 Plan-and-Execute + ReAct 的分步推理与工具调用示例结果。"
        )
        return final_answer
