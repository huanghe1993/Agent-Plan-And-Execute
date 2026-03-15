"""
Agent 状态与 Replan 策略（参考 LangGraph 的状态机 + 专用 replan 节点）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentState:
    """
    Plan-and-Execute Agent 的状态（状态机中的状态节点）。
    类似 LangGraph 的 State：在 plan / execute / replan 之间流转时携带的上下文。
    """
    # 用户问题
    user_query: str = ""
    # 计划
    plan: List[str] = field(default_factory=list)
    # 当前步骤索引
    current_step_index: int = 0
    # 每个已执行步骤的 transcript
    step_results: List[str] = field(default_factory=list)
    # 上一个步骤
    last_step: Optional[str] = None
    # 上一个步骤执行结果
    last_step_result: Optional[str] = None
    # 重规划次数
    replan_count: int = 0
    # 最大重规划次数
    max_replans: int = 2

    @property
    def has_next_step(self) -> bool:
        return self.current_step_index < len(self.plan)

    @property
    def current_step(self) -> Optional[str]:
        if not self.has_next_step:
            return None
        return self.plan[self.current_step_index]

    def advance_to_next_step(self, step_transcript: str) -> "AgentState":
        """执行完当前步骤后：记录结果并指向下一步。"""
        self.step_results.append(step_transcript)
        self.last_step = self.plan[self.current_step_index]
        self.last_step_result = step_transcript
        self.current_step_index += 1
        return self

    def trigger_replan(self, new_plan: List[str], keep_previous_results: bool = False) -> "AgentState":
        """
        触发「整份计划」重规划：用 new_plan 替换原计划，从步骤 0 重新执行全部步骤。
        默认清空 step_results。
        """
        self.plan = new_plan
        self.current_step_index = 0
        self.replan_count += 1
        self.last_step = None
        self.last_step_result = None
        if not keep_previous_results:
            self.step_results = []
        return self

    def trigger_incremental_replan(self, new_plan_for_remaining: List[str]) -> "AgentState":
        """
        触发「增量式」重规划（Incremental Replanning）：保留已成功步骤与结果，仅重塑剩余路径。
        - 调用前：刚执行完的一步失败，current_step_index 已指向“下一步”，last_step/last_step_result 为失败步。
        - 已完成步骤 = plan[:current_step_index-1]，其结果 = step_results[:current_step_index-1]。
        - 将计划改为：已完成步骤 + new_plan_for_remaining；丢弃失败步的结果；下一步从 new_plan_for_remaining[0] 执行。
        """
        # 失败步索引为 current_step_index - 1，已完成成功步骤数为 current_step_index - 1
        completed_count = self.current_step_index - 1
        if completed_count < 0:
            completed_count = 0
        self.plan = self.plan[:completed_count] + new_plan_for_remaining
        self.step_results = self.step_results[:completed_count]
        self.current_step_index = completed_count
        self.replan_count += 1
        self.last_step = None
        self.last_step_result = None
        return self


class ReplanPolicy:
    """
    是否触发 Replan 的策略（对应 LangGraph 中「从 execute 到 replan 节点」的边条件）。
    支持规则 + 可选 LLM 判断。
    mode: "full" 整份重规划（抛弃已执行，从步骤 0 重跑）；"incremental" 增量式重规划（保留已完成，只重规划剩余）。
    """
    use_llm: bool = False
    keywords_failure: Optional[List[str]] = None
    mode: str = "incremental"  # "full" | "incremental"

    def __init__(
        self,
        use_llm: bool = False,
        keywords_failure: Optional[List[str]] = None,
        mode: str = "incremental",
    ):
        self.use_llm = use_llm
        self.keywords_failure = keywords_failure or ["错误", "失败", "未找到", "不存在", "计算错误", "工具执行错误"]
        self.mode = mode if mode in ("full", "incremental") else "incremental"

    def should_replan(self, state: AgentState) -> bool:
        """
        根据当前状态判断是否需要进入 replan 节点。
        - 若已达最大重规划次数，不再 replan。
        - 若刚执行的步骤结果包含失败关键词，可触发 replan。
        - 若启用 use_llm，可再问 LLM「结果是否对步骤有帮助」。
        """
        if state.replan_count >= state.max_replans:
            return False
        # 如果上一个步骤执行结果为空，则不重规划
        if state.last_step_result is None:
            return False
        text = state.last_step_result
        # 如果上一个步骤执行结果包含失败关键词，则重规划
        if any(k in text for k in self.keywords_failure):
            return True
        # 如果启用use_llm，则用LLM判断是否需要重规划
        if self.use_llm:
            return self._llm_should_replan(state)
        # 否则不重规划
        return False

    def _llm_should_replan(self, state: AgentState) -> bool:
        """用 LLM 判断：当前步骤结果是否对任务无帮助，需要重规划。"""
        try:
            from .llm_client import get_api_key, get_llm_client
        except ImportError:
            return False
        if not get_api_key():
            return False
        client = get_llm_client()
        model = "deepseek-chat"  # 可与 executor 一致，这里简化
        prompt = f"""任务：{state.user_query}
                    当前步骤：{state.last_step}
                    步骤执行结果：{state.last_step_result}

                    请判断：该结果是否对完成本步骤/任务明显无帮助（如工具报错、未找到、答非所问）？
                    只回答一个字：是 或 否"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return "是" in answer
        except Exception:
            return False
