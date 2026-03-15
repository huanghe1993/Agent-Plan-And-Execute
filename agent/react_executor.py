"""
使用 LLM 执行标准 ReAct 循环：Thought -> Action -> Observation -> ... -> Final Answer。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import get_api_key, get_llm_client
from .tools import Tool


def _tools_description(tools: Dict[str, Tool]) -> str:
    """生成工具列表描述，供 prompt 使用。"""
    if not tools:
        return "（当前无可用工具）"
    lines = []
    for name, t in tools.items():
        lines.append(f"- {name}: {t.description}")
    return "\n".join(lines)


def _parse_react_turn(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析 LLM 一轮输出，提取 Thought、Action（tool_name, tool_input）或 Final Answer。
    返回 (thought, action_tool_name, action_tool_input) 或 (thought, None, final_answer)。
    """
    text = (text or "").strip()
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_input: Optional[str] = None
    final_answer: Optional[str] = None

    # Thought: ...（到下一关键字或结尾）
    thought_m = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if thought_m:
        thought = thought_m.group(1).strip()

    # Action: tool_name[tool_input]
    action_m = re.search(r"Action:\s*(\w+)\s*\[\s*([^\]]*)\s*\]", text, re.IGNORECASE)
    if action_m:
        action_name = action_m.group(1).strip()
        action_input = action_m.group(2).strip().strip('"').strip("'")

    # Final Answer: ...
    ans_m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if ans_m:
        final_answer = ans_m.group(1).strip()

    if final_answer is not None:
        return (thought, None, final_answer)
    if action_name:
        return (thought, action_name, action_input or "")
    return (thought, None, None)


@dataclass
class ReActExecutor:
    """
    使用 LLM 执行标准 ReAct：每轮由模型输出 Thought，再选择 Action（调用工具）或 Final Answer 结束。
    默认 Qwen；可通过 provider="deepseek" 或环境变量 LLM_PROVIDER=deepseek 切换为 DeepSeek。
    """

    tools: Dict[str, Tool] = field(default_factory=dict)
    max_react_rounds: int = 8
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = field(default=None)
    provider: Optional[str] = field(default=None, repr=False)  # "qwen" | "deepseek"
    model: str = "qwen-plus"
    use_llm: bool = True

    def _get_client(self):
        return get_llm_client(api_key=self.api_key, base_url=self.base_url, provider=self.provider)

    def _build_system_prompt(self, step: str, user_query: str) -> str:
        tools_desc = _tools_description(self.tools)
        return f"""你是一个 ReAct 推理助手。请针对当前子任务进行一步步推理，并在需要时调用工具。

                当前子任务：{step}
                用户原始问题：{user_query}

                可用工具：
                {tools_desc}

                你必须严格按以下格式输出（每轮只输出一次 Thought，然后二选一）：
                Thought: <你的推理>
                Action: <工具名>[<工具输入>]
                或
                Thought: <你的推理>
                Final Answer: <针对当前子任务的结论>

                若无需调用工具即可得出结论，请直接输出 Thought 与 Final Answer。"""

    def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        api_key = get_api_key(self.api_key, provider=self.provider)
        if not api_key:
            return None
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return None

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"错误：未找到工具「{tool_name}」。"
        try:
            out = tool(tool_input)
            return str(out)
        except Exception as e:
            return f"工具执行错误：{e}"

    def execute_step(self, step: str, user_query: str) -> str:
        """
        对单个 Plan 步骤执行标准 ReAct 循环，返回该步骤的完整推理过程与结论。
        """
        if not self.use_llm or not get_api_key(self.api_key, provider=self.provider):
            return self._fallback_execute_step(step, user_query)

        system = self._build_system_prompt(step, user_query)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请开始推理并给出第一轮 Thought，以及 Action 或 Final Answer。"},
        ]
        transcript: List[str] = []
        round_count = 0

        while round_count < self.max_react_rounds:
            round_count += 1
            response = self._call_llm(messages)
            if not response:
                transcript.append("（LLM 调用失败，结束本步骤）")
                break
            # 添加LLM的Thought 和 Action到transcript中
            transcript.append(f"Round {round_count}: {response}")
            _thought, action_name, action_input_or_answer = _parse_react_turn(response)
            
            # Final Answer -> 结束
            if action_name is None and isinstance(action_input_or_answer, str) and action_input_or_answer:
                print(f"Round {round_count}: Thought: {_thought}, Final Answer: {action_input_or_answer}")
                break

            # Action -> 执行工具并继续
            if action_name:
                print(f"Round {round_count}: Thought: {_thought}, action_name: {action_name}, action_input_or_answer: {action_input_or_answer}")
                observation = self._run_tool(action_name, action_input_or_answer or "")
                # 添加工具执行结果到transcript中
                transcript.append(f"Observation: {observation}")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续推理，给出下一轮 Thought，然后 Action 或 Final Answer。",
                })
                continue

            # 解析不到有效 Action 也非 Final Answer，提示模型重试
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "请严格按格式输出：Thought: ... 然后 Action: 工具名[输入] 或 Final Answer: ...",
            })

        return "\n".join(transcript)

    def _fallback_execute_step(self, step: str, user_query: str) -> str:
        """无 LLM 或未配置 Key 时的简单回退：仅输出说明，不调用工具。"""
        return (
            f"Thought: 当前未配置 LLM，无法执行标准 ReAct。子任务：{step}\n"
            f"Step-Conclusion: 请配置 DASHSCOPE_API_KEY 后使用 LLM 执行 ReAct。"
        )
