from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .llm_client import DASHSCOPE_BASE_URL, get_api_key, get_llm_client


def _fallback_plan(user_query: str) -> List[str]:
    """无 LLM 或调用失败时使用的规则回退计划。"""
    text = user_query
    if "对比" in text and "优缺点" in text:
        return [
            "澄清要对比的对象以及使用场景",
            "对每个对象分别检索基础信息",
            "整理每个对象的优点和缺点",
            "给出总结性结论和建议",
        ]
    if "计算" in text or "数学" in text or "公式" in text:
        return [
            "从问题中提取要计算的数学表达式",
            "使用计算工具得到结果",
            "用自然语言解释计算过程和结果",
        ]
    return [
        "理解用户问题的关键点",
        "检索相关的背景知识或信息",
        "整理并组织回答结构",
        "用清晰的自然语言给出最终答案",
    ]


def _parse_steps_from_llm(text: str) -> Optional[List[str]]:
    """
    从 LLM 返回的文本中解析步骤列表。
    支持格式：1. xxx  2. xxx  / 第一步 xxx  / - xxx 等。
    """
    if not text or not text.strip():
        return None
    lines = [s.strip() for s in text.strip().split("\n") if s.strip()]
    steps: List[str] = []
    # 匹配 "1. 内容" "第一步 内容" "- 内容" "1）内容"
    pattern = re.compile(
        r"^(?:\d+[\.\)、]\s*|第[一二三四五六七八九十\d]+步\s*|[-*]\s*)?(.+)$"
    )
    for line in lines:
        if not line:
            continue
        # 去掉行首编号/符号，取正文
        m = pattern.match(line)
        if m:
            step = m.group(1).strip()
            if step and len(step) > 1:
                steps.append(step)
        else:
            steps.append(line)
    return steps if steps else None


@dataclass
class Planner:
    """
    使用 LLM 根据用户任务生成多步计划。默认通义千问（Qwen）；
    可通过 provider="deepseek" 或环境变量 LLM_PROVIDER=deepseek 切换为 DeepSeek。
    无 API Key 或调用失败时回退到规则计划。
    """

    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = field(default=None)
    provider: Optional[str] = field(default=None, repr=False)  # "qwen" | "deepseek"
    model: str = "qwen-plus"
    use_llm: bool = True

    def _get_client(self):
        return get_llm_client(api_key=self.api_key, base_url=self.base_url, provider=self.provider)

    def _call_llm(self, user_query: str) -> Optional[List[str]]:
        client = self._get_client()
        api_key = get_api_key(self.api_key, provider=self.provider)
        if not api_key:
            return None
        system = """你是一个任务规划助手。根据用户的自然语言问题，输出一个简洁的、可执行的多步计划。
                    要求：
                    - 每行一个步骤，用数字编号（如 1. 步骤一 2. 步骤二）；
                    - 步骤要具体、可操作，便于后续用搜索、计算等工具执行；
                    - 只输出步骤列表，不要其他解释。"""
        user = f"用户问题：\n{user_query}\n\n请给出执行计划（每行一步，带编号）："
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()
            return _parse_steps_from_llm(content)
        except Exception:
            return None

    def make_plan(self, user_query: str) -> List[str]:
        if not self.use_llm:
            return _fallback_plan(user_query)
        steps = self._call_llm(user_query)
        if steps:
            return steps
        return _fallback_plan(user_query)
