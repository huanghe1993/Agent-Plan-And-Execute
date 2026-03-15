from __future__ import annotations

import abc
from typing import Any, Dict


class Tool(abc.ABC):
    """
    工具基类。
    """

    name: str
    description: str

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Any:  # pragma: no cover - demo 项目不写测试
        raise NotImplementedError


class SearchTool(Tool):
    """
    简单搜索工具（示例用，实际可以接入真正搜索 / 向量数据库）。
    """

    name = "search"
    description = "进行简单的知识检索，输入 query 字符串，输出简要说明"

    def __call__(self, query: str) -> str:
        # 此处仅为 Demo，可替换为真实搜索逻辑
        return f"[search-result] 关于『{query}』的基础信息：这是一个模拟搜索结果。"


class CalculatorTool(Tool):
    """
    简单计算器工具。
    """

    name = "calculator"
    description = "执行数学计算，输入表达式字符串，输出结果"

    def __call__(self, expression: str) -> str:
        try:
            # 使用受限的 eval 环境，避免安全问题（仅演示）
            result = eval(expression, {"__builtins__": {}})
            return f"[calc-result] {expression} = {result}"
        except Exception as e:  # noqa: BLE001
            return f"[calc-error] 计算失败: {e}"


def build_default_tools() -> Dict[str, Tool]:
    """
    构建一个默认工具字典，Agent 会使用它来执行 Action。
    """

    tools: Dict[str, Tool] = {
        SearchTool.name: SearchTool(),
        CalculatorTool.name: CalculatorTool(),
    }
    return tools

