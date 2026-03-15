"""
共享 LLM 客户端：支持通义千问（Qwen/DashScope）与 DeepSeek（OpenAI 兼容）。
供 Planner 与 ReActExecutor 复用。
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 通义千问 DashScope
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 可选：通过环境变量选择 provider，如 LLM_PROVIDER=deepseek
PROVIDER_ENV = "LLM_PROVIDER"


def _resolve_provider(provider: Optional[str] = None) -> str:
    """解析当前使用的 provider：qwen 或 deepseek。"""
    p = (provider or os.environ.get(PROVIDER_ENV) or "").strip().lower()
    if p == "deepseek":
        return "deepseek"
    return "qwen"


def get_llm_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> Any:
    """
    返回 OpenAI 兼容的 Chat 客户端。
    provider: "qwen"（默认）使用 DashScope，"deepseek" 使用 DeepSeek；
    也可通过环境变量 LLM_PROVIDER=deepseek 切换。
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("需要安装 openai：pip install openai") from e
    p = _resolve_provider(provider)
    if p == "deepseek":
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        base = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL
    else:
        key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = base_url or os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL
    kwargs: dict = {}
    if key:
        kwargs["api_key"] = key
    kwargs["base_url"] = base
    return OpenAI(**kwargs)


def get_api_key(api_key: Optional[str] = None, provider: Optional[str] = None) -> Optional[str]:
    """获取当前生效的 API Key（按 provider 选择 DASHSCOPE 或 DEEPSEEK）。"""
    p = _resolve_provider(provider)
    if p == "deepseek":
        return api_key or os.environ.get("DEEPSEEK_API_KEY")
    return api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
