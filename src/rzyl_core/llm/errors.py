"""大模型能力的错误类型。

客户端把 httpx 的底层异常收敛成下面这几个，让调用方（管道、装配层）不必认识
httpx。管道据此决定"整窗重试还是进死信"（见 issue #1 的故障处理）。
"""

from __future__ import annotations


class LLMError(Exception):
    """所有大模型能力错误的基类。"""


class LLMTimeoutError(LLMError):
    """请求超时（含重试后仍超时）。"""


class LLMRequestError(LLMError):
    """请求失败：网络错误，或重试后仍是可重试状态（429 / 5xx），或不可重试的 4xx。"""


class LLMResponseError(LLMError):
    """拿到了响应，但结构不符合预期（不是合法 JSON、缺字段、字段类型不对）。"""


__all__ = ["LLMError", "LLMRequestError", "LLMResponseError", "LLMTimeoutError"]
