"""聊天模型接口与结果对象。

接口只干一件事：输入系统提示词与用户内容，返回「文本 + token 用量」。它**不负责
记账落库**——用量随结果返回，由调用方（#7）决定怎么记。这样接口不依赖数据库，
`rzyl_core.llm` 也就不需要认识仓储或设置对象。

服务商、基址、模型名一律不硬编码默认值，全部由调用方传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ChatUsage:
    """一次聊天调用的 token 用量。

    单价与费用估算不在这里做：客户端只返回 token，估算与记账归 #7。
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ChatResult:
    """一次聊天调用的结果：文本、用量、以及产出它的模型名。

    ``model`` 一并返回，方便 #7 在 llm_call 记账里直接落库，不必再从客户端反查。
    """

    text: str
    usage: ChatUsage
    model: str


@runtime_checkable
class ChatModel(Protocol):
    """可替换的聊天模型接口。"""

    async def complete(self, system: str, user: str) -> ChatResult:
        """按系统提示词与用户内容取一次补全。"""
        ...


__all__ = ["ChatModel", "ChatResult", "ChatUsage"]
