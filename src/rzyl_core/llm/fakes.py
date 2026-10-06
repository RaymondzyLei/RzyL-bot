"""测试与离线回放共用的假实现。

放在 core 而不是 tests/，因为回放脚本要复用同一套，而不是把测试替身散在测试目录里。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rzyl_core.llm.chat import ChatResult


@dataclass(frozen=True, slots=True)
class ReceivedPrompt:
    """假模型收到的一次提示词，原样留存，供测试断言与 ``--dry-run`` 打印。"""

    system: str
    user: str


class FakeChatModel:
    """按预设顺序返回结果的假聊天模型。

    脚本里每一项要么是 ``ChatResult``（正常 JSON、格式错误文本、空数组都只是文本
    内容不同），要么是一个异常实例（用来模拟超时或报错）。脚本用完还没人取，
    就抛 ``RuntimeError``——宁可测试里明确炸掉，也不要静默重复上一条。
    """

    def __init__(self, script: Iterable[ChatResult | Exception]) -> None:
        self._script: list[ChatResult | Exception] = list(script)
        self._index = 0
        #: 依次留存的 (系统提示词, 用户内容)；即使那一步抛异常也会留存。
        self.received: list[ReceivedPrompt] = []

    async def complete(self, system: str, user: str) -> ChatResult:
        self.received.append(ReceivedPrompt(system=system, user=user))
        if self._index >= len(self._script):
            raise RuntimeError(
                f"假聊天模型的脚本已用完（共 {len(self._script)} 条），"
                f"但第 {self._index + 1} 次调用仍要求结果"
            )
        item = self._script[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def remaining(self) -> int:
        """脚本里还没被消费的条数。"""
        return len(self._script) - self._index


__all__ = ["FakeChatModel", "ReceivedPrompt"]
