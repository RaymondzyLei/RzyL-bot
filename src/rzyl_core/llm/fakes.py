"""测试与离线回放共用的假实现。

放在 core 而不是 tests/，因为回放脚本要复用同一套，而不是把测试替身散在测试目录里。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from rzyl_core.llm.chat import ChatResult, ChatUsage


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


#: 提示词里「本窗口消息」的编号行：``序号：发送者(QQ号) 内容``。
_MESSAGE_LINE = re.compile(r"^(\d+)：([^(]+)\((\d+)\)\s?(.*)$", re.MULTILINE)

#: 离线演示条目里 statement 的截断长度，避免一句超长原文把条目撑爆。
_STATEMENT_LIMIT = 120


class EchoChatModel:
    """离线演示用的假模型：把本窗口第一条消息原样回声成一条条目。

    它**不解析系统提示词**，只从用户内容里找第一行 ``序号：发送者(QQ号) 内容``，
    据此造一条合法 JSON 条目（``evidence`` 指向那一行、``person_refs`` 用行里的
    QQ 号与显示名）。用途是：没有 API key、没有网络时也能让整条链路——含校验、
    去重、入库——跑通并落库，作为里程碑 1 的验收入口。

    把假模型换成真模型只需改配置（设好 ``RZYL_CHAT_API_KEY``），不改任何代码。
    """

    #: 与真客户端返回的 ``model`` 字段对齐；记账里一眼能看出是离线演示。
    model_name = "echo-offline"

    def __init__(self) -> None:
        #: 依次留存的 (系统提示词, 用户内容)，供 dry-run 或测试回看。
        self.received: list[ReceivedPrompt] = []

    async def complete(self, system: str, user: str) -> ChatResult:
        self.received.append(ReceivedPrompt(system=system, user=user))
        match = _MESSAGE_LINE.search(user)
        if match is None:
            return self._result("[]", system=system, user=user)

        sequence, sender, user_id, content = match.groups()
        statement = (content.strip() or "（空消息）")[:_STATEMENT_LIMIT]
        items = [
            {
                "category": "event",
                "statement": statement,
                "detail": None,
                "confidence": 0.8,
                "evidence": [int(sequence)],
                "person_refs": [{"user_id": int(user_id), "nickname_snapshot": sender.strip()}],
                "occurred_at": None,
                "supersedes": [],
            }
        ]
        return self._result(json.dumps(items, ensure_ascii=False), system=system, user=user)

    def _result(self, text: str, *, system: str, user: str) -> ChatResult:
        # 粗略的字数折半当 token 估值，够记账演示用，不追求与真分词一致。
        return ChatResult(
            text=text,
            usage=ChatUsage(
                input_tokens=(len(system) + len(user)) // 2,
                output_tokens=len(text) // 2,
            ),
            model=self.model_name,
        )


__all__ = ["EchoChatModel", "FakeChatModel", "ReceivedPrompt"]
