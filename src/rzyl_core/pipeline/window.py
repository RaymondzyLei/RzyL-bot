"""窗口组装：把一个群的连续消息流攒成「可以直接交给模型的那一段」。

窗口的内容由三段构成，边界刻意保持清晰（见 issue #1、工单 #4）：

1. **上一窗口尾部**——只作上下文，让「他说的那个」能被解析；明确标注**不重复提取**。
   它不参与编号，模型的 ``evidence`` 不应引用它。
2. **上一窗口已记条目摘要**——从仓储按群取最近若干条，避免相邻窗口把同一件事记两遍。
   摘要里带 ``[#编号]``，模型的 ``supersedes`` 引用的就是它。
3. **本窗口消息**——每条带**窗口内序号**（1 起），模型的 ``evidence`` 引用的就是序号。

成窗规则：**三条任一成立即关**，每个群各自独立缓冲：

1. **条数上限**：本窗口条数达到 ``window_message_limit``——无条件成窗。
2. **时间到点**：新消息与缓冲首条的时间跨度达到 ``window_minutes``，**且**缓冲里条数不少于
   ``window_min_messages``——成窗。条数不够就继续攒着，避免稀疏流量下把只有一两条消息的
   缓冲也收掉、白调一次模型。
3. **兜底年龄**：缓冲里最老那条消息的年龄达到 ``window_max_minutes``——无条件成窗，不看条数。
   没有它，一个只发了一句话就安静下来的群，那条消息会无限期滞留、永远不会被提取。

时间判定看的是**消息自己的** ``sent_at``：在 :meth:`WindowAssembler.add` 里判定「已有缓冲
该不该关」时，"当下"取**本条新消息的** ``sent_at``（回放与掉线回补灌历史消息时这样才正确，
不必再「把时钟拨到消息时间」）；只有 :meth:`WindowAssembler.flush_expired` 在无人再来消息时
用注入时钟当"当下"。

给下游的契约（#7 提取入库要接）：

- :func:`assemble_window` 与 :class:`WindowAssembler` 产出 :class:`AssembledWindow`；
- ``AssembledWindow.messages`` 是本窗口消息，每条带 ``sequence``；
- ``AssembledWindow.sequence_map`` 给出 **窗口内序号 → 真实消息编号**（``message.id``）的映射，
  #7 把模型输出的 ``evidence`` 序号经 :meth:`AssembledWindow.resolve_evidence` 换回真实编号；
- :meth:`AssembledWindow.render` 直接渲染成 :class:`~rzyl_core.llm.prompts.RenderedPrompt`。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import cast

from rzyl_core.db.models import Memory, Message
from rzyl_core.db.repository import Repository
from rzyl_core.llm.prompts import (
    PromptTemplate,
    RenderedPrompt,
    current_prompt_version,
    get_prompt_template,
)
from rzyl_core.settings import Settings

#: 默认捎带上一个窗口的最后几条消息作为上下文。
DEFAULT_PREVIOUS_TAIL_SIZE = 3

#: 默认取上一个窗口「已记条目摘要」的条数。
DEFAULT_REMEMBERED_LIMIT = 10

#: 上一窗口尾部在提示词里的标注：只作上下文，不重复提取。
PREVIOUS_TAIL_HEADER = "（以下为上一窗口尾部，仅作上下文，不要重复提取）"


class UnknownSequenceError(ValueError):
    """模型的 ``evidence`` 引用了本窗口不存在的序号。"""


def sender_name(*, user_id: int, nickname: str | None, card: str | None) -> str:
    """取发送者显示名：群名片优先，其次昵称，都没有就用 QQ 号。"""
    return card or nickname or str(user_id)


@dataclass(frozen=True, slots=True)
class WindowMessage:
    """进入窗口组装的一条消息（组装器的**输入单元**，尚未成窗）。

    与 :class:`SequencedMessage` 的区别看「输入 / 输出」：本类是刚进缓冲的一条消息，
    只认数据库里的真实编号；:class:`SequencedMessage` 是**已成窗后窗口里的一行**，
    额外带窗口内序号。两者名字都像「窗口里的消息」，故在此点明。

    三个容易混的编号一次说清：

    - ``message_id``：``message`` 表的自增主键，即**真实消息编号**，全局唯一、不重用；
      #7 把模型引用的窗口内序号换回的就是它。
    - ``platform_message_id``：QQ / OneBot 的平台短 ID，只作参考，重启即失效，**不可**
      当长期标识（见 issue #1）。
    - ``message_seq``：OneBot ``get_group_msg_history`` 的翻页锚，只在历史来源里有值
      （见 :class:`~rzyl_core.pipeline.history.HistoryMessage`），也不当长期标识。

    仓储层的 :class:`~rzyl_core.db.models.Message` 用 :meth:`from_message` 适配；
    :meth:`WindowAssembler.add` 也直接收 ORM 对象，内部会自动适配。
    """

    message_id: int
    group_id: int
    user_id: int
    text: str
    sent_at: datetime
    nickname: str | None = None
    card: str | None = None

    @property
    def sender(self) -> str:
        """发送者显示名（群名片优先，其次昵称，再退到 QQ 号）。"""
        return sender_name(user_id=self.user_id, nickname=self.nickname, card=self.card)

    @classmethod
    def from_message(cls, message: Message) -> "WindowMessage":
        """从仓储的 ``Message`` 适配过来。

        SQLAlchemy 的 ``Mapped[...]`` 在 pyright 眼里不是内层类型，故显式 cast 一次；
        运行时就是普通标量。
        """
        return cls(
            message_id=cast(int, message.id),
            group_id=cast(int, message.group_id),
            user_id=cast(int, message.user_id),
            text=cast(str, message.text),
            sent_at=cast(datetime, message.sent_at),
            nickname=cast("str | None", message.nickname),
            card=cast("str | None", message.card),
        )


def _as_window_message(message: "WindowMessage | Message") -> WindowMessage:
    """把组装器的输入统一成 :class:`WindowMessage`。"""
    if isinstance(message, WindowMessage):
        return message
    return WindowMessage.from_message(message)


@dataclass(frozen=True, slots=True)
class SequencedMessage:
    """窗口里的一条消息（**已成窗**后窗口里的一行，组装器的输出）。

    ``sequence`` 是它在**本窗口**里的序号（1 起），也是提示词里给模型引用的编号。
    对上一窗口尾部而言，``sequence`` 是它在前一个窗口里的序号，仅信息性，
    **不可被 ``evidence`` 引用**。

    ``message_id`` 是真实消息编号（``message`` 表主键），语义与
    :attr:`WindowMessage.message_id` 完全一致；它**不是**平台短 ID，也不是翻页锚
    （后两者的区别见 :class:`WindowMessage` 的说明）。
    """

    sequence: int
    message_id: int
    group_id: int
    user_id: int
    sender: str
    text: str
    sent_at: datetime


@dataclass(frozen=True, slots=True)
class AssembledWindow:
    """一个可直接交给模型的窗口。

    同时携带三段内容与溯源映射；``remembered`` 保留原始条目对象，``remembered_summary``
    是渲染进提示词的文本。
    """

    group_id: int
    messages: tuple[SequencedMessage, ...]
    previous_tail: tuple[SequencedMessage, ...] = ()
    remembered: tuple[Memory, ...] = ()
    remembered_summary: str = ""
    prompt_version: str = field(default_factory=current_prompt_version)

    @property
    def started_at(self) -> datetime:
        """窗口第一条消息的发送时间。"""
        return self.messages[0].sent_at

    @property
    def ended_at(self) -> datetime:
        """窗口最后一条消息的发送时间。"""
        return self.messages[-1].sent_at

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def sequence_map(self) -> dict[int, int]:
        """窗口内序号 → 真实消息编号（``message.id``）。#7 换回真实编号就靠它。"""
        return {line.sequence: line.message_id for line in self.messages}

    def resolve_evidence(self, sequences: Iterable[int]) -> list[int]:
        """把模型引用的窗口内序号换回真实消息编号，保持原有顺序。

        引用了不存在的序号时抛 :class:`UnknownSequenceError`——宁可当场失败，
        也不要把编造的依据静默写进库。
        """
        mapping = self.sequence_map
        resolved: list[int] = []
        for sequence in sequences:
            try:
                resolved.append(mapping[sequence])
            except KeyError:
                raise UnknownSequenceError(
                    f"evidence 引用了窗口内不存在的序号 {sequence}；"
                    f"本窗口序号范围是 1..{self.message_count}"
                ) from None
        return resolved

    def render_window_messages(self) -> str:
        """渲染进 ``{{window_messages}}`` 的正文。

        上一窗口尾部在最前，带「仅作上下文」的标注且**不编号**（不进 ``evidence``）；
        本窗口消息在后，每条是 ``序号：发送者(QQ号) 内容``，与提示词里声明的格式一致。
        """
        blocks: list[str] = []
        if self.previous_tail:
            tail_lines = [PREVIOUS_TAIL_HEADER]
            tail_lines.extend(
                f"- {line.sender}({line.user_id}) {line.text}" for line in self.previous_tail
            )
            blocks.append("\n".join(tail_lines))
        blocks.append(
            "\n".join(
                f"{line.sequence}：{line.sender}({line.user_id}) {line.text}"
                for line in self.messages
            )
        )
        return "\n\n".join(blocks)

    def render(self, template: PromptTemplate | None = None) -> RenderedPrompt:
        """把窗口渲染成可直接交给 ``ChatModel`` 的提示词。

        默认用窗口自带的 ``prompt_version`` 取模板；也可传入别的版本来做对比回放。
        """
        resolved = template or get_prompt_template(self.prompt_version)
        return resolved.render(
            window_messages=self.render_window_messages(),
            remembered_summary=self.remembered_summary,
        )


def number_messages(messages: Sequence[WindowMessage]) -> tuple[SequencedMessage, ...]:
    """把一段消息按传入顺序编号成「窗口里的行」（序号 1 起）。

    本窗口正文与「上一窗口尾部」都靠它编号，所以只在这里实现一次——重试一个旧窗口时
    （见 :meth:`rzyl_core.runtime.Runtime._rebuild_window`）也要用它把尾部还原成同样的形状。
    """
    return tuple(
        SequencedMessage(
            sequence=index,
            message_id=message.message_id,
            group_id=message.group_id,
            user_id=message.user_id,
            sender=message.sender,
            text=message.text,
            sent_at=message.sent_at,
        )
        for index, message in enumerate(messages, start=1)
    )


def assemble_window(
    *,
    group_id: int,
    messages: Sequence[WindowMessage],
    previous_tail: Sequence[SequencedMessage] = (),
    remembered: Sequence[Memory] = (),
    remembered_summary: str | None = None,
    prompt_version: str | None = None,
) -> AssembledWindow:
    """纯函数：把一段消息装成一个窗口对象，序号按传入顺序 1 起编。

    不做任何 I/O——缓冲、仓储读取与时钟都在 :class:`WindowAssembler` 里，
    这里只负责把固定输入确定性地变成固定结构。
    """
    if not messages:
        raise ValueError("不能组装空窗口")
    return AssembledWindow(
        group_id=group_id,
        messages=number_messages(messages),
        previous_tail=tuple(previous_tail),
        remembered=tuple(remembered),
        remembered_summary=(
            remembered_summary if remembered_summary is not None else summarize_memories(remembered)
        ),
        prompt_version=prompt_version or current_prompt_version(),
    )


def summarize_memories(memories: Sequence[Memory]) -> str:
    """把上一窗口已记条目渲染成摘要文本：``[#编号] 类别 陈述``。

    编号是模型的 ``supersedes`` 要引用的锚点，因此必须出现。没有条目时返回空串，
    落在提示词里就是「已记条目摘要」这一段自然为空。
    """
    return "\n".join(
        f"[#{memory.id}] {memory.category.value} {memory.statement}" for memory in memories
    )


class WindowAssembler:
    """按群独立缓冲消息，满足三条关窗规则之一就成窗。

    规则：条数到 ``window_message_limit``；时间跨度到 ``window_minutes`` 且条数不少于
    ``window_min_messages``；或最老消息年龄到 ``window_max_minutes``（兜底）。

    构造时注入仓储、时钟与设置。时钟只供 :meth:`flush_expired`（群里没人说话时收尾）
    使用；:meth:`add` 的成窗时间判定完全基于消息自身的 ``sent_at``，与注入时钟无关。

        assembler = WindowAssembler(repository=repo, clock=clock, settings=settings)
        for message in stream:
            for window in await assembler.add(message):
                ...
        for window in await assembler.flush_expired():   # 定时任务调用
            ...

    缓冲**只在内存**里：进程重启时未满的窗口会丢，由里程碑 2 的掉线回补兜底。
    """

    def __init__(
        self,
        *,
        repository: Repository,
        clock: Callable[[], datetime],
        settings: Settings,
        previous_tail_size: int = DEFAULT_PREVIOUS_TAIL_SIZE,
        remembered_limit: int = DEFAULT_REMEMBERED_LIMIT,
        prompt_version: str | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._message_limit = settings.window_message_limit
        self._min_messages = settings.window_min_messages
        self._span = timedelta(minutes=settings.window_minutes)
        self._max_age = timedelta(minutes=settings.window_max_minutes)
        self._previous_tail_size = previous_tail_size
        self._remembered_limit = remembered_limit
        self._prompt_version = prompt_version or current_prompt_version()
        #: 每群一个未满窗口的缓冲；取出后清空。
        self._buffers: dict[int, list[WindowMessage]] = {}
        #: 每群上一个已出窗口的尾部，作为下一个窗口的上下文。
        self._tails: dict[int, tuple[SequencedMessage, ...]] = {}

    def buffered_count(self, group_id: int) -> int:
        """某群当前缓冲里还没成窗的消息条数。"""
        return len(self._buffers.get(group_id, ()))

    def pending_groups(self) -> list[int]:
        """当前有未满窗口的群号，升序。"""
        return sorted(group for group, buffer in self._buffers.items() if buffer)

    async def add(self, message: WindowMessage | Message) -> list[AssembledWindow]:
        """收一条消息，返回因此关闭的窗口（通常 0 或 1 个）。

        可直接传仓储的 ``Message``（内部会适配），也可传 :class:`WindowMessage`。

        收这条之前先判**已有缓冲**该不该关，"当下"取**这条新消息的** ``sent_at``：条数到
        ``window_message_limit``、跨度到 ``window_minutes`` 且条数不少于
        ``window_min_messages``、或最老消息年龄到 ``window_max_minutes``——任一成立就先把
        缓冲关掉再收这条（**新消息属于下一个窗口**）。判定只看消息自身的 ``sent_at``，
        不看注入时钟，所以回放与掉线回补灌历史消息时也能正确切窗。
        """
        normalized = _as_window_message(message)
        group_id = normalized.group_id
        buffer = self._buffers.setdefault(group_id, [])
        windows: list[AssembledWindow] = []

        if buffer and self._should_close(buffer, normalized.sent_at):
            windows.append(await self._close(group_id, buffer))
            buffer = self._buffers[group_id] = []

        buffer.append(normalized)
        if len(buffer) >= self._message_limit:
            windows.append(await self._close(group_id, buffer))
            self._buffers[group_id] = []

        return windows

    def _should_close(self, buffer: Sequence[WindowMessage], now: datetime) -> bool:
        """缓冲在 ``now`` 这一刻是否该关窗（三条规则任一成立）。

        ``now`` 在 :meth:`add` 里是本条新消息的 ``sent_at``，在 :meth:`flush_expired` 里是
        注入时钟给的当下。三条规则：条数到 ``window_message_limit``；跨度到
        ``window_minutes`` 且条数不少于 ``window_min_messages``；最老消息年龄到
        ``window_max_minutes``。
        """
        age = now - buffer[0].sent_at
        return (
            len(buffer) >= self._message_limit
            or (age >= self._span and len(buffer) >= self._min_messages)
            or age >= self._max_age
        )

    async def flush_group(self, group_id: int) -> AssembledWindow | None:
        """不管满没满，立刻把某群的缓冲关成一个窗口；没有缓冲则返回 ``None``。"""
        buffer = self._buffers.get(group_id)
        if not buffer:
            return None
        self._buffers[group_id] = []
        return await self._close(group_id, buffer)

    async def flush_expired(self) -> list[AssembledWindow]:
        """把该关的群窗口全部关掉。

        给后台定时任务用：群里没人说话时，也要有东西去触发成窗。"当下"取注入时钟——
        条数到上限、跨度到 ``window_minutes`` 且条数不少于 ``window_min_messages``、或最老
        消息年龄到 ``window_max_minutes``（兜底）任一成立就关。因此只有一两条消息的安静
        窗口最多滞留 ``window_max_minutes`` 就会被收掉，不会被无限期留在内存里。
        """
        now = self._clock()
        windows: list[AssembledWindow] = []
        for group_id in sorted(self._buffers):
            buffer = self._buffers[group_id]
            if buffer and self._should_close(buffer, now):
                self._buffers[group_id] = []
                windows.append(await self._close(group_id, buffer))
        return windows

    async def _close(self, group_id: int, buffer: Sequence[WindowMessage]) -> AssembledWindow:
        """关一个窗口：取上一次尾部与已记摘要，登记本次尾部。"""
        previous_tail = self._tails.get(group_id, ())
        remembered = await self._repository.recent_memories(
            group_id=group_id, limit=self._remembered_limit
        )
        window = assemble_window(
            group_id=group_id,
            messages=list(buffer),
            previous_tail=previous_tail,
            remembered=remembered,
            prompt_version=self._prompt_version,
        )
        if self._previous_tail_size > 0:
            self._tails[group_id] = window.messages[-self._previous_tail_size :]
        else:
            self._tails[group_id] = ()
        return window


__all__ = [
    "DEFAULT_PREVIOUS_TAIL_SIZE",
    "DEFAULT_REMEMBERED_LIMIT",
    "PREVIOUS_TAIL_HEADER",
    "AssembledWindow",
    "SequencedMessage",
    "UnknownSequenceError",
    "WindowAssembler",
    "WindowMessage",
    "assemble_window",
    "number_messages",
    "sender_name",
    "summarize_memories",
]
