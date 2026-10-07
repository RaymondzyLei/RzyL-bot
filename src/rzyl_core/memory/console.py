"""命令的实现：拿仓储与向量服务，把一条已解析的命令变成一段私聊文本。

与 :mod:`rzyl_core.memory.commands` 的分工是**数据 / 措辞**：那边只做解析与渲染（纯函数），
这边负责查库、写库、以及在查不到时怎么回话。因此这里的每个分支都能用真库（临时 SQLite）
加假模型测出来，不需要 NoneBot，也不需要 Runtime。

两个刻意的选择：

- **「今天」会先强制关窗**（``flush_pending``）。窗口最长可以攒到 ``window_max_minutes``，
  不关窗的话「今天」会漏掉最近一两个小时的内容——推送同理（见 issue #1 的推送约束）。
  关了几个窗口会写在回复里，不白花这笔模型钱还让人不知情。
- **不静默降级**：语义检索不可用、编号不存在、清空范围读不懂，一律在回复里说清楚。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from rzyl_core.db import (
    FeedbackKind,
    Memory,
    MemoryStatus,
    Repository,
)
from rzyl_core.llm import EmbeddingModel
from rzyl_core.memory.commands import (
    CommandKind,
    CommandResult,
    MemoryCommand,
    PurgeScope,
    format_help,
    format_search,
    format_sources,
)
from rzyl_core.memory.report import render_listing
from rzyl_core.memory.retrieval import DEFAULT_SEARCH_LIMIT, hybrid_search
from rzyl_core.pipeline.collector import merge_allowed_groups
from rzyl_core.settings import Settings
from rzyl_core.timeutil import local_day_bounds, resolve_timezone

logger = logging.getLogger(__name__)

#: 「今天 / 某个群」一次最多取多少条。这个量级远超实际（两天才 10 条），取一个宽松的
#: 上限只是为了防止某天刷屏把整库拉进内存；真到上限会在回复里说明。
LISTING_FETCH_LIMIT = 500

#: 「今天 / 某个群」最多列多少条，其余只报数。
LISTING_MAX_ENTRIES = 50

#: 人工列举时一并取回的状态：``active`` 与疑似重复都要能看见（故事 30「所有条目」），
#: 被推翻的 ``expired`` 不再列（它默认退出一切出口）。
LISTING_STATUSES: tuple[MemoryStatus, ...] = (
    MemoryStatus.ACTIVE,
    MemoryStatus.SUSPECT_DUPLICATE,
)


class MemoryConsole:
    """命令执行器。构造时注入仓储、向量实现与设置，外加一个「强制关窗」回调。"""

    def __init__(
        self,
        *,
        repository: Repository,
        embedding: EmbeddingModel,
        settings: Settings,
        clock: Callable[[], datetime],
        flush_pending: Callable[[], Awaitable[int]] | None = None,
    ) -> None:
        self._repository = repository
        self._embedding = embedding
        self._settings = settings
        self._clock = clock
        self._flush_pending = flush_pending
        self._timezone = resolve_timezone(settings.timezone)

    async def execute(self, command: MemoryCommand) -> CommandResult:
        """执行一条命令，返回要发回私聊的文本。任何异常都不外抛到插件层。"""
        if command.kind is CommandKind.HELP:
            return CommandResult(text=await self._help())
        if command.kind is CommandKind.INVALID:
            reason = command.error or "读不懂这条命令"
            return CommandResult(text=f"{reason}\n\n{await self._help()}", ok=False)
        if command.kind is CommandKind.SEARCH:
            return await self._search(command)
        if command.kind is CommandKind.TODAY:
            return await self._today()
        if command.kind is CommandKind.GROUP:
            return await self._group(command)
        if command.kind is CommandKind.SOURCES:
            return await self._sources(command)
        if command.kind is CommandKind.FALSE_POSITIVE:
            return await self._mark_false_positive(command)
        if command.kind is CommandKind.RESTORE:
            return await self._restore(command)
        if command.kind is CommandKind.MISSING:
            return await self._record_missing(command)
        if command.kind in (CommandKind.ENABLE, CommandKind.DISABLE):
            return await self._toggle(command)
        if command.kind is CommandKind.PURGE:
            return await self._purge(command)
        return CommandResult(text=f"未实现的命令：{command.kind}", ok=False)

    # —— 各条命令 ——

    async def _help(self) -> str:
        switches = await self._repository.group_settings()
        allowed = await self._allowed_groups()
        return format_help(
            configured=self._settings.group_whitelist,
            allowed=allowed,
            switches=switches,
        )

    async def _allowed_groups(self) -> frozenset[int]:
        """当前采集中的群：``(配置白名单 − 显式暂停) ∪ 显式开启``。

        与 :meth:`Runtime.allowed_groups` 同一条规则；命令侧要用它来回显状态与判断
        「这个群本来就在白名单里」这类提示。规则实现在 :mod:`rzyl_core.pipeline.collector`，
        两边都调它，不各写一遍。
        """
        return merge_allowed_groups(
            self._settings.group_whitelist,
            await self._repository.enabled_groups(),
            await self._repository.disabled_groups(),
        )

    async def _search(self, command: MemoryCommand) -> CommandResult:
        outcome = await hybrid_search(
            repository=self._repository,
            embedding=self._embedding,
            query=command.query,
            min_similarity=self._settings.retrieval_min_similarity,
            limit=DEFAULT_SEARCH_LIMIT,
        )
        return CommandResult(text=format_search(outcome, query=command.query, tz=self._timezone))

    async def _today(self) -> CommandResult:
        flushed = await self._flush()
        now = self._clock()
        since, until = local_day_bounds(now, self._timezone)
        rows = await self._repository.list_memories_in_range(
            since=since,
            until=until,
            statuses=LISTING_STATUSES,
            limit=LISTING_FETCH_LIMIT,
        )
        day = now.astimezone(self._timezone).strftime("%m-%d")
        footer = f"（查询前补关了 {flushed} 个窗口）" if flushed else None
        text = render_listing(
            rows,
            header=f"今天（{day}）新增的条目",
            tz=self._timezone,
            max_entries=LISTING_MAX_ENTRIES,
            empty_text="今天还没有条目。",
            footer=footer,
        )
        if len(rows) >= LISTING_FETCH_LIMIT:
            text += f"\n\n（已达单次读取上限 {LISTING_FETCH_LIMIT} 条，可能还有更早的没取到）"
        return CommandResult(text=text)

    async def _group(self, command: MemoryCommand) -> CommandResult:
        assert command.group_id is not None  # 解析层保证；assert 只为让类型收窄
        rows = await self._repository.list_memories_in_range(
            group_id=command.group_id,
            statuses=LISTING_STATUSES,
            limit=LISTING_FETCH_LIMIT,
        )
        text = render_listing(
            rows,
            header=f"群 {command.group_id} 的条目",
            tz=self._timezone,
            max_entries=LISTING_MAX_ENTRIES,
            empty_text="这个群还没有条目（或不在采集范围内）。",
            footer=await self._group_state_note(command.group_id),
        )
        if len(rows) >= LISTING_FETCH_LIMIT:
            text += f"\n\n（已达单次读取上限 {LISTING_FETCH_LIMIT} 条，可能还有更早的没取到）"
        return CommandResult(text=text)

    async def _group_state_note(self, group_id: int) -> str:
        """一句现况说明：这个群现在收不收、为什么。"""
        if group_id in await self._allowed_groups():
            return f"（群 {group_id} 当前采集中的）"
        if await self._repository.is_group_enabled(group_id):
            return f"（群 {group_id} 已开启采集）"
        switches = await self._repository.group_settings()
        if group_id in switches:
            return f"（群 {group_id} 已被暂停，不在采集范围；用 记忆 开启 {group_id} 恢复）"
        return f"（群 {group_id} 不在配置白名单里，也不在运行时开关里，当前不采集）"

    async def _sources(self, command: MemoryCommand) -> CommandResult:
        assert command.memory_id is not None
        memory = await self._repository.get_memory(command.memory_id)
        if memory is None:
            return CommandResult(
                text=f"没有编号 #{command.memory_id} 这条记忆（可能已被清空）。", ok=False
            )
        window = (
            await self._repository.get_window(int(memory.window_id))
            if memory.window_id is not None
            else None
        )
        messages = (
            await self._repository.list_messages_for_window(int(memory.window_id))
            if memory.window_id is not None
            else []
        )
        return CommandResult(
            text=format_sources(
                memory=memory, window=window, messages=messages, tz=self._timezone
            )
        )

    async def _mark_false_positive(self, command: MemoryCommand) -> CommandResult:
        assert command.memory_id is not None
        memory = await self._repository.get_memory(command.memory_id)
        if memory is None:
            return CommandResult(
                text=f"没有编号 #{command.memory_id} 这条记忆，标不了误报。", ok=False
            )
        await self._repository.add_feedback(
            kind=FeedbackKind.FALSE_POSITIVE,
            memory_id=int(memory.id),
            group_id=int(memory.group_id),
            note=self._feedback_note(command.note, memory),
        )
        # 标错的同时把它移出推送与检索：否则「标了误报但天天还在日报里」等于没用。
        # 复用 expired 这一个非活跃状态（数据模型不改），权威记录是上面那条 feedback
        # 样本；「恢复」会把状态改回 active。
        await self._repository.set_memory_status(int(memory.id), MemoryStatus.EXPIRED)
        note_line = f"\n备注：{command.note}" if command.note else ""
        return CommandResult(
            text=(
                f"已把 [#{memory.id}] 标为误报，并移出推送与检索。\n"
                f"陈述：{memory.statement}{note_line}\n"
                f"（原陈述已快照进错例集，这条记忆以后被清空也不影响样本；"
                f"用 记忆 恢复 {memory.id} 可撤回）"
            )
        )

    def _feedback_note(self, note: str | None, memory: Memory) -> str:
        """纠错样本的备注：把原陈述**快照**进去。

        理由与 ``person_refs`` 存快照一样：样本要在记忆被删掉之后仍然可用（故事 39 要拿
        它当提示词回归集），所以它不能只靠 ``memory_id`` 这个会断的引用。
        """
        snapshot = f"原陈述：{memory.statement}"
        return f"{note.strip()}\n{snapshot}" if note and note.strip() else snapshot

    async def _restore(self, command: MemoryCommand) -> CommandResult:
        assert command.memory_id is not None
        memory = await self._repository.get_memory(command.memory_id)
        if memory is None:
            return CommandResult(text=f"没有编号 #{command.memory_id} 这条记忆。", ok=False)
        if memory.status is MemoryStatus.ACTIVE:
            return CommandResult(text=f"[#{memory.id}] 本来就是活跃状态，没改。")
        previous = memory.status.value
        await self._repository.set_memory_status(int(memory.id), MemoryStatus.ACTIVE)
        had_link = await self._repository.clear_superseded_by(int(memory.id))
        extra = "（同时解开了指向它的「被推翻」引用）" if had_link else ""
        return CommandResult(
            text=f"已把 [#{memory.id}] 从 {previous} 恢复为 active{extra}，重新进入推送与检索。"
        )

    async def _record_missing(self, command: MemoryCommand) -> CommandResult:
        await self._repository.add_feedback(
            kind=FeedbackKind.FALSE_NEGATIVE,
            group_id=command.group_id,
            note=command.note.strip() if command.note else None,
        )
        where = f"（群 {command.group_id}）" if command.group_id else ""
        hint = (
            ""
            if command.note
            else "\n（建议下次带上备注：注明是哪条消息或哪个时间点，当回归样本时才知道要断言什么）"
        )
        return CommandResult(text=f"已记下一条漏报{where}，进错例集。{hint}")

    async def _toggle(self, command: MemoryCommand) -> CommandResult:
        assert command.group_id is not None
        enable = command.kind is CommandKind.ENABLE
        await self._repository.set_group_enabled(command.group_id, enable)
        configured = command.group_id in set(self._settings.group_whitelist)
        if enable:
            extra = "（它本来就在配置白名单里）" if configured else "（不在配置白名单里，开关存在库里，重启仍有效）"
            return CommandResult(text=f"已开启群 {command.group_id} 的采集{extra}。")
        extra = "（它写在配置白名单里，这一行开关会压过白名单）" if configured else ""
        return CommandResult(text=f"已暂停群 {command.group_id} 的采集{extra}。")

    async def _purge(self, command: MemoryCommand) -> CommandResult:
        scope = command.purge_scope
        if scope is PurgeScope.MEMORY:
            assert command.memory_id is not None
            memory = await self._repository.get_memory(command.memory_id)
            counts = await self._repository.purge(memory_ids=[command.memory_id])
            if counts.memories == 0:
                return CommandResult(
                    text=f"没有编号 #{command.memory_id} 这条记忆，什么都没删。", ok=False
                )
            statement = memory.statement if memory is not None else "（已取不到内容）"
            return CommandResult(
                text=(
                    f"已删掉 [#{command.memory_id}]：{statement}\n"
                    f"（只删这一条记忆；原文与窗口按保留期滚动，未受影响）"
                )
            )
        if scope is PurgeScope.GROUP:
            assert command.group_id is not None
            counts = await self._repository.purge(group_id=command.group_id)
            # 顺手暂停：否则白名单里的话下一分钟又开始收，等于没退出。
            previously_enabled = await self._repository.is_group_enabled(command.group_id)
            await self._repository.set_group_enabled(command.group_id, False)
            paused = "" if previously_enabled else "（原本就不在采集范围）"
            return CommandResult(
                text=(
                    f"已清空群 {command.group_id}：记忆 {counts.memories} 条、"
                    f"原文 {counts.messages} 条、窗口 {counts.windows} 个。\n"
                    f"该群已暂停采集{paused}——用 记忆 开启 {command.group_id} 可以重新开始。"
                    f"{self._unlink_note(counts.feedback_unlinked)}"
                )
            )
        counts = await self._repository.purge(everything=True)
        paused = await self._pause_known_groups()
        return CommandResult(
            text=(
                f"已清空全部：记忆 {counts.memories} 条、原文 {counts.messages} 条、"
                f"窗口 {counts.windows} 个。\n"
                f"已暂停 {paused} 个群的采集（含配置白名单里的），否则下一分钟就会重新开始收。"
                f"{self._unlink_note(counts.feedback_unlinked)}\n"
                f"用 记忆 开启 <群号> 可以重新开始。"
            )
        )

    def _unlink_note(self, unlinked: int) -> str:
        if not unlinked:
            return ""
        return f"\n（另解开了 {unlinked} 条纠错样本对已删条目的引用；样本本身保留）"

    async def _pause_known_groups(self) -> int:
        """把所有「当前会被采集」的群写成显式暂停，返回写了几行。

        清空全部之后如果不暂停，配置白名单里的群会立刻重新开始采集，「退出」就成了假动作。
        """
        groups = await self._allowed_groups()
        for group_id in sorted(groups):
            await self._repository.set_group_enabled(group_id, False)
        return len(groups)

    async def _flush(self) -> int:
        """强制关窗（查询前用），没有装配回调时返回 0。"""
        if self._flush_pending is None:
            return 0
        return await self._flush_pending()


__all__ = [
    "LISTING_FETCH_LIMIT",
    "LISTING_MAX_ENTRIES",
    "LISTING_STATUSES",
    "MemoryConsole",
]
