"""Runtime：把各部件装配成**唯一对外入口**，并持有它的生命周期。

issue #1 定的装配原则是「core 只通过一个 Runtime 对外暴露」——插件层与离线回放脚本
都只跟 Runtime 打交道，聊天模型、向量实现、时钟、数据库连接串在**构造时注入**，
所以测试与回放可以换成假实现而完全不改代码：

    runtime = Runtime(
        chat_model=..., embedding_model=..., settings=settings,
        clock=..., database_url="sqlite+aiosqlite:///data/rzyl.db",
    )
    await runtime.start()
    await runtime.ingest(group_id=..., user_id=..., text=..., sent_at=...)
    report = await runtime.replay(source=SampleHistorySource("history.json"), group_id=...)
    await runtime.stop()

内部后台任务的现状（工单 #5 与里程碑 2 明确要写明，避免"以为实现了"）：

- **保留期清理**（``cleanup_retention``）：**已实现**。按 ``settings.retention_days``
  删掉 ``sent_at`` 早于截止时刻的原文；只删原文，记忆条目永久。
- **向量补算**（``backfill_embeddings``）：**已实现**。取向量为空的活条目，批量算向量
  并写回；向量服务不可用（返回 ``None`` 或抛异常）时本轮什么都不做，条目照常保留。
  另提供 ``reembed=True``：连已有向量的条目也一起按当前模型重算，供换向量模型后把旧
  向量全部刷新（后台补算循环只用默认的 ``reembed=False``，不会自动重算全部向量）。
  启动与补算入口还会各跑一次「向量模型账本一致性」检查（同维度换模型也能查出，
  见 :meth:`Runtime.check_embedding_model_ledger`），同一进程只提醒一次。
- **窗口超时刷新**（``flush_expired``）：**已实现**。群里没人说话时，靠后台循环周期把
  未满的缓冲按时成窗并入库；间隔取 ``settings.window_flush_seconds``。
- **死信重试**（``retry_dead_windows``）：**已实现**。后台循环周期找出状态为 ``dead``
  的窗口重跑；护栏与可观察性见 :meth:`Runtime.retry_dead_windows` 的文档字符串。
  间隔取 ``settings.dead_letter_retry_seconds``。
- **启动对账**（``reconcile_uncovered_messages``）：**已实现**。``start()`` 打开
  ``settings.reconcile_on_startup``（默认开）时，用一个一次性后台任务把「没有被任何窗口
  时间区间覆盖」的消息重新送进窗口管道——补上「重启丢掉内存缓冲、而回补锚点又不会拉
  已存消息」这个洞。任务不阻塞启动，登记进 ``self._tasks`` 供 ``stop()`` 取消。
- **每日推送**（``push_daily_report_if_due``）：**已实现**。``start()`` 起一个循环，按
  ``settings.push_hour/push_minute``（``settings.timezone`` 计）到点触发：**先强制关窗**、
  再按类别渲染当天条目，交给已登记的投递器（``memory.push``）。core 不认识 QQ，所以发送
  这件事由插件登记进来；没登记就只记一条告警，绝不假装发出去了。

这五个循环共用 :meth:`Runtime._run_periodically` 一个骨架（间隔从设置或模块常量取），
所以「单轮失败只记日志、不悄悄停掉循环」这条约定只写一遍。

里程碑 3 还给了 Runtime 两个消费记忆的出口，供插件与脚本用：``execute_command``（一条
命令进、一段私聊文本出）与 ``build_daily_report``（关窗 + 渲染，不负责发送）。两者都只是
把 :mod:`rzyl_core.memory` 里的实现接上来，Runtime 依旧是**唯一对外入口**。

回放有两个模式，差别只在**写与不写**：

- 正常模式：消息与窗口结果都写进**配置的库**，从仓储能读回条目；
- ``dry_run``：消息写进一个临时库（窗口的编号映射需要真实的 ``message.id``），每个窗口
  只调一次模型、打印完整提示词与原始输出，**配置的库一条不写**，临时库用完即删。

回放不需要为时钟做特殊处理：窗口的成窗时间判定已经改为基于**消息自身**的 ``sent_at``
（见 ``pipeline/window.py``），所以实时链路、离线回放与里程碑 2 的掉线回补三条路径
共用同一个注入时钟即可，历史消息不会被当下时钟「瞬间顶成窗」。
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from rzyl_core.db import Message, Repository, Window, WindowStatus
from rzyl_core.llm import ChatModel, EmbeddingModel, embed_batch
from rzyl_core.memory.commands import CommandResult, MemoryCommand
from rzyl_core.memory.console import MemoryConsole
from rzyl_core.memory.push import get_report_sender, next_push_at, schedule_push_at
from rzyl_core.memory.report import DailyReport, render_daily_report
from rzyl_core.pipeline.collector import collect, merge_allowed_groups
from rzyl_core.pipeline.extract import ExtractionOutcome, ExtractionPipeline, PreviewOutcome
from rzyl_core.pipeline.history import HistorySource, HistoryMessage, message_dedupe_hash
from rzyl_core.pipeline.window import (
    DEFAULT_PREVIOUS_TAIL_SIZE,
    DEFAULT_REMEMBERED_LIMIT,
    AssembledWindow,
    WindowAssembler,
    WindowMessage,
    assemble_window,
)
from rzyl_core.settings import Settings
from rzyl_core.timeutil import local_day_bounds, resolve_timezone, utcnow

logger = logging.getLogger(__name__)

#: 保留期清理循环的间隔（秒）——暂时是模块常量，接实时链路时再挪进设置。
RETENTION_SWEEP_SECONDS = 3600

#: 向量补算循环的间隔（秒）。
EMBEDDING_SWEEP_SECONDS = 300

#: 每日推送循环醒来检查「到点没有」的间隔（秒）。
#:
#: 推送本身是每分钟才需要判一次的事，但用 30 秒是刻意的余量：真实推送要跑
#: ``flush_pending``（会调模型，可能几十秒），醒来晚一点就会把「每天 22:00」拖成 22:05。
PUSH_TICK_SECONDS = 30

#: 一次日报最多从库里取多少条。取一个宽松的上限只为防某天刷屏把整库拉进内存。
DAILY_REPORT_FETCH_LIMIT = 500

#: 日报投递失败后多久重试一次（秒）。
PUSH_RETRY_SECONDS = 600

#: 一份日报最多尝试投递几次；到顶就放弃到明天。最常见的失败原因是这一刻 QQ 没连上。
PUSH_MAX_ATTEMPTS = 3

#: 启动时的宽限：若启动时刻落在今天推送时刻之后的这段时间内，立刻补推一次。
#:
#: 推送循环每 30 秒才醒一次，所以「22:00:15 启动」会让严格的下一次变成明天——当天整份
#: 不推且毫无提示。宁可此时多推一份（重启前刚推过会重复），也不静默丢掉一天的日报。
PUSH_STARTUP_GRACE_SECONDS = 900

#: 一次向量补算最多处理多少条，避免一次拉太多进内存。
EMBEDDING_BACKFILL_BATCH = 100

#: 死信重试循环一轮最多处理多少个死信窗口，避免一次拉太多进内存。
DEAD_LETTER_RETRY_BATCH = 20

#: 向量调用写进 ``llm_call.purpose`` 的用途标签。
EMBED_PURPOSE = "embed"

#: 回放主循环返回的每窗口结果类型（预览或提取结果）。
T = TypeVar("T")


def _embedding_input_tokens(model: EmbeddingModel) -> int:
    """取向量实现最近一次上报的输入 token；没有可用用量信息时记 0，不编造。"""
    tokens = getattr(model, "last_input_tokens", None)
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        return tokens
    return 0


async def _preview_window(
    pipeline: ExtractionPipeline, window: AssembledWindow
) -> PreviewOutcome:
    """回放 dry-run 策略：只渲染 + 调一次模型，不写任何配置库。"""
    return await pipeline.preview_window(window)


async def _process_window(
    pipeline: ExtractionPipeline, window: AssembledWindow
) -> ExtractionOutcome:
    """回放正常策略：走完整的校验、去重与入库。"""
    return await pipeline.process_window(window)


async def _close_if_possible(resource: object) -> None:
    """对象提供 ``aclose`` 就关掉；没有就什么都不做（假实现都不带 aclose）。

    兼容同步与异步的 ``aclose``：调完若返回 awaitable 才 await。
    """
    aclose = getattr(resource, "aclose", None)
    if not callable(aclose):
        return
    result = aclose()
    if inspect.isawaitable(result):
        await result


@dataclass(frozen=True, slots=True)
class IngestResult:
    """一次 ``ingest`` 的结果：落库的那条消息、因此关闭的窗口、以及该群还在缓冲的条数。"""

    message: Message
    outcomes: tuple[ExtractionOutcome, ...]
    buffered: int


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """一次 ``replay`` 的结果。

    正常模式填 ``outcomes``（含窗口状态与新增条目编号），``dry_run`` 填 ``previews``
    （含完整提示词与模型原始输出）；两者不会同时非空。
    """

    group_id: int
    dry_run: bool
    message_count: int
    window_count: int
    outcomes: tuple[ExtractionOutcome, ...] = ()
    previews: tuple[PreviewOutcome, ...] = ()

    @property
    def memory_ids(self) -> tuple[int, ...]:
        """正常模式下本批新增的条目编号；dry-run 恒为空。"""
        return tuple(
            memory_id for outcome in self.outcomes for memory_id in outcome.memory_ids
        )


@dataclass(frozen=True, slots=True)
class GroupReconcile:
    """一个群的启动对账结果。"""

    group_id: int
    message_count: int
    window_count: int
    memory_ids: tuple[int, ...]
    capped: bool


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """一次 :meth:`Runtime.reconcile_uncovered_messages` 的结果：按群明细加汇总视图。"""

    groups: tuple[GroupReconcile, ...]

    @property
    def message_count(self) -> int:
        """本批重新送进管道的消息总条数。"""
        return sum(group.message_count for group in self.groups)

    @property
    def window_count(self) -> int:
        """本批新成的窗口总数。"""
        return sum(group.window_count for group in self.groups)

    @property
    def memory_ids(self) -> tuple[int, ...]:
        """本批新增的记忆条目编号。"""
        return tuple(
            memory_id for group in self.groups for memory_id in group.memory_ids
        )

    @property
    def capped_groups(self) -> tuple[int, ...]:
        """触到 ``reconcile_max_messages`` 上限的群号（还有未覆盖消息留着）。"""
        return tuple(group.group_id for group in self.groups if group.capped)


class Runtime:
    """装配好的一整套服务：仓储、窗口组装、提取管道，外加生命周期与后台任务。"""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
        database_url: str | None = None,
        provider: str | None = None,
        previous_tail_size: int = DEFAULT_PREVIOUS_TAIL_SIZE,
        remembered_limit: int = DEFAULT_REMEMBERED_LIMIT,
        prompt_version: str | None = None,
    ) -> None:
        self._chat = chat_model
        self._embedding = embedding_model
        self._settings = settings
        self._clock: Callable[[], datetime] = clock or utcnow
        self._database_url = database_url or settings.database_url
        self._provider = provider
        self._previous_tail_size = previous_tail_size
        self._remembered_limit = remembered_limit
        self._prompt_version = prompt_version
        self._repository: Repository | None = None
        self._assembler: WindowAssembler | None = None
        self._pipeline: ExtractionPipeline | None = None
        self._console: MemoryConsole | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False
        #: 下一次该推送的 UTC 时刻；``start()`` 后第一次醒来时才排定。
        self._next_push_at: datetime | None = None
        #: 本次日报已经投递失败了几次，用于给重试封顶。
        self._push_attempts = 0
        self._timezone = resolve_timezone(settings.timezone)
        #: 「向量模型账本一致性」检查同一进程内只做一次，免得后台循环每轮刷屏。
        self._embedding_ledger_checked = False
        self._embedding_ledger_model: str | None = None

    # —— 只读视图 ——

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def started(self) -> bool:
        return self._started

    @property
    def repository(self) -> Repository:
        """配置库的仓储；未 ``start`` 时访问是编程错误。"""
        return self._require_repository()

    @property
    def background_tasks(self) -> tuple[asyncio.Task[None], ...]:
        """当前在跑的后台任务（未启动或已停止时为空元组）。"""
        return tuple(self._tasks)

    # —— 生命周期 ——

    async def start(self, *, run_background_tasks: bool = True) -> None:
        """建库、装配窗口组装器与提取管道；可选拉起后台任务循环。

        ``run_background_tasks=False`` 供测试与脚本使用：它们要手动调
        ``cleanup_retention`` / ``backfill_embeddings``，不希望有循环常驻。
        """
        if self._started:
            return
        self._repository = await Repository.create(self._database_url, clock=self._clock)
        self._assembler = self._build_assembler(self._repository, self._clock)
        self._pipeline = self._build_pipeline(self._repository, self._clock)
        self._console = self._build_console(self._repository)
        self._started = True
        # 启动时核对一次向量模型账本：换过模型却没重算向量的库要尽早出声，而不是等到
        # 去重结果开始离谱。同一进程只提醒一次，后台循环不会重复刷屏。
        await self.check_embedding_model_ledger()
        if run_background_tasks:
            self._tasks = [
                asyncio.create_task(
                    self._run_periodically(
                        RETENTION_SWEEP_SECONDS, self.cleanup_retention, "保留期清理"
                    ),
                    name="rzyl-retention-sweep",
                ),
                asyncio.create_task(
                    self._run_periodically(
                        EMBEDDING_SWEEP_SECONDS, self.backfill_embeddings, "向量补算"
                    ),
                    name="rzyl-embedding-backfill",
                ),
                asyncio.create_task(
                    self._run_periodically(
                        self._settings.window_flush_seconds,
                        self._window_flush_tick,
                        "窗口超时刷新",
                    ),
                    name="rzyl-window-flush",
                ),
                asyncio.create_task(
                    self._run_periodically(
                        self._settings.dead_letter_retry_seconds,
                        self.retry_dead_windows,
                        "死信重试",
                    ),
                    name="rzyl-dead-letter-retry",
                ),
                asyncio.create_task(
                    self._run_periodically(
                        PUSH_TICK_SECONDS, self.push_daily_report_if_due, "每日推送"
                    ),
                    name="rzyl-daily-push",
                ),
            ]
            if self._settings.reconcile_on_startup:
                # 一次性任务：启动时不阻塞（对账可能调几十次模型），跑完自己结束；
                # 登记进 _tasks 只为 stop() 能取消它，不留下悬挂任务。
                self._tasks.append(
                    asyncio.create_task(
                        self._startup_reconcile(), name="rzyl-startup-reconcile"
                    )
                )

    async def stop(self) -> None:
        """取消后台任务、关仓储；对注入的模型，若它提供 ``aclose`` 也一并关闭。"""
        tasks = self._tasks
        self._tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._repository is not None:
            await self._repository.close()
        self._repository = None
        self._assembler = None
        self._pipeline = None
        self._console = None
        self._next_push_at = None
        self._push_attempts = 0
        self._started = False
        for model in (self._chat, self._embedding):
            await _close_if_possible(model)

    # —— 实时链路入口（里程碑 2 的插件会用它；回放也复用同一套落库逻辑）——

    async def ingest(
        self,
        *,
        group_id: int,
        user_id: int,
        text: str,
        sent_at: datetime,
        nickname: str | None = None,
        card: str | None = None,
        segments_json: str = "[]",
        platform_message_id: int | None = None,
    ) -> IngestResult:
        """收一条群消息：**先落库**、再进窗口，成窗的窗口立刻走提取入库。

        顺序是有讲究的：``AssembledWindow.sequence_map`` 的值是 ``message`` 表的自增
        编号，所以必须先 ``add_message``，窗口里的序号才能换回真实编号。
        """
        repository = self._require_repository()
        assembler = self._require_assembler()
        pipeline = self._require_pipeline()
        message = await repository.add_message(
            group_id=group_id,
            user_id=user_id,
            text=text,
            sent_at=sent_at,
            nickname=nickname,
            card=card,
            segments_json=segments_json,
            platform_message_id=platform_message_id,
            dedupe_hash=message_dedupe_hash(
                group_id=group_id, user_id=user_id, sent_at=sent_at, text=text
            ),
        )
        outcomes = [await pipeline.process_window(window) for window in await assembler.add(message)]
        return IngestResult(
            message=message,
            outcomes=tuple(outcomes),
            buffered=assembler.buffered_count(group_id),
        )

    async def allowed_groups(self) -> frozenset[int]:
        """当前允许采集的群号：``(配置白名单 − 显式暂停) ∪ 显式开启``。

        配置给初始值，里程碑 3 的 ``记忆 开启 / 暂停`` 写数据库里的每群开关。**数据库里
        有行的群以那一行为准**——暂停必须能压过配置白名单，否则命令是空操作（见
        :func:`~rzyl_core.pipeline.collector.merge_allowed_groups`）。
        """
        repository = self._require_repository()
        return merge_allowed_groups(
            self._settings.group_whitelist,
            await repository.enabled_groups(),
            await repository.disabled_groups(),
        )

    async def ingest_message(
        self,
        *,
        group_id: int,
        user_id: int,
        self_id: int | None,
        segments: Sequence[Any],
        sent_at: datetime,
        nickname: str | None = None,
        card: str | None = None,
        platform_message_id: int | None = None,
    ) -> IngestResult | None:
        """采集一条 OB11 群消息：判定 → 文本化 → 落库进窗；不该收时返回 ``None``。

        判定、文本化、入库全在 core（见 :mod:`rzyl_core.pipeline.collector`），插件只把
        事件的几样字段塞进来。``segments`` 是 OB11 消息段数组（``{"type": ..., "data": ...}``
        的列表），``self_id`` 是机器人自己的 QQ 号（用来跳过自己发的消息）。
        """
        rendered = collect(
            group_id=group_id,
            user_id=user_id,
            self_id=self_id,
            segments=segments,
            allowed_groups=await self.allowed_groups(),
        )
        if rendered is None:
            return None
        return await self.ingest(
            group_id=group_id,
            user_id=user_id,
            text=rendered.text,
            sent_at=sent_at,
            nickname=nickname,
            card=card,
            segments_json=rendered.segments_json,
            platform_message_id=platform_message_id,
        )

    async def flush_group(self, group_id: int) -> tuple[ExtractionOutcome, ...]:
        """立刻把某群未满的缓冲关成一个窗口并走提取；没有缓冲则返回空。"""
        assembler = self._require_assembler()
        pipeline = self._require_pipeline()
        window = await assembler.flush_group(group_id)
        if window is None:
            return ()
        return (await pipeline.process_window(window),)

    async def flush_expired(self) -> tuple[ExtractionOutcome, ...]:
        """把所有到时间上限的群窗口关掉并走提取。"""
        assembler = self._require_assembler()
        pipeline = self._require_pipeline()
        windows = await assembler.flush_expired()
        return tuple([await pipeline.process_window(window) for window in windows])

    async def flush_pending(self, group_id: int | None = None) -> int:
        """**不管满没满**，把缓冲里的窗口立刻关掉并走提取，返回关掉的窗口数。

        与 :meth:`flush_expired` 的区别就在「不管满没满」：到点才关是定时刷新的语义，
        而这里要的是「现在就要看到全部」——查当天条目与发日报前都要先跑一次，
        否则最近一两个小时的内容还躺在内存缓冲里，日报会漏（见 issue #1 的推送约束）。

        ``group_id`` 给定就只关那个群，缺省关所有有缓冲的群。代价要说清楚：每个被关的
        窗口都是一次真实的模型调用，所以调用方应当把关了几个窗口如实告诉用户。
        """
        assembler = self._require_assembler()
        pipeline = self._require_pipeline()
        groups = [group_id] if group_id is not None else assembler.pending_groups()
        closed = 0
        for target in groups:
            window = await assembler.flush_group(target)
            if window is None:
                continue
            outcome = await pipeline.process_window(window)
            closed += 1
            if outcome.status is WindowStatus.DEAD:
                logger.warning(
                    "强制关窗：群 %s 的窗口 %s 进了死信（%s），后台会重试",
                    target,
                    outcome.window_id,
                    outcome.error,
                )
        if closed:
            logger.info("强制关窗：关掉 %d 个窗口（群 %s）", closed, groups)
        return closed

    # —— 记忆出口（里程碑 3）——

    async def execute_command(self, command: MemoryCommand) -> CommandResult:
        """执行一条已解析的记忆命令，返回要发回私聊的文本。

        Runtime 只做转交：真正的实现在 :class:`~rzyl_core.memory.console.MemoryConsole`，
        它拿的是这个 Runtime 的仓储、向量实现与「强制关窗」回调。这样命令逻辑不必持有
        装配状态，也能脱离 Runtime 单测。
        """
        if self._console is None:
            raise RuntimeError("Runtime 尚未 start()，记忆命令不可用")
        return await self._console.execute(command)

    async def build_daily_report(self, now: datetime | None = None) -> DailyReport:
        """关掉全部未满窗口，再把「当天」的活跃条目渲染成一份日报（**不负责发送**）。

        「当天」按 ``settings.timezone`` 的自然日算，落在 ``memory.created_at`` 上。

        发送为什么不在这一层：core 不认识 QQ（见 ``memory.push``）。投递器由插件登记，
        :meth:`push_daily_report_if_due` 才负责到点触发并调用它。
        """
        repository = self._require_repository()
        moment = now or self._clock()
        flushed = await self.flush_pending()
        since, until = local_day_bounds(moment, self._timezone)
        memories = await repository.list_memories_in_range(
            since=since, until=until, limit=DAILY_REPORT_FETCH_LIMIT
        )
        report = render_daily_report(
            memories,
            day=moment.astimezone(self._timezone).date(),
            threshold=self._settings.confidence_threshold,
            max_entries=self._settings.push_max_entries,
            flushed_windows=flushed,
        )
        logger.info(
            "日报已生成：%s，列出 %d/%d 条（低置信度 %d 条未列），推送前补关 %d 个窗口",
            report.day.isoformat(),
            report.listed,
            report.total,
            report.low_confidence,
            flushed,
        )
        return report

    async def push_daily_report_if_due(self, now: datetime | None = None) -> DailyReport | None:
        """到点就生成并投递日报；没到点返回 ``None``。

        「到点」由 :func:`~rzyl_core.memory.push.next_push_at` 算，只看 ``settings.timezone``
        的墙上时间。第一次调用时排定下一次（见
        :func:`~rzyl_core.memory.push.schedule_push_at` 的启动宽限）。

        **失败一天都不丢**，这是本方法最要紧的性质，两处都按这个来设计：

        - 生成日报会真的调模型（先强制关窗），所以它也可能失败；失败时**不**把排班推到
          明天，而是在当天稍后重试（``PUSH_RETRY_SECONDS`` 一次，至多 ``PUSH_MAX_ATTEMPTS``
          次）。排班只在「这次不用再试了」——投递成功、没有投递器、或重试用尽——才推进。
        - 最常见的失败原因是这一刻 QQ 没连上（NapCat 掉线或正在重启），十分钟后可能就好了，
          所以重试落在**当天**而不是等到明天。

        三种结果都留痕：没登记投递器（告警，说清「生成了但没发出去」）、失败（异常日志 +
        重试或放弃，放弃时提示可以手工用 ``记忆 今天`` 看）、成功（INFO）。
        """
        if not self._settings.push_enabled:
            return None
        moment = now or self._clock()
        if self._next_push_at is None:
            self._next_push_at = schedule_push_at(
                moment,
                hour=self._settings.push_hour,
                minute=self._settings.push_minute,
                tz=self._timezone,
                startup_grace_seconds=PUSH_STARTUP_GRACE_SECONDS,
            )
            logger.info(
                "每日推送已排定：下一次 %s（%s）",
                self._next_push_at.astimezone(self._timezone).isoformat(),
                self._settings.timezone,
            )
        if moment < self._next_push_at:
            return None

        # 「下一次正常时刻」先算好但**先不落**：生成日报要关窗（真实调模型），它失败的话
        # 排到明天就等于把一整天丢了。所以排班只在「这次不用再试」时才推进到它。
        next_normal = next_push_at(
            moment,
            hour=self._settings.push_hour,
            minute=self._settings.push_minute,
            tz=self._timezone,
        )
        try:
            report = await self.build_daily_report(moment)
        except Exception:
            logger.exception("日报生成失败（关窗或渲染出错）")
            self._schedule_after_failure(moment, next_normal)
            return None

        sender = get_report_sender()
        if sender is None:
            logger.warning(
                "日报已生成但没登记投递器（插件没加载？），本次没有发出去：共 %d 条",
                report.total,
            )
            self._finish_attempt(next_normal)
            return report
        try:
            await sender(report)
        except Exception:
            logger.exception(
                "日报投递失败（共 %d 条，第 %d/%d 次）", report.total, self._push_attempts + 1,
                PUSH_MAX_ATTEMPTS,
            )
            self._schedule_after_failure(moment, next_normal)
            return report
        logger.info("日报已投递：%s，列出 %d 条", report.day.isoformat(), report.listed)
        self._finish_attempt(next_normal)
        return report

    def _schedule_after_failure(self, moment: datetime, next_normal: datetime) -> None:
        """一次失败之后：没到上限就在当天稍后重试，到顶就放弃到明天。

        计数在**放弃时清零**，否则第二天第一次投递失败会直接撞上昨天的旧计数、一次都不重试。
        """
        self._push_attempts += 1
        if self._push_attempts < PUSH_MAX_ATTEMPTS:
            self._next_push_at = moment + timedelta(seconds=PUSH_RETRY_SECONDS)
            logger.warning(
                "第 %d/%d 次尝试失败：%d 秒后重试（不是明天）",
                self._push_attempts,
                PUSH_MAX_ATTEMPTS,
                PUSH_RETRY_SECONDS,
            )
            return
        logger.error(
            "日报连续失败 %d 次，本次放弃，明天再推；已生成的条目不会丢，"
            "可以私聊发「记忆 今天」手工看",
            self._push_attempts,
        )
        self._finish_attempt(next_normal)

    def _finish_attempt(self, next_normal: datetime) -> None:
        """这次投递有了结论（成功、无处投递、或放弃重试）：排到下一个正常时刻，计数清零。"""
        self._push_attempts = 0
        self._next_push_at = next_normal

    # —— 后台循环的公共骨架 ——

    async def _run_periodically(
        self,
        interval_seconds: float,
        action: Callable[[], Awaitable[Any]],
        label: str,
    ) -> None:
        """每隔 ``interval_seconds`` 跑一次 ``action``，**单轮失败绝不退出循环**。

        五个后台循环（保留期清理、向量补算、窗口刷新、死信重试、每日推送）本来各写了一遍
        逐字相同的 ``while True: sleep / try / except CancelledError / except Exception``：
        抽成一处，是为了让「失败只记日志、不悄悄停掉这个循环」这条唯一的约定只写一遍——
        后台任务静默退出是那种要等一个月才发现少东西的失败。
        """
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await action()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s失败，下一轮再试", label)

    async def _window_flush_tick(self) -> None:
        """窗口超时刷新的一轮：关掉到点的缓冲，并在真关了东西时留一行日志。"""
        outcomes = await self.flush_expired()
        if outcomes:
            logger.info("窗口超时刷新关闭了 %d 个窗口", len(outcomes))

    # —— 后台任务：保留期清理 ——

    async def cleanup_retention(self) -> int:
        """删掉超过保留期的原文消息，返回删除条数（记忆条目不受影响）。"""
        repository = self._require_repository()
        cutoff = self._clock() - timedelta(days=self._settings.retention_days)
        removed = await repository.delete_messages_before(cutoff)
        if removed:
            logger.info("保留期清理删除了 %d 条原文（截止 %s）", removed, cutoff.isoformat())
        return removed

    # —— 后台任务：向量补算 ——

    async def check_embedding_model_ledger(self) -> str | None:
        """核对「向量模型账本」：库里最近一次成功 ``embed`` 用的模型 vs 当前配置。

        **为什么需要它**：维度守卫只挡得住维度变化。同维度的两个不同向量模型（例如
        SiliconFlow 上 ``BAAI/bge-m3`` 与 ``Qwen/Qwen3-Embedding-0.6B`` 都是 1024 维）
        混在一张表里，长度一模一样、余弦相似度照样算得出数，但两个向量空间不可比——
        去重会既漏报又**误报**（把真记忆判成疑似重复，而疑似重复按设计不进推送）。
        这类不一致从任何长度检查里都看不出来，只能靠 ``llm_call`` 里的模型名记账发现。

        返回账本里记录的旧模型名（库里一条成功的 ``embed`` 记录都没有时返回 ``None``）。
        不一致时记一条 ``WARNING`` 明确指明旧模型、新模型与补救动作（``reembed=True``）。
        **同一 Runtime 实例内只检查与提醒一次**，后台循环每轮调用也不会刷屏；
        ``start()`` 与 :meth:`backfill_embeddings` 入口都会走到这里，故可复用。
        """
        if self._embedding_ledger_checked:
            return self._embedding_ledger_model
        self._embedding_ledger_checked = True
        repository = self._require_repository()
        recorded = await repository.latest_successful_llm_call_model(purpose=EMBED_PURPOSE)
        self._embedding_ledger_model = recorded
        current = self._settings.embedding_model
        if recorded is not None and recorded != current:
            logger.warning(
                "向量模型已变（旧 %s → 新 %s）：请跑 backfill_embeddings(reembed=True) "
                "重算全部向量，否则新旧向量的相似度不可比、去重会既漏报又误报",
                recorded,
                current,
            )
        return recorded

    async def backfill_embeddings(
        self, *, limit: int = EMBEDDING_BACKFILL_BATCH, reembed: bool = False
    ) -> int:
        """给还没算向量的活条目补算并写回，返回成功写回的条数。

        ``reembed=False``（默认）：只补**空向量**的条目；
        ``reembed=True``：连**已有向量**的条目也一起按当前模型重算——换向量模型（维度
        或语义空间变了）后靠它把旧向量全部刷成新模型的结果。两种模式都只认
        ``active`` / ``suspect_duplicate``，已过期的条目不动。

        向量服务不可用（返回 ``None``、数量不符或抛异常）时本轮不改任何条目、返回 0；
        但这次**确已发生**的调用仍会记进 ``llm_call``（``purpose="embed"``），
        成功与失败都留痕，便于成本与可用性可见。

        可观察性：每轮都记日志说明模式、处理条数、写回条数与未写回的原因（整批不可用
        是 ``WARNING``，部分条目拿不到向量或写回失败也是 ``WARNING``）。入口会先核对
        一次向量模型账本（同维度换模型也会被查出，见 :meth:`check_embedding_model_ledger`）。
        """
        repository = self._require_repository()
        await self.check_embedding_model_ledger()
        if reembed:
            pending = await repository.list_memories_for_reembedding(limit=limit)
        else:
            pending = await repository.list_memories_missing_embedding(limit=limit)
        if not pending:
            return 0
        mode = "重算" if reembed else "补算"
        batch = await embed_batch(
            self._embedding, [str(memory.statement) for memory in pending]
        )
        await self._record_embedding_call(success=batch.ok, error=batch.error)
        if batch.vectors is None:
            logger.warning(
                "向量%s不可用，本轮跳过 %d 条：%s", mode, len(pending), batch.error
            )
            return 0
        filled = 0
        missing = 0
        for memory, vector in zip(pending, batch.vectors):
            if vector is None:
                missing += 1
                continue
            if await repository.update_memory_embedding(int(memory.id), vector):
                filled += 1
        failed = len(pending) - filled
        if filled:
            logger.info("向量%s写回 %d/%d 条", mode, filled, len(pending))
        if failed:
            logger.warning(
                "向量%s有 %d/%d 条未写回（其中 %d 条服务未返回向量）；原因：%s",
                mode,
                failed,
                len(pending),
                missing,
                batch.error or "服务对部分输入未返回向量或条目已不存在",
            )
        return filled

    async def _record_embedding_call(self, *, success: bool, error: str | None) -> None:
        """把一次向量服务调用记进 ``llm_call``。

        ``llm_call`` 在窗口事务之外独立提交是有意为之：每次真实发生、会被计费的调用
        都要留痕（窗口回滚也不该抹掉已发生的成本），从而回答「这个功能每月花多少」。
        向量接口不返回用量，故 token 取实现上报的输入 token，取不到记 0，绝不编造；
        费用按 ``settings.embedding_price`` 估算（拿不到 token 时自然为 0）。
        """
        repository = self._require_repository()
        tokens_in = _embedding_input_tokens(self._embedding)
        await repository.add_llm_call(
            purpose=EMBED_PURPOSE,
            model=self._settings.embedding_model,
            tokens_in=tokens_in,
            tokens_out=0,
            cost=self._settings.estimate_embedding_cost(tokens_in),
            success=success,
            error=error,
        )

    # —— 死信重试 ——

    async def retry_dead_windows(self, *, limit: int = DEAD_LETTER_RETRY_BATCH) -> int:
        """重试状态为 ``dead`` 的窗口，返回本轮真正重试（或判定放弃）的窗口数。

        **护栏（这是重试策略的完整说明）**：只有 ``retry_count < dead_letter_max_retries``
        的窗口才重试；每被后台重试一次，``retry_count`` 加一。首次失败时提取管道已把
        ``retry_count`` 记为尝试次数，所以默认 ``extract_max_attempts=3`` 配
        ``dead_letter_max_retries=5`` 大致是「首次失败后再重试两轮」，到顶就不再碰它——
        避免同一个窗口被无限重试。窗口按编号升序处理，老死信不排队。

        重试需要把 ``window`` 行还原成一段消息再重新组装（``evidence`` 的序号要换回真实
        编号）。若原文已被保留期清掉、还原不出任何消息，就把 ``retry_count`` 直接推到上限
        并记一条告警，避免每轮都白跑一次。可观察性：每次重试结果都写日志，``retry_count``
        与 ``error`` 落在窗口行上，仓储可读。
        """
        repository = self._require_repository()
        pipeline = self._require_pipeline()
        max_retries = self._settings.dead_letter_max_retries
        windows = await repository.list_windows_by_status(WindowStatus.DEAD, limit=limit)
        touched = 0
        for window in windows:
            window_id = int(window.id)
            base = int(window.retry_count or 0)
            if base >= max_retries:
                continue
            assembled = await self._rebuild_window(repository, window)
            if assembled is None:
                await repository.update_window_status(
                    window_id,
                    WindowStatus.DEAD,
                    retry_count=max_retries,
                    error="原文已被保留期清理，无法重建窗口",
                )
                logger.warning(
                    "死信窗口 %s（群 %s）的原文已不在，放弃重试", window_id, window.group_id
                )
                touched += 1
                continue
            outcome = await pipeline.process_window(assembled, window_id=window_id)
            if outcome.status is WindowStatus.DEAD:
                await repository.update_window_status(
                    window_id, WindowStatus.DEAD, retry_count=base + 1, error=outcome.error
                )
                logger.warning(
                    "死信窗口 %s 第 %d 次重试仍失败：%s", window_id, base + 1, outcome.error
                )
            else:
                logger.info("死信窗口 %s 重试后状态变为 %s", window_id, outcome.status.value)
            touched += 1
        return touched

    async def _rebuild_window(
        self, repository: Repository, window: Window
    ) -> AssembledWindow | None:
        """把一条 ``window`` 行按起止时间还原成可重跑的 :class:`AssembledWindow`。

        取该群 ``started_at`` 与 ``ended_at``（含两端）之间的原文；找不到（原文被保留期
        清掉）或窗口没记时间时返回 ``None``。超出 ``message_count`` 的部分砍掉，保证还原
        出的窗口与当初处理的那一段一致。
        """
        started_at = window.started_at
        ended_at = window.ended_at
        if started_at is None or ended_at is None:
            return None
        stored = await repository.list_messages_between(
            group_id=int(window.group_id),
            since=started_at,
            until=ended_at,
            limit=max(int(window.message_count or 0) + 5, 50),
        )
        count = int(window.message_count or 0)
        messages = [WindowMessage.from_message(message) for message in stored]
        if count:
            messages = messages[:count]
        if not messages:
            return None
        return assemble_window(
            group_id=int(window.group_id),
            messages=messages,
            prompt_version=window.prompt_version,
        )

    # —— 启动对账（里程碑 2）——

    async def reconcile_uncovered_messages(
        self, *, limit: int | None = None
    ) -> ReconcileReport:
        """把没有被任何窗口覆盖的消息重新送进窗口管道；每群处理完强制成窗。

        **要解决的问题**（真机实测）：窗口缓冲只在内存里（见 ``pipeline/window.py``），
        进程重启时未满的窗口会丢。那些消息**还在 ``message`` 表里**，但掉线回补的锚点是
        「该群最后一条已存消息的时间」（``latest_message_sent_at``），已经存过的消息不会
        再被拉回来，于是**永远不会被提取**。启动时按「``sent_at`` 不被任何同群窗口的
        ``[started_at, ended_at]`` 覆盖」找出这批消息，重新成窗、提取、入库。

        两个关键点：

        - **不重复写 ``message`` 表**：这些消息已经在库里，直接把仓储对象交给组装器
          （``WindowAssembler.add`` 会适配），于是 ``sequence_map`` 指向真实消息编号、
          ``evidence`` 正确，不需要任何特殊处理。
        - **用独立组装器**：实时采集（:meth:`ingest`）与后台刷新循环都在用
          ``self._assembler``，它按群维护内存缓冲；若共用，两边同时往同一个群的缓冲里塞
          消息会把窗口拼坏。这里用 :meth:`_build_assembler` 另建一个，缓冲互不干扰；
          内容万一重叠，入库时的去重 hash 会挡掉。

        每个群处理完调 ``flush_group`` 强制成窗——否则不足最小条数的尾巴会一直留在缓冲里，
        对账就等于没做。``limit`` 缺省取 ``settings.reconcile_max_messages``（每群上限）；
        触到上限记 ``WARNING``，剩下的留到下次启动或手动再调。

        幂等：处理完后这些消息就落在新窗口的时间区间里，所以下一次启动应查出接近 0 条。
        """
        repository = self._require_repository()
        pipeline = self._require_pipeline()
        per_group_limit = self._settings.reconcile_max_messages if limit is None else limit
        if per_group_limit <= 0:
            logger.warning("启动对账：每群上限为 %d，跳过本轮", per_group_limit)
            return ReconcileReport(groups=())
        # 独立组装器，绝不碰 self._assembler（理由见文档字符串）。
        assembler = self._build_assembler(repository, self._clock)
        groups: list[GroupReconcile] = []
        for group_id in sorted(await self.allowed_groups()):
            # 多取一条只为准确判断「是否真的还有剩下的」，处理时再砍回上限。
            fetched = await repository.list_uncovered_messages(
                group_id=group_id, limit=per_group_limit + 1
            )
            capped = len(fetched) > per_group_limit
            messages = fetched[:per_group_limit]
            if not messages:
                continue
            outcomes: list[ExtractionOutcome] = []
            for message in messages:
                for window in await assembler.add(message):
                    outcomes.append(await pipeline.process_window(window))
            remainder = await assembler.flush_group(group_id)
            if remainder is not None:
                outcomes.append(await pipeline.process_window(remainder))
            memory_ids = tuple(
                memory_id for outcome in outcomes for memory_id in outcome.memory_ids
            )
            groups.append(
                GroupReconcile(
                    group_id=group_id,
                    message_count=len(messages),
                    window_count=len(outcomes),
                    memory_ids=memory_ids,
                    capped=capped,
                )
            )
            logger.info(
                "启动对账：群 %s 重新处理 %d 条未覆盖消息，成窗 %d 个，新增条目 %d 条",
                group_id,
                len(messages),
                len(outcomes),
                len(memory_ids),
            )
            if capped:
                logger.warning(
                    "启动对账：群 %s 未覆盖消息已达上限 %d 条，仍有未覆盖的消息留着；"
                    "下次启动或手动调用 reconcile_uncovered_messages() 会继续处理",
                    group_id,
                    per_group_limit,
                )
        report = ReconcileReport(groups=tuple(groups))
        logger.info(
            "启动对账完成：%d 个群、%d 条消息、%d 个窗口、%d 条新增条目",
            len(report.groups),
            report.message_count,
            report.window_count,
            len(report.memory_ids),
        )
        return report

    async def _startup_reconcile(self) -> None:
        """启动对账的一次性后台任务：失败只记日志，不影响其余循环，也不阻塞启动。"""
        try:
            await self.reconcile_uncovered_messages()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("启动对账失败；未覆盖的消息留到下次启动或手动调用时再处理")

    # —— 离线回放 ——

    async def replay(
        self,
        *,
        source: HistorySource,
        group_id: int,
        since: datetime | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> ReplayReport:
        """把一段历史消息喂进整条管道。

        正常模式写进配置的库并返回每个窗口的提取结果；``dry_run`` 只产出提示词与模型
        原始输出，配置的库一条不写（消息落在临时库，用完即删，见模块 docstring）。
        ``dry_run`` 甚至**不要求 Runtime 已经 start**——它压根不碰配置的库。

        两种模式共用 :meth:`_replay_windows` 这一个循环，只差对每个窗口做什么
        （``preview_window`` 还是 ``process_window``）。
        """
        if not dry_run:
            self._require_repository()
        messages = await source.fetch(group_id=group_id, since=since, limit=limit)

        if dry_run:
            with tempfile.TemporaryDirectory(prefix="rzyl-dryrun-") as directory:
                temporary = await Repository.create(
                    f"sqlite+aiosqlite:///{Path(directory) / 'dryrun.db'}", clock=self._clock
                )
                try:
                    previews = await self._replay_windows(
                        repository=temporary,
                        messages=messages,
                        group_id=group_id,
                        handle=_preview_window,
                    )
                finally:
                    await temporary.close()
            return ReplayReport(
                group_id=group_id,
                dry_run=True,
                message_count=len(messages),
                window_count=len(previews),
                previews=tuple(previews),
            )

        repository = self._require_repository()
        outcomes = await self._replay_windows(
            repository=repository,
            messages=messages,
            group_id=group_id,
            handle=_process_window,
        )
        return ReplayReport(
            group_id=group_id,
            dry_run=False,
            message_count=len(messages),
            window_count=len(outcomes),
            outcomes=tuple(outcomes),
        )

    async def _replay_windows(
        self,
        *,
        repository: Repository,
        messages: Sequence[HistoryMessage],
        group_id: int,
        handle: Callable[[ExtractionPipeline, AssembledWindow], Awaitable[T]],
    ) -> list[T]:
        """回放主循环：逐条落库、成窗即交给 ``handle``；收尾再 flush 一次该群缓冲。

        成窗只依赖消息自身时间（见 ``pipeline/window.py``），所以时钟直接沿用 Runtime
        注入的那个即可，不需要再为回放临时改时钟。dry-run 与正常模式的差别只体现在
        传入的 ``handle`` 上——前者绝不写配置库，后者走完整入库。
        """
        assembler = self._build_assembler(repository, self._clock)
        pipeline = self._build_pipeline(repository, self._clock)
        results: list[T] = []
        for message in messages:
            stored = await self._store_message(repository, message)
            for window in await assembler.add(stored):
                results.append(await handle(pipeline, window))
        remainder = await assembler.flush_group(group_id)
        if remainder is not None:
            results.append(await handle(pipeline, remainder))
        return results

    async def _store_message(self, repository: Repository, message: HistoryMessage) -> Message:
        """把一条历史消息落库；去重 hash 用群号 + 发送者 + 时间 + 原文（供里程碑 2 回补去重）。"""
        return await repository.add_message(
            group_id=message.group_id,
            user_id=message.user_id,
            text=message.text,
            sent_at=message.sent_at,
            nickname=message.nickname,
            card=message.card,
            platform_message_id=message.platform_message_id,
            dedupe_hash=message_dedupe_hash(
                group_id=message.group_id,
                user_id=message.user_id,
                sent_at=message.sent_at,
                text=message.text,
            ),
        )

    # —— 内部装配 ——

    def _build_assembler(
        self, repository: Repository, clock: Callable[[], datetime]
    ) -> WindowAssembler:
        return WindowAssembler(
            repository=repository,
            clock=clock,
            settings=self._settings,
            previous_tail_size=self._previous_tail_size,
            remembered_limit=self._remembered_limit,
            prompt_version=self._prompt_version,
        )

    def _build_pipeline(
        self, repository: Repository, clock: Callable[[], datetime]
    ) -> ExtractionPipeline:
        return ExtractionPipeline(
            repository=repository,
            chat_model=self._chat,
            embedding_model=self._embedding,
            settings=self._settings,
            clock=clock,
            provider=self._provider,
        )

    def _build_console(self, repository: Repository) -> MemoryConsole:
        """装配命令执行器；「强制关窗」用 partial 绑到本实例的方法上。"""
        return MemoryConsole(
            repository=repository,
            embedding=self._embedding,
            settings=self._settings,
            clock=self._clock,
            flush_pending=functools.partial(self.flush_pending),
        )

    def _require_repository(self) -> Repository:
        if self._repository is None:
            raise RuntimeError("Runtime 尚未 start()，仓储不可用")
        return self._repository

    def _require_assembler(self) -> WindowAssembler:
        if self._assembler is None:
            raise RuntimeError("Runtime 尚未 start()，窗口组装器不可用")
        return self._assembler

    def _require_pipeline(self) -> ExtractionPipeline:
        if self._pipeline is None:
            raise RuntimeError("Runtime 尚未 start()，提取管道不可用")
        return self._pipeline


__all__ = [
    "DAILY_REPORT_FETCH_LIMIT",
    "DEAD_LETTER_RETRY_BATCH",
    "EMBEDDING_BACKFILL_BATCH",
    "EMBEDDING_SWEEP_SECONDS",
    "EMBED_PURPOSE",
    "PUSH_TICK_SECONDS",
    "RETENTION_SWEEP_SECONDS",
    "GroupReconcile",
    "IngestResult",
    "ReconcileReport",
    "ReplayReport",
    "Runtime",
    "get_runtime",
    "get_runtime_or_none",
    "set_runtime",
]


# —— 模块级 Runtime 注册表：插件取得 Runtime 的显式方式 ——
#
# NoneBot 插件与 core 之间不能互相隐式依赖，也不该往 driver 上乱挂属性。装配层
# （``bot.py``）构造好 Runtime 后 ``set_runtime(runtime)``，插件用 ``get_runtime()`` 取；
# 记忆功能因缺少密钥未启用时装配层 ``set_runtime(None)``，插件用
# ``get_runtime_or_none()`` 判空后静默跳过，机器人照常收发消息。

_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    """注册（或清空）进程内唯一的 Runtime；由装配层调用。"""
    global _runtime
    _runtime = runtime


def get_runtime_or_none() -> Runtime | None:
    """取已注册的 Runtime；未装配或记忆功能未启用时返回 ``None``。"""
    return _runtime


def get_runtime() -> Runtime:
    """取已注册的 Runtime；未装配时抛 ``RuntimeError``，明确告知而不是给个空壳。"""
    if _runtime is None:
        raise RuntimeError(
            "Runtime 尚未装配：bot.py 未调用 set_runtime()，或记忆功能因缺少聊天模型密钥未启用"
        )
    return _runtime
