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
- **窗口超时刷新**（``flush_expired``）：**已实现**。群里没人说话时，靠
  ``_window_flush_loop`` 周期把未满的缓冲按时成窗并入库；间隔取
  ``settings.window_flush_seconds``。
- **死信重试**（``retry_dead_windows``）：**已实现**。``_dead_letter_loop`` 周期找出
  状态为 ``dead`` 的窗口重跑；护栏与可观察性见 :meth:`Runtime.retry_dead_windows` 的
  文档字符串。间隔取 ``settings.dead_letter_retry_seconds``。
- 保留期清理与向量补算两个老循环的间隔仍是模块常量（``RETENTION_SWEEP_SECONDS`` /
  ``EMBEDDING_SWEEP_SECONDS``）；新增的两个循环按里程碑 2 的要求从设置读。

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
from rzyl_core.timeutil import utcnow

logger = logging.getLogger(__name__)

#: 保留期清理循环的间隔（秒）——暂时是模块常量，接实时链路时再挪进设置。
RETENTION_SWEEP_SECONDS = 3600

#: 向量补算循环的间隔（秒）。
EMBEDDING_SWEEP_SECONDS = 300

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
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False

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
        self._started = True
        if run_background_tasks:
            self._tasks = [
                asyncio.create_task(self._retention_loop(), name="rzyl-retention-sweep"),
                asyncio.create_task(self._embedding_loop(), name="rzyl-embedding-backfill"),
                asyncio.create_task(self._window_flush_loop(), name="rzyl-window-flush"),
                asyncio.create_task(self._dead_letter_loop(), name="rzyl-dead-letter-retry"),
            ]

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
        """当前允许采集的群号：设置里的白名单 **并上** 数据库里的每群运行开关。

        两个来源缺一不可——配置文件给初始值，里程碑 3 的开关命令写数据库；这里取并集，
        判定与插件启动日志都用它。
        """
        repository = self._require_repository()
        return merge_allowed_groups(
            self._settings.group_whitelist, await repository.enabled_groups()
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

    # —— 后台任务：保留期清理 ——

    async def cleanup_retention(self) -> int:
        """删掉超过保留期的原文消息，返回删除条数（记忆条目不受影响）。"""
        repository = self._require_repository()
        cutoff = self._clock() - timedelta(days=self._settings.retention_days)
        removed = await repository.delete_messages_before(cutoff)
        if removed:
            logger.info("保留期清理删除了 %d 条原文（截止 %s）", removed, cutoff.isoformat())
        return removed

    async def _retention_loop(self) -> None:
        """按固定间隔跑保留期清理；单轮失败不影响下一轮。"""
        while True:
            await asyncio.sleep(RETENTION_SWEEP_SECONDS)
            try:
                await self.cleanup_retention()
            except asyncio.CancelledError:
                raise
            except Exception:  # 后台任务绝不能因为一次失败就静默退出
                logger.exception("保留期清理失败，下一轮再试")

    # —— 后台任务：向量补算 ——

    async def backfill_embeddings(self, *, limit: int = EMBEDDING_BACKFILL_BATCH) -> int:
        """给还没算向量的活条目补算并写回，返回成功补上的条数。

        向量服务不可用（返回 ``None``、数量不符或抛异常）时本轮不改任何条目、返回 0；
        但这次**确已发生**的调用仍会记进 ``llm_call``（``purpose="embed"``），
        成功与失败都留痕，便于成本与可用性可见。
        """
        repository = self._require_repository()
        pending = await repository.list_memories_missing_embedding(limit=limit)
        if not pending:
            return 0
        batch = await embed_batch(
            self._embedding, [str(memory.statement) for memory in pending]
        )
        await self._record_embedding_call(success=batch.ok, error=batch.error)
        if batch.vectors is None:
            logger.warning("向量补算不可用，本轮跳过：%s", batch.error)
            return 0
        filled = 0
        for memory, vector in zip(pending, batch.vectors):
            if vector is not None and await repository.update_memory_embedding(
                int(memory.id), vector
            ):
                filled += 1
        if filled:
            logger.info("向量补算写回 %d 条", filled)
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

    async def _embedding_loop(self) -> None:
        """按固定间隔跑向量补算；单轮失败不影响下一轮。"""
        while True:
            await asyncio.sleep(EMBEDDING_SWEEP_SECONDS)
            try:
                await self.backfill_embeddings()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("向量补算失败，下一轮再试")

    # —— 后台任务：窗口超时刷新（里程碑 2）——

    async def _window_flush_loop(self) -> None:
        """按 ``settings.window_flush_seconds`` 周期调 ``flush_expired()``；单轮失败不退循环。

        群里没人说话时，未满的缓冲靠这个循环按时成窗——否则消息会一直卡在内存里，
        直到下一条消息到来或进程重启（重启就丢，只能靠回补兜底）。
        """
        while True:
            await asyncio.sleep(self._settings.window_flush_seconds)
            try:
                outcomes = await self.flush_expired()
                if outcomes:
                    logger.info("窗口超时刷新关闭了 %d 个窗口", len(outcomes))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("窗口超时刷新失败，下一轮再试")

    # —— 后台任务：死信重试（里程碑 2）——

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

    async def _dead_letter_loop(self) -> None:
        """按 ``settings.dead_letter_retry_seconds`` 周期跑死信重试；单轮失败不退循环。"""
        while True:
            await asyncio.sleep(self._settings.dead_letter_retry_seconds)
            try:
                await self.retry_dead_windows()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("死信重试失败，下一轮再试")

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
    "DEAD_LETTER_RETRY_BATCH",
    "EMBEDDING_BACKFILL_BATCH",
    "EMBEDDING_SWEEP_SECONDS",
    "EMBED_PURPOSE",
    "RETENTION_SWEEP_SECONDS",
    "IngestResult",
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
