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

内部后台任务的现状（工单 #5 明确要写明，避免"以为实现了"）：

- **保留期清理**（``cleanup_retention``）：**已实现**。按 ``settings.retention_days``
  删掉 ``sent_at`` 早于截止时刻的原文；只删原文，记忆条目永久。
- **向量补算**（``backfill_embeddings``）：**已实现**。取向量为空的活条目，批量算向量
  并写回；向量服务不可用（返回 ``None`` 或抛异常）时本轮什么都不做，条目照常保留。
- 两个后台循环（``_retention_loop`` / ``_embedding_loop``）只是按固定间隔调用上面两个
  方法；间隔是模块常量，将来接实时链路时再挪进设置。
- **尚未实现**：窗口超时刷新与死信重试的常驻循环属于里程碑 2 的实时链路，这里只提供
  ``flush_expired`` / ``flush_group`` 两个手动入口给回放与测试用。

回放有两个模式，差别只在**写与不写**：

- 正常模式：消息与窗口结果都写进**配置的库**，从仓储能读回条目；
- ``dry_run``：消息写进一个临时库（窗口的编号映射需要真实的 ``message.id``），每个窗口
  只调一次模型、打印完整提示词与原始输出，**配置的库一条不写**，临时库用完即删。

回放还解决了一个务必做对的点（上游已在 ``pipeline/window.py`` 写明）：**时钟要跟着
「当前正在回放的那条消息」走**。窗口的时间成窗判定是拿注入时钟与缓冲首条消息的时间比；
回放若沿用真实当下，历史消息会瞬间全部超时成窗。所以回放内部自建一个 :class:`_ReplayClock`，
每条消息进来前把它拨到该消息的发送时间。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rzyl_core.db import Message, Repository
from rzyl_core.llm import ChatModel, EmbeddingModel
from rzyl_core.pipeline.extract import ExtractionOutcome, ExtractionPipeline, PreviewOutcome
from rzyl_core.pipeline.history import HistorySource, HistoryMessage, message_dedupe_hash
from rzyl_core.pipeline.window import (
    DEFAULT_PREVIOUS_TAIL_SIZE,
    DEFAULT_REMEMBERED_LIMIT,
    WindowAssembler,
)
from rzyl_core.settings import Settings

logger = logging.getLogger(__name__)

#: 保留期清理循环的间隔（秒）——暂时是模块常量，接实时链路时再挪进设置。
RETENTION_SWEEP_SECONDS = 3600

#: 向量补算循环的间隔（秒）。
EMBEDDING_SWEEP_SECONDS = 300

#: 一次向量补算最多处理多少条，避免一次拉太多进内存。
EMBEDDING_BACKFILL_BATCH = 100


def _utcnow() -> datetime:
    """默认时间源：当前 UTC 时刻（带时区）。"""
    return datetime.now(timezone.utc)


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


class _ReplayClock:
    """回放专用时钟：值由回放逐条拨到「当前回放消息」的发送时间。

    窗口的成窗判定读它而不是读真实时间，历史消息才不会瞬间全部成窗（见模块 docstring）。
    """

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


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
        self._clock: Callable[[], datetime] = clock or _utcnow
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

        向量服务不可用（返回 ``None``、数量不符或抛异常）时本轮返回 0、什么都不改。
        """
        repository = self._require_repository()
        pending = await repository.list_memories_missing_embedding(limit=limit)
        if not pending:
            return 0
        try:
            vectors = await self._embedding.embed([str(memory.statement) for memory in pending])
        except Exception:
            logger.exception("向量补算调用失败，本轮跳过")
            return 0
        if vectors is None or len(vectors) != len(pending):
            return 0
        filled = 0
        for memory, vector in zip(pending, vectors):
            if vector and await repository.update_memory_embedding(int(memory.id), list(vector)):
                filled += 1
        if filled:
            logger.info("向量补算写回 %d 条", filled)
        return filled

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
        """
        if not dry_run:
            self._require_repository()
        messages = await source.fetch(group_id=group_id, since=since, limit=limit)
        replay_clock = _ReplayClock(self._clock())

        if dry_run:
            with tempfile.TemporaryDirectory(prefix="rzyl-dryrun-") as directory:
                temporary = await Repository.create(
                    f"sqlite+aiosqlite:///{Path(directory) / 'dryrun.db'}", clock=replay_clock
                )
                try:
                    assembler = self._build_assembler(temporary, replay_clock)
                    pipeline = self._build_pipeline(temporary, replay_clock)
                    previews: list[PreviewOutcome] = []
                    for message in messages:
                        replay_clock.now = message.sent_at
                        stored = await self._store_message(temporary, message)
                        for window in await assembler.add(stored):
                            previews.append(await pipeline.preview_window(window))
                    remainder = await assembler.flush_group(group_id)
                    if remainder is not None:
                        previews.append(await pipeline.preview_window(remainder))
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
        assembler = self._build_assembler(repository, replay_clock)
        pipeline = self._build_pipeline(repository, replay_clock)
        outcomes: list[ExtractionOutcome] = []
        for message in messages:
            replay_clock.now = message.sent_at
            stored = await self._store_message(repository, message)
            for window in await assembler.add(stored):
                outcomes.append(await pipeline.process_window(window))
        remainder = await assembler.flush_group(group_id)
        if remainder is not None:
            outcomes.append(await pipeline.process_window(remainder))
        return ReplayReport(
            group_id=group_id,
            dry_run=False,
            message_count=len(messages),
            window_count=len(outcomes),
            outcomes=tuple(outcomes),
        )

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
    "EMBEDDING_BACKFILL_BATCH",
    "EMBEDDING_SWEEP_SECONDS",
    "RETENTION_SWEEP_SECONDS",
    "IngestResult",
    "ReplayReport",
    "Runtime",
]
