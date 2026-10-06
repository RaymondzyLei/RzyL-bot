"""Runtime 的行为测试（工单 #5）。

seam 是 *Runtime*：外部依赖（聊天模型、向量、时钟、数据库连接串）在构造时注入，
测试从仓储读 API 断言记忆条目、窗口状态与后台任务的结果。不碰 Runtime 内部字段，
也不给内部协作者打桩——只换掉注入的假实现与时钟。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import Category, Repository
from rzyl_core.llm import ChatModel, DeterministicEmbedding, EmbeddingModel, NullEmbedding
from rzyl_core.llm.fakes import EchoChatModel
from rzyl_core.pipeline import assemble_window
from rzyl_core.pipeline.window import WindowMessage
from rzyl_core.runtime import Runtime
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

GROUP = 111
USER = 10001


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


class _Clock:
    """可手动推进的时钟：测窗口超时不用真等 5 分钟。"""

    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + timedelta(minutes=minutes)


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _runtime(
    tmp_path: Path,
    clock: _Clock,
    *,
    chat_model: ChatModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    **settings: object,
) -> Runtime:
    return Runtime(
        chat_model=chat_model or EchoChatModel(),
        embedding_model=embedding_model or DeterministicEmbedding(8),
        settings=_settings(**settings),
        clock=clock,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )


# —— 生命周期 ——


async def test_runtime_start_and_stop_own_the_repository(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock)

    with pytest.raises(RuntimeError):
        _ = runtime.repository

    await runtime.start(run_background_tasks=False)
    repo = runtime.repository
    assert isinstance(repo, Repository)

    await runtime.stop()

    with pytest.raises(RuntimeError):
        _ = runtime.repository


async def test_runtime_background_task_skeletons_are_started_and_cancelled(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock)

    await runtime.start(run_background_tasks=True)
    assert len(runtime.background_tasks) == 2
    assert all(not task.done() for task in runtime.background_tasks)

    await runtime.stop()
    assert runtime.background_tasks == ()


async def test_ingest_before_start_is_a_programming_error(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock)

    with pytest.raises(RuntimeError):
        await runtime.ingest(group_id=GROUP, user_id=USER, text="还没启动", sent_at=BEGIN)


# —— 灌入消息、成窗、落库 ——


async def test_ingest_stores_the_message_before_windowing_and_pipeline_maps_it_back(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=2)
    await runtime.start(run_background_tasks=False)
    try:
        first = await runtime.ingest(
            group_id=GROUP, user_id=USER, text="甲", sent_at=BEGIN, nickname="小A"
        )
        assert first.outcomes == ()
        assert first.buffered == 1

        second = await runtime.ingest(
            group_id=GROUP, user_id=USER, text="乙", sent_at=BEGIN + timedelta(minutes=1), nickname="小A"
        )

        assert len(second.outcomes) == 1
        outcome = second.outcomes[0]
        assert outcome.status.value == "done"
        assert len(outcome.memory_ids) == 1

        memory = await runtime.repository.get_memory(outcome.memory_ids[0])
        assert memory is not None
        # 消息先落库再进窗口，所以 evidence 里的窗口序号能换回真实 message.id。
        assert memory.evidence == [first.message.id]
        assert memory.group_id == GROUP

        window = await runtime.repository.get_window(outcome.window_id)
        assert window is not None
        assert window.status.value == "done"
        assert window.message_count == 2
    finally:
        await runtime.stop()


async def test_flush_expired_closes_a_quiet_window_through_the_pipeline(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30)
    await runtime.start(run_background_tasks=False)
    try:
        await runtime.ingest(group_id=GROUP, user_id=USER, text="一个人说话", sent_at=BEGIN)
        assert await runtime.flush_expired() == ()

        clock.advance(6)
        outcomes = await runtime.flush_expired()

        assert len(outcomes) == 1
        assert outcomes[0].status.value == "done"
        assert len(outcomes[0].memory_ids) == 1
    finally:
        await runtime.stop()


# —— 后台任务：保留期清理 ——


async def test_retention_cleanup_deletes_only_messages_older_than_the_cutoff(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, retention_days=30)
    await runtime.start(run_background_tasks=False)
    try:
        repo = runtime.repository
        await repo.add_message(
            group_id=GROUP, user_id=USER, text="很久以前", sent_at=BEGIN - timedelta(days=40)
        )
        await repo.add_message(group_id=GROUP, user_id=USER, text="最近", sent_at=BEGIN)

        removed = await runtime.cleanup_retention()

        assert removed == 1
        remaining = await repo.list_messages(group_id=GROUP)
        assert [message.text for message in remaining] == ["最近"]
    finally:
        await runtime.stop()


# —— 后台任务：向量补算 ——


async def test_embedding_backfill_fills_memories_without_a_vector(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, embedding_model=DeterministicEmbedding(8))
    await runtime.start(run_background_tasks=False)
    try:
        memory = await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="结论：先跑基线再调参",
            confidence=0.9,
            prompt_version="v1",
        )
        assert memory.embedding is None

        filled = await runtime.backfill_embeddings()

        assert filled == 1
        reloaded = await runtime.repository.get_memory(memory.id)
        assert reloaded is not None and reloaded.embedding is not None
    finally:
        await runtime.stop()


async def test_embedding_backfill_is_a_noop_without_a_vector_service(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, embedding_model=NullEmbedding())
    await runtime.start(run_background_tasks=False)
    try:
        memory = await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="没有向量服务",
            confidence=0.9,
            prompt_version="v1",
        )

        assert await runtime.backfill_embeddings() == 0

        reloaded = await runtime.repository.get_memory(memory.id)
        assert reloaded is not None and reloaded.embedding is None
    finally:
        await runtime.stop()


# —— 离线假模型 ——


async def test_echo_chat_model_returns_one_valid_entry_for_the_first_message() -> None:
    from rzyl_core.pipeline.extract import parse_extraction

    window = assemble_window(
        group_id=GROUP,
        messages=[
            WindowMessage(
                message_id=101,
                group_id=GROUP,
                user_id=10001,
                text="实验课改到周五下午三点",
                sent_at=BEGIN,
                nickname="小A",
            ),
            WindowMessage(
                message_id=102,
                group_id=GROUP,
                user_id=10002,
                text="好的",
                sent_at=BEGIN + timedelta(seconds=1),
                nickname="小B",
            ),
        ],
    )
    rendered = window.render()

    result = await EchoChatModel().complete(rendered.system, rendered.user)
    items = parse_extraction(result.text)

    assert len(items) == 1
    assert items[0].evidence == [1]
    assert items[0].statement == "实验课改到周五下午三点"
    assert items[0].person_refs[0].user_id == 10001
    assert items[0].person_refs[0].nickname_snapshot == "小A"
