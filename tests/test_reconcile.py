"""启动对账的行为测试：把「没有被任何窗口时间区间覆盖」的消息重新送进窗口管道。

背景（真机实测，不是设想）：窗口缓冲活在内存里，进程重启时未满的窗口会丢。那些消息
还在 ``message`` 表里，但掉线回补的锚点是「该群最后一条已存消息的时间」，已经存过的
消息不会再被拉回来，于是**永远不会被提取**。对账就是补这个洞：启动时找出没有被任何
窗口覆盖的消息，用一条独立路径重新成窗、提取、入库。

seam 是 ``Repository``（读 API）与 ``Runtime``（装配入口）。模型用 ``EchoChatModel`` /
``FakeChatModel``，绝不碰真实 API；所有时间从注入时钟或消息自身的 ``sent_at`` 来。

幂等性是这套东西成立的前提：对账处理完后，那些消息就落在新窗口的时间区间里了，所以
第二次启动应该查不出任何东西——这条有专门的测试守着。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import Repository, WindowStatus
from rzyl_core.llm import DeterministicEmbedding
from rzyl_core.llm.fakes import EchoChatModel
from rzyl_core.runtime import Runtime
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

GROUP = 111
USER = 10001


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


class _Clock:
    """可手动推进的时钟；对账本身不依赖当下时刻，这里只为满足注入契约。"""

    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _runtime(tmp_path: Path, clock: _Clock, **settings: object) -> Runtime:
    return Runtime(
        chat_model=EchoChatModel(),
        embedding_model=DeterministicEmbedding(8),
        settings=_settings(group_whitelist=[GROUP], **settings),
        clock=clock,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )


async def _seed_covered_gap_covered(tmp_path: Path, *, gap: int = 3) -> list[int]:
    """往库文件里直接播场景：一段被窗口覆盖的消息、一段空隙、又一段被覆盖的消息。

    返回**空隙里那几条消息的编号**（升序）——它们没有任何窗口覆盖，正是对账要捡回来的
    对象。先用独立仓储播好再开 Runtime，这样 ``run_background_tasks=True`` 的自动对账
    一启动就有东西可处理。
    """
    repo = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    try:
        for index in range(3):
            await repo.add_message(
                group_id=GROUP,
                user_id=USER,
                text=f"覆盖甲{index}",
                sent_at=BEGIN + timedelta(seconds=index),
            )
        await repo.add_window(
            group_id=GROUP,
            started_at=BEGIN,
            ended_at=BEGIN + timedelta(seconds=2),
            message_count=3,
            status=WindowStatus.DONE,
        )

        gap_ids: list[int] = []
        for index in range(gap):
            message = await repo.add_message(
                group_id=GROUP,
                user_id=USER,
                text=f"空隙{index}",
                sent_at=BEGIN + timedelta(minutes=10, seconds=index),
                nickname="小A",
            )
            gap_ids.append(message.id)

        for index in range(2):
            await repo.add_message(
                group_id=GROUP,
                user_id=USER,
                text=f"覆盖乙{index}",
                sent_at=BEGIN + timedelta(minutes=20, seconds=index),
            )
        await repo.add_window(
            group_id=GROUP,
            started_at=BEGIN + timedelta(minutes=20),
            ended_at=BEGIN + timedelta(minutes=20, seconds=1),
            message_count=2,
            status=WindowStatus.DONE,
        )
        return gap_ids
    finally:
        await repo.close()


# —— Repository.list_uncovered_messages ——


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'repo.db'}")
    yield instance
    await instance.close()


async def test_uncovered_query_returns_only_the_gap_in_time_order(repo: Repository) -> None:
    await repo.add_message(group_id=GROUP, user_id=USER, text="覆盖甲", sent_at=BEGIN)
    first = await repo.add_message(
        group_id=GROUP, user_id=USER, text="空隙一", sent_at=BEGIN + timedelta(minutes=10)
    )
    second = await repo.add_message(
        group_id=GROUP, user_id=USER, text="空隙二", sent_at=BEGIN + timedelta(minutes=11)
    )
    await repo.add_message(
        group_id=GROUP, user_id=USER, text="覆盖乙", sent_at=BEGIN + timedelta(minutes=20)
    )
    # 两个窗口分别盖住头尾两条；中间两条落在空隙里。
    await repo.add_window(
        group_id=GROUP,
        started_at=BEGIN,
        ended_at=BEGIN,
        message_count=1,
        status=WindowStatus.DONE,
    )
    await repo.add_window(
        group_id=GROUP,
        started_at=BEGIN + timedelta(minutes=20),
        ended_at=BEGIN + timedelta(minutes=20),
        message_count=1,
        status=WindowStatus.DONE,
    )

    uncovered = await repo.list_uncovered_messages(group_id=GROUP)

    assert [message.id for message in uncovered] == [first.id, second.id]


async def test_uncovered_query_ignores_windows_without_an_end(repo: Repository) -> None:
    """覆盖要求 ``started_at`` 与 ``ended_at`` 均非空：只有开头的窗口不算覆盖。"""
    message = await repo.add_message(
        group_id=GROUP, user_id=USER, text="没结尾的窗口", sent_at=BEGIN
    )
    await repo.add_window(
        group_id=GROUP, started_at=BEGIN, ended_at=None, message_count=1, status=WindowStatus.PENDING
    )

    uncovered = await repo.list_uncovered_messages(group_id=GROUP)

    assert [item.id for item in uncovered] == [message.id]


async def test_uncovered_query_is_per_group_and_respects_limit(repo: Repository) -> None:
    first = await repo.add_message(
        group_id=GROUP, user_id=USER, text="本群一", sent_at=BEGIN + timedelta(minutes=1)
    )
    await repo.add_message(
        group_id=GROUP, user_id=USER, text="本群二", sent_at=BEGIN + timedelta(minutes=2)
    )
    # 别的群的未覆盖消息不该串进来。
    await repo.add_message(
        group_id=GROUP + 1, user_id=USER, text="别群", sent_at=BEGIN + timedelta(minutes=1)
    )

    assert [item.id for item in await repo.list_uncovered_messages(group_id=GROUP)] == [
        first.id,
        first.id + 1,
    ]
    assert [item.id for item in await repo.list_uncovered_messages(group_id=GROUP, limit=1)] == [
        first.id
    ]


# —— Runtime.reconcile_uncovered_messages ——


async def test_reconcile_windows_and_extracts_the_gap_messages(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30)
    await runtime.start(run_background_tasks=False)
    try:
        gap_ids = await _seed_covered_gap_covered(tmp_path)

        report = await runtime.reconcile_uncovered_messages()

        assert report.message_count == 3
        assert report.window_count == 1
        assert len(report.memory_ids) == 1

        # evidence 必须换回真实消息编号，而不是窗口内序号。
        memory = await runtime.repository.get_memory(report.memory_ids[0])
        assert memory is not None
        assert memory.group_id == GROUP
        assert memory.evidence == [gap_ids[0]]

        # 新窗口的时间区间正好盖住那一整段空隙。
        windows = await runtime.repository.list_windows_by_status(WindowStatus.DONE)
        gap_window = next(
            window
            for window in windows
            if window.started_at == BEGIN + timedelta(minutes=10)
            and window.ended_at == BEGIN + timedelta(minutes=10, seconds=2)
        )
        assert gap_window.message_count == 3
    finally:
        await runtime.stop()


async def test_reconcile_is_idempotent(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30)
    await runtime.start(run_background_tasks=False)
    try:
        gap_ids = await _seed_covered_gap_covered(tmp_path)

        first = await runtime.reconcile_uncovered_messages()
        assert first.message_count == 3
        assert len(first.memory_ids) == 1
        windows_before = len(await runtime.repository.list_windows_by_status(WindowStatus.DONE))
        memories_before = len(await runtime.repository.list_group_memories(group_id=GROUP))

        # 处理完的消息已经落在新窗口的时间区间里，第二次应查不出任何东西。
        second = await runtime.reconcile_uncovered_messages()

        assert second.message_count == 0
        assert second.window_count == 0
        assert second.memory_ids == ()
        assert (
            len(await runtime.repository.list_windows_by_status(WindowStatus.DONE)) == windows_before
        )
        assert (
            len(await runtime.repository.list_group_memories(group_id=GROUP)) == memories_before
        )
        assert gap_ids  # 场景确实造出了空隙

        # 幂等也可以直接从覆盖查询看出来：一条未覆盖的都不剩。
        assert await runtime.repository.list_uncovered_messages(group_id=GROUP) == []
    finally:
        await runtime.stop()


async def test_reconcile_warns_when_the_per_group_cap_is_hit(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30, reconcile_max_messages=2)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_covered_gap_covered(tmp_path, gap=3)

        with caplog.at_level(logging.WARNING, logger="rzyl_core.runtime"):
            report = await runtime.reconcile_uncovered_messages()

        assert report.message_count == 2
        assert report.capped_groups == (GROUP,)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any(str(GROUP) in message and "上限" in message for message in warnings)
    finally:
        await runtime.stop()


# —— 启动时的自动对账（开关与后台任务）——


async def test_startup_reconcile_runs_as_a_registered_background_task(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=True)
    try:
        names = {task.get_name() for task in runtime.background_tasks}
        # 五个常驻循环 + 一个一次性对账任务。断言名字集合而不是条数：多一个任务时要能一眼
        # 看出是哪个，条数变了只会看到一个数字（用户与对账都靠名字认任务）。
        assert names == {
            "rzyl-retention-sweep",
            "rzyl-embedding-backfill",
            "rzyl-window-flush",
            "rzyl-dead-letter-retry",
            "rzyl-daily-push",
            "rzyl-startup-reconcile",
        }
    finally:
        await runtime.stop()
    assert runtime.background_tasks == ()


async def test_startup_reconcile_processes_uncovered_messages_without_blocking_start(
    tmp_path: Path, clock: _Clock
) -> None:
    gap_ids = await _seed_covered_gap_covered(tmp_path)
    runtime = _runtime(tmp_path, clock, window_message_limit=30)

    await runtime.start(run_background_tasks=True)
    try:
        task = next(
            task
            for task in runtime.background_tasks
            if task.get_name() == "rzyl-startup-reconcile"
        )
        await asyncio.wait_for(asyncio.shield(task), timeout=5)

        memories = await runtime.repository.list_group_memories(group_id=GROUP)
        assert len(memories) == 1
        assert memories[0].evidence == [gap_ids[0]]
    finally:
        await runtime.stop()


async def test_startup_reconcile_is_skipped_when_disabled(
    tmp_path: Path, clock: _Clock
) -> None:
    await _seed_covered_gap_covered(tmp_path)
    runtime = _runtime(tmp_path, clock, reconcile_on_startup=False)

    await runtime.start(run_background_tasks=True)
    try:
        await asyncio.sleep(0.05)
        # 开关关掉：没有对账任务，五个常驻循环照常（里程碑 3 起多了每日推送那个）。
        assert "rzyl-startup-reconcile" not in {
            task.get_name() for task in runtime.background_tasks
        }
        assert len(runtime.background_tasks) == 5
        assert await runtime.repository.list_group_memories(group_id=GROUP) == []
    finally:
        await runtime.stop()


async def test_reconcile_does_not_run_without_background_tasks(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_covered_gap_covered(tmp_path)

        # 测试 / 脚本模式不自动跑：库里那些空隙消息原封不动（场景只播了两个覆盖窗口）。
        assert runtime.background_tasks == ()
        assert await runtime.repository.list_group_memories(group_id=GROUP) == []
        assert len(await runtime.repository.list_windows_by_status(WindowStatus.DONE)) == 2

        # 但方法本身可以被手动调用。
        report = await runtime.reconcile_uncovered_messages()
        assert report.message_count == 3
    finally:
        await runtime.stop()
