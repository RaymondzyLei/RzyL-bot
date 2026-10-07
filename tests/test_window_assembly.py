"""窗口组装的行为测试（工单 #4）。

只断言**外部可观察的结构**：一个群的消息流进来，攒成窗口后拿到三段（上一窗口尾部、
上一窗口已记条目摘要、本窗口消息带窗口内序号）与「序号 → 真实消息编号」的映射。
不碰内部缓冲、不断言内部调用顺序。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import Category, Repository
from rzyl_core.pipeline import WindowMessage
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)


def _message(
    message_id: int,
    *,
    group_id: int = 111,
    text: str | None = None,
    minutes: int = 0,
    seconds: int = 0,
) -> WindowMessage:
    return WindowMessage(
        message_id=message_id,
        group_id=group_id,
        user_id=10000 + message_id,
        text=text if text is not None else f"第{message_id}条",
        sent_at=BEGIN + timedelta(minutes=minutes, seconds=seconds),
        nickname=f"用户{message_id}",
    )


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    yield instance
    await instance.close()


def _settings(
    *,
    limit: int = 30,
    minutes: int = 5,
    min_messages: int = 5,
    max_minutes: int = 60,
) -> Settings:
    """窗口参数可调、其余用默认值的设置对象。"""
    # ``_env_file`` 是 pydantic-settings 的运行时开关，类型签名里没有，故忽略告警。
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        window_message_limit=limit,
        window_minutes=minutes,
        window_min_messages=min_messages,
        window_max_minutes=max_minutes,
    )


class _Clock:
    """可手动推进的时钟：测试不必真等 5 分钟。"""

    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + timedelta(minutes=minutes)



def test_assembled_window_numbers_messages_and_maps_back_to_real_ids() -> None:
    from rzyl_core.pipeline import assemble_window

    window = assemble_window(
        group_id=111,
        messages=[_message(101), _message(102), _message(103)],
    )

    assert [line.sequence for line in window.messages] == [1, 2, 3]
    assert window.sequence_map == {1: 101, 2: 102, 3: 103}
    assert window.resolve_evidence([3, 1]) == [103, 101]


async def test_message_limit_closes_the_window_and_clears_the_buffer(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(repository=repo, clock=_Clock(), settings=_settings(limit=3))

    assert await assembler.add(_message(1)) == []
    assert await assembler.add(_message(2)) == []

    windows = await assembler.add(_message(3))

    assert len(windows) == 1
    assert windows[0].group_id == 111
    assert [line.message_id for line in windows[0].messages] == [1, 2, 3]
    assert assembler.buffered_count(111) == 0


async def test_groups_buffer_independently(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(repository=repo, clock=_Clock(), settings=_settings(limit=3))

    await assembler.add(_message(1, group_id=111))
    await assembler.add(_message(2, group_id=111))
    await assembler.add(_message(10, group_id=222))
    await assembler.add(_message(11, group_id=222))
    windows = await assembler.add(_message(12, group_id=222))

    assert [window.group_id for window in windows] == [222]
    assert assembler.buffered_count(111) == 2
    assert assembler.buffered_count(222) == 0
    assert assembler.pending_groups() == [111]


async def test_time_limit_closes_the_window_before_the_new_message(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    clock = _Clock()
    assembler = WindowAssembler(repository=repo, clock=clock, settings=_settings(limit=30))

    # 时间规则要求缓冲里至少有 window_min_messages 条，先攒满 5 条。
    for index in range(5):
        assert await assembler.add(_message(index + 1, minutes=index)) == []
    clock.advance(5)

    windows = await assembler.add(_message(6, minutes=5))

    assert len(windows) == 1
    assert [line.message_id for line in windows[0].messages] == [1, 2, 3, 4, 5]
    assert assembler.buffered_count(111) == 1


async def test_flush_expired_closes_a_quiet_window(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    clock = _Clock()
    assembler = WindowAssembler(repository=repo, clock=clock, settings=_settings(limit=30))

    # 条数够 window_min_messages，时间到点才成窗。
    for index in range(5):
        await assembler.add(_message(index + 1, minutes=index))
    assert await assembler.flush_expired() == []

    clock.advance(5)
    windows = await assembler.flush_expired()

    assert [window.message_count for window in windows] == [5]
    assert assembler.buffered_count(111) == 0
    assert await assembler.flush_expired() == []


async def test_sparse_traffic_keeps_buffering_until_the_minimum_is_reached(
    repo: Repository,
) -> None:
    """稀疏流量：每 30 分钟才来一条，够 window_min_messages 之前都不成窗。

    把 window_max_minutes 调大，把「最小条数」这条规则单独隔离出来测。
    """
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(
        repository=repo,
        clock=_Clock(),
        settings=_settings(limit=30, minutes=5, min_messages=5, max_minutes=240),
    )

    # 每 30 分钟一条，前 5 条都留在同一个缓冲里：跨度早早超过 window_minutes，
    # 但条数不够，所以一条都不成窗。
    windows = []
    for index in range(5):
        windows.extend(await assembler.add(_message(index + 1, minutes=30 * index)))

    assert windows == []
    assert assembler.buffered_count(111) == 5

    # 第 6 条到达时，缓冲已有 5 条（够最小条数）、跨度 150 分钟（到时间点）→ 成窗。
    windows = await assembler.add(_message(6, minutes=150))

    assert len(windows) == 1
    assert [line.message_id for line in windows[0].messages] == [1, 2, 3, 4, 5]
    assert assembler.buffered_count(111) == 1


async def test_max_minutes_fallback_flushes_a_lonely_message(repo: Repository) -> None:
    """兜底：只来一条就安静，window_max_minutes 之前不收、之后无条件收。"""
    from rzyl_core.pipeline import WindowAssembler

    clock = _Clock()
    assembler = WindowAssembler(
        repository=repo,
        clock=clock,
        settings=_settings(limit=30, minutes=5, min_messages=5, max_minutes=60),
    )

    await assembler.add(_message(1, minutes=0))

    # 59 分钟：时间规则因条数不够不触发，兜底也还差一点。
    clock.advance(59)
    assert await assembler.flush_expired() == []
    assert assembler.buffered_count(111) == 1

    # 到 60 分钟：兜底无条件关窗，哪怕只有 1 条。
    clock.advance(1)
    windows = await assembler.flush_expired()

    assert [window.message_count for window in windows] == [1]
    assert [line.message_id for line in windows[0].messages] == [1]
    assert assembler.buffered_count(111) == 0


async def test_message_limit_wins_over_the_minimum_within_an_hour(repo: Repository) -> None:
    """上限优先：一小时内猛灌 30 条，条数上限先触发，不受最小条数影响。"""
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(
        repository=repo,
        clock=_Clock(),
        settings=_settings(limit=30, minutes=5, min_messages=5, max_minutes=60),
    )

    # 30 条压在 5 分钟内（每 8 秒一条）：时间规则来不及触发，只有条数上限会先到。
    windows = []
    for index in range(30):
        windows.extend(await assembler.add(_message(index + 1, seconds=8 * index)))

    assert len(windows) == 1
    assert windows[0].message_count == 30
    assert assembler.buffered_count(111) == 0


async def test_windowing_follows_message_times_not_a_frozen_clock(repo: Repository) -> None:
    """掉线回补回归：时钟停住不动，一批间隔很小的历史消息仍按各自时间切窗。

    旧实现拿「当下时钟 − 缓冲首条时间」判超时：回补时时钟是当下、消息却是历史，
    每条都远超窗口，于是每条各自成窗（回补彻底失效）。成窗改为看**消息自身时间**后，
    这 9 条各隔 1 分钟的消息应切成 [5 条, 4 条] 两个窗口，而不是 9 个单条窗口。
    """
    from rzyl_core.pipeline import WindowAssembler

    frozen_now = _Clock(BEGIN + timedelta(days=1))
    assembler = WindowAssembler(
        repository=repo, clock=frozen_now, settings=_settings(limit=100, minutes=5)
    )

    windows = []
    for minute in range(9):
        windows.extend(await assembler.add(_message(minute + 1, minutes=minute)))
    remainder = await assembler.flush_group(111)
    if remainder is not None:
        windows.append(remainder)

    assert [window.message_count for window in windows] == [5, 4]
    assert assembler.buffered_count(111) == 0


async def test_flush_group_closes_a_partial_window(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(repository=repo, clock=_Clock(), settings=_settings(limit=30))

    assert await assembler.flush_group(999) is None

    await assembler.add(_message(1))
    window = await assembler.flush_group(111)

    assert window is not None
    assert window.message_count == 1
    assert await assembler.flush_group(111) is None


async def test_previous_tail_is_the_tail_of_the_window_just_closed(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(
        repository=repo,
        clock=_Clock(),
        settings=_settings(limit=2),
        previous_tail_size=2,
    )

    first = await assembler.add(_message(1))
    assert first == []
    first_windows = await assembler.add(_message(2))
    assert await assembler.add(_message(3)) == []
    second_windows = await assembler.add(_message(4))

    assert first_windows[0].previous_tail == ()
    assert [line.message_id for line in second_windows[0].previous_tail] == [1, 2]
    assert [line.message_id for line in second_windows[0].messages] == [3, 4]


async def test_remembered_summary_is_empty_when_nothing_was_remembered(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    assembler = WindowAssembler(repository=repo, clock=_Clock(), settings=_settings(limit=1))

    windows = await assembler.add(_message(1))

    assert windows[0].remembered == ()
    assert windows[0].remembered_summary == ""


async def test_remembered_summary_comes_from_the_same_group_only(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    older = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="实验课改到周三",
        confidence=0.9,
        prompt_version="v1",
    )
    newer = await repo.add_memory(
        group_id=111,
        category=Category.RESOURCE,
        statement="群友分享了讲义网盘链接",
        confidence=0.9,
        prompt_version="v1",
    )
    await repo.add_memory(
        group_id=222,
        category=Category.RESOURCE,
        statement="别的群的资料",
        confidence=0.9,
        prompt_version="v1",
    )
    assembler = WindowAssembler(repository=repo, clock=_Clock(), settings=_settings(limit=1))

    windows = await assembler.add(_message(1, group_id=111))

    window = windows[0]
    assert {memory.id for memory in window.remembered} == {older.id, newer.id}
    assert f"[#{older.id}]" in window.remembered_summary
    assert f"[#{newer.id}]" in window.remembered_summary
    assert "别的群" not in window.remembered_summary


async def test_remembered_limit_keeps_only_the_most_recent(repo: Repository) -> None:
    from rzyl_core.pipeline import WindowAssembler

    for index in range(3):
        await repo.add_memory(
            group_id=111,
            category=Category.KNOWLEDGE,
            statement=f"第{index}条已记",
            confidence=0.9,
            prompt_version="v1",
        )
    assembler = WindowAssembler(
        repository=repo,
        clock=_Clock(),
        settings=_settings(limit=1),
        remembered_limit=1,
    )

    windows = await assembler.add(_message(1))

    assert len(windows[0].remembered) == 1
    assert "第2条已记" in windows[0].remembered_summary


async def test_assembler_takes_repository_messages_and_maps_their_real_ids(repo: Repository) -> None:
    """仓储的 ``Message`` 天然满足输入形状；映射里的编号就是库里的自增 id。"""
    from rzyl_core.pipeline import WindowAssembler

    stored = [
        await repo.add_message(
            group_id=111,
            user_id=222,
            text=f"第{index}条",
            sent_at=BEGIN + timedelta(minutes=index),
        )
        for index in range(3)
    ]
    assembler = WindowAssembler(
        repository=repo,
        clock=_Clock(BEGIN + timedelta(minutes=2)),
        settings=_settings(limit=3),
    )

    windows = [window for message in stored for window in await assembler.add(message)]

    window = windows[0]
    assert window.sequence_map == {1: stored[0].id, 2: stored[1].id, 3: stored[2].id}
    assert window.started_at == stored[0].sent_at
    assert window.ended_at == stored[2].sent_at


def test_unknown_evidence_sequence_is_rejected() -> None:
    from rzyl_core.pipeline import UnknownSequenceError, assemble_window

    window = assemble_window(group_id=111, messages=[_message(101)])

    with pytest.raises(UnknownSequenceError):
        window.resolve_evidence([1, 2])


def test_assembling_an_empty_window_is_a_programming_error() -> None:
    from rzyl_core.pipeline import assemble_window

    with pytest.raises(ValueError):
        assemble_window(group_id=111, messages=[])



