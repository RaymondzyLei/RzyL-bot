"""每日推送与推送前的强制关窗（故事 33-36 + 里程碑 3 的推送约束）。

seam 是 Runtime：注入假模型与假时钟，投递器通过 ``set_report_sender`` 登记成一段记录
用的假函数（真的发送是插件的事，core 里只需要一个可替换的接收端）。断言的是：到点才推、
推送前把未满窗口关掉、投递失败在当天重试、没投递器要说出来。
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import Category, MemoryStatus
from rzyl_core.llm import ChatResult, ChatUsage, NullEmbedding
from rzyl_core.llm.fakes import FakeChatModel
from rzyl_core.memory import DailyReport, schedule_push_at
from rzyl_core.memory.push import set_report_sender
from rzyl_core.runtime import PUSH_STARTUP_GRACE_SECONDS, Runtime
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
UTC = timezone.utc
SHANGHAI = "Asia/Shanghai"
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

GROUP = 111
USER = 10001


def _item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "category": "knowledge",
        "statement": "一条结论",
        "detail": None,
        "confidence": 0.9,
        "evidence": [1],
        "person_refs": [],
        "occurred_at": None,
        "supersedes": None,
    }
    item.update(overrides)
    return item


def _reply(items: list[dict[str, object]]) -> ChatResult:
    return ChatResult(
        text=json.dumps(items, ensure_ascii=False),
        usage=ChatUsage(input_tokens=100, output_tokens=40),
        model="scripted",
    )


#: 一个窗口的提取结果：一条高置信度的「事务」+ 一条低置信度的「知识」。
#: 两条的 evidence 都指向序号 1，所以单条消息的窗口也能通过校验。
SCRIPT = [
    _reply(
        [
            _item(category="event", statement="实验改到周五下午三点", confidence=0.95),
            _item(statement="虹鳟的脂肪线只有养得好才有", confidence=0.4),
        ]
    )
]


class _Clock:
    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, moment: datetime) -> None:
        self.now = moment


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _runtime(tmp_path: Path, clock: _Clock, **settings: object) -> Runtime:
    return Runtime(
        chat_model=FakeChatModel(list(SCRIPT)),
        embedding_model=NullEmbedding(),
        settings=_settings(
            push_hour=22,
            push_minute=0,
            timezone=SHANGHAI,
            window_message_limit=20,
            **settings,
        ),
        clock=clock,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )


@pytest.fixture(autouse=True)
def _isolated_sender() -> Generator[None, None, None]:
    """投递器是进程级登记处，串味会让用例互相影响，所以每个用例前后都清一次。"""
    set_report_sender(None)
    yield
    set_report_sender(None)


async def _seed_buffered_message(runtime: Runtime) -> None:
    """灌一条消息但**不**让它成窗：留在内存缓冲里，正是推送前要补关的那种。"""
    await runtime.ingest(
        group_id=GROUP, user_id=USER, text="实验改到周五下午三点", sent_at=BEGIN
    )


def _recorder(sink: list[DailyReport] | None = None):
    """造一个记录用的假投递器。"""
    delivered: list[DailyReport] = sink if sink is not None else []

    async def _deliver(report: DailyReport) -> None:
        delivered.append(report)

    return _deliver, delivered


# —— 首次排班的宽限 ——


def test_startup_shortly_after_the_hour_still_pushes_today() -> None:
    """22:00:15 启动时严格的下一次是明天，那一天就整份不推了——所以宽限内立刻补推。"""
    now = datetime(2026, 10, 7, 22, 0, 15, tzinfo=CST)

    moment = schedule_push_at(
        now, hour=22, minute=0, tz=timezone(timedelta(hours=8)),
        startup_grace_seconds=PUSH_STARTUP_GRACE_SECONDS,
    )

    assert moment == now


def test_startup_hours_after_the_hour_waits_for_tomorrow() -> None:
    now = datetime(2026, 10, 7, 23, 30, tzinfo=CST)

    moment = schedule_push_at(
        now, hour=22, minute=0, tz=timezone(timedelta(hours=8)),
        startup_grace_seconds=PUSH_STARTUP_GRACE_SECONDS,
    )

    assert moment == datetime(2026, 10, 8, 22, 0, tzinfo=CST)


def test_startup_before_the_hour_waits_for_today() -> None:
    now = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

    moment = schedule_push_at(
        now, hour=22, minute=0, tz=timezone(timedelta(hours=8)),
        startup_grace_seconds=PUSH_STARTUP_GRACE_SECONDS,
    )

    assert moment == datetime(2026, 10, 7, 22, 0, tzinfo=CST)


# —— 强制关窗 ——


async def test_flush_pending_closes_buffers_that_flush_expired_leaves_alone(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_buffered_message(runtime)

        assert await runtime.flush_expired() == ()  # 没到点，定时刷新不管它
        assert await runtime.flush_pending() == 1  # 强制关窗才关得掉
        assert await runtime.flush_pending() == 0  # 关过就没了

        assert len(await runtime.repository.list_group_memories(group_id=GROUP)) == 2
    finally:
        await runtime.stop()


async def test_flush_pending_can_target_one_group(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_buffered_message(runtime)
        await runtime.ingest(group_id=222, user_id=USER, text="别的群", sent_at=BEGIN)

        assert await runtime.flush_pending(222) == 1
        assert runtime._require_assembler().buffered_count(GROUP) == 1  # pyright: ignore[reportPrivateUsage]
    finally:
        await runtime.stop()


# —— 日报内容 ——


async def test_build_daily_report_flushes_first_so_today_is_not_missed(
    tmp_path: Path, clock: _Clock
) -> None:
    """窗口最长能攒到 ``window_max_minutes``；不先关窗，日报会漏掉刚发生的事。"""
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_buffered_message(runtime)

        report = await runtime.build_daily_report()

        assert report.flushed_windows == 1
        assert report.total == 2
        assert "实验改到周五下午三点" in report.text
    finally:
        await runtime.stop()


async def test_daily_report_lists_only_entries_at_or_above_the_threshold(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_buffered_message(runtime)

        report = await runtime.build_daily_report()

        assert "实验改到周五下午三点" in report.text
        assert "虹鳟的脂肪线只有养得好才有" not in report.text
        assert (report.listed, report.total, report.low_confidence) == (1, 2, 1)
    finally:
        await runtime.stop()


async def test_daily_report_excludes_suspect_duplicates(tmp_path: Path, clock: _Clock) -> None:
    """故事 25：疑似重复保留但不推送。"""
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        window = await runtime.repository.add_window(
            group_id=GROUP, started_at=BEGIN, ended_at=BEGIN, message_count=1,
            prompt_version="v2",
        )
        for statement, status in (
            ("疑似重复的那条", MemoryStatus.SUSPECT_DUPLICATE),
            ("正常的那条", MemoryStatus.ACTIVE),
        ):
            await runtime.repository.add_memory(
                window_id=int(window.id),
                group_id=GROUP,
                category=Category.KNOWLEDGE,
                statement=statement,
                confidence=0.99,
                prompt_version="v2",
                status=status,
            )

        report = await runtime.build_daily_report()

        assert "正常的那条" in report.text
        assert "疑似重复的那条" not in report.text
        assert report.total == 1
    finally:
        await runtime.stop()


async def test_daily_report_on_a_quiet_day_still_says_something(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        report = await runtime.build_daily_report()

        assert report.total == 0
        assert "今天没有值得记的。" in report.text
    finally:
        await runtime.stop()


async def test_daily_report_covers_the_local_day_not_the_utc_day(
    tmp_path: Path, clock: _Clock
) -> None:
    """两个时刻在 UTC 下同属 10-07，但上海时间分属 10-07 与 10-08。"""
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        repo = runtime.repository
        for moment, statement in (
            (datetime(2026, 10, 7, 15, 30, tzinfo=UTC), "上海时间的今天"),
            (datetime(2026, 10, 7, 16, 30, tzinfo=UTC), "上海时间的明天"),
            (datetime(2026, 10, 6, 10, 0, tzinfo=UTC), "上海时间的昨天"),
        ):
            repo.clock = lambda moment=moment: moment  # pyright: ignore[reportUnknownLambdaType]
            await repo.add_memory(
                group_id=GROUP, category=Category.KNOWLEDGE, statement=statement,
                confidence=0.99, prompt_version="v2",
            )

        clock.set(datetime(2026, 10, 7, 15, 0, tzinfo=UTC))  # 上海 10-07 23:00
        report = await runtime.build_daily_report()

        assert "上海时间的今天" in report.text
        assert "上海时间的明天" not in report.text
        assert "上海时间的昨天" not in report.text
        assert report.total == 1
    finally:
        await runtime.stop()


# —— 排程与投递 ——


async def test_push_does_not_fire_before_the_hour(tmp_path: Path, clock: _Clock) -> None:
    deliver, delivered = _recorder()
    set_report_sender(deliver)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 21, 59, tzinfo=CST))

        assert await runtime.push_daily_report_if_due() is None
        assert delivered == []
    finally:
        await runtime.stop()


async def test_push_fires_at_the_hour_and_hands_the_report_to_the_sender(
    tmp_path: Path, clock: _Clock
) -> None:
    deliver, delivered = _recorder()
    set_report_sender(deliver)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await _seed_buffered_message(runtime)
        clock.set(datetime(2026, 10, 7, 21, 30, tzinfo=CST))
        assert await runtime.push_daily_report_if_due() is None  # 排班，还没到点

        clock.set(datetime(2026, 10, 7, 22, 0, tzinfo=CST))
        report = await runtime.push_daily_report_if_due()

        assert report is not None
        assert delivered == [report]
        assert report.flushed_windows == 1  # 推送前补关了窗口
        assert "实验改到周五下午三点" in delivered[0].text
        # 同一天不会推第二次。
        clock.set(datetime(2026, 10, 7, 22, 1, tzinfo=CST))
        assert await runtime.push_daily_report_if_due() is None
        assert len(delivered) == 1
    finally:
        await runtime.stop()


async def test_push_is_skipped_entirely_when_disabled(tmp_path: Path, clock: _Clock) -> None:
    deliver, delivered = _recorder()
    set_report_sender(deliver)
    runtime = _runtime(tmp_path, clock, push_enabled=False)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 22, 0, tzinfo=CST))

        assert await runtime.push_daily_report_if_due() is None
        assert delivered == []
    finally:
        await runtime.stop()


async def test_push_warns_when_nothing_can_deliver_the_report(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """「生成了但没发出去」必须看得见——否则用户以为当天真的没有条目。"""
    set_report_sender(None)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 22, 0, tzinfo=CST))

        with caplog.at_level("WARNING", logger="rzyl_core.runtime"):
            report = await runtime.push_daily_report_if_due()

        assert report is not None
        assert any("没登记投递器" in record.getMessage() for record in caplog.records)
    finally:
        await runtime.stop()


async def test_push_retries_within_the_same_day_after_a_delivery_failure(
    tmp_path: Path, clock: _Clock
) -> None:
    """最常见的失败原因是这一刻 QQ 没连上，十分钟后可能就好了——不该等到明天。"""
    attempts: list[DailyReport] = []

    async def flaky(report: DailyReport) -> None:
        attempts.append(report)
        if len(attempts) == 1:
            raise RuntimeError("NapCat 没连上")

    set_report_sender(flaky)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 21, 30, tzinfo=CST))
        await runtime.push_daily_report_if_due()
        clock.set(datetime(2026, 10, 7, 22, 0, tzinfo=CST))
        await runtime.push_daily_report_if_due()

        # 十分钟内不再打搅；到点后重试并成功。
        clock.set(datetime(2026, 10, 7, 22, 5, tzinfo=CST))
        assert await runtime.push_daily_report_if_due() is None
        clock.set(datetime(2026, 10, 7, 22, 10, tzinfo=CST))
        await runtime.push_daily_report_if_due()

        assert len(attempts) == 2
    finally:
        await runtime.stop()


async def test_push_gives_up_after_the_attempt_cap(tmp_path: Path, clock: _Clock) -> None:
    attempts: list[DailyReport] = []

    async def always_fails(report: DailyReport) -> None:
        attempts.append(report)
        raise RuntimeError("NapCat 一直没连上")

    set_report_sender(always_fails)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 21, 30, tzinfo=CST))
        await runtime.push_daily_report_if_due()
        for minute in (0, 10, 20, 30, 40):
            clock.set(datetime(2026, 10, 7, 22, minute, tzinfo=CST))
            await runtime.push_daily_report_if_due()

        # 到达上限后放弃，不再整夜每十分钟刷一条异常。
        assert len(attempts) == 3
    finally:
        await runtime.stop()


async def test_push_reopens_a_fresh_schedule_on_the_next_day(
    tmp_path: Path, clock: _Clock
) -> None:
    deliver, delivered = _recorder()
    set_report_sender(deliver)
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        clock.set(datetime(2026, 10, 7, 21, 30, tzinfo=CST))
        await runtime.push_daily_report_if_due()
        clock.set(datetime(2026, 10, 7, 22, 0, tzinfo=CST))
        await runtime.push_daily_report_if_due()

        clock.set(datetime(2026, 10, 8, 22, 0, tzinfo=CST))
        await runtime.push_daily_report_if_due()

        assert [report.day.isoformat() for report in delivered] == ["2026-10-07", "2026-10-08"]
    finally:
        await runtime.stop()
