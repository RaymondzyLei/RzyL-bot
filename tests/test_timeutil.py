"""时间工具：自然日边界与「下一次推送时刻」。

两个都是纯函数，但一处偏差就足以让「今天」错一天、让日报晚一天，所以边界逐条断言：
本地午夜的这一秒算哪天、正好到点的那一秒算今天还是明天。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from rzyl_core.memory.push import next_push_at
from rzyl_core.timeutil import local_day_bounds, resolve_timezone, utcnow

UTC = timezone.utc
SHANGHAI = ZoneInfo("Asia/Shanghai")
CST = timezone(timedelta(hours=8))


def test_utcnow_is_aware_utc() -> None:
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_local_day_bounds_are_the_local_natural_day() -> None:
    # UTC 15:00 = 上海 23:00，属于 10-07 这一天；它的起止换成 UTC 是 10-06 16:00 起。
    start, end = local_day_bounds(datetime(2026, 10, 7, 15, 0, tzinfo=UTC), SHANGHAI)

    assert start == datetime(2026, 10, 6, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 10, 7, 16, 0, tzinfo=UTC)
    assert end - start == timedelta(days=1)


def test_local_day_bounds_put_local_midnight_in_the_new_day() -> None:
    # 上海 10-08 00:00（= UTC 10-07 16:00）已经是新的一天，且正好是前一天的右端点。
    start, end = local_day_bounds(datetime(2026, 10, 7, 16, 0, tzinfo=UTC), SHANGHAI)

    assert start == datetime(2026, 10, 7, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 10, 8, 16, 0, tzinfo=UTC)


def test_local_day_bounds_are_half_open_and_aware() -> None:
    start, end = local_day_bounds(datetime(2026, 10, 7, 15, 0, tzinfo=UTC), CST)

    assert start.tzinfo is UTC and end.tzinfo is UTC
    # 半开区间：右端点属于「第二天」，不属于这一天。
    next_start, _ = local_day_bounds(end, CST)
    assert next_start == end


def test_resolve_timezone_falls_back_to_utc_instead_of_raising() -> None:
    assert str(resolve_timezone("Asia/Shanghai")) == "Asia/Shanghai"
    assert resolve_timezone("这里没有这个时区") == UTC


def test_next_push_at_is_today_when_the_time_is_still_ahead() -> None:
    # 上海 13:00 → 今天 22:00（= UTC 14:00）
    moment = next_push_at(datetime(2026, 10, 7, 5, 0, tzinfo=UTC), hour=22, minute=0, tz=SHANGHAI)

    assert moment == datetime(2026, 10, 7, 14, 0, tzinfo=UTC)


def test_next_push_at_rolls_to_tomorrow_once_passed() -> None:
    # 上海 23:00，今天 22:00 已经过了。
    moment = next_push_at(datetime(2026, 10, 7, 15, 0, tzinfo=UTC), hour=22, minute=0, tz=SHANGHAI)

    assert moment == datetime(2026, 10, 8, 14, 0, tzinfo=UTC)


def test_next_push_at_at_the_exact_moment_rolls_to_tomorrow() -> None:
    # 正好到点（上海 22:00:00）算「已经到过」，下一次是明天——否则循环每轮都会再触发一次。
    moment = next_push_at(datetime(2026, 10, 7, 14, 0, tzinfo=UTC), hour=22, minute=0, tz=SHANGHAI)

    assert moment == datetime(2026, 10, 8, 14, 0, tzinfo=UTC)


def test_next_push_at_keeps_the_wall_clock_across_consecutive_days() -> None:
    moment = datetime(2026, 10, 7, 5, 0, tzinfo=UTC)
    for _ in range(3):
        moment = next_push_at(moment, hour=22, minute=0, tz=SHANGHAI)

    assert moment == datetime(2026, 10, 9, 14, 0, tzinfo=UTC)


def test_next_push_at_honours_the_minute_and_non_utc_offset() -> None:
    # 上海 09:10 时，「每天 09:30」应该在今天 09:30（= UTC 01:30）。
    moment = next_push_at(
        datetime(2026, 10, 7, 1, 10, tzinfo=UTC), hour=9, minute=30, tz=SHANGHAI
    )

    assert moment == datetime(2026, 10, 7, 1, 30, tzinfo=UTC)
