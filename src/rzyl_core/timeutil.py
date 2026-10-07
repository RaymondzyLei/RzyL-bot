"""内部公共工具：默认时间源与自然日换算。

``utcnow`` 是仓储与 Runtime 共用的默认时钟——原先两处各写了一遍逐字相同的实现，
现收敛到一处，时间语义（带时区、取 UTC）只在这里定义一次。

「今天」这类**按人所在时区的自然日**的说法，也在这一处换算：库里存的一律是 UTC，
只有展示与判定「当日新增」时才折回本地时区。两处各算一遍必然漂移，所以只留一个实现。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = ["local_day_bounds", "resolve_timezone", "utcnow"]


def utcnow() -> datetime:
    """默认时间源：当前 UTC 时刻（带时区）。"""
    return datetime.now(timezone.utc)


def resolve_timezone(name: str) -> tzinfo:
    """把时区名解析成 ``tzinfo``；认不出来就退回 UTC，不因此中断调用方。

    配置里写错时区名是可能的（手写 ``.env``），为此让整条管道报错太脆——退回 UTC
    最多让「今天」的边界差几小时，而且调用方仍能照常工作。
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def local_day_bounds(now: datetime, tz: tzinfo) -> tuple[datetime, datetime]:
    """``now`` 在 ``tz`` 时区里所在自然日的 ``[当天 00:00, 次日 00:00)``，换算回 UTC。

    返回的区间是**左闭右开**的：``since <= created_at < until``。用它去查库时不要再
    加减一秒——半开区间本身就不会把午夜那条消息算进两天。
    """
    local = now.astimezone(tz)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
