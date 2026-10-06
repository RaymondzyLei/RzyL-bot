"""内部公共工具：默认时间源。

``utcnow`` 是仓储与 Runtime 共用的默认时钟——原先两处各写了一遍逐字相同的实现，
现收敛到一处，时间语义（带时区、取 UTC）只在这里定义一次。
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """默认时间源：当前 UTC 时刻（带时区）。"""
    return datetime.now(timezone.utc)


__all__ = ["utcnow"]
