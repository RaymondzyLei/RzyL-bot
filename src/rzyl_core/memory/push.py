"""推送：算下一次该推的时刻，以及「谁去把这行字发出去」的登记处。

推送**算时刻**与**投递**必须分开，因为 core 不认识 QQ：它只会算出「现在该推了」并
渲染好文本，真正的发送是插件的事。两边的接口就是 :data:`ReportSender`。

登记走模块级函数（``set_report_sender`` / ``get_report_sender``），与 ``set_runtime``
同一套做法，理由也一样：插件在 ``load_plugins`` 时就已经执行，那时装配层还没调用
``set_runtime``；用一个模块级的登记处，插件在 import 期就能把投递器装好，不必和
``driver.on_startup`` 的注册顺序赛跑。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, tzinfo

from rzyl_core.memory.report import DailyReport

#: 投递器：拿到一份渲染好的日报，把它发给管理员。抛异常由调用方记日志，不影响循环。
ReportSender = Callable[[DailyReport], Awaitable[None]]

_report_sender: ReportSender | None = None


def set_report_sender(sender: ReportSender | None) -> None:
    """登记（或清空）日报投递器；由插件层调用。"""
    global _report_sender
    _report_sender = sender


def get_report_sender() -> ReportSender | None:
    """取已登记的投递器；没有登记时返回 ``None``。"""
    return _report_sender


def next_push_at(now: datetime, *, hour: int, minute: int, tz: tzinfo) -> datetime:
    """算 ``now`` 之后**下一次**推送时刻（返回 UTC 时刻），按 ``tz`` 的墙上时间定。

    今天该时刻还没到就返回今天，已经过了（或正好到）就返回明天同一钟点。用
    ``local.replace(...)`` 而不是构造 naive 时间，是为了让 ``tzinfo`` 自己处理夏令时
    这类偏移变化；加了时区之后 ``+timedelta(days=1)`` 保持的是**墙上钟点**不变，
    正是「每天 22:00」想要的语义。
    """
    local = now.astimezone(tz)
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local:
        candidate = candidate + timedelta(days=1)
    return candidate


def schedule_push_at(
    now: datetime, *, hour: int, minute: int, tz: tzinfo, startup_grace_seconds: int
) -> datetime:
    """**首次**排班用：正常给「下一次」时刻；若此刻就在今天的推送时刻之后一小段宽限内，
    就直接给``now``，也就是立刻补推。

    为什么需要宽限：推送循环每 :data:`~rzyl_core.runtime.PUSH_TICK_SECONDS` 秒才醒来
    一次，所以「22:00:15 才启动」这种情况里，严格的下一次已经变成明天——当天就整份
    不推了，而且**没有任何提示**。宁可在这种情况下多推一份（重启前刚推过的话会重复），
    也不要静默丢掉一整天的日报；宽限窗口只有十来分钟，重复的代价是一次可辨认的多余消息。
    """
    local = now.astimezone(tz)
    today = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if today <= local <= today + timedelta(seconds=startup_grace_seconds):
        return now
    return next_push_at(now, hour=hour, minute=minute, tz=tz)


__all__ = [
    "ReportSender",
    "get_report_sender",
    "next_push_at",
    "schedule_push_at",
    "set_report_sender",
]
