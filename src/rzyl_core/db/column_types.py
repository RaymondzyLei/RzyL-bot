"""SQLAlchemy 自定义列类型。

SQLite 没有原生 datetime 与向量类型，这里补两个：

- :class:`TZDateTime` —— 存 ISO8601 字符串，**统一归一化到 UTC** 再落库。归一化让
  字符串的字典序等于时间先后，所以按时间范围的过滤可以直接比较列值。读回来一定是
  带时区的 ``datetime``；写入不带时区的时间是调用方错误，直接报错。
- 向量用 ``LargeBinary`` 存 float32，编解码见 :mod:`rzyl_core.db.vectors`。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator


class TZDateTime(TypeDecorator[datetime]):
    """带时区的 datetime，落库前归一化到 UTC。"""

    impl = String(40)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("时间必须带时区（naive datetime 不允许入库）")
        return value.astimezone(timezone.utc).isoformat()

    def process_result_value(self, value: str | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
