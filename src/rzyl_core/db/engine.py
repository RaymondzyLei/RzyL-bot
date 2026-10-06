"""异步引擎的构造与 SQLite 调参。

SQLite 开 WAL（读写不互相阻塞）、外键约束与忙等待超时。库是一个文件而不是内存库：
容器通过 bind mount 写进去，宿主机才读得到、备份得了。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import event, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def _sqlite_on_connect(dbapi_connection: Any, connection_record: Any) -> None:
    """连接建立时设置 PRAGMA。journal_mode 是库级持久设置，设一次即生效。"""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def open_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """按连接串建一个异步引擎；SQLite 文件库会自动补齐父目录。"""
    url = make_url(database_url)

    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(database_url, echo=echo, future=True)
    if url.get_backend_name() == "sqlite":
        event.listen(engine.sync_engine, "connect", _sqlite_on_connect)
    return engine
