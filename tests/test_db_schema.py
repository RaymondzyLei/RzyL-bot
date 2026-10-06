"""数据库底座的行为测试（工单 #2）：建表、FTS、触发器、WAL、版本号。

seam 是 ``rzyl_core.db`` 的公开函数：给一个连接串就能建出可用的库。这里只从
``sqlite_master`` / ``PRAGMA`` 这些外部可观察的事实断言，不碰内部实现。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from rzyl_core.db import Category, WindowStatus, apply_schema, open_engine

BUSINESS_TABLES = {"message", "window", "memory", "llm_call", "feedback", "group_setting"}

MEMORY_COLUMNS = {
    "id",
    "window_id",
    "group_id",
    "category",
    "statement",
    "detail",
    "confidence",
    "evidence_json",
    "person_refs_json",
    "occurred_at",
    "prompt_version",
    "dedupe_hash",
    "embedding",
    "status",
    "superseded_by",
    "model",
    "tokens_in",
    "tokens_out",
    "cost",
    "created_at",
}


def _file_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}"


async def _names(engine: AsyncEngine, kind: str) -> set[str]:
    async with engine.connect() as conn:
        rows = (await conn.execute(text("SELECT name FROM sqlite_master WHERE type = :kind"), {"kind": kind})).scalars()
        return set(rows)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine, None]:
    instance = open_engine(_file_url(tmp_path))
    await apply_schema(instance)
    yield instance
    await instance.dispose()


async def test_schema_creates_six_business_tables_and_fts(engine: AsyncEngine) -> None:
    tables = await _names(engine, "table")
    assert BUSINESS_TABLES <= tables
    assert "memory_fts" in tables
    assert "schema_version" in tables


async def test_memory_table_has_every_required_column(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        rows = (await conn.execute(text("PRAGMA table_info('memory')"))).all()
    assert MEMORY_COLUMNS <= {row.name for row in rows}


async def test_fts_triggers_keep_index_in_sync(engine: AsyncEngine) -> None:
    triggers = await _names(engine, "trigger")
    assert {"memory_fts_ai", "memory_fts_ad", "memory_fts_au"} <= triggers


async def test_schema_creation_is_idempotent(engine: AsyncEngine) -> None:
    await apply_schema(engine)  # 第二次不应报错
    async with engine.connect() as conn:
        versions = (await conn.execute(text("SELECT version FROM schema_version"))).scalars().all()
    assert len(versions) == 1


async def test_database_is_wal_and_on_disk(engine: AsyncEngine, tmp_path: Path) -> None:
    async with engine.connect() as conn:
        journal_mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
    assert str(journal_mode).lower() == "wal"
    assert (tmp_path / "rzyl.db").is_file()


async def test_schema_version_is_recorded(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        version = (await conn.execute(text("SELECT version FROM schema_version"))).scalar()
    assert isinstance(version, int) and version >= 1


def test_category_enum_has_exactly_four_values_and_no_other() -> None:
    assert {item.value for item in Category} == {"resource", "knowledge", "event", "request"}
    assert not any(item.value in {"other", "misc", "unknown"} for item in Category)


def test_window_status_covers_pending_done_dead() -> None:
    assert {"pending", "done", "dead"} <= {item.value for item in WindowStatus}
