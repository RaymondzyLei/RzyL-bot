"""带版本号的手写 DDL。

此处是**建表的唯一权威**：表的形状写死在下面的语句里，不靠 ``metadata.create_all``
生成，因为 FTS5 虚表与同步触发器表达不出来。模型（``models.py``）只是给这些列套一层
带类型的映射，两边必须一致。

版本号记在 ``schema_version`` 表里。表结构变更先在这里追加带新版本号的语句，不上
Alembic——脚手架阶段的迁移不需要那么重的工具。所有语句都是 ``IF NOT EXISTS``，重复
执行幂等。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

SCHEMA_VERSION = 1
"""当前表结构版本。改了下面的 DDL 就要 +1。"""

_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS message (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id            INTEGER NOT NULL,
        user_id             INTEGER NOT NULL,
        nickname            TEXT,
        card                TEXT,
        sent_at             TEXT    NOT NULL,
        text                TEXT    NOT NULL DEFAULT '',
        segments_json       TEXT    NOT NULL DEFAULT '[]',
        platform_message_id INTEGER,
        dedupe_hash         TEXT    NOT NULL DEFAULT '',
        created_at          TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_message_group_sent_at ON message (group_id, sent_at)",
    "CREATE INDEX IF NOT EXISTS ix_message_dedupe_hash ON message (dedupe_hash)",
    """
    CREATE TABLE IF NOT EXISTS window (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id       INTEGER NOT NULL,
        started_at     TEXT    NOT NULL,
        ended_at       TEXT,
        message_count  INTEGER NOT NULL DEFAULT 0,
        status         TEXT    NOT NULL DEFAULT 'pending',
        prompt_version TEXT,
        retry_count    INTEGER NOT NULL DEFAULT 0,
        error          TEXT,
        created_at     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_window_group_status ON window (group_id, status)",
    """
    CREATE TABLE IF NOT EXISTS memory (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        window_id        INTEGER REFERENCES window(id),
        group_id         INTEGER NOT NULL,
        category         TEXT    NOT NULL,
        statement        TEXT    NOT NULL,
        detail           TEXT,
        confidence       REAL    NOT NULL,
        evidence_json    TEXT    NOT NULL DEFAULT '[]',
        person_refs_json TEXT    NOT NULL DEFAULT '[]',
        occurred_at      TEXT,
        prompt_version   TEXT    NOT NULL,
        dedupe_hash      TEXT    NOT NULL DEFAULT '',
        embedding        BLOB,
        status           TEXT    NOT NULL DEFAULT 'active',
        superseded_by    INTEGER REFERENCES memory(id),
        model            TEXT,
        tokens_in        INTEGER,
        tokens_out       INTEGER,
        cost             REAL,
        created_at       TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_memory_group_created ON memory (group_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_memory_category ON memory (category)",
    "CREATE INDEX IF NOT EXISTS ix_memory_dedupe_hash ON memory (dedupe_hash)",
    """
    CREATE TABLE IF NOT EXISTS llm_call (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        purpose    TEXT    NOT NULL,
        provider   TEXT,
        model      TEXT,
        tokens_in  INTEGER NOT NULL DEFAULT 0,
        tokens_out INTEGER NOT NULL DEFAULT 0,
        cost       REAL    NOT NULL DEFAULT 0,
        latency_ms INTEGER,
        success    INTEGER NOT NULL DEFAULT 1,
        error      TEXT,
        created_at TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT    NOT NULL,
        memory_id  INTEGER REFERENCES memory(id),
        group_id   INTEGER,
        note       TEXT,
        created_at TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS group_setting (
        group_id   INTEGER PRIMARY KEY,
        enabled    INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    )
    """,
)

_FTS: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        statement,
        detail,
        content='memory',
        content_rowid='id',
        tokenize='trigram'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory BEGIN
        INSERT INTO memory_fts (rowid, statement, detail)
        VALUES (new.id, new.statement, new.detail);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory BEGIN
        INSERT INTO memory_fts (memory_fts, rowid, statement, detail)
        VALUES ('delete', old.id, old.statement, old.detail);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory BEGIN
        INSERT INTO memory_fts (memory_fts, rowid, statement, detail)
        VALUES ('delete', old.id, old.statement, old.detail);
        INSERT INTO memory_fts (rowid, statement, detail)
        VALUES (new.id, new.statement, new.detail);
    END
    """,
)

SCHEMA_STATEMENTS: tuple[str, ...] = _TABLES + _FTS


async def apply_schema(engine: AsyncEngine) -> None:
    """建表、装触发器、记录版本号；重复调用幂等。"""
    async with engine.begin() as conn:
        for statement in SCHEMA_STATEMENTS:
            await conn.exec_driver_sql(statement)
        # 外部内容表的索引在触发器安装前可能是空的，重建一次保证与 memory 表一致。
        await conn.exec_driver_sql("INSERT INTO memory_fts (memory_fts) VALUES ('rebuild')")
        count = (await conn.exec_driver_sql("SELECT COUNT(*) FROM schema_version")).scalar()
        if not count:
            await conn.exec_driver_sql("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
