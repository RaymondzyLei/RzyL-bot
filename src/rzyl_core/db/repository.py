"""仓储层：对外唯一可读可写的持久化 API。

仓储是「对外可读的 API」，不只是写入口——后面每一步（窗口摘要、提取入库、检索、
推送）都靠它断言结果，所以读方法和写方法一样齐。构造只需要一个连接串：

    repo = await Repository.create("sqlite+aiosqlite:///data/rzyl.db")
    ...
    await repo.close()

时间源可注入（``clock``），测试窗口超时不用真等；默认取当前 UTC 时刻。所有写方法
自动补 ``created_at`` / ``updated_at``。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from .engine import open_engine
from .enums import Category, FeedbackKind, MemoryStatus, WindowStatus
from .models import Feedback, GroupSetting, LlmCall, Memory, Message, PersonRef, Window
from .schema import apply_schema
from .vectors import decode_vector, encode_vector


def _utcnow() -> datetime:
    """默认时间源：当前 UTC 时刻（带时区）。"""
    return datetime.now(timezone.utc)


def _require_aware(value: datetime | None, field: str) -> datetime | None:
    """在 API 边界挡住 naive datetime，报错比落库时才发现更清楚。"""
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} 必须带时区")
    return value


class Repository:
    """一个连接串背后的一整套读写操作。"""

    def __init__(self, engine: AsyncEngine, *, clock: Callable[[], datetime] | None = None) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.clock: Callable[[], datetime] = clock or _utcnow

    @classmethod
    async def create(
        cls,
        database_url: str,
        *,
        echo: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> "Repository":
        """按连接串建库（含建表）、返回仓储。建表幂等。"""
        engine = open_engine(database_url, echo=echo)
        await apply_schema(engine)
        return cls(engine, clock=clock)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def close(self) -> None:
        await self._engine.dispose()

    # —— 群消息 ——

    async def add_message(
        self,
        *,
        group_id: int,
        user_id: int,
        text: str,
        sent_at: datetime,
        nickname: str | None = None,
        card: str | None = None,
        segments_json: str = "[]",
        platform_message_id: int | None = None,
        dedupe_hash: str = "",
    ) -> Message:
        message = Message(
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            card=card,
            sent_at=_require_aware(sent_at, "sent_at"),
            text=text,
            segments_json=segments_json,
            platform_message_id=platform_message_id,
            dedupe_hash=dedupe_hash,
            created_at=self.clock(),
        )
        async with self._sessions() as session:
            session.add(message)
            await session.commit()
        return message

    # —— 处理窗口 ——

    async def add_window(
        self,
        *,
        group_id: int,
        started_at: datetime,
        ended_at: datetime | None = None,
        message_count: int = 0,
        status: WindowStatus = WindowStatus.PENDING,
        prompt_version: str | None = None,
        retry_count: int = 0,
        error: str | None = None,
    ) -> Window:
        now = self.clock()
        window = Window(
            group_id=group_id,
            started_at=_require_aware(started_at, "started_at"),
            ended_at=_require_aware(ended_at, "ended_at"),
            message_count=message_count,
            status=status,
            prompt_version=prompt_version,
            retry_count=retry_count,
            error=error,
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session:
            session.add(window)
            await session.commit()
        return window

    async def update_window_status(
        self,
        window_id: int,
        status: WindowStatus,
        *,
        retry_count: int | None = None,
        error: str | None = None,
    ) -> None:
        async with self._sessions() as session:
            window = await session.get(Window, window_id)
            if window is None:
                return
            window.status = status
            if retry_count is not None:
                window.retry_count = retry_count
            if error is not None:
                window.error = error
            window.updated_at = self.clock()
            await session.commit()

    async def get_window(self, window_id: int) -> Window | None:
        async with self._sessions() as session:
            return await session.get(Window, window_id)

    # —— 记忆条目 ——

    async def add_memory(
        self,
        *,
        group_id: int,
        category: Category,
        statement: str,
        confidence: float,
        prompt_version: str,
        window_id: int | None = None,
        detail: str | None = None,
        evidence: Sequence[int] = (),
        person_refs: Sequence[PersonRef] = (),
        occurred_at: datetime | None = None,
        dedupe_hash: str = "",
        embedding: Sequence[float] | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        superseded_by: int | None = None,
        model: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost: float | None = None,
    ) -> Memory:
        memory = Memory(
            window_id=window_id,
            group_id=group_id,
            category=category,
            statement=statement,
            detail=detail,
            confidence=confidence,
            evidence=list(evidence),
            person_refs=list(person_refs),
            occurred_at=_require_aware(occurred_at, "occurred_at"),
            prompt_version=prompt_version,
            dedupe_hash=dedupe_hash,
            embedding=encode_vector(list(embedding)) if embedding else None,
            status=status,
            superseded_by=superseded_by,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            created_at=self.clock(),
        )
        async with self._sessions() as session:
            session.add(memory)
            await session.commit()
        return memory

    async def get_memory(self, memory_id: int) -> Memory | None:
        async with self._sessions() as session:
            return await session.get(Memory, memory_id)

    async def get_source_messages(self, memory_id: int) -> list[Message]:
        """取某条目引用的来源消息，按发送时间排序。"""
        memory = await self.get_memory(memory_id)
        if memory is None:
            return []
        ids = list(memory.evidence or [])
        if not ids:
            return []
        async with self._sessions() as session:
            result = await session.execute(
                select(Message).where(Message.id.in_(ids)).order_by(Message.sent_at, Message.id)
            )
            return list(result.scalars().all())

    # —— 检索 ——

    async def search_memories(
        self,
        query: str,
        *,
        category: Category | None = None,
        group_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """FTS5 关键词检索，可按类别 / 群 / 时间范围过滤。

        trigram 分词对中文按子串匹配，但匹配词至少 3 个字；两字及以下的词（如「论文」）
        走 LIKE 兜底。多个空格分隔的词之间是「与」的关系。
        """
        terms = [term for term in query.split() if term]
        if not terms:
            return []

        long_terms = [term for term in terms if len(term) >= 3]
        short_terms = [term for term in terms if len(term) < 3]

        statement = select(Memory)
        if long_terms:
            match_expression = " AND ".join('"' + term.replace('"', '""') + '"' for term in long_terms)
            statement = statement.where(
                text(
                    "memory.id IN (SELECT rowid FROM memory_fts WHERE memory_fts MATCH :fts_query)"
                ).bindparams(fts_query=match_expression)
            )
        for term in short_terms:
            pattern = f"%{term}%"
            statement = statement.where(or_(Memory.statement.like(pattern), Memory.detail.like(pattern)))
        if category is not None:
            statement = statement.where(Memory.category == category)
        if group_id is not None:
            statement = statement.where(Memory.group_id == group_id)
        if since is not None:
            statement = statement.where(Memory.created_at >= since)
        if until is not None:
            statement = statement.where(Memory.created_at <= until)
        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(limit)

        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def recent_memories(
        self,
        *,
        group_id: int,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> list[Memory]:
        """取某群最近的条目，新的在前。默认排除过期 / 疑似重复的条目。"""
        statement = select(Memory).where(Memory.group_id == group_id)
        if not include_inactive:
            statement = statement.where(Memory.status == MemoryStatus.ACTIVE)
        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def list_embeddings(
        self,
        *,
        group_id: int | None = None,
        include_inactive: bool = False,
    ) -> list[tuple[int, list[float]]]:
        """取出全部已算好的向量，供暴力余弦使用。"""
        statement = select(Memory.id, Memory.embedding).where(func.length(Memory.embedding) > 0)
        if not include_inactive:
            statement = statement.where(Memory.status == MemoryStatus.ACTIVE)
        if group_id is not None:
            statement = statement.where(Memory.group_id == group_id)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
            return [(row.id, decode_vector(row.embedding)) for row in rows]

    # —— 记账与纠错 ——

    async def add_llm_call(
        self,
        *,
        purpose: str,
        provider: str | None = None,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        latency_ms: int | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> LlmCall:
        call = LlmCall(
            purpose=purpose,
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=latency_ms,
            success=success,
            error=error,
            created_at=self.clock(),
        )
        async with self._sessions() as session:
            session.add(call)
            await session.commit()
        return call

    async def add_feedback(
        self,
        *,
        kind: FeedbackKind | str,
        memory_id: int | None = None,
        group_id: int | None = None,
        note: str | None = None,
    ) -> Feedback:
        feedback = Feedback(
            kind=FeedbackKind(kind),
            memory_id=memory_id,
            group_id=group_id,
            note=note,
            created_at=self.clock(),
        )
        async with self._sessions() as session:
            session.add(feedback)
            await session.commit()
        return feedback

    # —— 每群开关 ——

    async def set_group_enabled(self, group_id: int, enabled: bool) -> None:
        async with self._sessions() as session:
            setting = await session.get(GroupSetting, group_id)
            now = self.clock()
            if setting is None:
                session.add(GroupSetting(group_id=group_id, enabled=enabled, updated_at=now))
            else:
                setting.enabled = enabled
                setting.updated_at = now
            await session.commit()

    async def is_group_enabled(self, group_id: int) -> bool:
        async with self._sessions() as session:
            setting = await session.get(GroupSetting, group_id)
            return bool(setting is not None and setting.enabled)

    async def enabled_groups(self) -> list[int]:
        async with self._sessions() as session:
            result = await session.execute(
                select(GroupSetting.group_id)
                .where(GroupSetting.enabled == True)  # noqa: E712 —— SQLAlchemy 表达式，不是 Python 真值判断
                .order_by(GroupSetting.group_id)
            )
            return list(result.scalars().all())
