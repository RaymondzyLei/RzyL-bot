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

from sqlalchemy import delete, func, or_, select, text
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

    async def add_window_result(
        self,
        *,
        window_id: int,
        memories: Sequence[Memory] = (),
        supersedings: Sequence[tuple[int, Memory]] = (),
        status: WindowStatus = WindowStatus.DONE,
        retry_count: int | None = None,
        error: str | None = None,
    ) -> list[Memory]:
        """一个事务内写入本窗口的全部条目、标记被推翻的旧条目，并推进窗口状态。

        ``memories`` 是调用方（提取管道）构造好、尚未入库的 ``Memory`` 对象；
        ``supersedings`` 每项是（被推翻的旧条目编号, 指向它的新条目对象）——新条目编号
        要 flush 之后才存在，所以传对象而不是编号。新条目不做反向修改，只有旧条目被置为
        ``expired`` 并记住 ``superseded_by``。

        整段要么全成、要么全不成：任一步报错都会回滚，窗口行保持调用前的状态（可重试）。
        """
        async with self._sessions() as session:
            session.add_all(list(memories))
            await session.flush()
            for old_id, replacement in supersedings:
                old = await session.get(Memory, old_id)
                if old is None:
                    continue
                old.status = MemoryStatus.EXPIRED
                old.superseded_by = replacement.id
            window = await session.get(Window, window_id)
            if window is not None:
                window.status = status
                if retry_count is not None:
                    window.retry_count = retry_count
                if error is not None:
                    window.error = error
                window.updated_at = self.clock()
            await session.commit()
        return list(memories)

    async def get_memory(self, memory_id: int) -> Memory | None:
        async with self._sessions() as session:
            return await session.get(Memory, memory_id)

    async def list_group_memories(
        self, *, group_id: int, include_expired: bool = False
    ) -> list[Memory]:
        """取某群的**全部**条目，新的在前。

        ``recent_memories`` 只给最近若干条，够窗口摘要用；去重比对照的是同群全部条目，
        所以另开这一个读方法。默认排除已过期（``expired``）的条目。
        """
        statement = select(Memory).where(Memory.group_id == group_id)
        if not include_expired:
            statement = statement.where(Memory.status != MemoryStatus.EXPIRED)
        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc())
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

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

    async def list_llm_calls(
        self, *, purpose: str | None = None, limit: int = 50
    ) -> list[LlmCall]:
        """取最近的调用记账，新的在前；可按用途过滤。"""
        statement = select(LlmCall)
        if purpose is not None:
            statement = statement.where(LlmCall.purpose == purpose)
        statement = statement.order_by(LlmCall.created_at.desc(), LlmCall.id.desc()).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

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

    # —— 原文消息的读与清理（以下为工单 #5 追加）——

    async def list_messages(
        self, *, group_id: int | None = None, limit: int = 100
    ) -> list[Message]:
        """取原文消息，按发送时间升序（同刻按编号升序）。

        回放脚本与保留期清理的验收都靠它把「库里现在有什么」读回来；按群过滤时
        就是该群的一份时间线。
        """
        statement = select(Message)
        if group_id is not None:
            statement = statement.where(Message.group_id == group_id)
        statement = statement.order_by(Message.sent_at, Message.id).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def delete_messages_before(self, cutoff: datetime) -> int:
        """删掉 ``sent_at`` 严格早于 ``cutoff`` 的原文消息，返回删除条数。

        只删原文，不动记忆条目——原文滚动保留、结论永久，这是 issue #1 的保留策略。
        注意 ``message`` 上没有外键指向它的行（``memory.evidence`` 存的是编号数组而非
        外键），所以这里删得干净，不需要级联。
        """
        async with self._sessions() as session:
            result = await session.execute(delete(Message).where(Message.sent_at < cutoff))
            await session.commit()
            # DELETE 返回的是 CursorResult，但公共签名里只有 Result，故用 getattr 取 rowcount。
            return int(getattr(result, "rowcount", 0) or 0)

    # —— 向量补算（以下为工单 #5 追加）——

    async def list_memories_missing_embedding(self, *, limit: int = 100) -> list[Memory]:
        """取还没算向量的记忆条目（向量为空或空字节串），按编号升序。

        后台补算任务据此分批取活干；只认 ``active`` / ``suspect_duplicate``，已过期
        的条目不再补算。
        """
        statement = (
            select(Memory)
            .where(Memory.status != MemoryStatus.EXPIRED)
            .where(or_(Memory.embedding.is_(None), func.length(Memory.embedding) == 0))
            .order_by(Memory.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def update_memory_embedding(self, memory_id: int, embedding: Sequence[float]) -> bool:
        """把补算出的向量写回某条记忆；条目不存在返回 ``False``。"""
        async with self._sessions() as session:
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return False
            memory.embedding = encode_vector(list(embedding))
            await session.commit()
            return True
