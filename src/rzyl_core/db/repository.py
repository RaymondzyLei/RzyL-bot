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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from rzyl_core.timeutil import utcnow

from .engine import open_engine
from .enums import Category, FeedbackKind, MemoryStatus, WindowStatus
from .models import Feedback, GroupSetting, LlmCall, Memory, Message, PersonRef, Window
from .schema import apply_schema
from .vectors import decode_vector, encode_vector


def _require_aware(value: datetime | None, field: str) -> datetime | None:
    """在 API 边界挡住 naive datetime，报错比落库时才发现更清楚。"""
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} 必须带时区")
    return value


def _rowcount(result: object) -> int:
    """取一条 DELETE / UPDATE 的影响行数。

    ``exec_driver_sql`` 与 Core 的 ``delete()`` 返回的都是 ``CursorResult``，但公共签名
    里只承诺 ``Result``，故用 ``getattr`` 取 ``rowcount``。
    """
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass(frozen=True, slots=True)
class PurgeCounts:
    """一次清空实际删掉了多少行，逐表列出——清空是不可逆操作，效果必须看得见。"""

    memories: int = 0
    messages: int = 0
    windows: int = 0
    feedback_unlinked: int = 0
    """被摘掉 ``memory_id`` 的纠错样本条数（样本本身保留，见 :meth:`Repository.purge`）。"""


def _with_statuses(statement: Any, statuses: Sequence[MemoryStatus] | None) -> Any:
    """按状态过滤 ``memory`` 查询。

    约定（三处语义各有用处，写在类型上而不是靠调用方记）：

    - ``None``（默认）：只取 ``active``——检索与推送的默认语义（故事 22、25）；
    - 空序列：**不加状态条件**，全部状态都取；
    - 非空序列：只取列出的状态。
    """
    if statuses is None:
        return statement.where(Memory.status == MemoryStatus.ACTIVE)
    if not statuses:
        return statement
    return statement.where(Memory.status.in_(list(statuses)))


def _statuses_of(include_inactive: bool) -> Sequence[MemoryStatus] | None:
    """把旧的 ``include_inactive`` 开关翻译成 :func:`_with_statuses` 的三档语义。

    两套写法并存是历史：里程碑 1 只需要「要不要带上不活跃的」，里程碑 3 需要「正好这几种
    状态」。状态判定只在一处实现（见 :func:`_with_statuses` 的约定），这里只做翻译，免得
    同一个「什么算活跃」在很多个方法里各写一遍、日后慢慢漂移。
    """
    return () if include_inactive else None


def _person_exists(person_id: int):
    """「``person_refs_json`` 里含这个 QQ 号」的 SQL 条件。

    相关人存的是 JSON 数组（快照，不是外键），所以用 ``json_each`` 展开后比对
    ``user_id``。SQLite 3.38 起 JSON 函数内置，本项目的库版本远高于此。
    """
    return text(
        "EXISTS (SELECT 1 FROM json_each(memory.person_refs_json) AS person "
        "WHERE json_extract(person.value, '$.user_id') = :person_id)"
    ).bindparams(person_id=person_id)


class Repository:
    """一个连接串背后的一整套读写操作。"""

    def __init__(self, engine: AsyncEngine, *, clock: Callable[[], datetime] | None = None) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.clock: Callable[[], datetime] = clock or utcnow

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
            self._apply_window_status(window, status, retry_count=retry_count, error=error)
            await session.commit()

    def _apply_window_status(
        self,
        window: Window,
        status: WindowStatus,
        *,
        retry_count: int | None,
        error: str | None,
    ) -> None:
        """把 status / retry_count / error 三字段与 ``updated_at`` 落到窗口行上。

        建窗口结果（``add_window_result``）与单改状态（``update_window_status``）
        共用同一段更新逻辑，避免两处漂移。
        """
        window.status = status
        if retry_count is not None:
            window.retry_count = retry_count
        if error is not None:
            window.error = error
        window.updated_at = self.clock()

    async def get_window(self, window_id: int) -> Window | None:
        async with self._sessions() as session:
            return await session.get(Window, window_id)

    # —— 按状态列举窗口（以下为里程碑 2 追加）——

    async def list_windows_by_status(
        self,
        status: WindowStatus,
        *,
        limit: int = 100,
        group_id: int | None = None,
    ) -> list[Window]:
        """取指定状态的窗口，**编号升序**（最老的先处理），可选按群过滤。

        死信重试（``Runtime.retry_dead_windows``）据此取 ``status=dead`` 的窗口；
        编号升序保证老死信不会被新死信一直插队。
        """
        statement = select(Window).where(Window.status == status)
        if group_id is not None:
            statement = statement.where(Window.group_id == group_id)
        statement = statement.order_by(Window.id).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

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
                self._apply_window_status(window, status, retry_count=retry_count, error=error)
            await session.commit()
        return list(memories)

    async def get_memory(self, memory_id: int) -> Memory | None:
        async with self._sessions() as session:
            return await session.get(Memory, memory_id)

    async def list_group_memories(
        self, *, group_id: int, include_inactive: bool = False
    ) -> list[Memory]:
        """取某群的**全部**条目，新的在前。

        ``recent_memories`` 只给最近若干条，够窗口摘要用；去重比对照的是同群全部条目，
        所以另开这一个读方法（提取管道传 ``include_inactive=True``）。

        默认只给 ``active`` 的条目：被 supersede 的旧条目（``expired``）与疑似重复
        （``suspect_duplicate``）默认退出检索与推送（故事 22），显式开关可一并取回。
        """
        statement = _with_statuses(
            select(Memory).where(Memory.group_id == group_id), _statuses_of(include_inactive)
        )
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

    async def list_messages_for_window(self, window_id: int) -> list[Message]:
        """取某窗口按起止时间框住的**全部**原文，升序。

        与 ``memory.evidence`` 的区别：``evidence`` 是模型挑出来当依据的那几条，这里是
        窗口覆盖的整段原文——「记忆 来源」要看的是后者（故事 31：追到原文窗口）。
        窗口不存在、或没记时间区间（``ended_at`` 为空）时返回空列表。
        """
        async with self._sessions() as session:
            window = await session.get(Window, window_id)
            if window is None:
                return []
            started_at = window.started_at
            ended_at = window.ended_at
            group_id = int(window.group_id)
        if started_at is None or ended_at is None:
            return []
        return await self.list_messages_between(
            group_id=group_id, since=started_at, until=ended_at, limit=500
        )

    async def list_memories_by_id(self, memory_ids: Sequence[int]) -> list[Memory]:
        """按编号取条目，**返回顺序与传入顺序一致**；不存在的编号直接跳过。

        RRF 融合出来的名次要看顺序，而 SQL 的 ``IN`` 不保证顺序，所以这里在 Python 侧
        按传入顺序重排。
        """
        cleaned = [int(memory_id) for memory_id in memory_ids]
        if not cleaned:
            return []
        statement = select(Memory).where(Memory.id.in_(cleaned))
        async with self._sessions() as session:
            result = await session.execute(statement)
            by_id = {int(memory.id): memory for memory in result.scalars().all()}
        return [by_id[memory_id] for memory_id in cleaned if memory_id in by_id]

    async def list_memories_in_range(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        group_id: int | None = None,
        category: Category | None = None,
        person_id: int | None = None,
        statuses: Sequence[MemoryStatus] | None = None,
        limit: int = 200,
    ) -> list[Memory]:
        """按时间区间（落在 ``created_at`` 上，左闭右开）与各种前置条件取条目，新的在前。

        与 :meth:`search_memories` 的区别是**没有关键词**：用于「今天」「某个群」这类
        纯列举场景。``statuses`` 的三档语义见 :func:`_with_statuses`。
        """
        statement = select(Memory)
        statement = _with_statuses(statement, statuses)
        if since is not None:
            statement = statement.where(Memory.created_at >= _require_aware(since, "since"))
        if until is not None:
            statement = statement.where(Memory.created_at < _require_aware(until, "until"))
        if group_id is not None:
            statement = statement.where(Memory.group_id == group_id)
        if category is not None:
            statement = statement.where(Memory.category == category)
        if person_id is not None:
            statement = statement.where(_person_exists(person_id))
        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def set_memory_status(self, memory_id: int, status: MemoryStatus) -> bool:
        """改一条记忆的状态；条目不存在返回 ``False``。

        只动 ``status`` 一列：``superseded_by`` 之类的引用关系由写入方自己维护，
        免得「恢复一条被推翻的旧条目」把它的引用链也一并改掉。
        """
        async with self._sessions() as session:
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return False
            memory.status = status
            await session.commit()
            return True

    async def clear_superseded_by(self, memory_id: int) -> bool:
        """清掉**这条记忆自己**的 ``superseded_by``，返回是否真的改动了。

        「记忆 恢复」要用：被 supersede 的旧条目身上带着一个「我是被 #N 推翻的」的引用，
        它被人工恢复成 ``active`` 之后这条引用就没有意义了——留着会让「``expired`` 且
        没有 ``superseded_by``」这个判据（人工标为误报）失真。

        注意方向：这里清的是**指向别人的那一列**，不是「指向它的别人」；后者是清空时
        解外键要做的事，在 :meth:`purge` 里。
        """
        async with self._sessions() as session:
            memory = await session.get(Memory, memory_id)
            if memory is None or memory.superseded_by is None:
                return False
            memory.superseded_by = None
            await session.commit()
            return True

    # —— 检索 ——

    async def search_memories(
        self,
        query: str,
        *,
        category: Category | None = None,
        group_id: int | None = None,
        person_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        include_inactive: bool = False,
    ) -> list[Memory]:
        """FTS5 关键词检索，可按类别 / 群 / 相关人 / 时间范围过滤。

        trigram 分词对中文按子串匹配，但匹配词至少 3 个字；两字及以下的词（如「论文」）
        走 LIKE 兜底。多个空格分隔的词之间是「与」的关系。

        默认只检索 ``active`` 条目：被 supersede 的旧条目（``expired``）与疑似重复
        （``suspect_duplicate``）默认退出检索（故事 22）；``include_inactive=True``
        才一并取回，供「一键恢复 / 回看历史」这类场景使用。

        时间区间是**左闭右开** ``[since, until)``，与 :meth:`list_memories_in_range` 一致：
        「今天」的边界正好能表达成「今天 00:00 到明天 00:00」，不必再为午夜那条消息
        加减一秒。
        """
        terms = [term for term in query.split() if term]
        if not terms:
            return []

        long_terms = [term for term in terms if len(term) >= 3]
        short_terms = [term for term in terms if len(term) < 3]

        statement = _with_statuses(select(Memory), _statuses_of(include_inactive))
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
        if person_id is not None:
            statement = statement.where(_person_exists(person_id))
        if since is not None:
            statement = statement.where(Memory.created_at >= _require_aware(since, "since"))
        if until is not None:
            statement = statement.where(Memory.created_at < _require_aware(until, "until"))
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
        statement = _with_statuses(
            select(Memory).where(Memory.group_id == group_id), _statuses_of(include_inactive)
        )
        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(limit)
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def list_embeddings(
        self,
        *,
        category: Category | None = None,
        group_id: int | None = None,
        person_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_inactive: bool = False,
    ) -> list[tuple[int, list[float]]]:
        """取出已算好向量的条目的 ``(编号, 向量)``，供暴力余弦使用。

        前置过滤项与 :meth:`search_memories` 一一对应：混合检索的两路必须**看到同一批
        候选**，否则 RRF 会把「被过滤掉、却在另一路里排第一」的条目捞回来。
        """
        statement = _with_statuses(
            select(Memory.id, Memory.embedding).where(func.length(Memory.embedding) > 0),
            _statuses_of(include_inactive),
        )
        if category is not None:
            statement = statement.where(Memory.category == category)
        if group_id is not None:
            statement = statement.where(Memory.group_id == group_id)
        if person_id is not None:
            statement = statement.where(_person_exists(person_id))
        if since is not None:
            statement = statement.where(Memory.created_at >= _require_aware(since, "since"))
        if until is not None:
            statement = statement.where(Memory.created_at < _require_aware(until, "until"))
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
            return [(int(row.id), decode_vector(row.embedding)) for row in rows]

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
        """记一次聊天 / 向量调用，``purpose`` 区分用途（如 ``extract`` / ``embed``）。

        这条记账**独立于窗口事务、自行提交**（对比 ``add_window_result``）：窗口写库
        整体回滚时账目仍保留。这是有意为之——每次真实发生、会被计费的调用都要留痕，
        否则回滚会把已花的钱从账上抹掉，「这个功能每月花多少」就答不准了。
        """
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

    async def latest_successful_llm_call_model(self, *, purpose: str) -> str | None:
        """取某用途最近一次**成功**调用记录里的模型名；没有成功记录时返回 ``None``。

        给运行时「向量模型账本一致性」检查用：维度守卫只挡得住维度变化，同维度的不同
        模型（例如两个都是 1024 维的向量模型）混在一张表里，任何长度检查都看不出来，
        只能靠模型名记账来发现。
        """
        statement = (
            select(LlmCall.model)
            .where(LlmCall.purpose == purpose, LlmCall.success.is_(True))
            .order_by(LlmCall.created_at.desc(), LlmCall.id.desc())
            .limit(1)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return result.scalars().first()

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

    async def list_feedback(self, *, limit: int = 50) -> list[Feedback]:
        """取最近的纠错样本，新的在前。

        误报样本连同原陈述的**快照**一起留在这里，供日后当提示词回归集用；
        ``memory_id`` 可能已被清空摘掉，所以样本自身必须读得懂。
        """
        statement = (
            select(Feedback).order_by(Feedback.created_at.desc(), Feedback.id.desc()).limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

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

    async def disabled_groups(self) -> list[int]:
        """被**显式关掉**的群号。

        与 :meth:`enabled_groups` 一起构成「数据库里的每群开关」，两者都不含没有行的群。
        有行就是显式表态，于是运行时的「暂停」才能真正压过配置里的白名单——否则
        ``记忆 暂停`` 对一个写在 ``RZYL_GROUP_WHITELIST`` 里的群是个空操作。
        """
        async with self._sessions() as session:
            result = await session.execute(
                select(GroupSetting.group_id)
                .where(GroupSetting.enabled == False)  # noqa: E712 —— 同上
                .order_by(GroupSetting.group_id)
            )
            return list(result.scalars().all())

    async def group_settings(self) -> dict[int, bool]:
        """每群开关的完整快照：``{群号: 是否开启}``，只含**有行**的群。"""
        async with self._sessions() as session:
            result = await session.execute(select(GroupSetting.group_id, GroupSetting.enabled))
            return {int(row.group_id): bool(row.enabled) for row in result.all()}

    # —— 清空（里程碑 3 的「记忆 清空」）——

    async def purge(
        self,
        *,
        memory_ids: Sequence[int] | None = None,
        group_id: int | None = None,
        everything: bool = False,
    ) -> PurgeCounts:
        """按范围彻底删除记忆、原文与窗口，返回逐表删除行数。

        三个范围**互斥且必须显式给一个**（这是不可逆操作，不给默认值）：

        - ``memory_ids``：只删这几条记忆，「原文与窗口不动」。单条记忆引用的原文常常
          与同群其它条目共用，删原文会连累别人。
        - ``group_id``：删这个群的记忆、原文与窗口——「这个群我不看了」，痕迹一起清掉。
        - ``everything``：全库清空。

        两处**引用**必须先解开，否则 ``PRAGMA foreign_keys=ON`` 会直接拒绝删除：

        1. ``memory.superseded_by`` 指向即将被删的条目时置空；
        2. ``feedback.memory_id`` 指向即将被删的条目时置空——**样本本身保留**。误报样本
           是调提示词的回归材料（故事 39），不能因为「删掉了那条记忆」就跟着消失；所以
           记误报时会把原陈述**快照**进 ``note``（与 ``person_refs`` 存快照同理）。

        删除顺序：记忆 → 窗口 → 原文。``memory.window_id`` 是外键，所以窗口必须等记忆
        删完；原文没有任何外键指向它，放最后只是为了读起来顺。
        """
        provided = [
            memory_ids is not None and len(memory_ids) > 0,
            group_id is not None,
            everything,
        ]
        if sum(1 for item in provided if item) != 1:
            raise ValueError(
                "purge 必须且只能指定一个范围：memory_ids / group_id / everything"
            )

        async with self._sessions() as session:
            targets = select(Memory.id)
            if memory_ids is not None:
                targets = targets.where(Memory.id.in_([int(value) for value in memory_ids]))
            elif group_id is not None:
                targets = targets.where(Memory.group_id == group_id)
            target_ids = [int(value) for value in (await session.execute(targets)).scalars().all()]

            unlinked = 0
            if target_ids:
                # 先解开两处引用：外键开着，直接删会被拒绝。
                unlinked = _rowcount(
                    await session.execute(
                        update(Feedback)
                        .where(Feedback.memory_id.in_(target_ids))
                        .values(memory_id=None)
                    )
                )
                await session.execute(
                    update(Memory)
                    .where(Memory.superseded_by.in_(target_ids))
                    .values(superseded_by=None)
                )

            memory_statement = delete(Memory)
            if memory_ids is not None:
                # 按编号点名：点名里不存在的编号自然什么都不删（空列表也不必发语句）。
                memories_deleted = (
                    _rowcount(
                        await session.execute(memory_statement.where(Memory.id.in_(target_ids)))
                    )
                    if target_ids
                    else 0
                )
            else:
                if group_id is not None:
                    memory_statement = memory_statement.where(Memory.group_id == group_id)
                memories_deleted = _rowcount(await session.execute(memory_statement))

            messages_deleted = 0
            windows_deleted = 0
            if everything or group_id is not None:
                window_statement = delete(Window)
                message_statement = delete(Message)
                if group_id is not None:
                    window_statement = window_statement.where(Window.group_id == group_id)
                    message_statement = message_statement.where(Message.group_id == group_id)
                windows_deleted = _rowcount(await session.execute(window_statement))
                messages_deleted = _rowcount(await session.execute(message_statement))

            await session.commit()

        return PurgeCounts(
            memories=memories_deleted,
            messages=messages_deleted,
            windows=windows_deleted,
            feedback_unlinked=unlinked,
        )

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

    async def list_messages_between(
        self,
        *,
        group_id: int,
        since: datetime,
        until: datetime,
        limit: int = 200,
    ) -> list[Message]:
        """取某群 ``since`` 与 ``until``（含两端）之间的原文，按发送时间升序。

        死信重试要靠它把 ``window`` 行还原回一段消息、重新组装窗口（序号→真实编号的
        映射必须对得上）。
        """
        statement = (
            select(Message)
            .where(Message.group_id == group_id)
            .where(Message.sent_at >= since)
            .where(Message.sent_at <= until)
            .order_by(Message.sent_at, Message.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def list_messages_before(
        self, *, group_id: int, before: datetime, limit: int = 3
    ) -> list[Message]:
        """取某群 ``sent_at`` 严格早于 ``before`` 的**最近** ``limit`` 条原文，按时间升序。

        重试一个旧窗口时用它还原「上一窗口尾部」——那段上下文只作理解用、不进 ``evidence``，
        所以取最近几条即可（与实时链路里 ``previous_tail_size`` 的语义一致）。
        """
        statement = (
            select(Message)
            .where(Message.group_id == group_id)
            .where(Message.sent_at < _require_aware(before, "before"))
            .order_by(Message.sent_at.desc(), Message.id.desc())
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(reversed(list(result.scalars().all())))

    async def latest_message_sent_at(self, group_id: int) -> datetime | None:
        """某群**已存消息里最新一条**的发送时间；一条都没有时返回 ``None``。

        掉线回补的锚点（见 issue #1「以该群最后一条已处理消息的时间戳为锚」）——
        平台短 ID 是内存映射、重启即失效，只能靠时间戳。
        """
        statement = select(func.max(Message.sent_at)).where(Message.group_id == group_id)
        async with self._sessions() as session:
            value = (await session.execute(statement)).scalar()
        return value if isinstance(value, datetime) else None

    async def list_uncovered_messages(
        self, *, group_id: int, limit: int = 100
    ) -> list[Message]:
        """取某群里**没有被任何窗口的时间区间覆盖**的消息，按 ``sent_at`` 升序（同刻按编号）。

        覆盖的判定：存在一个同群的窗口，其 ``started_at`` 与 ``ended_at`` **均非空**，且
        这条消息的 ``sent_at`` 落在 ``[started_at, ended_at]``（含两端）之内。不区分窗口
        状态——只要窗口记下了时间区间，这条消息就算有归属。

        启动对账用它找出「重启时随内存缓冲一起丢掉、却因回补锚点而永远不会被提取」的消息
        （见 :meth:`~rzyl_core.runtime.Runtime.reconcile_uncovered_messages`）。

        **已知局限**：覆盖是按时间区间判定的，所以一条真正没进过管道的消息，如果它的
        ``sent_at`` 恰好落在**别的**窗口的时间区间内，这里就检测不出来。之所以仍然够用，
        是因为丢失的内存缓冲总是位于两个窗口之间的**空隙**里——那一段没有任何窗口盖住。
        换言之：能检测到的是「时间上没有被任何窗口跨过的消息」，而不是「从未参与过任何
        窗口的消息」；后者需要记录每条消息的处理归属，超出当前 ``window`` 表的表达能力。
        """
        # 相关子查询：同群、时间区间非空、且把本消息的 sent_at 夹在区间里。
        covered = (
            select(Window.id)
            .where(Window.group_id == Message.group_id)
            .where(Window.started_at.is_not(None))
            .where(Window.ended_at.is_not(None))
            .where(Message.sent_at >= Window.started_at)
            .where(Message.sent_at <= Window.ended_at)
            .exists()
        )
        statement = (
            select(Message)
            .where(Message.group_id == group_id)
            .where(~covered)
            .order_by(Message.sent_at, Message.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def existing_message_hashes(self, hashes: Sequence[str]) -> set[str]:
        """在给定的一批消息去重 hash 里，返回**库中已存在**的那些。

        掉线回补据此跳过已经存过的消息（内容 hash = 群号 + 发送者 + 时间 + 文本，
        见 ``pipeline.history.message_dedupe_hash``），避免重启回补把同一条记两遍。
        """
        cleaned = [value for value in hashes if value]
        if not cleaned:
            return set()
        statement = select(Message.dedupe_hash).where(Message.dedupe_hash.in_(cleaned))
        async with self._sessions() as session:
            result = await session.execute(statement)
            return set(result.scalars().all())

    async def delete_messages_before(self, cutoff: datetime) -> int:
        """删掉 ``sent_at`` 严格早于 ``cutoff`` 的原文消息，返回删除条数。

        只删原文，不动记忆条目——原文滚动保留、结论永久，这是 issue #1 的保留策略。
        注意 ``message`` 上没有外键指向它的行（``memory.evidence`` 存的是编号数组而非
        外键），所以这里删得干净，不需要级联。
        """
        async with self._sessions() as session:
            result = await session.execute(delete(Message).where(Message.sent_at < cutoff))
            await session.commit()
            return _rowcount(result)

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

    async def list_memories_for_reembedding(self, *, limit: int = 100) -> list[Memory]:
        """取需要**按新模型重算向量**的条目（**含已有向量**），按编号升序。

        与 :meth:`list_memories_missing_embedding` 的唯一区别是不筛掉已有向量的条目：
        换向量模型（维度或语义空间变了）后要把全部条目重算一遍，靠的就是这个查询。
        只认 ``active`` / ``suspect_duplicate``，已过期的条目不再重算。
        """
        statement = (
            select(Memory)
            .where(Memory.status != MemoryStatus.EXPIRED)
            .order_by(Memory.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())
