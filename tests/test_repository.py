"""仓储层的行为测试（工单 #2）。

seam 是 ``Repository`` 的公开读写 API：只给一个连接串就能构造，写入后按编号 / 关键词 /
最近条目 / 向量等方式取回。断言的是**外部可观察的行为**——写进去什么，读出来什么——
不碰会话、语句这些内部细节。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import (
    Category,
    MemoryStatus,
    Repository,
    WindowStatus,
    decode_vector,
)

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    yield instance
    await instance.close()


async def test_repository_only_needs_a_connection_string(repo: Repository) -> None:
    assert repo is not None


async def test_message_round_trips_and_keeps_timezone(repo: Repository) -> None:
    message = await repo.add_message(
        group_id=111,
        user_id=222,
        text="这张图里的公式",
        sent_at=BEGIN,
        nickname="小明",
        card="小明(实验班)",
        segments_json='[{"type":"text","data":{"text":"这张图里的公式"}}]',
        platform_message_id=987654,
        dedupe_hash="hash-1",
    )

    assert message.id > 0
    assert message.group_id == 111
    assert message.text == "这张图里的公式"
    assert message.nickname == "小明"
    # 归一化到 UTC 后读回来仍是带时区的 datetime，且表示同一时刻
    assert message.sent_at.tzinfo is not None
    assert message.sent_at == BEGIN


async def test_memory_round_trips_every_column(repo: Repository) -> None:
    window = await repo.add_window(
        group_id=111,
        started_at=BEGIN,
        ended_at=BEGIN + timedelta(minutes=5),
        message_count=30,
        prompt_version="extract-v1",
    )
    first = await repo.add_message(group_id=111, user_id=222, text="链接在这", sent_at=BEGIN)
    second = await repo.add_message(group_id=111, user_id=333, text="收到了", sent_at=BEGIN + timedelta(minutes=1))

    record = await repo.add_memory(
        window_id=window.id,
        group_id=111,
        category=Category.RESOURCE,
        statement="群友分享了一份分布式系统讲义",
        detail="网盘链接，三天内有效",
        confidence=0.82,
        evidence=[first.id, second.id],
        person_refs=[{"user_id": 222, "nickname_snapshot": "小明"}],
        occurred_at=BEGIN - timedelta(days=1),
        prompt_version="extract-v1",
        dedupe_hash="dedupe-1",
        status=MemoryStatus.ACTIVE,
        model="deepseek-chat",
        tokens_in=1200,
        tokens_out=300,
        cost=0.0018,
    )

    loaded = await repo.get_memory(record.id)
    assert loaded is not None
    assert loaded.id == record.id
    assert loaded.window_id == window.id
    assert loaded.group_id == 111
    assert loaded.category is Category.RESOURCE
    assert loaded.statement == "群友分享了一份分布式系统讲义"
    assert loaded.detail == "网盘链接，三天内有效"
    assert loaded.confidence == pytest.approx(0.82)
    assert loaded.evidence == [first.id, second.id]
    assert loaded.person_refs == [{"user_id": 222, "nickname_snapshot": "小明"}]
    assert loaded.occurred_at is not None and loaded.occurred_at.tzinfo is not None
    assert loaded.occurred_at == BEGIN - timedelta(days=1)
    assert loaded.prompt_version == "extract-v1"
    assert loaded.dedupe_hash == "dedupe-1"
    assert loaded.embedding is None
    assert loaded.status is MemoryStatus.ACTIVE
    assert loaded.superseded_by is None
    assert loaded.model == "deepseek-chat"
    assert loaded.tokens_in == 1200
    assert loaded.tokens_out == 300
    assert loaded.cost == pytest.approx(0.0018)
    assert loaded.created_at.tzinfo is not None


async def test_get_memory_returns_none_for_unknown_id(repo: Repository) -> None:
    assert await repo.get_memory(9999) is None


async def test_source_messages_follow_evidence(repo: Repository) -> None:
    first = await repo.add_message(group_id=111, user_id=222, text="一", sent_at=BEGIN)
    middle = await repo.add_message(group_id=111, user_id=222, text="二", sent_at=BEGIN + timedelta(minutes=1))
    last = await repo.add_message(group_id=111, user_id=222, text="三", sent_at=BEGIN + timedelta(minutes=2))
    record = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="引用第一条与第三条，不引用第二条",
        confidence=0.9,
        evidence=[last.id, first.id],
        prompt_version="extract-v1",
    )

    sources = await repo.get_source_messages(record.id)

    assert [m.id for m in sources] == [first.id, last.id]
    assert middle.id not in {m.id for m in sources}


async def test_window_status_can_become_dead(repo: Repository) -> None:
    window = await repo.add_window(group_id=111, started_at=BEGIN, message_count=3)
    assert window.status is WindowStatus.PENDING

    await repo.update_window_status(window.id, WindowStatus.DEAD, retry_count=3, error="模型连续失败")

    reloaded = await repo.get_window(window.id)
    assert reloaded is not None
    assert reloaded.status is WindowStatus.DEAD
    assert reloaded.retry_count == 3
    assert reloaded.error == "模型连续失败"


async def test_keyword_search_finds_chinese_and_filters(repo: Repository) -> None:
    await repo.add_memory(
        group_id=111,
        category=Category.RESOURCE,
        statement="这个网盘链接里有分布式系统讲义",
        confidence=0.9,
        prompt_version="extract-v1",
        dedupe_hash="a",
    )
    await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="实验课改到周五下午三点",
        confidence=0.8,
        prompt_version="extract-v1",
        dedupe_hash="b",
    )
    await repo.add_memory(
        group_id=222,
        category=Category.RESOURCE,
        statement="另一个群里的分布式系统讲义",
        confidence=0.9,
        prompt_version="extract-v1",
        dedupe_hash="c",
    )

    hits = await repo.search_memories("分布式系统")
    assert len(hits) == 2

    scoped = await repo.search_memories("分布式系统", group_id=111)
    assert [m.statement for m in scoped] == ["这个网盘链接里有分布式系统讲义"]

    by_category = await repo.search_memories("分布式系统", category=Category.EVENT)
    assert by_category == []


async def test_keyword_search_handles_short_terms(repo: Repository) -> None:
    """两字词在 trigram FTS 里匹配不到，走 LIKE 兜底——中文常用词必须能搜到。"""
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="这篇论文给了新的证明",
        confidence=0.9,
        prompt_version="extract-v1",
    )
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="无关的一句话",
        confidence=0.9,
        prompt_version="extract-v1",
    )

    hits = await repo.search_memories("论文")

    assert [m.statement for m in hits] == ["这篇论文给了新的证明"]


async def test_search_filters_by_time_range(repo: Repository) -> None:
    repo.clock = lambda: datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
    old = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="旧的分布式系统结论",
        confidence=0.9,
        prompt_version="extract-v1",
    )
    repo.clock = lambda: datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)
    new = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="新的分布式系统结论",
        confidence=0.9,
        prompt_version="extract-v1",
    )

    hits = await repo.search_memories(
        "分布式系统",
        since=datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc),
    )

    assert [m.id for m in hits] == [new.id]
    assert old.id not in {m.id for m in hits}


async def test_recent_memories_are_per_group_newest_first(repo: Repository) -> None:
    stamp = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
    for index in range(3):
        repo.clock = lambda index=index: stamp + timedelta(minutes=index)
        await repo.add_memory(
            group_id=111,
            category=Category.KNOWLEDGE,
            statement=f"群一的第{index}条",
            confidence=0.9,
            prompt_version="extract-v1",
        )
    repo.clock = lambda: stamp + timedelta(hours=1)
    await repo.add_memory(
        group_id=222,
        category=Category.KNOWLEDGE,
        statement="群二的条目",
        confidence=0.9,
        prompt_version="extract-v1",
    )

    recent = await repo.recent_memories(group_id=111, limit=2)

    assert [m.statement for m in recent] == ["群一的第2条", "群一的第1条"]


async def test_recent_memories_exclude_expired_by_default(repo: Repository) -> None:
    active = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="仍然有效的结论",
        confidence=0.9,
        prompt_version="extract-v1",
    )
    await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="被推翻的结论",
        confidence=0.9,
        prompt_version="extract-v1",
        status=MemoryStatus.EXPIRED,
    )

    default = await repo.recent_memories(group_id=111)
    assert [m.id for m in default] == [active.id]

    everything = await repo.recent_memories(group_id=111, include_inactive=True)
    assert len(everything) == 2


async def test_embeddings_are_stored_and_returned_for_cosine(repo: Repository) -> None:
    vector = [0.5, -1.0, 2.0, 0.25]
    with_vector = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="带向量的条目",
        confidence=0.9,
        prompt_version="extract-v1",
        embedding=vector,
    )
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="还没算向量的条目",
        confidence=0.9,
        prompt_version="extract-v1",
    )

    candidates = await repo.list_embeddings()

    assert [memory_id for memory_id, _ in candidates] == [with_vector.id]
    assert decode_vector((await repo.get_memory(with_vector.id)).embedding) == pytest.approx(vector)  # type: ignore[union-attr]


async def test_group_switch_persists(repo: Repository) -> None:
    assert await repo.enabled_groups() == []
    assert await repo.is_group_enabled(111) is False

    await repo.set_group_enabled(111, True)
    await repo.set_group_enabled(222, False)
    await repo.set_group_enabled(111, True)  # 幂等

    assert await repo.enabled_groups() == [111]
    assert await repo.is_group_enabled(111) is True
    assert await repo.is_group_enabled(222) is False


async def test_llm_call_and_feedback_are_recorded(repo: Repository) -> None:
    call = await repo.add_llm_call(
        purpose="extract",
        provider="deepseek",
        model="deepseek-chat",
        tokens_in=1000,
        tokens_out=200,
        cost=0.001,
        latency_ms=850,
        success=True,
    )
    assert call.id > 0 and call.success is True

    memory = await repo.add_memory(
        group_id=111,
        category=Category.REQUEST,
        statement="有人求一份讲义",
        confidence=0.5,
        prompt_version="extract-v1",
    )
    feedback = await repo.add_feedback(
        kind="false_positive",
        memory_id=memory.id,
        group_id=111,
        note="把玩笑当成承诺了",
    )
    assert feedback.id > 0
    assert feedback.kind.value == "false_positive"


async def test_naive_datetime_is_rejected(repo: Repository) -> None:
    with pytest.raises(ValueError):
        await repo.add_message(group_id=111, user_id=222, text="无时区", sent_at=datetime(2026, 10, 7, 9, 0))
