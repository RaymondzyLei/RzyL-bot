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
    FeedbackKind,
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


async def test_search_memories_excludes_inactive_entries_by_default(repo: Repository) -> None:
    """故事 22：被判为过期 / 疑似重复的条目默认退出检索，显式开关才取回。"""
    active = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="分布式系统仍然有效的结论",
        confidence=0.9,
        prompt_version="extract-v1",
    )
    expired = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="分布式系统的旧结论已被推翻",
        confidence=0.9,
        prompt_version="extract-v1",
        status=MemoryStatus.EXPIRED,
    )
    suspect = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="分布式系统的疑似重复结论",
        confidence=0.9,
        prompt_version="extract-v1",
        status=MemoryStatus.SUSPECT_DUPLICATE,
    )

    default = await repo.search_memories("分布式系统")
    assert [m.id for m in default] == [active.id]

    everything = await repo.search_memories("分布式系统", include_inactive=True)
    assert {m.id for m in everything} == {active.id, expired.id, suspect.id}


async def test_list_group_memories_excludes_inactive_by_default(repo: Repository) -> None:
    """名字统一后语义也统一：四个读方法默认都只给 ``active``，开关一次放开全部。"""
    active = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="仍然有效的结论",
        confidence=0.9,
        prompt_version="extract-v1",
    )
    expired = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="被推翻的结论",
        confidence=0.9,
        prompt_version="extract-v1",
        status=MemoryStatus.EXPIRED,
    )
    suspect = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="疑似重复的结论",
        confidence=0.9,
        prompt_version="extract-v1",
        status=MemoryStatus.SUSPECT_DUPLICATE,
    )

    default = await repo.list_group_memories(group_id=111)
    assert [m.id for m in default] == [active.id]

    everything = await repo.list_group_memories(group_id=111, include_inactive=True)
    assert {m.id for m in everything} == {active.id, expired.id, suspect.id}


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


# —— 按状态列举窗口（里程碑 2 的死信重试要用）——


async def test_list_windows_by_status_filters_and_orders_oldest_first(repo: Repository) -> None:
    first = await repo.add_window(group_id=111, started_at=BEGIN, message_count=1)
    second = await repo.add_window(group_id=222, started_at=BEGIN, message_count=1)
    await repo.add_window(group_id=333, started_at=BEGIN, message_count=1)
    await repo.update_window_status(first.id, WindowStatus.DEAD, retry_count=3)
    await repo.update_window_status(second.id, WindowStatus.DEAD, retry_count=1)

    dead = await repo.list_windows_by_status(WindowStatus.DEAD)

    # 编号升序：最老的先重试，避免新死信一直插队。
    assert [window.id for window in dead] == [first.id, second.id]
    assert all(window.status is WindowStatus.DEAD for window in dead)


async def test_list_windows_by_status_respects_limit_and_can_filter_by_group(repo: Repository) -> None:
    for index in range(4):
        window = await repo.add_window(group_id=111, started_at=BEGIN, message_count=1)
        await repo.update_window_status(window.id, WindowStatus.DEAD, retry_count=index)

    assert len(await repo.list_windows_by_status(WindowStatus.DEAD, limit=2)) == 2
    assert await repo.list_windows_by_status(WindowStatus.DEAD, group_id=999) == []


# —— 回补锚点：某群最后一条已存消息的时间 ——


async def test_latest_message_sent_at_returns_the_newest_or_none(repo: Repository) -> None:
    assert await repo.latest_message_sent_at(111) is None

    await repo.add_message(group_id=111, user_id=1, text="早", sent_at=BEGIN - timedelta(minutes=5))
    await repo.add_message(group_id=111, user_id=2, text="晚", sent_at=BEGIN)
    await repo.add_message(group_id=222, user_id=3, text="别的群", sent_at=BEGIN + timedelta(days=1))

    assert await repo.latest_message_sent_at(111) == BEGIN
    assert await repo.latest_message_sent_at(222) == BEGIN + timedelta(days=1)


async def test_list_messages_between_returns_the_window_slice(repo: Repository) -> None:
    await repo.add_message(group_id=111, user_id=1, text="窗口前", sent_at=BEGIN - timedelta(minutes=1))
    first = await repo.add_message(group_id=111, user_id=1, text="窗口首", sent_at=BEGIN)
    last = await repo.add_message(group_id=111, user_id=2, text="窗口尾", sent_at=BEGIN + timedelta(minutes=1))
    await repo.add_message(group_id=111, user_id=3, text="窗口后", sent_at=BEGIN + timedelta(minutes=2))

    slice_ = await repo.list_messages_between(
        group_id=111, since=BEGIN, until=BEGIN + timedelta(minutes=1)
    )

    assert [message.id for message in slice_] == [first.id, last.id]


async def test_existing_message_hashes_reports_only_known_hashes(repo: Repository) -> None:
    await repo.add_message(group_id=111, user_id=1, text="甲", sent_at=BEGIN, dedupe_hash="h1")
    await repo.add_message(group_id=111, user_id=2, text="乙", sent_at=BEGIN, dedupe_hash="h2")

    found = await repo.existing_message_hashes(["h2", "h3", ""])

    assert found == {"h2"}
    assert await repo.existing_message_hashes([]) == set()


# —— 里程碑 3 追加的读与清空 API ——


async def test_list_memories_in_range_filters_by_time_and_status(repo: Repository) -> None:
    repo.clock = lambda: datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
    active = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="今天的结论",
        confidence=0.9, prompt_version="v2",
    )
    suspect = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="今天的疑似重复",
        confidence=0.9, prompt_version="v2", status=MemoryStatus.SUSPECT_DUPLICATE,
    )
    expired = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="今天的过期条目",
        confidence=0.9, prompt_version="v2", status=MemoryStatus.EXPIRED,
    )
    repo.clock = lambda: datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)
    await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="明天的结论",
        confidence=0.9, prompt_version="v2",
    )

    window = (
        datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc),
    )
    default = await repo.list_memories_in_range(since=window[0], until=window[1])
    everything = await repo.list_memories_in_range(
        since=window[0], until=window[1], statuses=()
    )
    non_expired = await repo.list_memories_in_range(
        since=window[0],
        until=window[1],
        statuses=(MemoryStatus.ACTIVE, MemoryStatus.SUSPECT_DUPLICATE),
    )

    assert [m.id for m in default] == [active.id]
    assert {m.id for m in everything} == {active.id, suspect.id, expired.id}
    assert {m.id for m in non_expired} == {active.id, suspect.id}


async def test_list_memories_in_range_is_half_open_at_the_upper_bound(repo: Repository) -> None:
    """左闭右开：正好落在右端点上的条目属于下一天，不属于这一天。"""
    boundary = datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc)
    repo.clock = lambda: boundary
    await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="午夜整点那条",
        confidence=0.9, prompt_version="v2",
    )

    in_day = await repo.list_memories_in_range(
        since=datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc), until=boundary
    )
    next_day = await repo.list_memories_in_range(since=boundary, until=None)

    assert in_day == []
    assert len(next_day) == 1


async def test_list_memories_by_id_keeps_the_requested_order(repo: Repository) -> None:
    first = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="甲",
        confidence=0.9, prompt_version="v2",
    )
    second = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="乙",
        confidence=0.9, prompt_version="v2",
    )

    ordered = await repo.list_memories_by_id([int(second.id), int(first.id), 999])

    assert [m.statement for m in ordered] == ["乙", "甲"]
    assert await repo.list_memories_by_id([]) == []


async def test_memories_can_be_filtered_by_person(repo: Repository) -> None:
    """相关人是 JSON 快照，所以这条也在验 ``json_each`` 那条查询真的成立。"""
    await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="小明说的",
        confidence=0.9, prompt_version="v2",
        person_refs=[{"user_id": 10001, "nickname_snapshot": "小明"}],
    )
    await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="小红说的",
        confidence=0.9, prompt_version="v2",
        person_refs=[{"user_id": 10002, "nickname_snapshot": "小红"}],
    )

    hits = await repo.list_memories_in_range(person_id=10001)

    assert [m.statement for m in hits] == ["小明说的"]


async def test_list_messages_for_window_reaches_the_whole_window(repo: Repository) -> None:
    window = await repo.add_window(
        group_id=111,
        started_at=BEGIN,
        ended_at=BEGIN + timedelta(minutes=5),
        message_count=2,
        prompt_version="v2",
    )
    await repo.add_message(group_id=111, user_id=1, text="窗口之前", sent_at=BEGIN - timedelta(minutes=1))
    inside = await repo.add_message(group_id=111, user_id=1, text="窗口之内", sent_at=BEGIN)
    await repo.add_message(group_id=111, user_id=1, text="窗口之后", sent_at=BEGIN + timedelta(minutes=6))

    messages = await repo.list_messages_for_window(int(window.id))

    assert [m.id for m in messages] == [inside.id]
    assert await repo.list_messages_for_window(9999) == []


async def test_group_switches_expose_enabled_and_disabled_sets(repo: Repository) -> None:
    await repo.set_group_enabled(111, True)
    await repo.set_group_enabled(222, False)

    assert await repo.enabled_groups() == [111]
    assert await repo.disabled_groups() == [222]
    assert await repo.group_settings() == {111: True, 222: False}


async def test_purge_unlinks_references_before_deleting(repo: Repository) -> None:
    """外键开着（``PRAGMA foreign_keys=ON``），两处引用必须先解开才能删。"""
    old = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="被推翻的旧条目",
        confidence=0.9, prompt_version="v2",
    )
    new = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="推翻它的新条目",
        confidence=0.9, prompt_version="v2",
    )
    window = await repo.add_window(
        group_id=111, started_at=BEGIN, ended_at=BEGIN, message_count=1, prompt_version="v2"
    )
    replacement = await repo.get_memory(int(new.id))
    assert replacement is not None
    await repo.add_window_result(
        window_id=int(window.id), supersedings=[(int(old.id), replacement)], status=WindowStatus.DONE
    )
    await repo.add_feedback(
        kind=FeedbackKind.FALSE_POSITIVE, memory_id=int(old.id), group_id=111, note="原陈述：…"
    )

    counts = await repo.purge(memory_ids=[int(old.id)])

    assert counts.memories == 1
    assert counts.feedback_unlinked == 1
    assert counts.messages == 0 and counts.windows == 0  # 只删记忆，原文与窗口不动
    assert await repo.get_memory(int(old.id)) is None
    assert await repo.get_window(int(window.id)) is not None
    samples = await repo.list_feedback()
    assert len(samples) == 1 and samples[0].memory_id is None
    assert samples[0].note == "原陈述：…"


async def test_purge_by_group_removes_windows_before_the_text(repo: Repository) -> None:
    window = await repo.add_window(
        group_id=111, started_at=BEGIN, ended_at=BEGIN, message_count=1, prompt_version="v2"
    )
    await repo.add_memory(
        window_id=int(window.id), group_id=111, category=Category.KNOWLEDGE,
        statement="本群的", confidence=0.9, prompt_version="v2",
    )
    await repo.add_message(group_id=111, user_id=1, text="本群的原文", sent_at=BEGIN)
    await repo.add_message(group_id=222, user_id=1, text="别群的原文", sent_at=BEGIN)

    counts = await repo.purge(group_id=111)

    assert (counts.memories, counts.messages, counts.windows) == (1, 1, 1)
    assert await repo.list_group_memories(group_id=111) == []
    assert await repo.get_window(int(window.id)) is None
    assert [message.group_id for message in await repo.list_messages()] == [222]


async def test_purge_everything_clears_the_whole_library(repo: Repository) -> None:
    await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="任意一条",
        confidence=0.9, prompt_version="v2",
    )
    await repo.add_message(group_id=222, user_id=1, text="任意原文", sent_at=BEGIN)

    counts = await repo.purge(everything=True)

    assert (counts.memories, counts.messages) == (1, 1)
    assert await repo.list_messages() == []
    assert await repo.list_memories_in_range(statuses=()) == []


async def test_set_memory_status_reports_a_missing_memory(repo: Repository) -> None:
    memory = await repo.add_memory(
        group_id=111, category=Category.KNOWLEDGE, statement="存在的一条",
        confidence=0.9, prompt_version="v2",
    )

    assert await repo.set_memory_status(int(memory.id), MemoryStatus.EXPIRED) is True
    assert await repo.set_memory_status(9999, MemoryStatus.EXPIRED) is False
    reloaded = await repo.get_memory(int(memory.id))
    assert reloaded is not None and reloaded.status is MemoryStatus.EXPIRED


async def test_list_messages_before_returns_the_nearest_older_messages(repo: Repository) -> None:
    """重试旧窗口时靠它还原「上一窗口尾部」：取最近的几条、按时间升序回来。"""
    for index in range(5):
        await repo.add_message(
            group_id=111,
            user_id=1,
            text=f"第 {index} 条",
            sent_at=BEGIN + timedelta(minutes=index),
        )
    await repo.add_message(group_id=222, user_id=1, text="别的群", sent_at=BEGIN)

    tail = await repo.list_messages_before(
        group_id=111, before=BEGIN + timedelta(minutes=3), limit=2
    )

    assert [message.text for message in tail] == ["第 1 条", "第 2 条"]
    assert await repo.list_messages_before(group_id=111, before=BEGIN, limit=2) == []
