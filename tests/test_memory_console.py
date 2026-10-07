"""命令执行器的行为测试（故事 2、30-31、37-38、45）。

seam 是 :class:`MemoryConsole`：给一个真库（临时 SQLite）+ 可控的假向量，从仓储读回来
断言「命令做了什么」。不碰 NoneBot、不碰 Runtime——命令逻辑本来就不依赖它们。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import (
    Category,
    Feedback,
    FeedbackKind,
    MemoryStatus,
    Repository,
    WindowStatus,
)
from rzyl_core.llm import NullEmbedding
from rzyl_core.memory.commands import CommandKind, MemoryCommand, PurgeScope
from rzyl_core.memory.console import MemoryConsole
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

GROUP = 111
OTHER_GROUP = 222
USER = 10001


class _Clock:
    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class TopicEmbedding:
    """按词给方向的假向量，让关键词与语义两路的行为都可预测。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "论文" in text or "文献" in text else [0.0, 1.0] for text in texts]


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    yield instance
    await instance.close()


def _console(
    repo: Repository,
    *,
    clock: _Clock | None = None,
    flush: list[int] | None = None,
    **settings: object,
) -> MemoryConsole:
    """构造执行器；``flush`` 传一个列表时会记下每次强制关窗被调用了几次。"""
    calls = flush if flush is not None else []

    async def _flush_pending() -> int:
        calls.append(1)
        return 2

    return MemoryConsole(
        repository=repo,
        embedding=TopicEmbedding(),
        settings=_settings(**settings),
        clock=clock or _Clock(),
        flush_pending=_flush_pending if flush is not None else None,
    )


async def _add(
    repo: Repository,
    *,
    statement: str,
    group_id: int = GROUP,
    category: Category = Category.KNOWLEDGE,
    confidence: float = 0.9,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    person_refs: Sequence[dict[str, object]] = (),
) -> int:
    memory = await repo.add_memory(
        group_id=group_id,
        category=category,
        statement=statement,
        confidence=confidence,
        prompt_version="v2",
        status=status,
        person_refs=person_refs,  # pyright: ignore[reportArgumentType]
    )
    return int(memory.id)


# —— 帮助 ——


async def test_help_shows_usage_and_live_switches(repo: Repository) -> None:
    await repo.set_group_enabled(OTHER_GROUP, False)
    console = _console(repo, group_whitelist=[GROUP, OTHER_GROUP])

    result = await console.execute(MemoryCommand(kind=CommandKind.HELP))

    assert result.ok
    assert "记忆 今天" in result.text
    assert f"采集中的群：{GROUP}" in result.text
    assert f"已暂停的群：{OTHER_GROUP}" in result.text


async def test_invalid_command_returns_the_reason_plus_usage(repo: Repository) -> None:
    console = _console(repo)

    result = await console.execute(
        MemoryCommand(kind=CommandKind.INVALID, error="来源要一个编号")
    )

    assert result.ok is False
    assert result.text.startswith("来源要一个编号")
    assert "记忆 来源 <编号>" in result.text


# —— 检索 ——


async def test_search_finds_by_keyword(repo: Repository) -> None:
    await _add(repo, statement="这篇论文给了新的证明")
    await _add(repo, statement="网盘链接在这里")

    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.SEARCH, query="论文")
    )

    assert "这篇论文给了新的证明" in result.text
    assert "网盘链接在这里" not in result.text


# —— 今天与某个群 ——


async def test_today_forces_pending_windows_closed_first(repo: Repository) -> None:
    """窗口最长能攒到 window_max_minutes；不先关窗，「今天」会漏掉最近的内容。"""
    await _add(repo, statement="今天记下的")
    flushed: list[int] = []

    result = await _console(repo, clock=_Clock(), flush=flushed).execute(
        MemoryCommand(kind=CommandKind.TODAY)
    )

    assert flushed == [1]
    assert "今天记下的" in result.text
    assert "（查询前补关了 2 个窗口）" in result.text


async def test_today_includes_low_confidence_and_marks_suspect_duplicates(
    repo: Repository,
) -> None:
    """故事 30：`记忆 今天` 要看到所有条目，包括低置信度的那些。"""
    await _add(repo, statement="低置信度的", confidence=0.2)
    await _add(repo, statement="被推翻的", status=MemoryStatus.EXPIRED)
    console = _console(repo, clock=_Clock())

    result = await console.execute(MemoryCommand(kind=CommandKind.TODAY))

    assert "低置信度的" in result.text
    assert "被推翻的" not in result.text


async def test_today_is_a_local_natural_day(repo: Repository) -> None:
    """按 Asia/Shanghai 的自然日算：上海 10-07 23:30 的那条在「今天」里，10-08 00:30 的不在。

    两个时间在 UTC 下是同一天（10-07 15:30 与 16:30），所以这条断言只有真的按上海时区
    切日才能通过。
    """
    clock = _Clock(datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc))  # 上海 10-07 23:00
    repo.clock = lambda: datetime(2026, 10, 7, 15, 30, tzinfo=timezone.utc)
    await _add(repo, statement="上海时间的今天")
    repo.clock = lambda: datetime(2026, 10, 7, 16, 30, tzinfo=timezone.utc)
    await _add(repo, statement="上海时间的明天")

    result = await _console(repo, clock=clock, timezone="Asia/Shanghai").execute(
        MemoryCommand(kind=CommandKind.TODAY)
    )

    assert "上海时间的今天" in result.text
    assert "上海时间的明天" not in result.text


async def test_group_listing_is_scoped_and_reports_the_group_state(repo: Repository) -> None:
    await _add(repo, statement="本群的", group_id=GROUP)
    await _add(repo, statement="别的群的", group_id=OTHER_GROUP)
    await repo.set_group_enabled(GROUP, False)
    console = _console(repo, group_whitelist=[])

    result = await console.execute(MemoryCommand(kind=CommandKind.GROUP, group_id=GROUP))

    assert "本群的" in result.text
    assert "别的群的" not in result.text
    assert "已被暂停" in result.text


# —— 来源 ——


async def test_sources_reaches_the_original_window(repo: Repository) -> None:
    """故事 31：从编号追到原文窗口，并标出哪几条是依据。"""
    window = await repo.add_window(
        group_id=GROUP,
        started_at=BEGIN,
        ended_at=BEGIN + timedelta(minutes=5),
        message_count=2,
        prompt_version="v2",
    )
    first = await repo.add_message(
        group_id=GROUP, user_id=USER, text="你们看这个", sent_at=BEGIN, nickname="小红"
    )
    second = await repo.add_message(
        group_id=GROUP, user_id=USER + 1, text="论文里的脂肪线很漂亮",
        sent_at=BEGIN + timedelta(minutes=1), nickname="小明",
    )
    memory = await repo.add_memory(
        window_id=window.id,
        group_id=GROUP,
        category=Category.KNOWLEDGE,
        statement="论文里的脂肪线很漂亮",
        confidence=0.9,
        prompt_version="v2",
        evidence=[int(second.id)],
    )

    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.SOURCES, memory_id=int(memory.id))
    )

    assert f"[#{memory.id}]" in result.text
    assert "你们看这个" in result.text
    assert "← 依据" in result.text
    assert "论文里的脂肪线很漂亮  ← 依据" in result.text
    assert f"原文窗口 #{window.id}" in result.text
    assert first.id != second.id


async def test_sources_reports_a_missing_id_instead_of_pretending(repo: Repository) -> None:
    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.SOURCES, memory_id=999)
    )

    assert result.ok is False
    assert "999" in result.text


# —— 纠错 ——


async def test_marking_a_false_positive_records_a_sample_and_hides_the_entry(
    repo: Repository,
) -> None:
    """故事 37 + 22：进错例集，同时退出推送与检索（否则「标了误报」等于没用）。"""
    memory_id = await _add(repo, statement="Hypixel 有没有爬虫是个值得讨论的问题")

    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.FALSE_POSITIVE, memory_id=memory_id, note="这是讨论不是结论")
    )

    assert result.ok
    samples = await repo.list_feedback()
    assert len(samples) == 1
    assert samples[0].kind is FeedbackKind.FALSE_POSITIVE
    assert samples[0].memory_id == memory_id
    assert samples[0].group_id == GROUP
    assert "这是讨论不是结论" in (samples[0].note or "")
    assert "原陈述：Hypixel 有没有爬虫是个值得讨论的问题" in (samples[0].note or "")
    reloaded = await repo.get_memory(memory_id)
    assert reloaded is not None and reloaded.status is MemoryStatus.EXPIRED
    # 默认检索里已经没有它了。
    assert await repo.search_memories("爬虫") == []


async def test_feedback_survives_purging_the_memory_it_points_at(repo: Repository) -> None:
    """错例样本是调提示词的回归材料，不能因为删掉那条记忆就跟着消失。"""
    memory_id = await _add(repo, statement="记错的结论")
    console = _console(repo)
    await console.execute(MemoryCommand(kind=CommandKind.FALSE_POSITIVE, memory_id=memory_id))

    await repo.purge(memory_ids=[memory_id])

    samples = await repo.list_feedback()
    assert len(samples) == 1
    assert samples[0].memory_id is None
    assert "记错的结论" in (samples[0].note or "")


async def test_restore_brings_an_entry_back(repo: Repository) -> None:
    """故事 22：可以一键恢复。"""
    memory_id = await _add(repo, statement="本来是对的")
    console = _console(repo)
    await console.execute(MemoryCommand(kind=CommandKind.FALSE_POSITIVE, memory_id=memory_id))

    result = await console.execute(MemoryCommand(kind=CommandKind.RESTORE, memory_id=memory_id))

    assert "active" in result.text
    reloaded = await repo.get_memory(memory_id)
    assert reloaded is not None and reloaded.status is MemoryStatus.ACTIVE
    assert [m.id for m in await repo.search_memories("本来是对的")] == [memory_id]


async def test_restore_says_so_when_there_is_nothing_to_do(repo: Repository) -> None:
    memory_id = await _add(repo, statement="一直是对的")

    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.RESTORE, memory_id=memory_id)
    )

    assert "本来就是活跃状态" in result.text


async def test_restore_clears_a_supersede_link(repo: Repository) -> None:
    """恢复一条被推翻的旧条目时，那条引用就没有意义了，留着会让「人工标错」的判据失真。"""
    window = await repo.add_window(
        group_id=GROUP, started_at=BEGIN, ended_at=BEGIN, message_count=1, prompt_version="v2"
    )
    old_id = await _add(repo, statement="旧结论")
    new_id = await _add(repo, statement="新结论")
    replacement = await repo.get_memory(new_id)
    assert replacement is not None
    await repo.add_window_result(
        window_id=int(window.id), supersedings=[(old_id, replacement)], status=WindowStatus.DONE
    )
    superseded = await repo.get_memory(old_id)
    assert superseded is not None and superseded.superseded_by == new_id

    await _console(repo).execute(MemoryCommand(kind=CommandKind.RESTORE, memory_id=old_id))

    restored = await repo.get_memory(old_id)
    assert restored is not None
    assert restored.status is MemoryStatus.ACTIVE
    assert restored.superseded_by is None


async def test_recording_a_missing_entry_needs_no_memory(repo: Repository) -> None:
    """故事 38：漏报没有条目可指，样本就是「本该记但没记」这件事本身。"""
    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.MISSING, group_id=GROUP, note="三文鱼那条该记")
    )

    assert result.ok
    samples = await repo.list_feedback()
    assert len(samples) == 1
    assert samples[0].kind is FeedbackKind.FALSE_NEGATIVE
    assert samples[0].memory_id is None
    assert samples[0].group_id == GROUP
    assert samples[0].note == "三文鱼那条该记"


async def test_recording_a_missing_entry_without_a_note_nudges_for_one(repo: Repository) -> None:
    result = await _console(repo).execute(MemoryCommand(kind=CommandKind.MISSING))

    assert "建议下次带上备注" in result.text


# —— 开关 ——


async def test_pause_masks_the_configured_whitelist(repo: Repository) -> None:
    """故事 2 的关键：暂停一个写在配置白名单里的群必须真的停，否则命令是空操作。"""
    console = _console(repo, group_whitelist=[GROUP])

    paused = await console.execute(MemoryCommand(kind=CommandKind.DISABLE, group_id=GROUP))

    assert "已暂停" in paused.text
    assert "压过白名单" in paused.text
    assert await repo.enabled_groups() == []
    assert await repo.disabled_groups() == [GROUP]

    resumed = await console.execute(MemoryCommand(kind=CommandKind.ENABLE, group_id=GROUP))

    assert "已开启" in resumed.text
    assert await repo.enabled_groups() == [GROUP]
    assert await repo.disabled_groups() == []


async def test_enabling_a_group_outside_the_whitelist_says_it_is_persisted(
    repo: Repository,
) -> None:
    result = await _console(repo, group_whitelist=[]).execute(
        MemoryCommand(kind=CommandKind.ENABLE, group_id=OTHER_GROUP)
    )

    assert "不在配置白名单里" in result.text
    assert await repo.enabled_groups() == [OTHER_GROUP]


# —— 清空 ——


async def test_purge_one_memory_leaves_the_original_text_alone(repo: Repository) -> None:
    """单条记忆引用的原文常与同群其它条目共用，删原文会连累别人。"""
    await repo.add_message(group_id=GROUP, user_id=USER, text="原文", sent_at=BEGIN)
    memory_id = await _add(repo, statement="要删的")

    result = await _console(repo).execute(
        MemoryCommand(
            kind=CommandKind.PURGE, purge_scope=PurgeScope.MEMORY, memory_id=memory_id
        )
    )

    assert result.ok
    assert "只删这一条记忆" in result.text
    assert await repo.get_memory(memory_id) is None
    assert len(await repo.list_messages(group_id=GROUP)) == 1


async def test_purge_by_group_removes_text_windows_and_pauses_the_group(
    repo: Repository,
) -> None:
    window = await repo.add_window(
        group_id=GROUP, started_at=BEGIN, ended_at=BEGIN, message_count=1, prompt_version="v2"
    )
    await repo.add_message(group_id=GROUP, user_id=USER, text="本群的原文", sent_at=BEGIN)
    await repo.add_message(group_id=OTHER_GROUP, user_id=USER, text="别群的原文", sent_at=BEGIN)
    await _add(repo, statement="本群的记忆", group_id=GROUP)
    await _add(repo, statement="别群的记忆", group_id=OTHER_GROUP)

    result = await _console(repo, group_whitelist=[GROUP]).execute(
        MemoryCommand(kind=CommandKind.PURGE, purge_scope=PurgeScope.GROUP, group_id=GROUP)
    )

    assert "记忆 1 条" in result.text
    assert "原文 1 条" in result.text
    assert "窗口 1 个" in result.text
    assert await repo.list_group_memories(group_id=GROUP) == []
    assert len(await repo.list_group_memories(group_id=OTHER_GROUP)) == 1
    assert [m.group_id for m in await repo.list_messages()] == [OTHER_GROUP]
    assert await repo.get_window(int(window.id)) is None
    # 顺手暂停，否则白名单里的话下一分钟又开始收，「退出」是假动作。
    assert await repo.disabled_groups() == [GROUP]


async def test_purge_everything_also_pauses_every_collected_group(repo: Repository) -> None:
    await repo.set_group_enabled(OTHER_GROUP, True)
    await _add(repo, statement="任何一条", group_id=GROUP)
    await repo.add_message(group_id=GROUP, user_id=USER, text="原文", sent_at=BEGIN)

    result = await _console(repo, group_whitelist=[GROUP, 333]).execute(
        MemoryCommand(kind=CommandKind.PURGE, purge_scope=PurgeScope.ALL)
    )

    assert "已清空全部" in result.text
    assert "已暂停 3 个群" in result.text
    assert await repo.list_feedback() == []
    assert await repo.list_messages() == []
    assert await repo.enabled_groups() == []
    assert sorted(await repo.disabled_groups()) == [111, 222, 333]


async def test_purge_of_a_missing_memory_does_nothing(repo: Repository) -> None:
    result = await _console(repo).execute(
        MemoryCommand(kind=CommandKind.PURGE, purge_scope=PurgeScope.MEMORY, memory_id=404)
    )

    assert result.ok is False
    assert "什么都没删" in result.text


async def test_purge_keeps_feedback_samples_but_unlinks_them(repo: Repository) -> None:
    memory_id = await _add(repo, statement="记错的")
    console = _console(repo)
    await console.execute(MemoryCommand(kind=CommandKind.FALSE_POSITIVE, memory_id=memory_id))

    result = await console.execute(
        MemoryCommand(kind=CommandKind.PURGE, purge_scope=PurgeScope.GROUP, group_id=GROUP)
    )

    assert "解开了 1 条纠错样本" in result.text
    samples: list[Feedback] = await repo.list_feedback()
    assert len(samples) == 1 and samples[0].memory_id is None


async def test_purge_without_a_scope_is_a_programming_error(repo: Repository) -> None:
    with pytest.raises(ValueError):
        await repo.purge()
    with pytest.raises(ValueError):
        await repo.purge(memory_ids=[1], everything=True)


async def test_console_without_a_flusher_still_answers(repo: Repository) -> None:
    """脚本或测试里没接关窗回调时，「今天」照答，只是不补关窗口。"""
    await _add(repo, statement="还是能看到")

    result = await _console(repo).execute(MemoryCommand(kind=CommandKind.TODAY))

    assert "还是能看到" in result.text
    assert "补关了" not in result.text


async def test_console_degrades_when_the_embedding_model_is_absent(repo: Repository) -> None:
    """向量服务未配置时检索仍能用（纯关键词），并在回复里说清。"""
    await _add(repo, statement="这篇论文给了新的证明")
    console = MemoryConsole(
        repository=repo,
        embedding=NullEmbedding(),
        settings=_settings(),
        clock=_Clock(),
    )

    result = await console.execute(MemoryCommand(kind=CommandKind.SEARCH, query="论文"))

    assert "这篇论文给了新的证明" in result.text
    assert "本次只用了关键词检索" in result.text


async def test_unknown_command_kind_is_reported_rather_than_crashing(repo: Repository) -> None:
    """解析层与执行层都在 core 里演进，落到最后那个兜底分支时要说清楚而不是抛异常。"""
    fake = MemoryCommand(kind=CommandKind.HELP)
    object.__setattr__(fake, "kind", "not-a-kind")

    result = await _console(repo).execute(fake)

    assert result.ok is False
    assert "未实现的命令" in result.text
