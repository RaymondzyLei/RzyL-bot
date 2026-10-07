"""命令的解析与渲染（故事 27-31、37-38、45）。

解析是纯函数，所以用表格把每种命令的每种写法与每种错法都钉住；渲染同样直接断言文本。
真正查库的部分在 tests/test_memory_console.py。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rzyl_core.db import Category, Memory, MemoryStatus, Message, Window, WindowStatus
from rzyl_core.memory.commands import (
    CommandKind,
    PurgeScope,
    format_help,
    format_search,
    format_sources,
    parse_memory_command,
)
from rzyl_core.memory.retrieval import RetrievalHit, RetrievalOutcome

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)


# —— 解析 ——


@pytest.mark.parametrize(
    "text, kind",
    [
        ("记忆", CommandKind.HELP),
        ("记忆 帮助", CommandKind.HELP),
        ("记忆 今天", CommandKind.TODAY),
        ("记忆 论文", CommandKind.SEARCH),
        ("记忆 分布式 系统 讲义", CommandKind.SEARCH),
        ("记忆 4242", CommandKind.SEARCH),
        ("记忆 搜 群", CommandKind.SEARCH),
        ("记忆 群 100200300", CommandKind.GROUP),
        ("记忆 来源 12", CommandKind.SOURCES),
        ("记忆 误报 12", CommandKind.FALSE_POSITIVE),
        ("记忆 误报 12 这是讨论不是结论", CommandKind.FALSE_POSITIVE),
        ("记忆 恢复 12", CommandKind.RESTORE),
        ("记忆 漏了", CommandKind.MISSING),
        ("记忆 漏了 741 那条三文鱼有用", CommandKind.MISSING),
        ("记忆 漏了 没记下三文鱼那条", CommandKind.MISSING),
        ("记忆 开启 111", CommandKind.ENABLE),
        ("记忆 暂停 111", CommandKind.DISABLE),
        ("记忆 清空 12", CommandKind.PURGE),
        ("记忆 清空 群 111", CommandKind.PURGE),
        ("记忆 清空 全部", CommandKind.PURGE),
    ],
)
def test_each_documented_form_parses(text: str, kind: CommandKind) -> None:
    command = parse_memory_command(text)

    assert command is not None
    assert command.kind is kind


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "在吗",
        "今天天气不错",
        "记一下这个链接",
        "你记忆真好",  # 「记忆」不在开头
    ],
)
def test_non_commands_are_left_alone(text: str) -> None:
    assert parse_memory_command(text) is None


def test_anything_beginning_with_the_prefix_is_treated_as_a_command() -> None:
    """刻意的取舍：不要求「记忆」后面跟空格。

    ``记忆今天`` 这种不走空格的写法在中文里很自然，而**一声不响**最容易让人以为机器人
    坏了；把 ``记忆是个好词`` 当成搜「是个好词」只是多一次无害的检索，两害相权取轻。
    """
    without_space = parse_memory_command("记忆今天")
    sentence = parse_memory_command("记忆是个好词")

    assert without_space is not None and without_space.kind is CommandKind.TODAY
    assert sentence is not None and sentence.kind is CommandKind.SEARCH
    assert sentence.query == "是个好词"


def test_bare_prefix_asks_for_help_rather_than_guessing() -> None:
    bare = parse_memory_command("记忆")
    padded = parse_memory_command("  记忆  ")

    assert bare is not None and bare.kind is CommandKind.HELP
    assert padded is not None and padded.kind is CommandKind.HELP


def test_reserved_word_only_wins_when_it_stands_alone() -> None:
    """刻意的不对称：`记忆 今天有什么` 是搜这几个字，不是日报。"""
    command = parse_memory_command("记忆 今天有什么活动")
    assert command is not None
    assert command.kind is CommandKind.SEARCH
    assert command.query == "今天有什么活动"


def test_explicit_search_escapes_a_reserved_first_word() -> None:
    command = parse_memory_command("记忆 搜 群 号怎么加")
    assert command is not None
    assert command.kind is CommandKind.SEARCH
    assert command.query == "群 号怎么加"


def test_group_and_number_arguments_are_attached() -> None:
    group = parse_memory_command("记忆 群 100200300")
    sources = parse_memory_command("记忆 来源 42")
    purge_group = parse_memory_command("记忆 清空 群 100200300")
    purge_memory = parse_memory_command("记忆 清空 42")

    assert group is not None and group.group_id == 100200300
    assert sources is not None and sources.memory_id == 42
    assert purge_group is not None
    assert (purge_group.purge_scope, purge_group.group_id) == (PurgeScope.GROUP, 100200300)
    assert purge_memory is not None
    assert (purge_memory.purge_scope, purge_memory.memory_id) == (PurgeScope.MEMORY, 42)


def test_missing_takes_a_leading_group_number_otherwise_it_is_all_note() -> None:
    with_group = parse_memory_command("记忆 漏了 100200300 那条三文鱼")
    note_only = parse_memory_command("记忆 漏了 没记下三文鱼那条")

    assert with_group is not None
    assert (with_group.group_id, with_group.note) == (100200300, "那条三文鱼")
    assert note_only is not None
    assert (note_only.group_id, note_only.note) == (None, "没记下三文鱼那条")


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("记忆 群", "群号"),
        ("记忆 群 abc", "群号"),
        ("记忆 来源", "编号"),
        ("记忆 来源 #12", "编号"),
        ("记忆 误报", "编号"),
        ("记忆 误报 零", "编号"),
        ("记忆 恢复 12 多余的话", "多余"),
        ("记忆 开启", "群号"),
        ("记忆 清空", "范围"),
        ("记忆 清空 不知道", "读不懂"),
        ("记忆 清空 群", "群号"),
        ("记忆 搜", "搜什么"),
    ],
)
def test_sloppy_forms_are_rejected_with_a_reason(text: str, fragment: str) -> None:
    """读不懂就回一句原因——用户明确打了命令却什么都不发生，比回用法糟得多。"""
    command = parse_memory_command(text)

    assert command is not None
    assert command.kind is CommandKind.INVALID
    assert command.error is not None and fragment in command.error


# —— 渲染 ——


def test_help_shows_usage_and_the_current_collection_state() -> None:
    text = format_help(
        configured=[111, 222],
        allowed=[111],
        switches={111: True, 222: False},
    )

    assert "记忆 今天" in text
    assert "采集中的群：111" in text
    assert "配置白名单：111、222" in text
    assert "已暂停的群：222" in text


def _memory(memory_id: int = 7, **overrides: object) -> Memory:
    values: dict[str, object] = {
        "id": memory_id,
        "group_id": 111,
        "category": Category.KNOWLEDGE,
        "statement": "虹鳟的脂肪线",
        "confidence": 0.9,
        "prompt_version": "v2",
        "created_at": BEGIN,
    }
    values.update(overrides)
    return Memory(**values)  # pyright: ignore[reportArgumentType]


def test_search_reply_reports_both_arms() -> None:
    outcome = RetrievalOutcome(
        hits=(RetrievalHit(memory=_memory(), score=0.03, keyword_rank=1, vector_rank=2),),
        keyword_hits=1,
        vector_hits=1,
        semantic_available=True,
    )

    text = format_search(outcome, query="论文", tz=CST)

    assert "检索「论文」：命中 1 条" in text
    assert "关键词 1 条 · 语义 1 条" in text
    assert "[#7] 虹鳟的脂肪线" in text


def test_search_reply_says_when_semantics_did_not_run() -> None:
    """语义不可用要说出来：只跑了关键词这件事，用户有权知道。"""
    outcome = RetrievalOutcome(
        hits=(), keyword_hits=0, vector_hits=0, semantic_available=False,
        semantic_error="向量服务不可用",
    )

    text = format_search(outcome, query="三文鱼", tz=CST)

    assert "没有找到条目。" in text
    assert "本次只用了关键词检索" in text
    # 搜不到时给一句出路：把「搜错词」和「确实没记过」分开，也避免误当成机器人坏了。
    assert "发「记忆」看用法" in text


def test_search_reply_flags_partial_degradation_on_a_hit() -> None:
    outcome = RetrievalOutcome(
        hits=(RetrievalHit(memory=_memory(), score=0.01, keyword_rank=1),),
        keyword_hits=1,
        vector_hits=0,
        semantic_available=False,
        semantic_error="向量服务不可用",
    )

    text = format_search(outcome, query="虹鳟", tz=CST)

    assert "本次只用了关键词检索" in text


def test_sources_reply_includes_the_whole_window_and_marks_evidence() -> None:
    """故事 31：要能追到原文窗口，并看出哪几条是这条记忆的依据。"""
    memory = _memory(
        7,
        detail="补充说明",
        evidence=[2],
        person_refs=[{"user_id": 10001, "nickname_snapshot": "小明"}],
    )
    window = Window(
        id=8,
        group_id=111,
        started_at=BEGIN,
        ended_at=BEGIN + timedelta(minutes=5),
        message_count=2,
        status=WindowStatus.DONE,
        created_at=BEGIN,
        updated_at=BEGIN,
    )
    messages = [
        Message(
            id=1, group_id=111, user_id=10002, text="你们看这个",
            sent_at=BEGIN, nickname="小红", created_at=BEGIN,
        ),
        Message(
            id=2, group_id=111, user_id=10001, text="虹鳟的脂肪线很漂亮",
            sent_at=BEGIN + timedelta(minutes=1), card="小明(实验班)", created_at=BEGIN,
        ),
    ]

    text = format_sources(memory=memory, window=window, messages=messages, tz=CST)

    assert "[#7] 虹鳟的脂肪线" in text
    assert "补充：补充说明" in text
    assert "相关人：小明(10001)" in text
    assert "原文窗口 #8" in text and "2 条消息" in text
    assert "小明(实验班)(10001)：虹鳟的脂肪线很漂亮  ← 依据" in text
    assert "你们看这个" in text
    assert "你们看这个  ← 依据" not in text


def test_sources_reply_degrades_clearly_when_the_window_is_gone() -> None:
    text = format_sources(memory=_memory(), window=None, messages=[], tz=CST)

    assert "没有关联的处理窗口" in text


def test_sources_reply_degrades_clearly_when_the_text_expired() -> None:
    window = Window(
        id=8, group_id=111, started_at=BEGIN, ended_at=BEGIN,
        message_count=0, status=WindowStatus.DONE, created_at=BEGIN, updated_at=BEGIN,
    )
    text = format_sources(memory=_memory(), window=window, messages=[], tz=CST)

    assert "已超出保留期" in text


def test_sources_reply_truncates_a_long_window_with_a_visible_note() -> None:
    window = Window(
        id=8, group_id=111, started_at=BEGIN, ended_at=BEGIN,
        message_count=40, status=WindowStatus.DONE, created_at=BEGIN, updated_at=BEGIN,
    )
    messages = [
        Message(
            id=index, group_id=111, user_id=10001, text=f"第 {index} 句",
            sent_at=BEGIN, nickname="小明", created_at=BEGIN,
        )
        for index in range(1, 41)
    ]

    text = format_sources(memory=_memory(), window=window, messages=messages, tz=CST)

    assert "第 20 句" in text
    assert "第 21 句" not in text
    assert "（还有 20 条未列）" in text


def test_sources_reply_truncates_a_single_huge_message_with_an_ellipsis() -> None:
    window = Window(
        id=8, group_id=111, started_at=BEGIN, ended_at=BEGIN,
        message_count=1, status=WindowStatus.DONE, created_at=BEGIN, updated_at=BEGIN,
    )
    messages = [
        Message(
            id=1, group_id=111, user_id=10001, text="很长" * 300,
            sent_at=BEGIN, nickname="小明", created_at=BEGIN,
        )
    ]

    text = format_sources(memory=_memory(), window=window, messages=messages, tz=CST)

    assert "…" in text
    assert "很长" * 300 not in text


def test_dead_letters_are_labelled_in_the_sources_header() -> None:
    window = Window(
        id=9, group_id=111, started_at=BEGIN, ended_at=BEGIN, message_count=1,
        status=WindowStatus.DEAD, created_at=BEGIN, updated_at=BEGIN,
    )
    text = format_sources(memory=_memory(), window=window, messages=[], tz=CST)

    assert "死信" in text


def test_sources_reply_handles_a_memory_with_no_evidence_or_window() -> None:
    text = format_sources(memory=_memory(status=MemoryStatus.EXPIRED), window=None, messages=[], tz=CST)

    assert "[#7] [已过期]" in text
