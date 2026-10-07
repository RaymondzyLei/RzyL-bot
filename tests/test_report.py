"""日报与列举的渲染（故事 30、33-36）。

渲染是纯函数，所以这里直接给条目对象、断言文本——不建库、不调模型。断言的都是
用户能看到的东西：分节顺序、编号、低置信度只报数不列出、当天没有也要有一句话。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from rzyl_core.db import Category, Memory, MemoryStatus
from rzyl_core.memory.report import (
    CATEGORY_SECTIONS,
    format_memory_line,
    render_daily_report,
    render_listing,
)

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)
DAY = date(2026, 10, 7)


def _memory(
    memory_id: int,
    *,
    category: Category = Category.KNOWLEDGE,
    statement: str = "一条结论",
    confidence: float = 0.9,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    group_id: int = 111,
    created_at: datetime = BEGIN,
    detail: str | None = None,
    person_refs: list[dict[str, object]] | None = None,
) -> Memory:
    return Memory(
        id=memory_id,
        group_id=group_id,
        category=category,
        statement=statement,
        detail=detail,
        confidence=confidence,
        person_refs=person_refs or [],
        prompt_version="v2",
        status=status,
        created_at=created_at,
    )


def test_sections_follow_the_declared_order() -> None:
    """故事 36：事务与请求排在最前——它们最像提醒。"""
    assert [category for category, _ in CATEGORY_SECTIONS] == [
        Category.EVENT,
        Category.REQUEST,
        Category.RESOURCE,
        Category.KNOWLEDGE,
    ]


def test_daily_report_lists_entries_by_section_with_ids() -> None:
    report = render_daily_report(
        [
            _memory(3, category=Category.KNOWLEDGE, statement="虹鳟的脂肪线"),
            _memory(1, category=Category.EVENT, statement="实验改到周五"),
            _memory(2, category=Category.REQUEST, statement="有人求一份讲义"),
        ],
        day=DAY,
        threshold=0.7,
    )

    assert "10-07 记忆日报" in report.text
    event_at = report.text.index("【事务】")
    request_at = report.text.index("【请求】")
    knowledge_at = report.text.index("【知识】")
    assert event_at < request_at < knowledge_at
    assert report.text.index("[#1] 实验改到周五") < report.text.index("[#2] 有人求一份讲义")
    assert report.listed == 3
    assert report.total == 3
    assert report.memory_ids == (1, 2, 3)


def test_daily_report_omits_low_confidence_entries_but_counts_them() -> None:
    """故事 34：低置信度的存档但不打扰，末尾要报数。"""
    report = render_daily_report(
        [
            _memory(1, statement="可信的", confidence=0.9),
            _memory(2, statement="勉强记下的", confidence=0.4),
            _memory(3, statement="更没把握的", confidence=0.2),
        ],
        day=DAY,
        threshold=0.7,
    )

    assert "可信的" in report.text
    assert "勉强记下的" not in report.text
    assert "更没把握的" not in report.text
    assert report.low_confidence == 2
    assert report.listed == 1
    assert "今天共 3 条，另有 2 条低置信度未列。" in report.text


def test_daily_report_says_so_when_there_is_nothing() -> None:
    """故事 35：没有也要发一句，否则「真没东西」与「机器人挂了」长得一样。"""
    report = render_daily_report([], day=DAY, threshold=0.7)

    assert report.text == "📋 10-07 记忆日报\n\n今天没有值得记的。"
    assert report.total == 0 and report.listed == 0 and report.low_confidence == 0


def test_daily_report_reports_the_overflow_instead_of_truncating_silently() -> None:
    memories = [_memory(index, statement=f"第 {index} 条") for index in range(1, 7)]

    report = render_daily_report(memories, day=DAY, threshold=0.0, max_entries=4)

    assert report.listed == 4
    assert report.dropped == 2
    assert "还有 2 条因超出单条消息上限未列" in report.text


def test_daily_report_mentions_forced_flushes() -> None:
    report = render_daily_report(
        [_memory(1)], day=DAY, threshold=0.7, flushed_windows=3
    )

    assert "（推送前补关了 3 个窗口）" in report.text
    assert report.flushed_windows == 3


def test_daily_report_with_only_low_confidence_entries_has_no_section() -> None:
    report = render_daily_report([_memory(1, confidence=0.1)], day=DAY, threshold=0.7)

    assert "【" not in report.text
    assert "今天共 1 条，另有 1 条低置信度未列。" in report.text


def test_format_memory_line_carries_group_confidence_and_status_mark() -> None:
    active = _memory(7, statement="三文鱼", confidence=0.85, group_id=100200301)
    suspect = _memory(8, statement="疑似重复的那条", status=MemoryStatus.SUSPECT_DUPLICATE)
    expired = _memory(9, statement="过期的", status=MemoryStatus.EXPIRED)

    assert format_memory_line(active) == "[#7] 三文鱼（群 100200301 · 置信度 0.85）"
    assert format_memory_line(suspect).startswith("[#8] [疑似重复] ")
    assert format_memory_line(expired).startswith("[#9] [已过期] ")


def test_format_memory_line_adds_local_time_only_when_asked() -> None:
    memory = _memory(5, created_at=datetime(2026, 10, 7, 9, 30, tzinfo=CST))

    assert "10-07 09:30" not in format_memory_line(memory)
    assert "10-07 09:30" in format_memory_line(memory, tz=CST, with_date=True)


def test_listing_includes_low_confidence_and_marks_suspect_duplicates() -> None:
    """故事 30：`记忆 今天` 要看得到所有条目，包括低置信度的那些。"""
    text = render_listing(
        [
            _memory(2, statement="低置信度但记下了", confidence=0.2),
            _memory(1, statement="正常的", confidence=0.95),
            _memory(3, statement="疑似重复", status=MemoryStatus.SUSPECT_DUPLICATE),
        ],
        header="今天（10-07）新增的条目",
        tz=CST,
    )

    assert "低置信度但记下了" in text
    assert "[疑似重复]" in text
    # 按编号升序：编号就是时间顺序，输出可复现。
    assert text.index("[#1]") < text.index("[#2]") < text.index("[#3]")


def test_listing_reports_an_empty_day_and_an_overflow() -> None:
    assert render_listing([], header="今天新增的条目", empty_text="今天还没有条目。").endswith(
        "今天还没有条目。"
    )

    many = [_memory(index, statement=f"第 {index} 条") for index in range(1, 6)]
    text = render_listing(many, header="某个群", max_entries=3)
    assert "还有 2 条未列" in text
    assert "第 5 条" not in text


def test_listing_appends_the_caller_supplied_footer() -> None:
    text = render_listing(
        [_memory(1)], header="今天新增的条目", footer="（查询前补关了 2 个窗口）"
    )
    assert text.endswith("（查询前补关了 2 个窗口）")


@pytest.mark.parametrize("confidence", [0.7, 0.99, 1.0])
def test_threshold_is_inclusive(confidence: float) -> None:
    """>= 阈值就列出：边界上的条目不该因为浮点写法被悄悄丢掉。"""
    report = render_daily_report([_memory(1, confidence=confidence)], day=DAY, threshold=0.7)
    assert report.listed == 1
