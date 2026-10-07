"""日报与列举的渲染：把条目列表变成一段能直接发出去的私聊文本。

全是**纯函数**（输入条目、输出字符串），所以不依赖 Runtime、不碰网络，可以直接断言。
本节措辞本身就是产品决策，逐条说明理由：

- **分节顺序**是 事务 → 请求 → 资源 → 知识（故事 36）：前两类最像提醒，先看它们。
- **低置信度不进日报**但在末尾报数（故事 34）：既不打扰，也知道自己的噪声水平。
- **当天没有也发一句**（故事 35）：否则「真的没东西」和「机器人挂了」长得一样。
- **条数超上限时说清还有多少**：绝不静默截断——截断不报是本项目最忌讳的失败方式。
- 状态标记：疑似重复标 ``[疑似重复]``、被推翻的标 ``[已过期]``，人工列举时看得见质量。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, tzinfo

from rzyl_core.db import Category, Memory, MemoryStatus

#: 日报分节的顺序与中文标题。
CATEGORY_SECTIONS: tuple[tuple[Category, str], ...] = (
    (Category.EVENT, "事务"),
    (Category.REQUEST, "请求"),
    (Category.RESOURCE, "资源"),
    (Category.KNOWLEDGE, "知识"),
)

#: 条目状态在文本里的标记；``active`` 不带标记。
STATUS_MARKS: dict[MemoryStatus, str] = {
    MemoryStatus.SUSPECT_DUPLICATE: "[疑似重复] ",
    MemoryStatus.EXPIRED: "[已过期] ",
}


@dataclass(frozen=True, slots=True)
class DailyReport:
    """一次日报的产物：文本 + 可核对的计数。

    计数单独留着，是为了让「日报正文」与「到底有多少条」可以分别断言——正文里少一条
    可能是截断、也可能是过滤，只有计数能说清是哪一种。
    """

    day: date
    text: str
    total: int
    """当天**活跃**条目总数（含低于置信度阈值、未列出的那些）。"""

    listed: int
    """正文里实际列出的条数。"""

    low_confidence: int
    """因低于阈值而未列出的条数。"""

    dropped: int = 0
    """因超出单条消息条数上限而未列出的条数。"""

    flushed_windows: int = 0
    """推送前为了不漏当日内容而强制关掉的缓冲窗口数。"""

    memory_ids: tuple[int, ...] = ()
    """正文里列出的条目编号，便于人工核对与后续操作。"""


def format_memory_line(
    memory: Memory, *, tz: tzinfo | None = None, with_date: bool = False
) -> str:
    """一行一条：``[#编号] 陈述（群 123 · 置信度 0.90）``。

    ``tz`` 与 ``with_date`` 只在跨时间的列举（检索结果）里有意义：日报按天分节，日期在
    标题上，行里再写一遍只是噪声。
    """
    mark = STATUS_MARKS.get(memory.status, "")
    parts = [f"群 {memory.group_id}"]
    if with_date and tz is not None and memory.created_at is not None:
        parts.append(memory.created_at.astimezone(tz).strftime("%m-%d %H:%M"))
    parts.append(f"置信度 {memory.confidence:.2f}")
    return f"[#{memory.id}] {mark}{memory.statement}（{' · '.join(parts)}）"


def _confidence_footer(total: int, low_confidence: int, dropped: int) -> str:
    """日报末尾那一句：共几条、另有多少条没列、为什么没列。"""
    parts = [f"今天共 {total} 条"]
    if low_confidence:
        parts.append(f"另有 {low_confidence} 条低置信度未列")
    if dropped:
        parts.append(f"还有 {dropped} 条因超出单条消息上限未列")
    return "，".join(parts) + "。"


def render_daily_report(
    memories: Sequence[Memory],
    *,
    day: date,
    threshold: float,
    max_entries: int = 50,
    flushed_windows: int = 0,
) -> DailyReport:
    """把某天的**活跃**条目渲染成一条日报。

    ``memories`` 只放 ``active`` 的条目（疑似重复按设计不推送，故事 25）；本函数再按
    置信度分两拨：不低于 ``threshold`` 的列进正文，其余只计数。同类别内按编号升序——
    编号就是时间顺序，且不受列表传入顺序影响，输出可复现。
    """
    header = f"📋 {day.strftime('%m-%d')} 记忆日报"
    low_confidence_total = sum(1 for memory in memories if memory.confidence < threshold)
    if not memories:
        text = f"{header}\n\n今天没有值得记的。"
        return DailyReport(day=day, text=text, total=0, listed=0, low_confidence=0,
                           flushed_windows=flushed_windows)

    listed_pool = sorted(
        (memory for memory in memories if memory.confidence >= threshold),
        key=lambda memory: int(memory.id),
    )
    listed = listed_pool[: max(0, max_entries)]
    dropped = len(listed_pool) - len(listed)

    blocks: list[str] = [header]
    for category, label in CATEGORY_SECTIONS:
        section = [memory for memory in listed if memory.category is category]
        if not section:
            continue
        lines = "\n".join(format_memory_line(memory) for memory in section)
        blocks.append(f"【{label}】\n{lines}")

    footer = _confidence_footer(len(memories), low_confidence_total, dropped)
    if flushed_windows:
        footer += f"（推送前补关了 {flushed_windows} 个窗口）"
    blocks.append(footer)

    return DailyReport(
        day=day,
        text="\n\n".join(blocks),
        total=len(memories),
        listed=len(listed),
        low_confidence=low_confidence_total,
        dropped=dropped,
        flushed_windows=flushed_windows,
        memory_ids=tuple(int(memory.id) for memory in listed),
    )


def render_listing(
    memories: Sequence[Memory],
    *,
    header: str,
    tz: tzinfo | None = None,
    max_entries: int = 50,
    empty_text: str = "没有找到条目。",
    footer: str | None = None,
) -> str:
    """把一组条目渲染成一段列举文本，用于「记忆 今天」「记忆 群」。

    与日报的区别：不按置信度过滤（故事 30 明确要求今天能看到低置信度的），只按编号
    升序列全；超出上限时说明还剩多少。
    """
    ordered = sorted(memories, key=lambda memory: int(memory.id))
    if not ordered:
        return f"{header}\n\n{empty_text}"
    shown = ordered[: max(0, max_entries)]
    lines = "\n".join(format_memory_line(memory, tz=tz, with_date=True) for memory in shown)
    blocks = [header, lines]
    if len(ordered) > len(shown):
        blocks.append(f"（还有 {len(ordered) - len(shown)} 条未列，可缩小范围再查）")
    if footer:
        blocks.append(footer)
    return "\n\n".join(blocks)


__all__ = [
    "CATEGORY_SECTIONS",
    "STATUS_MARKS",
    "DailyReport",
    "format_memory_line",
    "render_daily_report",
    "render_listing",
]
