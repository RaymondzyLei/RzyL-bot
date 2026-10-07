"""记忆的出口：检索、日报渲染、推送投递、命令解析与执行。

这是 issue #1 里 ``memory`` 模块的落点，也是「记忆被消费」的唯一入口——上游写进来
（``pipeline``），这里读出去。对外四样东西：

- :func:`~rzyl_core.memory.retrieval.hybrid_search` —— 关键词 + 语义，RRF 融合；
- :func:`~rzyl_core.memory.report.render_daily_report` —— 日报渲染（纯函数）；
- :mod:`rzyl_core.memory.push` —— 下一次该推的时刻，以及投递器登记处（core 不认识 QQ）；
- :class:`~rzyl_core.memory.console.MemoryConsole` —— 把一条命令变成一段私聊文本。

这一层不依赖 NoneBot（架构约束由 ``tests/test_package_layout.py`` 守着）。
"""

from __future__ import annotations

from rzyl_core.memory.commands import (
    COMMAND_PREFIX,
    CommandKind,
    CommandResult,
    MemoryCommand,
    PurgeScope,
    format_help,
    format_search,
    format_sources,
    parse_memory_command,
)
from rzyl_core.memory.console import MemoryConsole
from rzyl_core.memory.push import (
    ReportSender,
    get_report_sender,
    next_push_at,
    schedule_push_at,
    set_report_sender,
)
from rzyl_core.memory.report import (
    CATEGORY_LABELS,
    CATEGORY_SECTIONS,
    DailyReport,
    format_memory_line,
    render_daily_report,
    render_listing,
)
from rzyl_core.memory.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    RetrievalHit,
    RetrievalOutcome,
    hybrid_search,
    rrf_fuse,
)

__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_SECTIONS",
    "COMMAND_PREFIX",
    "DEFAULT_SEARCH_LIMIT",
    "RRF_K",
    "CommandKind",
    "CommandResult",
    "DailyReport",
    "MemoryCommand",
    "MemoryConsole",
    "PurgeScope",
    "ReportSender",
    "RetrievalHit",
    "RetrievalOutcome",
    "format_help",
    "format_memory_line",
    "format_search",
    "format_sources",
    "get_report_sender",
    "hybrid_search",
    "next_push_at",
    "parse_memory_command",
    "render_daily_report",
    "render_listing",
    "rrf_fuse",
    "schedule_push_at",
    "set_report_sender",
]
