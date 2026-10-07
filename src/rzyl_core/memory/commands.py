"""命令的解析与渲染：``记忆 …`` 这行字 ↔ 一个可执行的结构。

这一层是**纯的**：解析只把字符串变成 :class:`MemoryCommand`，渲染只把已经查好的数据
变成文本，不碰仓储、不碰网络。真正去查库的部分在 :mod:`rzyl_core.memory.console`。

解析规则有两条刻意的不对称，先说清楚：

1. **保留字只有在「单独成词且带齐它的参数」时才是保留字**。所以 ``记忆 今天`` 是日报，
   而 ``记忆 今天有什么活动`` 是搜这几个字——不然用户想搜「今天」这个词就无路可走。
   真要搜的词恰好以保留字开头，用 ``记忆 搜 <关键词>``。
2. **凡是以「记忆」开头的私聊文本都算命令**（不要求后面跟空格）。中文里不走空格的写法
   很自然（``记忆今天``），而**没有任何回复**是最容易让人以为「机器人坏了」的结果；
   反过来，把 ``记忆是个好词`` 当成搜「是个好词」只是多一次无害的检索。两害相权，
   取「宁可多答一次」。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from enum import StrEnum

from rzyl_core.db import Memory, Message, Window, WindowStatus
from rzyl_core.memory.report import format_memory_line
from rzyl_core.memory.retrieval import RetrievalOutcome

#: 命令前缀。私聊里以它开头的消息才会被当成记忆命令。
COMMAND_PREFIX = "记忆"

#: 单条「来源」回复里最多列多少条原文、每条截断到多少字、整段软上限多少字。
SOURCES_MAX_MESSAGES = 20
SOURCES_MESSAGE_MAX_CHARS = 120
SOURCES_TOTAL_MAX_CHARS = 1800


class CommandKind(StrEnum):
    """一条记忆命令的种类。``INVALID`` 是「看着像命令但读不懂」，要回用法而不是静默。"""

    HELP = "help"
    SEARCH = "search"
    TODAY = "today"
    GROUP = "group"
    SOURCES = "sources"
    FALSE_POSITIVE = "false_positive"
    RESTORE = "restore"
    MISSING = "missing"
    ENABLE = "enable"
    DISABLE = "disable"
    PURGE = "purge"
    INVALID = "invalid"


class PurgeScope(StrEnum):
    """``记忆 清空`` 的范围。三者不可逆，所以必须显式给一个。"""

    MEMORY = "memory"
    GROUP = "group"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class MemoryCommand:
    """一条已解析的命令。字段按种类取用，用不到的一律保持默认值。"""

    kind: CommandKind
    query: str = ""
    group_id: int | None = None
    memory_id: int | None = None
    note: str | None = None
    purge_scope: PurgeScope | None = None
    error: str | None = None
    """``INVALID`` 时说明哪里读不懂，直接回给用户。"""

    @property
    def raw(self) -> str:
        """原始入参的简短回显（用于「没找到抄错的编号」这类回复）。"""
        if self.query:
            return self.query
        if self.memory_id is not None:
            return str(self.memory_id)
        if self.group_id is not None:
            return str(self.group_id)
        return ""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """一条命令的执行结果：要发回私聊的文本，以及它是否成功。"""

    text: str
    ok: bool = True


#: 无参数（或参数在行尾）的保留字。
_BARE_KEYWORDS: dict[str, CommandKind] = {
    "今天": CommandKind.TODAY,
    "帮助": CommandKind.HELP,
    "help": CommandKind.HELP,
    "?": CommandKind.HELP,
}

#: 需要一个群号参数的保留字。
_GROUP_KEYWORDS: dict[str, CommandKind] = {
    "群": CommandKind.GROUP,
    "开启": CommandKind.ENABLE,
    "打开": CommandKind.ENABLE,
    "暂停": CommandKind.DISABLE,
    "关闭": CommandKind.DISABLE,
}

_ALL_WORDS = frozenset({"全部", "所有", "all"})
_GROUP_WORDS = frozenset({"群", "群号"})


def _positive_int(text: str) -> int | None:
    """把一串数字读成正整数；读不出或不是正数就返回 ``None``。"""
    stripped = text.strip()
    if not stripped.isdigit():
        return None
    value = int(stripped)
    return value if value > 0 else None


def _split_head(rest: str) -> tuple[str, str]:
    """把「第一个词」与「剩下的部分」分开；两者都去空白。"""
    head, _, tail = rest.partition(" ")
    return head.strip(), tail.strip()


def _parse_purge(tail: str) -> MemoryCommand:
    """``清空`` 的三个范围：``<编号>`` / ``群 <群号>`` / ``全部``。"""
    head, rest = _split_head(tail)
    if not head:
        return MemoryCommand(
            kind=CommandKind.INVALID,
            error="清空要指定范围：记忆 清空 <编号> / 记忆 清空 群 <群号> / 记忆 清空 全部",
        )
    if head in _ALL_WORDS and not rest:
        return MemoryCommand(kind=CommandKind.PURGE, purge_scope=PurgeScope.ALL)
    if head in _GROUP_WORDS:
        group_id = _positive_int(rest)
        if group_id is None:
            return MemoryCommand(
                kind=CommandKind.INVALID, error=f"清空群要一个群号，收到：{rest or '（空）'}"
            )
        return MemoryCommand(
            kind=CommandKind.PURGE, purge_scope=PurgeScope.GROUP, group_id=group_id
        )
    memory_id = _positive_int(head)
    if memory_id is not None and not rest:
        return MemoryCommand(
            kind=CommandKind.PURGE, purge_scope=PurgeScope.MEMORY, memory_id=memory_id
        )
    return MemoryCommand(
        kind=CommandKind.INVALID,
        error=f"清空的参数读不懂：{tail}（用 <编号>、群 <群号> 或 全部）",
    )


def parse_memory_command(text: str) -> MemoryCommand | None:
    """解析一条私聊文本；不是记忆命令（不以「记忆」开头）时返回 ``None``。

    开头是「记忆」、内容读不懂时返回 ``kind=INVALID`` 的命令（带一句原因），由调用方
    回用法——**不静默**：用户明确打了命令却什么也不发生，比回一句用法糟得多。
    """
    stripped = text.strip()
    if not stripped.startswith(COMMAND_PREFIX):
        return None
    rest = stripped[len(COMMAND_PREFIX) :].strip()
    if not rest:
        return MemoryCommand(kind=CommandKind.HELP)

    head, tail = _split_head(rest)

    # 明确的关键词检索：查的词恰好是保留字时走这条。
    if head in {"搜", "搜索", "查"}:
        if not tail:
            return MemoryCommand(kind=CommandKind.INVALID, error="搜什么？例：记忆 搜 论文")
        return MemoryCommand(kind=CommandKind.SEARCH, query=tail)

    if not tail and head in _BARE_KEYWORDS:
        return MemoryCommand(kind=_BARE_KEYWORDS[head])

    if head in _GROUP_KEYWORDS:
        group_id = _positive_int(tail)
        if group_id is None:
            return MemoryCommand(
                kind=CommandKind.INVALID,
                error=f"「{head}」要一个群号，收到：{tail or '（空）'}",
            )
        return MemoryCommand(kind=_GROUP_KEYWORDS[head], group_id=group_id)

    if head == "来源":
        memory_id = _positive_int(tail)
        if memory_id is None:
            return MemoryCommand(
                kind=CommandKind.INVALID, error=f"来源要一个编号，收到：{tail or '（空）'}"
            )
        return MemoryCommand(kind=CommandKind.SOURCES, memory_id=memory_id)

    if head in {"误报", "恢复"}:
        kind = CommandKind.FALSE_POSITIVE if head == "误报" else CommandKind.RESTORE
        first, remainder = _split_head(tail)
        memory_id = _positive_int(first)
        if memory_id is None:
            return MemoryCommand(
                kind=CommandKind.INVALID,
                error=f"「{head}」要一个编号，收到：{first or '（空）'}——编号在推送与检索结果里是 [#N]",
            )
        if kind is CommandKind.RESTORE and remainder:
            return MemoryCommand(
                kind=CommandKind.INVALID, error=f"「恢复」只接一个编号，多余的读不懂：{remainder}"
            )
        return MemoryCommand(kind=kind, memory_id=memory_id, note=remainder or None)

    if head == "漏了":
        # 规则：行首是纯数字就当成群号，其余整段是备注。写清楚了就不是猜测。
        first, remainder = _split_head(tail)
        group_id = _positive_int(first)
        if group_id is not None:
            return MemoryCommand(kind=CommandKind.MISSING, group_id=group_id,
                                 note=remainder or None)
        return MemoryCommand(kind=CommandKind.MISSING, note=tail or None)

    if head == "清空":
        return _parse_purge(tail)

    # 其余一律当关键词检索——包括「记忆 42」这种纯数字，用户想搜什么就搜什么。
    return MemoryCommand(kind=CommandKind.SEARCH, query=rest)


def format_help(
    *,
    configured: Collection[int],
    allowed: Collection[int],
    switches: Mapping[int, bool],
) -> str:
    """用法说明，附上当前采集状态——没有这一句，用户不知道「暂停」该对哪个群号用。"""
    lives = "、".join(str(group) for group in sorted(allowed)) or "（无）"
    config = "、".join(str(group) for group in sorted(configured)) or "（空）"
    paused = [str(group) for group, enabled in sorted(switches.items()) if not enabled]
    lines = [
        "记忆 · 用法（仅超管、仅私聊；记忆内容只在私聊出现）",
        "",
        "  记忆 <关键词>              混合检索（关键词 + 语义，RRF 融合）",
        "  记忆 搜 <关键词>           明确按关键词搜（要搜的词是保留字时用它）",
        "  记忆 今天                  今天新增的条目（含低置信度的；先补关未满的窗口）",
        "  记忆 群 <群号>             某个群的条目",
        "  记忆 来源 <编号>           这条记忆的原文窗口",
        "  记忆 误报 <编号> [备注]    标为记错：进错例集，并移出推送与检索",
        "  记忆 恢复 <编号>           撤回一次标记",
        "  记忆 漏了 [群号] [备注]    记一条「本该记但没记」（行首纯数字当群号）",
        "  记忆 开启 <群号>           开始采集某个群",
        "  记忆 暂停 <群号>           停止采集（会压过配置白名单）",
        "  记忆 清空 <编号> | 群 <群号> | 全部   彻底删除，不可逆",
        "",
        f"采集中的群：{lives}",
        f"配置白名单：{config}",
    ]
    if paused:
        lines.append(f"已暂停的群：{'、'.join(paused)}")
    return "\n".join(lines)


def _semantic_note(outcome: RetrievalOutcome) -> str | None:
    """语义那一路没参与时的一句说明；参与了就返回 ``None``。

    必须说清**为什么**（``semantic_error``，例如「向量服务未配置」）：只跑了关键词这件事
    不是一个看不见的细节——语义检索的漏检在这里是看不出来的，用户得知道自己拿到的
    结果是哪一种。
    """
    if outcome.semantic_available:
        return None
    reason = outcome.semantic_error or "原因未知"
    return f"（本次只用了关键词检索：语义那一路没跑，{reason}）"


def format_search(outcome: RetrievalOutcome, *, query: str, tz: tzinfo) -> str:
    """渲染一次混合检索的结果，并如实说明语义那一路有没有参与。"""
    header = f"检索「{query}」：命中 {outcome.total} 条"
    detail = f"关键词 {outcome.keyword_hits} 条 · 语义 {outcome.vector_hits} 条"
    note = _semantic_note(outcome)
    if not outcome.hits:
        lines = [header, detail, "", "没有找到条目。", "（如果这不是你想搜的，发「记忆」看用法）"]
        if note:
            lines.append(note)
        return "\n".join(lines)
    lines = [header, detail, "", "\n".join(
        format_memory_line(hit.memory, tz=tz, with_date=True) for hit in outcome.hits
    )]
    if note:
        lines.append(note)
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    """超长截断并留下可见的省略号——截断不报是比截断更糟的事。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_sources(
    *,
    memory: Memory,
    window: Window | None,
    messages: Sequence[Message],
    tz: tzinfo,
) -> str:
    """渲染一条记忆的来源：条目本身 + 它的原文窗口，标出哪几条是它的依据。

    只看 ``evidence`` 那几条是不够的——故事 31 要的是「追到原文窗口」，能不能判断对错
    取决于上下文在不在。所以整窗都列出来，模型挑中的那几条加一个 ``← 依据`` 标记。
    """
    header = format_memory_line(memory, tz=tz, with_date=True)
    blocks = [header]
    if memory.detail:
        blocks.append(f"补充：{memory.detail}")
    if memory.person_refs:
        people = "、".join(
            f"{person.get('nickname_snapshot') or '（无昵称）'}({person.get('user_id')})"
            for person in memory.person_refs
        )
        blocks.append(f"相关人：{people}")

    if window is None:
        blocks.append("（这条记忆没有关联的处理窗口，可能是数据被清理过）")
        return "\n\n".join(blocks)

    span = ""
    if window.started_at is not None and window.ended_at is not None:
        start = window.started_at.astimezone(tz).strftime("%m-%d %H:%M")
        end = window.ended_at.astimezone(tz).strftime("%H:%M")
        span = f"{start}–{end}"
    status = "死信" if window.status is WindowStatus.DEAD else "已处理"
    blocks.append(
        f"原文窗口 #{window.id}（{span or '时间未知'}，{window.message_count} 条消息，{status}）"
    )

    if not messages:
        blocks.append("（窗口原文已超出保留期，已被清理）")
        return "\n\n".join(blocks)

    evidence = {int(value) for value in (memory.evidence or [])}
    lines: list[str] = []
    for message in messages[:SOURCES_MAX_MESSAGES]:
        sender = message.card or message.nickname or str(message.user_id)
        stamp = message.sent_at.astimezone(tz).strftime("%H:%M")
        marker = "  ← 依据" if int(message.id) in evidence else ""
        text = _truncate(message.text.replace("\n", " "), SOURCES_MESSAGE_MAX_CHARS)
        lines.append(f"  [{stamp}] {sender}({message.user_id})：{text}{marker}")
    if len(messages) > SOURCES_MAX_MESSAGES:
        lines.append(f"  （还有 {len(messages) - SOURCES_MAX_MESSAGES} 条未列）")
    blocks.append("\n".join(lines))

    text = "\n\n".join(blocks)
    if len(text) > SOURCES_TOTAL_MAX_CHARS:
        text = _truncate(text, SOURCES_TOTAL_MAX_CHARS) + "\n\n（回复过长，已截断）"
    return text


__all__ = [
    "COMMAND_PREFIX",
    "SOURCES_MAX_MESSAGES",
    "SOURCES_MESSAGE_MAX_CHARS",
    "SOURCES_TOTAL_MAX_CHARS",
    "CommandKind",
    "CommandResult",
    "MemoryCommand",
    "PurgeScope",
    "format_help",
    "format_search",
    "format_sources",
    "parse_memory_command",
]
