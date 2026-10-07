"""核心流水线：采集缓冲、窗口组装、提取、去重、入库、历史来源。

工单 #4 交付**窗口组装**（``rzyl_core.pipeline.window``）；工单 #7 交付**提取入库**
（``rzyl_core.pipeline.extract``）；工单 #5 追加**历史来源**（``rzyl_core.pipeline.history``）
与 dry-run 的 ``PreviewOutcome``。

    from rzyl_core.pipeline import WindowAssembler, ExtractionPipeline
"""

from __future__ import annotations

from rzyl_core.pipeline.collector import (
    SEGMENT_PLACEHOLDERS,
    RenderedSegments,
    collect,
    merge_allowed_groups,
    render_segments,
    should_collect,
)
from rzyl_core.pipeline.extract import (
    LLM_PURPOSE,
    ExtractedMemory,
    ExtractedPersonRef,
    ExtractionOutcome,
    ExtractionParseError,
    ExtractionPipeline,
    PreviewOutcome,
    cosine_similarity,
    dedupe_hash,
    normalize_statement,
    parse_extraction,
)
from rzyl_core.pipeline.history import (
    GROUP_MESSAGE_HISTORY_ACTION,
    HistoryFetchError,
    HistoryMessage,
    HistorySource,
    OneBotHistorySource,
    SampleHistorySource,
    extract_history_messages,
    fetch_onebot_history,
    message_dedupe_hash,
    parse_onebot_message,
    render_message_segments,
    resolve_backfill_since,
)
from rzyl_core.pipeline.window import (
    DEFAULT_PREVIOUS_TAIL_SIZE,
    DEFAULT_REMEMBERED_LIMIT,
    PREVIOUS_TAIL_HEADER,
    AssembledWindow,
    SequencedMessage,
    UnknownSequenceError,
    WindowAssembler,
    WindowMessage,
    assemble_window,
    sender_name,
    summarize_memories,
)

__all__ = [
    "DEFAULT_PREVIOUS_TAIL_SIZE",
    "DEFAULT_REMEMBERED_LIMIT",
    "GROUP_MESSAGE_HISTORY_ACTION",
    "HistoryFetchError",
    "HistoryMessage",
    "HistorySource",
    "LLM_PURPOSE",
    "OneBotHistorySource",
    "PREVIOUS_TAIL_HEADER",
    "PreviewOutcome",
    "SampleHistorySource",
    "AssembledWindow",
    "ExtractedMemory",
    "ExtractedPersonRef",
    "ExtractionOutcome",
    "ExtractionParseError",
    "ExtractionPipeline",
    "RenderedSegments",
    "SEGMENT_PLACEHOLDERS",
    "SequencedMessage",
    "UnknownSequenceError",
    "WindowAssembler",
    "WindowMessage",
    "assemble_window",
    "collect",
    "cosine_similarity",
    "dedupe_hash",
    "extract_history_messages",
    "fetch_onebot_history",
    "merge_allowed_groups",
    "message_dedupe_hash",
    "normalize_statement",
    "parse_extraction",
    "parse_onebot_message",
    "render_message_segments",
    "render_segments",
    "resolve_backfill_since",
    "sender_name",
    "should_collect",
    "summarize_memories",
]
