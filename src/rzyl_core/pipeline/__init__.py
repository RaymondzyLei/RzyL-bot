"""核心流水线：采集缓冲、窗口组装、提取、去重、入库。

工单 #4 交付**窗口组装**（``rzyl_core.pipeline.window``）；工单 #7 交付**提取入库**
（``rzyl_core.pipeline.extract``）。对外只暴露这两个模块里的公开名字。

    from rzyl_core.pipeline import WindowAssembler, ExtractionPipeline
"""

from __future__ import annotations

from rzyl_core.pipeline.extract import (
    LLM_PURPOSE,
    ExtractedMemory,
    ExtractedPersonRef,
    ExtractionOutcome,
    ExtractionParseError,
    ExtractionPipeline,
    cosine_similarity,
    dedupe_hash,
    normalize_statement,
    parse_extraction,
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
    "LLM_PURPOSE",
    "PREVIOUS_TAIL_HEADER",
    "AssembledWindow",
    "ExtractedMemory",
    "ExtractedPersonRef",
    "ExtractionOutcome",
    "ExtractionParseError",
    "ExtractionPipeline",
    "SequencedMessage",
    "UnknownSequenceError",
    "WindowAssembler",
    "WindowMessage",
    "assemble_window",
    "cosine_similarity",
    "dedupe_hash",
    "normalize_statement",
    "parse_extraction",
    "sender_name",
    "summarize_memories",
]
