"""核心流水线：采集缓冲、窗口组装、（后续的）提取、去重、入库。

工单 #4 先交付**窗口组装**：把一个群的连续消息流攒成可直接交给模型的一段。
对外只暴露 ``rzyl_core.pipeline.window`` 里的公开名字，见该模块 docstring 的契约说明。

    from rzyl_core.pipeline import WindowAssembler, assemble_window
"""

from __future__ import annotations

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
    "PREVIOUS_TAIL_HEADER",
    "AssembledWindow",
    "SequencedMessage",
    "UnknownSequenceError",
    "WindowAssembler",
    "WindowMessage",
    "assemble_window",
    "sender_name",
    "summarize_memories",
]
