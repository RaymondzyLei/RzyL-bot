"""持久化底座：设置无关的 SQLite 库、模型与仓储。

对外只暴露这一层：连接串进来，引擎 / 建表 / 仓储出去。业务代码不直接碰 SQLAlchemy
的会话与语句。
"""

from __future__ import annotations

from .column_types import TZDateTime
from .engine import open_engine
from .enums import Category, FeedbackKind, MemoryStatus, WindowStatus
from .models import Base, Feedback, GroupSetting, LlmCall, Memory, Message, PersonRef, Window
from .repository import PurgeCounts, Repository
from .schema import SCHEMA_VERSION, apply_schema
from .vectors import decode_vector, encode_vector

__all__ = [
    "SCHEMA_VERSION",
    "Base",
    "Category",
    "Feedback",
    "FeedbackKind",
    "GroupSetting",
    "LlmCall",
    "Memory",
    "MemoryStatus",
    "Message",
    "PersonRef",
    "PurgeCounts",
    "Repository",
    "TZDateTime",
    "Window",
    "WindowStatus",
    "apply_schema",
    "decode_vector",
    "encode_vector",
    "open_engine",
]
