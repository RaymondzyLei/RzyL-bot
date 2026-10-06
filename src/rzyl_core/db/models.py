"""SQLAlchemy 2.0 声明式模型：六张业务表。

这些模型只负责**映射与类型**（给仓储层与调用方一个带类型的读写对象）；建表语句是
:mod:`rzyl_core.db.schema` 里带版本号的手写 DDL。两边必须同步，仓储层的往返测试会
在它们漂移时报错。

几个刻意的选择：

- ``evidence`` / ``person_refs`` 在库里是 ``*_json`` 文本列，Python 侧是结构化的
  ``list``——省掉一张关联表，也省掉懒加载。
- ``Memory.id`` 就是自增主键，将来也是推送里的 ``[#N]``，所以编号不重用。
- 相关人存**快照**（QQ 号 + 当时的昵称）而不是外键：对方改名之后仍要知道当时是谁。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from sqlalchemy import JSON, Enum as SAEnum, Float, ForeignKey, Integer, LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .column_types import TZDateTime
from .enums import Category, FeedbackKind, MemoryStatus, WindowStatus


class Base(DeclarativeBase):
    """所有模型的声明式基类。"""


class PersonRef(TypedDict):
    """记忆条目里的相关人：QQ 号加当时的昵称快照。"""

    user_id: int
    nickname_snapshot: str


def _value_enum(enum_cls: type[Any]) -> SAEnum:
    """按**枚举值**（resource / done / …）落库，而不是成员名。"""
    return SAEnum(
        enum_cls,
        native_enum=False,
        values_callable=lambda cls: [member.value for member in cls],
        validate_strings=True,
    )


class Message(Base):
    """一条原始群消息。原文必须落库：平台短 ID 重启即失效，不能只存 ID。"""

    __tablename__ = "message"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    nickname: Mapped[str | None] = mapped_column(Text, default=None)
    card: Mapped[str | None] = mapped_column(Text, default=None)
    sent_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    text: Mapped[str] = mapped_column(Text, default="")
    segments_json: Mapped[str] = mapped_column(Text, default="[]")
    platform_message_id: Mapped[int | None] = mapped_column(Integer, default=None)
    dedupe_hash: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)


class Window(Base):
    """一个处理窗口。状态为 ``dead`` 的窗口即死信。"""

    __tablename__ = "window"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(TZDateTime, default=None)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[WindowStatus] = mapped_column(_value_enum(WindowStatus), default=WindowStatus.PENDING)
    prompt_version: Mapped[str | None] = mapped_column(Text, default=None)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)


class Memory(Base):
    """一条记忆条目。"""

    __tablename__ = "memory"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    window_id: Mapped[int | None] = mapped_column(ForeignKey("window.id"), default=None)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[Category] = mapped_column(_value_enum(Category), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, default=None)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[list[int]] = mapped_column("evidence_json", JSON, default=list)
    person_refs: Mapped[list[PersonRef]] = mapped_column("person_refs_json", JSON, default=list)
    occurred_at: Mapped[datetime | None] = mapped_column(TZDateTime, default=None)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    dedupe_hash: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    status: Mapped[MemoryStatus] = mapped_column(_value_enum(MemoryStatus), default=MemoryStatus.ACTIVE)
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey("memory.id"), default=None)
    model: Mapped[str | None] = mapped_column(Text, default=None)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    cost: Mapped[float | None] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)


class LlmCall(Base):
    """逐次模型调用的记账，用来回答「这个功能每月花多少」。"""

    __tablename__ = "llm_call"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str | None] = mapped_column(Text, default=None)
    model: Mapped[str | None] = mapped_column(Text, default=None)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    success: Mapped[bool] = mapped_column(default=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)


class Feedback(Base):
    """纠错样本：误报与漏报，日后当提示词的回归集。"""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kind: Mapped[FeedbackKind] = mapped_column(_value_enum(FeedbackKind), nullable=False)
    memory_id: Mapped[int | None] = mapped_column(ForeignKey("memory.id"), default=None)
    group_id: Mapped[int | None] = mapped_column(Integer, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)


class GroupSetting(Base):
    """每群的监听开关，重启后保留。"""

    __tablename__ = "group_setting"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
