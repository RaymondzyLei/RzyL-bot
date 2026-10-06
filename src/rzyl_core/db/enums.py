"""数据库里用到的枚举。

这些取值是**写进库、也写进提示词与推送**的稳定字面量，改动等于数据迁移，谨慎增删。
"""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    """记忆条目的四类之一。刻意没有「其他」这一档：归不了类就不记。"""

    RESOURCE = "resource"
    KNOWLEDGE = "knowledge"
    EVENT = "event"
    REQUEST = "request"


class WindowStatus(StrEnum):
    """处理窗口的状态。死信就是 ``DEAD`` 的窗口，不另开一张表。"""

    PENDING = "pending"
    DONE = "done"
    DEAD = "dead"


class MemoryStatus(StrEnum):
    """记忆条目的状态。``EXPIRED`` 是被新证据推翻的旧条目，标记而非删除。"""

    ACTIVE = "active"
    SUSPECT_DUPLICATE = "suspect_duplicate"
    EXPIRED = "expired"


class FeedbackKind(StrEnum):
    """纠错样本的两类：误报（记错了）与漏报（本该记但没记）。"""

    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
