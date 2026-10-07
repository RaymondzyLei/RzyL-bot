"""插件接线的隐私底线：命令只在私聊、只对超管（issue #1「群内永不回显」）。

这一条本来属于「不测 NoneBot 插件层」的例外——它不是业务判定，而是**隐私边界**：
规则写错的后果是把别人的群聊记录回显到群里，或者对任何私聊者开放查询。这种错误没有
别的办法发现，所以用合成事件把边界钉死（写这套测试时就真的抓到一个：规则只做了类型
标注、没有 ``isinstance`` 判定，群里发「记忆 今天」会被当成命令）。

只断言接线（规则与权限），不测处理函数本身：那部分只是把文本递给 Runtime。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPERUSER = "10001"

# 引入仓库根目录：插件包 `plugins/` 在那里，靠隐式命名空间包导入。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def plugin() -> Iterator[ModuleType]:
    """初始化一个最小 NoneBot 环境后导入插件包（进程级全局状态，不拆除）。"""
    nonebot = pytest.importorskip("nonebot")
    nonebot.init(superusers={SUPERUSER})
    import plugins.memory_bridge.commands as plugin_commands  # noqa: PLC0415 —— 必须在 init 之后

    yield plugin_commands


def _private(text: str):
    from nonebot.adapters.onebot.v11 import Message, PrivateMessageEvent  # noqa: PLC0415
    from nonebot.adapters.onebot.v11.event import Sender  # noqa: PLC0415

    return PrivateMessageEvent(
        time=1_760_000_000,
        self_id=999,
        post_type="message",
        sub_type="friend",
        user_id=int(SUPERUSER),
        message_id=1,
        message_type="private",
        raw_message=text,
        font=0,
        sender=Sender(user_id=int(SUPERUSER), nickname="管理员"),
        message=Message(text),
        original_message=Message(text),
    )


def _group(text: str):
    from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message  # noqa: PLC0415
    from nonebot.adapters.onebot.v11.event import Sender  # noqa: PLC0415

    return GroupMessageEvent(
        time=1_760_000_000,
        self_id=999,
        post_type="message",
        sub_type="normal",
        user_id=int(SUPERUSER),
        message_id=1,
        message_type="group",
        group_id=100200300,
        raw_message=text,
        font=0,
        sender=Sender(user_id=int(SUPERUSER), nickname="管理员"),
        message=Message(text),
        original_message=Message(text),
        anonymous=None,
    )


def test_commands_only_answer_in_private_chat(plugin: ModuleType) -> None:
    """群里发「记忆 今天」绝不能被当成命令——记忆内容是别人的聊天记录。"""
    assert plugin.looks_like_memory_command(_private("记忆 今天")) is True
    assert plugin.looks_like_memory_command(_group("记忆 今天")) is False


def test_non_commands_in_private_are_left_alone(plugin: ModuleType) -> None:
    assert plugin.looks_like_memory_command(_private("今天吃什么")) is False


def test_the_rule_also_rejects_non_message_events(plugin: ModuleType) -> None:
    """规则先按事件类型挡一道，与解析器无关——解析器只该看到私聊消息。"""
    from nonebot.adapters.onebot.v11 import FriendRecallNoticeEvent  # noqa: PLC0415

    notice = FriendRecallNoticeEvent(
        time=1_760_000_000,
        self_id=999,
        post_type="notice",
        notice_type="friend_recall",
        user_id=int(SUPERUSER),
        message_id=1,
    )

    assert plugin.looks_like_memory_command(notice) is False


def test_the_matcher_is_gated_on_the_frameworks_superuser_permission(plugin: ModuleType) -> None:
    """权限挂在 matcher 上，且用的就是 NoneBot 的 ``SUPERUSER`` 检查器。

    处理函数里没有再判一次权限，所以这里认错了对象就等于把查询开放给任何私聊者。
    NoneBot 装配时会把权限包一层（``Permission() | permission``），因此断言检查器的类型
    而不是对象标识。
    """
    from nonebot.permission import SuperUser  # noqa: PLC0415

    matcher = plugin.matcher
    checkers = list(matcher.permission.checkers)
    assert any(isinstance(getattr(checker, "call", None), SuperUser) for checker in checkers)
    assert matcher.block is True
    # 命令优先级高于采集（priority=100），不会被别的手续挡在后面。
    assert matcher.priority <= 10


def test_the_daily_report_sender_is_registered_on_import(plugin: ModuleType) -> None:
    """core 不认识 QQ，投递器必须由插件在 import 时登记好（不能等 on_startup）。"""
    from rzyl_core.memory import get_report_sender  # noqa: PLC0415

    assert plugin is not None
    assert get_report_sender() is not None
