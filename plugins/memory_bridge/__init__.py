"""记忆桥插件：把 OneBot v11 群消息交给 ``rzyl_core`` 的 Runtime，并在重连后回补。

这个插件**只做薄适配**（issue #1：「插件保持薄」）：不做判定、不做文本化、不碰数据库，
只把 NoneBot 事件的几样字段拆出来交给 Runtime，剩下全在 core。依赖方向是硬的：

    plugins/memory_bridge → rzyl_core     （插件可以依赖 core，core 不依赖插件）

两个入口：

- ``on_message``（低优先级、**不 block**）——把群消息交给 ``Runtime.ingest_message``。
  不 block 是为了不影响 echo 等已有插件；低优先级让它排在别人后面。
- ``driver.on_bot_connect``——机器人启动 / 掉线重连后：打一条白名单与可用群列表的日志，
  再对白名单里的群做一次掉线回补（见 :mod:`.backfill`）。

Runtime 由 ``bot.py`` 装配后通过 ``rzyl_core.runtime.set_runtime`` 注册，插件用
``get_runtime_or_none()`` 取。记忆功能因缺少密钥未启用时注册的是 ``None``，插件静默
跳过，机器人照常收发消息。插件之间不互相 import，本插件内部的两个模块除外。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from nonebot import get_driver, on_message
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import Bot as OneBotV11Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.plugin import PluginMetadata

from rzyl_core.runtime import get_runtime_or_none

from .backfill import backfill_all_groups, group_ids_from_group_list

__plugin_meta__ = PluginMetadata(
    name="记忆桥",
    description="把白名单群的消息交给 rzyl_core 记忆管道，并在掉线重连后回补。",
    usage="无需命令：由 bot.py 装配 Runtime 后自动工作。",
    type="application",
    supported_adapters={"~onebot.v11"},
)

logger = logging.getLogger("rzyl.plugin.memory_bridge")

driver = get_driver()

# 低优先级、不 block：不干扰 echo 之类已有插件。
_matcher = on_message(priority=100, block=False)


def _segments_from_event(event: GroupMessageEvent) -> list[dict[str, Any]]:
    """把 OneBot v11 的 ``Message`` 拆回 OB11 原始段落 dict 列表。"""
    return [{"type": segment.type, "data": dict(segment.data)} for segment in event.message]


@_matcher.handle()
async def _on_group_message(bot: OneBotV11Bot, event: Event) -> None:
    """群消息进来就交给 core；判定与入库全在 Runtime。"""
    runtime = get_runtime_or_none()
    if runtime is None or not isinstance(event, GroupMessageEvent):
        return
    await runtime.ingest_message(
        group_id=event.group_id,
        user_id=event.user_id,
        self_id=int(event.self_id),
        segments=_segments_from_event(event),
        sent_at=datetime.fromtimestamp(event.time, tz=timezone.utc),
        nickname=event.sender.nickname,
        card=event.sender.card or None,
        platform_message_id=event.message_id,
    )


@driver.on_bot_connect
async def _on_bot_connect(bot: Bot) -> None:
    """启动 / 重连：打白名单与可用群日志，并对白名单里的群回补掉线期间的消息。"""
    if not isinstance(bot, OneBotV11Bot):
        return
    runtime = get_runtime_or_none()
    if runtime is None:
        logger.warning("未装配 Runtime（多半是未配置聊天模型密钥），记忆功能未启用，跳过回补")
        return

    allowed = await runtime.allowed_groups()
    try:
        group_list = await bot.call_api("get_group_list")
    except Exception:  # 群列表拿不到不应挡住启动
        logger.exception("取群列表失败，本轮跳过回补")
        return

    available = group_ids_from_group_list(group_list)
    logger.info(
        "记忆采集白名单（配置 ∪ 运行时开关）：%s；机器人当前在的群：%s",
        sorted(allowed) or "（空）",
        sorted(available) or "（空）",
    )

    backfilled = await backfill_all_groups(
        bot, runtime, self_id=int(bot.self_id), now=datetime.now(timezone.utc)
    )
    # 灌入 0 条也要打这一行：否则「回补跑了但没拉到东西」与「回补根本没跑」在日志里长得
    # 一模一样，排查时会白绕一圈（这个坑踩过一次）。
    logger.info("掉线回补完成：本次共灌入 %d 条消息", backfilled)
