"""日报投递：core 渲染好文本，这里把它发给超管。

这是整个记忆功能里**唯一**需要认识 QQ 的一步，所以它待在插件里。core 侧只留一个
投递器的登记处（``rzyl_core.memory.push``），插件在 import 时就登记好——这样不必和
``driver.on_startup`` 的执行顺序赛跑（Runtime 的推送循环要到 start 之后才会醒来）。

投递失败一律**抛出**：让 Runtime 去记日志并安排重试（``PUSH_RETRY_SECONDS``），
插件不自己吞掉失败——「生成了但没发出去」必须是可见的。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from nonebot import get_bots, get_driver
from nonebot.adapters.onebot.v11 import Bot as OneBotV11Bot

from rzyl_core.memory import DailyReport, set_report_sender

logger = logging.getLogger("rzyl.plugin.memory_bridge.push")

#: 单条私聊消息的字符上限（保守取值）。QQ 对文本长度有硬限制，而日报列满 50 条时会远超；
#: 超了就在**行边界**上切成几条发，而不是截断——截断不报是本项目最忌讳的失败方式。
MAX_PRIVATE_MESSAGE_CHARS = 1200


def split_for_send(text: str, *, limit: int = MAX_PRIVATE_MESSAGE_CHARS) -> list[str]:
    """按行把一段文本切成不超过 ``limit`` 字符的若干段。

    优先在换行处切（日报的每条记忆本来就是一行）。**单行自身超限时会硬切**：把一句陈述
    从中间断开确实难看，但让那一条发送失败、连带整份日报投递不出去并反复重试到放弃，
    是更糟的失败方式——内容一个字都不能丢。
    """
    if limit <= 0:
        raise ValueError(f"limit 必须为正数，收到 {limit}")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if current and len(current) + 1 + len(line) > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
        while len(current) > limit:
            chunks.append(current[:limit])
            current = current[limit:]
    if current:
        chunks.append(current)
    return chunks


def superuser_ids() -> list[int]:
    """配置里 ``SUPERUSERS`` 的 QQ 号，升序。读不出数字的条目直接跳过并记一条告警。"""
    raw: Iterable[str] = get_driver().config.superusers
    recipients: list[int] = []
    for value in raw:
        try:
            recipients.append(int(value))
        except (TypeError, ValueError):
            logger.warning("SUPERUSERS 里有读不成 QQ 号的条目，已跳过：%r", value)
    return sorted(recipients)


async def deliver_daily_report(report: DailyReport) -> None:
    """把日报发给每个超管；任一收件人失败都算投递失败（抛给 Runtime 重试）。"""
    bots = [bot for bot in get_bots().values() if isinstance(bot, OneBotV11Bot)]
    if not bots:
        raise RuntimeError("当前没有已连接的 OneBot 机器人，日报无法投递")
    recipients = superuser_ids()
    if not recipients:
        raise RuntimeError("配置里没有 SUPERUSERS，日报没有收件人")
    bot = bots[0]
    chunks = split_for_send(report.text)
    failures: list[str] = []
    for user_id in recipients:
        for index, chunk in enumerate(chunks, start=1):
            try:
                await bot.call_api("send_private_msg", user_id=user_id, message=chunk)
            except Exception as exc:  # 收件人之间互不影响：一个失败不挡另一个
                failures.append(f"{user_id}（第 {index}/{len(chunks)} 段）：{exc}")
                logger.exception("日报发往 %s 失败（第 %d/%d 段）", user_id, index, len(chunks))
    logger.info(
        "日报投递：收件人 %s，每份 %d 段，失败 %d 处",
        recipients,
        len(chunks),
        len(failures),
    )
    if failures:
        raise RuntimeError("部分日报未送达：" + "；".join(failures))


# 插件 import 即登记：Runtime 的后台推送循环在 start() 之后才可能触发，此时早已就绪。
set_report_sender(deliver_daily_report)
