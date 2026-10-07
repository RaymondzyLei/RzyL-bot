"""掉线回补：机器人启动 / 重连后，把白名单群里掉线期间的消息补进管道。

容器里没有 OneBot 的 HTTP 端点（``OneBotHistorySource`` 那套 httpx 用不上），所以这里走
插件侧的 ``bot.call_api("get_group_msg_history")``；但**翻页与解析一行都不重写**——
复用 ``rzyl_core.pipeline.history`` 的 :func:`fetch_onebot_history` 与
:func:`parse_onebot_message`（``fetch_page`` 换成 ``bot.call_api`` 而已）。

两条护栏（issue #1「至多 24 小时或至多 N 条，超出记告警」）：

- 时间：起点取「该群最后一条已存消息时间」与 ``现在 - backfill_max_hours`` 的较晚者，
  被护栏截断时记一条警告（:func:`resolve_backfill_since`）。
- 条数：一次最多 ``backfill_max_messages`` 条，达到上限记一条警告。

去重：同一个群的同一内容按 ``(群号, 发送者, 时间, 文本)`` 算 hash，回补前先查库中已存的
hash 跳过。文本用 **collector 的渲染**（与实时链路同一套），所以「实时收过的文件段消息」
在回补时也能对上，不会被记两遍。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from nonebot.adapters.onebot.v11 import Bot

from rzyl_core.pipeline.collector import render_segments
from rzyl_core.pipeline.history import (
    HistoryMessage,
    extract_history_messages,
    fetch_onebot_history,
    message_dedupe_hash,
    resolve_backfill_since,
)
from rzyl_core.runtime import Runtime

logger = logging.getLogger("rzyl.plugin.memory_bridge.backfill")


def group_ids_from_group_list(group_list: object) -> set[int]:
    """从 ``get_group_list`` 的返回里取出群号集合，容忍形状差异。"""
    if not isinstance(group_list, list):
        return set()
    ids: set[int] = set()
    for item in group_list:
        if not isinstance(item, dict):
            continue
        group_id = item.get("group_id")
        if isinstance(group_id, int) and not isinstance(group_id, bool):
            ids.add(group_id)
    return ids


async def backfill_all_groups(
    bot: Bot, runtime: Runtime, *, self_id: int, now: datetime
) -> int:
    """对白名单里的每个群回补一次，返回灌入管道的消息总数。``now`` 由调用方注入。"""
    allowed = await runtime.allowed_groups()
    if not allowed:
        logger.info("白名单为空（配置与运行时开关都没有群），跳过回补")
        return 0
    total = 0
    for group_id in sorted(allowed):
        total += await backfill_group(
            bot, runtime, group_id=group_id, self_id=self_id, now=now
        )
    return total


async def backfill_group(
    bot: Bot, runtime: Runtime, *, group_id: int, self_id: int, now: datetime
) -> int:
    """回补一个群，返回真正灌入管道的消息条数（去重后）。

    拉历史失败只记日志、返回 0，不让一个群的问题挡住其余群与机器人启动。
    """
    settings = runtime.settings
    repository = runtime.repository
    anchor = await repository.latest_message_sent_at(group_id)
    since, clamped = resolve_backfill_since(
        anchor=anchor, now=now, max_hours=settings.backfill_max_hours
    )
    if clamped:
        logger.warning(
            "群 %s 最后一条消息早于回补时间护栏（%d 小时），只补最近窗口",
            group_id,
            settings.backfill_max_hours,
        )
    limit = settings.backfill_max_messages

    async def fetch_page(message_seq: int) -> list[dict[str, Any]]:
        data = await bot.call_api(
            "get_group_msg_history",
            group_id=group_id,
            message_seq=message_seq,
            count=settings.backfill_page_size,
        )
        if not isinstance(data, dict):
            return []
        return extract_history_messages(data) or []

    try:
        messages = await fetch_onebot_history(
            group_id=group_id, fetch_page=fetch_page, since=since, limit=limit
        )
    except Exception:  # 单群失败不影响其它群
        logger.exception("群 %s 回补拉历史失败，跳过该群", group_id)
        return 0

    if len(messages) >= limit:
        logger.warning("群 %s 回补达到条数上限 %d，可能有遗漏", group_id, limit)

    # 每条算出与实时链路一致的 (文本, 段落, hash)，先过滤掉库里已存的。
    candidates: list[tuple[str, list[dict[str, Any]], HistoryMessage]] = []
    for message in messages:
        if message.segments:
            segments = list(message.segments)
            text = render_segments(segments).text
        else:
            text = message.text
            segments = [{"type": "text", "data": {"text": message.text}}]
        dedupe_hash = message_dedupe_hash(
            group_id=group_id, user_id=message.user_id, sent_at=message.sent_at, text=text
        )
        candidates.append((dedupe_hash, segments, message))

    known = await repository.existing_message_hashes(
        [dedupe_hash for dedupe_hash, _, _ in candidates]
    )
    ingested = 0
    for dedupe_hash, segments, message in candidates:
        if dedupe_hash in known:
            continue
        known.add(dedupe_hash)
        result = await runtime.ingest_message(
            group_id=group_id,
            user_id=message.user_id,
            self_id=self_id,
            segments=segments,
            sent_at=message.sent_at,
            nickname=message.nickname,
            card=message.card,
            platform_message_id=message.platform_message_id,
        )
        if result is not None:
            ingested += 1
    logger.info("群 %s 回补灌入 %d 条（拉取 %d 条）", group_id, ingested, len(messages))
    return ingested
