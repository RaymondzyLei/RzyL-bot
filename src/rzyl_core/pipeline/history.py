"""历史来源：把「某个群的一段历史消息」从两处取进来（工单 #5）。

回放脚本要能对两样东西跑：**固定样本文件**（JSON，离线自足）与**真实 OneBot**。
两者对外是同一个协议 :class:`HistorySource`——``fetch(group_id, since, limit)`` 返回
按时间升序的 :class:`HistoryMessage` 列表，下游（Runtime / 回放）不需要知道它从哪来。

两处刻意的选择：

- **样本文件那条路完全自足**：不需要 QQ、不需要网络、不需要任何 API key，纯读一个
  JSON 文件，所以离线回放与验收能一键复现。
- **OneBot HTTP 是自带的薄客户端**：直接 ``POST {api_root}/{action}`` 调
  ``get_group_msg_history``。它**不是 OneBot v11 规范的一部分**，是 NapCat 的
  go-cqhttp 兼容扩展，返回 ``{"messages": [...]}``（有的版本包在 ``data`` 下）。
  翻页靠每条消息的 ``message_seq`` 与时间戳向更早走，不用平台短 ID 当锚。httpx 的
  ``transport`` 可注入，测试用 ``httpx.MockTransport`` **不发真实请求**。

``since`` 的语义是「只取更晚的」——回补时以该群最后一条已处理消息的时间戳为锚；
``limit`` 取**最新的**若干条。返回前一律按时间升序排好。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from rzyl_core.settings import Settings

#: ``get_group_msg_history`` 的 action 名（NapCat 的 go-cqhttp 兼容扩展）。
GROUP_MESSAGE_HISTORY_ACTION = "get_group_msg_history"

#: 非文本消息段落成占位符，保证对话不断层（图片 / 语音本身的内容属于里程碑 4）。
_SEGMENT_PLACEHOLDERS: dict[str, str] = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[合并转发]",
    "json": "[卡片]",
    "xml": "[卡片]",
    "poke": "[戳一戳]",
    "music": "[音乐]",
}


class HistoryFetchError(RuntimeError):
    """历史拉取失败：HTTP 错误、响应不合预期、消息字段缺失等。"""


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """一条历史消息（历史来源的输出单元）。

    ``sent_at`` 一律带时区；``message_seq`` 只在 OneBot 来源里有值，是翻页用的锚，
    ``platform_message_id`` 只作参考（重启即失效，不能当长期标识）。
    """

    group_id: int
    user_id: int
    text: str
    sent_at: datetime
    nickname: str | None = None
    card: str | None = None
    platform_message_id: int | None = None
    message_seq: int | None = None
    segments: tuple[dict[str, Any], ...] = ()
    """原始消息段（OB11 形状）；只有记录里带 ``message`` 段时才有值。

    掉线回补用它渲染出与**实时链路一致**的文本与 ``segments_json``——否则同一张图/
    同一个文件在两条路上渲染成的文本不同，内容 hash 去重会失效。样本文件来源不填。
    """


def message_dedupe_hash(
    *, group_id: int, user_id: int, sent_at: datetime, text: str
) -> str:
    """一条原始消息的去重 hash：群号 + 发送者 + 时间 + 原文（见 issue #1 的回补去重）。

    时间先归一成 UTC ISO 串，避免同一时刻因时区写法不同而算出两个 hash。
    """
    material = f"{group_id}|{user_id}|{sent_at.astimezone(timezone.utc).isoformat()}|{text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def resolve_backfill_since(
    *, anchor: datetime | None, now: datetime, max_hours: int
) -> tuple[datetime, bool]:
    """算出回补的起点时间，并返回是否被时间护栏截断。

    锚点是该群最后一条已存消息的时间；护栏是「至多回补最近 ``max_hours`` 小时」。
    起点取锚点与 ``now - max_hours`` 里**较晚**的一个：

    - 锚点比护栏下限更早（或压根没有锚点）→ 用下限，第二项为 ``True``（只有真的因为
      护栏截断才为真；没有锚点不算「超了护栏」）；调用方据此记一条告警。
    - 锚点在护栏之内 → 用锚点，第二项为 ``False``。

    纯函数、可注入 ``now``，所以护栏行为能脱离时钟单测。
    """
    floor = now - timedelta(hours=max_hours)
    if anchor is None:
        return floor, False
    if anchor < floor:
        return floor, True
    return anchor, False


@runtime_checkable
class HistorySource(Protocol):
    """可替换的历史来源。"""

    async def fetch(
        self,
        *,
        group_id: int,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HistoryMessage]:
        """取某群的历史消息，按时间升序；``since`` 只取更晚的，``limit`` 取最新若干。"""
        ...


class SampleHistorySource:
    """固定样本文件来源：读一个 JSON 数组，完全离线自足。

    文件形状（``nickname`` / ``card`` / ``platform_message_id`` 可省）::

        [{"group_id": 100200300, "user_id": 10001, "sent_at": "2026-10-07T09:00:00+08:00",
          "text": "……", "nickname": "小A", "card": "阿A", "platform_message_id": 900001}, …]

    ``sent_at`` 必须是带时区的 ISO 8601 字符串——库的写入口对 naive 时间零容忍，
    在这里就报错比落库时才发现清楚。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    async def fetch(
        self,
        *,
        group_id: int,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HistoryMessage]:
        messages = [
            self._parse(index, record)
            for index, record in enumerate(self._load())
            if record.get("group_id") == group_id
        ]
        return _apply_since_and_limit(messages, since=since, limit=limit)

    def _load(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"读不出样本文件 {self._path}：{exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"样本文件 {self._path} 顶层必须是 JSON 数组")
        records: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"样本文件第 {index} 条不是对象")
            records.append(item)
        return records

    @staticmethod
    def _parse(index: int, record: dict[str, Any]) -> HistoryMessage:
        try:
            group_id = int(record["group_id"])
            user_id = int(record["user_id"])
            sent_at = datetime.fromisoformat(str(record["sent_at"]))
            text = str(record["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"样本第 {index} 条记录缺少必需字段或类型不对：{exc}") from exc
        if sent_at.tzinfo is None:
            raise ValueError(f"样本第 {index} 条的 sent_at 必须带时区：{record['sent_at']!r}")

        raw_message_id = record.get("platform_message_id")
        platform_message_id = int(raw_message_id) if raw_message_id is not None else None
        nickname = record.get("nickname")
        card = record.get("card")
        return HistoryMessage(
            group_id=group_id,
            user_id=user_id,
            text=text,
            sent_at=sent_at,
            nickname=str(nickname) if nickname is not None else None,
            card=str(card) if card is not None else None,
            platform_message_id=platform_message_id,
        )


class OneBotHistorySource:
    """OneBot（NapCat）HTTP 历史来源：调 ``get_group_msg_history`` 向更早翻页。

    ``api_root`` 与 ``access_token`` 走设置对象（``RZYL_ONEBOT_API_ROOT`` /
    ``RZYL_ONEBOT_ACCESS_TOKEN``），用 :meth:`from_settings` 构造；token 为空时不发
    ``Authorization`` 头。``transport`` 可注入以便测试。

    翻页规则：首请求 ``message_seq=0``（取最新一页），之后用本页最老的 ``message_seq - 1``
    继续向更早要；落到 0 或本页没消息就停。页面内顺序按 ``message_seq`` 升序假定，
    落地前再按时间排一次并去重，避免实现细节差异导致乱序或重复。
    """

    def __init__(
        self,
        *,
        api_root: str,
        access_token: str = "",
        page_size: int = 20,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_root.strip():
            raise ValueError("api_root 不能为空；配置 RZYL_ONEBOT_API_ROOT 后才可用")
        if page_size < 1:
            raise ValueError(f"page_size 至少为 1，收到 {page_size}")
        self._api_root = api_root.rstrip("/")
        self._access_token = access_token
        self._page_size = page_size
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        page_size: int = 20,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "OneBotHistorySource":
        """按设置构造；没配 ``onebot_api_root`` 时明确失败，不回退到假地址。"""
        if not settings.onebot_api_root.strip():
            raise ValueError(
                "未配置 RZYL_ONEBOT_API_ROOT，无法从 OneBot 拉历史；"
                "本机没有可用端点时请改用样本文件回放"
            )
        return cls(
            api_root=settings.onebot_api_root,
            access_token=settings.onebot_access_token,
            page_size=page_size,
            timeout=timeout,
            transport=transport,
        )

    @property
    def api_root(self) -> str:
        return self._api_root

    @property
    def access_token(self) -> str:
        return self._access_token

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OneBotHistorySource":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch(
        self,
        *,
        group_id: int,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HistoryMessage]:
        return await fetch_onebot_history(
            group_id=group_id,
            fetch_page=lambda message_seq: self._request_page(
                group_id=group_id, message_seq=message_seq
            ),
            since=since,
            limit=limit,
        )

    async def _request_page(self, *, group_id: int, message_seq: int) -> list[dict[str, Any]]:
        url = f"{self._api_root}/{GROUP_MESSAGE_HISTORY_ACTION}"
        payload = {
            "group_id": group_id,
            "message_seq": message_seq,
            "count": self._page_size,
        }
        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise HistoryFetchError(f"拉取群 {group_id} 历史失败：{exc}") from exc
        if response.status_code >= 400:
            raise HistoryFetchError(
                f"拉取群 {group_id} 历史返回 {response.status_code}：{response.text[:200]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise HistoryFetchError(f"历史响应不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise HistoryFetchError(f"历史响应不是对象：{type(data).__name__}")
        retcode = data.get("retcode")
        if isinstance(retcode, int) and not isinstance(retcode, bool) and retcode != 0:
            raise HistoryFetchError(f"历史接口 retcode={retcode}：{data.get('message', '')}")
        messages = _extract_messages(data)
        if messages is None:
            raise HistoryFetchError("历史响应缺少 messages 列表")
        return messages

    @staticmethod
    def _parse(group_id: int, record: dict[str, Any]) -> HistoryMessage:
        """兼容入口：等价于模块级 :func:`parse_onebot_message`。"""
        return parse_onebot_message(group_id, record)


def parse_onebot_message(group_id: int, record: dict[str, Any]) -> HistoryMessage:
    """把一条 OneBot 原始消息 dict 转成内部 :class:`HistoryMessage`。

    HTTP 历史来源与插件侧 ``bot.call_api("get_group_msg_history")`` 的回补共用这一处：
    时间取 unix 秒并按 UTC 落带时区的 ``datetime``；发送者取 ``sender.user_id``
    （退回顶层 ``user_id``）；文本按消息段渲染（没有 ``message`` 段时退回 ``raw_message``）。
    缺必需字段就抛 :class:`HistoryFetchError`，宁可当场失败也不落一条残缺消息。
    """
    if not isinstance(record, dict):
        raise HistoryFetchError(f"历史消息不是对象：{type(record).__name__}")
    raw_time = record.get("time")
    if not isinstance(raw_time, int | float) or isinstance(raw_time, bool):
        raise HistoryFetchError(f"历史消息缺少可用的 time 字段：{raw_time!r}")
    sent_at = datetime.fromtimestamp(int(raw_time), tz=timezone.utc)

    sender = record.get("sender")
    sender = sender if isinstance(sender, dict) else {}
    user_id = sender.get("user_id", record.get("user_id"))
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise HistoryFetchError(f"历史消息缺少可用的 sender.user_id：{user_id!r}")

    raw_message_id = record.get("message_id")
    raw_seq = record.get("message_seq")
    nickname = sender.get("nickname")
    card = sender.get("card")
    raw_segments = record.get("message")
    segments: tuple[dict[str, Any], ...] = (
        tuple(item for item in raw_segments if isinstance(item, dict))
        if isinstance(raw_segments, list)
        else ()
    )
    return HistoryMessage(
        group_id=group_id,
        user_id=user_id,
        text=_render_record_text(record),
        sent_at=sent_at,
        nickname=nickname if isinstance(nickname, str) else None,
        card=card if isinstance(card, str) and card else None,
        platform_message_id=(
            raw_message_id
            if isinstance(raw_message_id, int) and not isinstance(raw_message_id, bool)
            else None
        ),
        message_seq=(
            raw_seq if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else None
        ),
        segments=segments,
    )


async def fetch_onebot_history(
    *,
    group_id: int,
    fetch_page: Callable[[int], Awaitable[list[dict[str, Any]]]],
    since: datetime | None = None,
    limit: int | None = None,
) -> list[HistoryMessage]:
    """按 ``message_seq`` 向更早翻页拉某群历史，返回时间升序的 :class:`HistoryMessage`。

    ``fetch_page(message_seq)`` 返回一页原始消息（``0`` 表示最新一页）。翻页规则与
    :class:`OneBotHistorySource` 完全一致：首请求取最新一页，之后用本页最老的
    ``message_seq - 1`` 继续向更早要，落到 0 或本页没消息就停；落地前按时间排一次并
    去重。把 HTTP 与翻页解耦，是为了让插件侧的 ``bot.call_api`` 回补复用同一段逻辑
    （容器内没有 OneBot HTTP 端点，``OneBotHistorySource`` 用不上）。
    """
    collected: list[HistoryMessage] = []
    seen: set[Any] = set()
    cursor = 0
    while True:
        page = await fetch_page(cursor)
        parsed: list[HistoryMessage] = []
        for record in page:
            key = record.get("message_seq", record.get("message_id"))
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            parsed.append(parse_onebot_message(group_id, record))
        if not parsed:
            break

        collected = parsed + collected
        oldest = min(parsed, key=lambda message: message.sent_at)
        if since is not None and oldest.sent_at <= since:
            break
        if limit is not None and len(collected) >= limit:
            break
        sequences = [
            message.message_seq for message in parsed if message.message_seq is not None
        ]
        next_cursor = (min(sequences) - 1) if sequences else 0
        if next_cursor <= 0:
            break
        cursor = next_cursor

    return _apply_since_and_limit(_sort_by_time(collected), since=since, limit=limit)


def render_message_segments(segments: Sequence[Any]) -> str:
    """把一个 OneBot 消息段数组渲染成纯文本：文本段原样拼接，非文本段落占位符。"""
    parts: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("type")
        data = segment.get("data")
        data = data if isinstance(data, dict) else {}
        if kind == "text":
            parts.append(str(data.get("text", "")))
        elif kind == "at":
            target: Any = data.get("name") or data.get("qq") or ""
            parts.append(f"@{target}")
        elif isinstance(kind, str):
            parts.append(_SEGMENT_PLACEHOLDERS.get(kind, f"[{kind}]"))
    return "".join(parts)


def _render_record_text(record: dict[str, Any]) -> str:
    """优先按消息段渲染；没有消息段就退回 ``raw_message``。"""
    segments = record.get("message")
    if isinstance(segments, list):
        return render_message_segments(segments)
    raw = record.get("raw_message")
    return raw if isinstance(raw, str) else ""


def _extract_messages(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """兼容两种包裹：顶层 ``messages`` 与 ``data.messages``。"""
    top = data.get("messages")
    if isinstance(top, list):
        return [item for item in top if isinstance(item, dict)]
    nested = data.get("data")
    if isinstance(nested, dict):
        inner = nested.get("messages")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return None


def extract_history_messages(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """从历史响应里取出消息列表，兼容顶层 ``messages`` 与 ``data.messages``。

    插件侧 ``bot.call_api("get_group_msg_history")`` 拿到的是已解包的 ``data``，形状与
    HTTP 响应不完全一致，靠这个公开入口统一。
    """
    return _extract_messages(data)


def _sort_by_time(messages: Sequence[HistoryMessage]) -> list[HistoryMessage]:
    return sorted(messages, key=lambda message: message.sent_at)


def _apply_since_and_limit(
    messages: Sequence[HistoryMessage],
    *,
    since: datetime | None,
    limit: int | None,
) -> list[HistoryMessage]:
    """过滤 ``since``（只留更晚的）、取最新 ``limit`` 条，返回时间升序列表。"""
    ordered = _sort_by_time(messages)
    if since is not None:
        ordered = [message for message in ordered if message.sent_at > since]
    if limit is not None:
        ordered = ordered[-limit:]
    return ordered


__all__ = [
    "GROUP_MESSAGE_HISTORY_ACTION",
    "HistoryFetchError",
    "HistoryMessage",
    "HistorySource",
    "OneBotHistorySource",
    "SampleHistorySource",
    "extract_history_messages",
    "fetch_onebot_history",
    "message_dedupe_hash",
    "parse_onebot_message",
    "render_message_segments",
    "resolve_backfill_since",
]
