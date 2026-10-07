"""历史来源的行为测试（工单 #5）。

两个实现共用一条 seam：``fetch(group_id, since, limit)`` 返回**按时间升序**的
:class:`HistoryMessage`。样本文件来源纯离线；OneBot 来源用 ``httpx.MockTransport``
把 NapCat 的 ``get_group_msg_history`` 扩展接口模拟成一个按 ``message_seq`` 向更早
翻页的假服务，**不发真实网络请求**。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from rzyl_core.pipeline.history import (
    HistoryFetchError,
    OneBotHistorySource,
    SampleHistorySource,
    extract_history_messages,
    fetch_onebot_history,
    parse_onebot_message,
    resolve_backfill_since,
)
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_history.json"
GROUP = 100200300


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _naive_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "naive.json"
    path.write_text(
        json.dumps(
            [
                {
                    "group_id": 1,
                    "user_id": 2,
                    "sent_at": "2026-10-07T09:00:00",
                    "text": "没有时区",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# —— 样本文件来源 ——


async def test_sample_source_reads_the_fixture_in_chronological_order() -> None:
    source = SampleHistorySource(SAMPLE)

    messages = await source.fetch(group_id=GROUP)

    assert len(messages) == 8
    assert [message.text for message in messages[:3]] == [
        "大家看下这个网盘链接 https://pan.example.com/s/abc123 里面是实验讲义",
        "收到，谢谢",
        "总结一下：实验课改到周五下午三点，地点还是老地方",
    ]
    assert all(message.sent_at.tzinfo is not None for message in messages)
    assert messages[0].sent_at == datetime(2026, 10, 7, 9, 0, tzinfo=CST)
    assert messages[0].nickname == "小A"
    assert messages[0].card == "阿A"
    assert messages[0].platform_message_id == 900001


async def test_sample_source_filters_by_group_id(tmp_path: Path) -> None:
    path = tmp_path / "two_groups.json"
    path.write_text(
        json.dumps(
            [
                {"group_id": 111, "user_id": 1, "sent_at": "2026-10-07T09:00:00+08:00", "text": "甲的"},
                {"group_id": 222, "user_id": 1, "sent_at": "2026-10-07T09:01:00+08:00", "text": "乙的"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    messages = await SampleHistorySource(path).fetch(group_id=222)

    assert [message.text for message in messages] == ["乙的"]


async def test_sample_source_applies_since_and_keeps_the_newest_limit() -> None:
    source = SampleHistorySource(SAMPLE)

    messages = await source.fetch(
        group_id=GROUP,
        since=datetime(2026, 10, 7, 9, 1, tzinfo=CST),
        limit=2,
    )

    # since 是「只取更晚的」，limit 取最新的若干条，返回仍是升序。
    assert [message.text for message in messages] == ["收藏了", "另外考试范围老师说是前三章"]


async def test_sample_source_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await SampleHistorySource(_naive_fixture(tmp_path)).fetch(group_id=1)


async def test_sample_source_rejects_a_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps([{"group_id": 1, "text": "少了 sender 和时间"}]), encoding="utf-8")

    with pytest.raises(ValueError):
        await SampleHistorySource(path).fetch(group_id=1)


async def test_sample_source_is_fully_offline(tmp_path: Path) -> None:
    """拷贝一份样本到临时目录也能读——不依赖仓库、网络或任何 key。"""
    copy = tmp_path / "copy.json"
    copy.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    messages = await SampleHistorySource(copy).fetch(group_id=GROUP, limit=1)

    assert len(messages) == 1


# —— OneBot HTTP 来源 ——

Handler = Callable[[httpx.Request], httpx.Response]


def _onebot_message(
    seq: int,
    *,
    user_id: int = 10001,
    text: str = "消息",
    at: str = "2026-10-07T09:00:00+08:00",
    segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """造一条 NapCat 风格的历史消息；时间用 unix 秒。"""
    timestamp = int(datetime.fromisoformat(at).timestamp())
    payload: dict[str, Any] = {
        "message_id": seq,
        "real_id": seq,
        "message_seq": seq,
        "time": timestamp,
        "sender": {"user_id": user_id, "nickname": "小A", "card": "阿A"},
        "raw_message": text,
    }
    if segments is not None:
        payload["message"] = segments
    else:
        payload["message"] = [{"type": "text", "data": {"text": text}}]
    return payload


def _history_server(all_messages: list[dict[str, Any]], *, retcode: int = 0) -> Handler:
    """按 ``message_seq`` 向更早翻页的假 NapCat：0 表示最新一页。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seq = int(body.get("message_seq", 0))
        count = int(body.get("count", 20))
        if seq <= 0:
            page = all_messages[-count:]
        else:
            page = [message for message in all_messages if message["message_seq"] <= seq][-count:]
        return httpx.Response(
            200,
            json={"status": "ok", "retcode": retcode, "data": {"messages": page}},
        )

    return handler


def _source(handler: Handler, *, page_size: int = 3) -> OneBotHistorySource:
    return OneBotHistorySource(
        api_root="http://onebot.invalid",
        access_token="test-token",
        page_size=page_size,
        transport=httpx.MockTransport(handler),
    )


async def test_onebot_source_pages_backwards_until_the_start_of_history() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _history_server(
            [_onebot_message(seq, text=f"第{seq}条", at=f"2026-10-07T09:0{seq}:00+08:00") for seq in range(1, 10)]
        )(request)

    async with _source(handler) as source:
        messages = await source.fetch(group_id=GROUP)

    assert [message.platform_message_id for message in messages] == list(range(1, 10))
    assert [message.text for message in messages][:2] == ["第1条", "第2条"]
    # 请求以 0（最新）起，之后拿本页最老的 seq 当锚点向更早翻，本页没有新东西就停。
    # 锚点不复位减一（见 fetch_onebot_history 里的注释：减一会被 NapCat 判成"消息不存在"）。
    assert [request["message_seq"] for request in seen] == [0, 7, 5, 3, 1]
    assert all(request["group_id"] == GROUP for request in seen)


async def test_onebot_source_sends_bearer_token_to_the_extension_action() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _history_server([_onebot_message(1)])(request)

    async with _source(handler) as source:
        await source.fetch(group_id=GROUP)

    assert requests[0].url.path == "/get_group_msg_history"
    assert requests[0].headers["authorization"] == "Bearer test-token"


async def test_onebot_source_accepts_a_top_level_messages_container() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "ok", "retcode": 0, "messages": [_onebot_message(1, text="顶层容器")]},
        )

    async with _source(handler) as source:
        messages = await source.fetch(group_id=GROUP)

    assert [message.text for message in messages] == ["顶层容器"]


async def test_onebot_source_renders_non_text_segments_as_placeholders() -> None:
    segments: list[dict[str, Any]] = [
        {"type": "text", "data": {"text": "看这张图"}},
        {"type": "image", "data": {"file": "x.png"}},
        {"type": "text", "data": {"text": "和这个文件"}},
        {"type": "file", "data": {"name": "讲义.pdf"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _history_server([_onebot_message(1, text="看这张图", segments=segments)])(request)

    async with _source(handler) as source:
        messages = await source.fetch(group_id=GROUP)

    # 占位符落在原段落位置，保持对话顺序。
    assert messages[0].text == "看这张图[图片]和这个文件[文件]"


async def test_onebot_source_trims_to_the_newest_limit() -> None:
    all_messages = [
        _onebot_message(seq, text=f"第{seq}条", at=f"2026-10-07T09:0{seq}:00+08:00")
        for seq in range(1, 10)
    ]

    async with _source(_history_server(all_messages)) as source:
        messages = await source.fetch(group_id=GROUP, limit=4)

    assert [message.platform_message_id for message in messages] == [6, 7, 8, 9]


async def test_onebot_source_filters_by_since_and_stops_paging() -> None:
    all_messages = [
        _onebot_message(seq, text=f"第{seq}条", at=f"2026-10-07T09:0{seq}:00+08:00")
        for seq in range(1, 10)
    ]
    since = datetime(2026, 10, 7, 9, 5, tzinfo=CST)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _history_server(all_messages)(request)

    async with _source(handler) as source:
        messages = await source.fetch(group_id=GROUP, since=since)

    assert [message.platform_message_id for message in messages] == [6, 7, 8, 9]
    # 第 2 页里最老的已经 <= since，无需再向更早翻。
    assert len(requests) == 2


async def test_onebot_source_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    async with _source(handler) as source:
        with pytest.raises(HistoryFetchError):
            await source.fetch(group_id=GROUP)


async def test_onebot_source_raises_on_a_nonzero_retcode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "failed", "retcode": 100, "data": {"messages": []}},
        )

    async with _source(handler) as source:
        with pytest.raises(HistoryFetchError):
            await source.fetch(group_id=GROUP)


def test_onebot_source_from_settings_requires_an_api_root() -> None:
    with pytest.raises(ValueError):
        OneBotHistorySource.from_settings(_settings())


def test_onebot_source_from_settings_reads_root_and_token() -> None:
    source = OneBotHistorySource.from_settings(
        _settings(onebot_api_root="http://127.0.0.1:3000", onebot_access_token="tok")
    )

    assert source.api_root == "http://127.0.0.1:3000"
    assert source.access_token == "tok"


# —— 共享的翻页与解析（插件侧 bot.call_api 回补复用）——


def test_parse_onebot_message_extracts_sender_time_and_segments() -> None:
    record = _onebot_message(
        7,
        user_id=10002,
        segments=[
            {"type": "text", "data": {"text": "看"}},
            {"type": "image", "data": {"file": "x.png"}},
        ],
    )

    message = parse_onebot_message(GROUP, record)

    assert message.group_id == GROUP
    assert message.user_id == 10002
    assert message.text == "看[图片]"
    assert message.message_seq == 7
    assert message.nickname == "小A"
    assert message.sent_at.tzinfo is not None
    # 原始段落一并带回，供回补渲染出与实时链路一致的文本。
    assert message.segments[1]["type"] == "image"


def test_parse_onebot_message_rejects_a_record_without_time() -> None:
    with pytest.raises(HistoryFetchError):
        parse_onebot_message(GROUP, {"sender": {"user_id": 1}})


def test_extract_history_messages_accepts_both_wrappers() -> None:
    assert extract_history_messages({"messages": [{"seq": 1}]}) == [{"seq": 1}]
    assert extract_history_messages({"data": {"messages": [{"seq": 2}]}}) == [{"seq": 2}]
    assert extract_history_messages({}) is None


async def test_fetch_onebot_history_pages_with_an_injected_page_fetcher() -> None:
    """翻页逻辑与 HTTP 解耦：注入一个 fetch_page 即可复用（插件侧就是 bot.call_api）。"""
    all_messages = [
        _onebot_message(seq, text=f"第{seq}条", at=f"2026-10-07T09:0{seq}:00+08:00")
        for seq in range(1, 10)
    ]
    seen: list[int] = []

    async def fetch_page(message_seq: int) -> list[dict[str, Any]]:
        seen.append(message_seq)
        if message_seq <= 0:
            return all_messages[-3:]
        return [m for m in all_messages if m["message_seq"] <= message_seq][-3:]

    messages = await fetch_onebot_history(group_id=GROUP, fetch_page=fetch_page)

    assert [message.platform_message_id for message in messages] == list(range(1, 10))
    # 游标是本页最老那条**自己**（不复位减一）：每页与上一页重叠一条，由 seen 挡掉。
    # 减一版本的游标是 [0, 6, 3]——那正是实测里 NapCat 抛"消息 X 不存在"的病根。
    assert seen == [0, 7, 5, 3, 1]


async def test_fetch_onebot_history_stops_at_since_and_keeps_newest_limit() -> None:
    all_messages = [
        _onebot_message(seq, text=f"第{seq}条", at=f"2026-10-07T09:0{seq}:00+08:00")
        for seq in range(1, 10)
    ]

    async def fetch_page(message_seq: int) -> list[dict[str, Any]]:
        if message_seq <= 0:
            return all_messages[-3:]
        return [m for m in all_messages if m["message_seq"] <= message_seq][-3:]

    since = datetime(2026, 10, 7, 9, 5, tzinfo=CST)
    messages = await fetch_onebot_history(group_id=GROUP, fetch_page=fetch_page, since=since)

    assert [message.platform_message_id for message in messages] == [6, 7, 8, 9]

    limited = await fetch_onebot_history(group_id=GROUP, fetch_page=fetch_page, limit=4)
    assert [message.platform_message_id for message in limited] == [6, 7, 8, 9]


# —— 回补护栏：起点取锚点与 24 小时下限里的较晚者 ——


def test_resolve_backfill_since_uses_the_anchor_when_inside_the_guard() -> None:
    now = datetime(2026, 10, 7, 12, 0, tzinfo=CST)
    anchor = now - timedelta(hours=2)

    since, clamped = resolve_backfill_since(anchor=anchor, now=now, max_hours=24)

    assert since == anchor
    assert clamped is False


def test_resolve_backfill_since_clamps_an_old_anchor_and_reports_it() -> None:
    now = datetime(2026, 10, 7, 12, 0, tzinfo=CST)
    anchor = now - timedelta(hours=30)

    since, clamped = resolve_backfill_since(anchor=anchor, now=now, max_hours=24)

    assert since == now - timedelta(hours=24)
    assert clamped is True


def test_resolve_backfill_since_without_an_anchor_uses_the_floor_silently() -> None:
    now = datetime(2026, 10, 7, 12, 0, tzinfo=CST)

    since, clamped = resolve_backfill_since(anchor=None, now=now, max_hours=24)

    assert since == now - timedelta(hours=24)
    assert clamped is False
