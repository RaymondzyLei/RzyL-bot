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
    # 请求以 0（最新）起，然后按「本页最老的 seq - 1」向更早翻；落到 0 就停。
    assert [request["message_seq"] for request in seen] == [0, 6, 3]
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
