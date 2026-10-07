"""离线回放的行为测试（工单 #5）。

回放走 Runtime 这条 seam：一段历史消息（样本文件或 OneBot）喂进去，正常模式从仓储
读回条目与窗口状态；``dry_run`` 只产出提示词与模型原始输出，仓储里空空如也。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from rzyl_core.llm import DeterministicEmbedding
from rzyl_core.llm.fakes import EchoChatModel
from rzyl_core.llm.prompts import DEFAULT_PROMPT_VERSION
from rzyl_core.pipeline.history import OneBotHistorySource, SampleHistorySource
from rzyl_core.runtime import Runtime
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_history.json"
GROUP = 100200300


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


class _Clock:
    """壁钟停在很远以后，用来证明成窗不看注入时钟、只看消息自身的时间。

    旧实现拿「当下时钟 − 缓冲首条时间」判超时，壁钟停在未来会让历史消息瞬间全部成窗；
    现在成窗只看消息自身的 ``sent_at``，这个时钟再怎么偏都不影响切窗。
    """

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _runtime(tmp_path: Path) -> Runtime:
    runtime = Runtime(
        chat_model=EchoChatModel(),
        embedding_model=DeterministicEmbedding(8),
        settings=_settings(window_message_limit=3, window_minutes=5),
        clock=_Clock(datetime(2030, 1, 1, 0, 0, tzinfo=CST)),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )
    await runtime.start(run_background_tasks=False)
    return runtime


# —— 正常模式 ——


async def test_replay_over_the_sample_history_persists_entries(tmp_path: Path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        report = await runtime.replay(source=SampleHistorySource(SAMPLE), group_id=GROUP)

        assert report.dry_run is False
        assert report.message_count == 8
        assert report.window_count == 3
        assert len(report.memory_ids) == 3
        assert all(outcome.status.value == "done" for outcome in report.outcomes)

        memories = await runtime.repository.list_group_memories(group_id=GROUP)
        assert len(memories) == 3
        statements = {memory.statement for memory in memories}
        assert "求一份上周的数据集，有人有吗" in statements

        messages = await runtime.repository.list_messages(group_id=GROUP)
        assert len(messages) == 8
    finally:
        await runtime.stop()


async def test_replay_windows_by_the_replayed_timeline_not_the_wall_clock(tmp_path: Path) -> None:
    """成窗跟着回放时间线（消息自身时间）走：09:20 那条把前一段 09:03–09:05 顶成一个窗口。"""
    runtime = await _runtime(tmp_path)
    try:
        report = await runtime.replay(source=SampleHistorySource(SAMPLE), group_id=GROUP)

        window_lengths = [outcome.window_id for outcome in report.outcomes]
        assert len(window_lengths) == 3
        windows = [await runtime.repository.get_window(window_id) for window_id in window_lengths]
        assert [window.message_count for window in windows if window is not None] == [3, 2, 3]
    finally:
        await runtime.stop()


async def test_replay_can_pull_history_from_onebot(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seq = int(body.get("message_seq", 0))
        count = int(body.get("count", 20))
        all_messages: list[dict[str, Any]] = [
            {
                "message_id": index,
                "real_id": index,
                "message_seq": index,
                "time": int(
                    datetime(2026, 10, 7, 9, 0, tzinfo=CST).timestamp() + index * 30
                ),
                "sender": {"user_id": 10000 + index, "nickname": f"用户{index}", "card": ""},
                "message": [{"type": "text", "data": {"text": f"第{index}条"}}],
            }
            for index in range(1, 5)
        ]
        page = all_messages[-count:] if seq <= 0 else [
            message for message in all_messages if message["message_seq"] <= seq
        ][-count:]
        return httpx.Response(200, json={"status": "ok", "retcode": 0, "data": {"messages": page}})

    source = OneBotHistorySource(
        api_root="http://onebot.invalid",
        access_token="tok",
        page_size=2,
        transport=httpx.MockTransport(handler),
    )
    runtime = await _runtime(tmp_path)
    try:
        async with source:
            report = await runtime.replay(source=source, group_id=GROUP)

        assert report.message_count == 4
        assert len(report.memory_ids) >= 1
        assert len(await runtime.repository.list_messages(group_id=GROUP)) == 4
    finally:
        await runtime.stop()


# —— --dry-run ——


async def test_dry_run_prints_the_prompt_and_raw_output_without_writing(tmp_path: Path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        report = await runtime.replay(
            source=SampleHistorySource(SAMPLE), group_id=GROUP, dry_run=True
        )

        assert report.dry_run is True
        assert report.memory_ids == ()
        assert report.message_count == 8
        assert report.window_count == 3
        assert len(report.previews) == 3

        first = report.previews[0]
        assert first.error is None
        assert first.rendered.system
        assert "本窗口消息" in first.rendered.user
        assert first.raw_output is not None
        assert json.loads(first.raw_output)[0]["evidence"] == [1]

        # 库是干净的：没有条目、没有窗口、没有消息、没有记账。
        repository = runtime.repository
        assert await repository.list_group_memories(group_id=GROUP) == []
        assert await repository.list_messages(group_id=GROUP) == []
        assert await repository.list_llm_calls() == []
        assert report.previews[0].rendered.version == DEFAULT_PROMPT_VERSION
    finally:
        await runtime.stop()


async def test_dry_run_renders_the_full_prompt_including_the_previous_tail(tmp_path: Path) -> None:
    """完整提示词：第二个窗口起应带上一窗口尾部（仅上下文、不编号）。

    已记条目摘要这一段在 dry-run 里是空的——本模式不往任何库写条目，也就没有可摘要
    的既往条目；尾部则来自组装器的内存态，照常出现。
    """
    runtime = await _runtime(tmp_path)
    try:
        report = await runtime.replay(
            source=SampleHistorySource(SAMPLE), group_id=GROUP, dry_run=True
        )

        assert "（以下为上一窗口尾部，仅作上下文，不要重复提取）" not in report.previews[0].rendered.user
        assert "（以下为上一窗口尾部，仅作上下文，不要重复提取）" in report.previews[1].rendered.user
        assert "# 已记条目摘要" in report.previews[1].rendered.user
    finally:
        await runtime.stop()


async def test_dry_run_touches_no_database_at_all(tmp_path: Path) -> None:
    """dry-run 连配置的库都不需要：不 start、不建库文件，照样能出完整提示词。"""
    runtime = Runtime(
        chat_model=EchoChatModel(),
        embedding_model=DeterministicEmbedding(8),
        settings=_settings(window_message_limit=3, window_minutes=5),
        clock=_Clock(datetime(2030, 1, 1, 0, 0, tzinfo=CST)),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
    )

    report = await runtime.replay(source=SampleHistorySource(SAMPLE), group_id=GROUP, dry_run=True)

    assert report.window_count == 3
    assert not (tmp_path / "rzyl.db").exists()


async def test_replay_with_an_empty_source_is_a_noop(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    runtime = await _runtime(tmp_path)
    try:
        report = await runtime.replay(source=SampleHistorySource(empty), group_id=GROUP)

        assert report.message_count == 0
        assert report.window_count == 0
        assert report.memory_ids == ()
    finally:
        await runtime.stop()
