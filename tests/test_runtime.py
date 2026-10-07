"""Runtime 的行为测试（工单 #5）。

seam 是 *Runtime*：外部依赖（聊天模型、向量、时钟、数据库连接串）在构造时注入，
测试从仓储读 API 断言记忆条目、窗口状态与后台任务的结果。不碰 Runtime 内部字段，
也不给内部协作者打桩——只换掉注入的假实现与时钟。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rzyl_core.db import Category, Repository, WindowStatus, decode_vector
from rzyl_core.llm import (
    ChatModel,
    ChatResult,
    ChatUsage,
    DeterministicEmbedding,
    EmbeddingModel,
    FakeChatModel,
    NullEmbedding,
)
from rzyl_core.llm.fakes import EchoChatModel
from rzyl_core.pipeline import assemble_window
from rzyl_core.pipeline.window import WindowMessage
from rzyl_core.runtime import Runtime
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)

GROUP = 111
USER = 10001


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


class _Clock:
    """可手动推进的时钟：测窗口超时不用真等 5 分钟。"""

    def __init__(self, now: datetime = BEGIN) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + timedelta(minutes=minutes)


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _runtime(
    tmp_path: Path,
    clock: _Clock,
    *,
    chat_model: ChatModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    **settings: object,
) -> Runtime:
    return Runtime(
        chat_model=chat_model or EchoChatModel(),
        embedding_model=embedding_model or DeterministicEmbedding(8),
        settings=_settings(**settings),
        clock=clock,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )


# —— 生命周期 ——


async def test_runtime_start_and_stop_own_the_repository(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock)

    with pytest.raises(RuntimeError):
        _ = runtime.repository

    await runtime.start(run_background_tasks=False)
    repo = runtime.repository
    assert isinstance(repo, Repository)

    await runtime.stop()

    with pytest.raises(RuntimeError):
        _ = runtime.repository


async def test_runtime_background_task_skeletons_are_started_and_cancelled(
    tmp_path: Path, clock: _Clock
) -> None:
    # 关掉启动对账，专心验证四个常驻循环；对账任务另有 tests/test_reconcile.py 守着。
    runtime = _runtime(tmp_path, clock, reconcile_on_startup=False)

    await runtime.start(run_background_tasks=True)
    # 四个常驻循环：保留期清理、向量补算、窗口超时刷新、死信重试。
    assert len(runtime.background_tasks) == 4
    assert all(not task.done() for task in runtime.background_tasks)

    await runtime.stop()
    assert runtime.background_tasks == ()


async def test_ingest_before_start_is_a_programming_error(tmp_path: Path, clock: _Clock) -> None:
    runtime = _runtime(tmp_path, clock)

    with pytest.raises(RuntimeError):
        await runtime.ingest(group_id=GROUP, user_id=USER, text="还没启动", sent_at=BEGIN)


# —— 灌入消息、成窗、落库 ——


async def test_ingest_stores_the_message_before_windowing_and_pipeline_maps_it_back(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=2)
    await runtime.start(run_background_tasks=False)
    try:
        first = await runtime.ingest(
            group_id=GROUP, user_id=USER, text="甲", sent_at=BEGIN, nickname="小A"
        )
        assert first.outcomes == ()
        assert first.buffered == 1

        second = await runtime.ingest(
            group_id=GROUP, user_id=USER, text="乙", sent_at=BEGIN + timedelta(minutes=1), nickname="小A"
        )

        assert len(second.outcomes) == 1
        outcome = second.outcomes[0]
        assert outcome.status.value == "done"
        assert len(outcome.memory_ids) == 1

        memory = await runtime.repository.get_memory(outcome.memory_ids[0])
        assert memory is not None
        # 消息先落库再进窗口，所以 evidence 里的窗口序号能换回真实 message.id。
        assert memory.evidence == [first.message.id]
        assert memory.group_id == GROUP

        window = await runtime.repository.get_window(outcome.window_id)
        assert window is not None
        assert window.status.value == "done"
        assert window.message_count == 2
    finally:
        await runtime.stop()


async def test_flush_expired_closes_a_quiet_window_through_the_pipeline(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=30)
    await runtime.start(run_background_tasks=False)
    try:
        # 时间规则要求缓冲里有 window_min_messages 条，先攒够 5 条再等时间到点。
        for index in range(5):
            await runtime.ingest(group_id=GROUP, user_id=USER, text=f"第{index}条", sent_at=BEGIN)
        assert await runtime.flush_expired() == ()

        clock.advance(6)
        outcomes = await runtime.flush_expired()

        assert len(outcomes) == 1
        assert outcomes[0].status.value == "done"
        assert len(outcomes[0].memory_ids) == 1
    finally:
        await runtime.stop()


# —— 后台任务：保留期清理 ——


async def test_retention_cleanup_deletes_only_messages_older_than_the_cutoff(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, retention_days=30)
    await runtime.start(run_background_tasks=False)
    try:
        repo = runtime.repository
        await repo.add_message(
            group_id=GROUP, user_id=USER, text="很久以前", sent_at=BEGIN - timedelta(days=40)
        )
        await repo.add_message(group_id=GROUP, user_id=USER, text="最近", sent_at=BEGIN)

        removed = await runtime.cleanup_retention()

        assert removed == 1
        remaining = await repo.list_messages(group_id=GROUP)
        assert [message.text for message in remaining] == ["最近"]
    finally:
        await runtime.stop()


# —— 后台任务：向量补算 ——


async def test_embedding_backfill_fills_memories_without_a_vector(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, embedding_model=DeterministicEmbedding(8))
    await runtime.start(run_background_tasks=False)
    try:
        memory = await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="结论：先跑基线再调参",
            confidence=0.9,
            prompt_version="v1",
        )
        assert memory.embedding is None

        filled = await runtime.backfill_embeddings()

        assert filled == 1
        reloaded = await runtime.repository.get_memory(memory.id)
        assert reloaded is not None and reloaded.embedding is not None
    finally:
        await runtime.stop()


async def test_reembed_recomputes_memories_that_already_have_a_vector(
    tmp_path: Path, clock: _Clock
) -> None:
    """换向量模型后：默认补算不动已有向量，``reembed=True`` 才连已有向量一起重算。"""
    old = _runtime(tmp_path, clock, embedding_model=DeterministicEmbedding(4))
    await old.start(run_background_tasks=False)
    try:
        memory = await old.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="结论：先跑基线再调参",
            confidence=0.9,
            prompt_version="v1",
            embedding=[1.0, 0.0, 0.0, 0.0],  # 4 维旧向量，模拟换模型前的历史行
        )
    finally:
        await old.stop()

    new = _runtime(tmp_path, clock, embedding_model=DeterministicEmbedding(8))
    await new.start(run_background_tasks=False)
    try:
        # 默认只补空向量：已有向量的条目不动，维度仍是 4。
        assert await new.backfill_embeddings() == 0
        untouched = await new.repository.get_memory(memory.id)
        assert untouched is not None and untouched.embedding is not None
        assert len(decode_vector(untouched.embedding)) == 4

        # reembed=True：连已有向量一起按新模型重算，维度变成 8。
        assert await new.backfill_embeddings(reembed=True) == 1
        reloaded = await new.repository.get_memory(memory.id)
        assert reloaded is not None and reloaded.embedding is not None
        assert len(decode_vector(reloaded.embedding)) == 8
    finally:
        await new.stop()


async def test_embedding_backfill_is_a_noop_without_a_vector_service(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, embedding_model=NullEmbedding())
    await runtime.start(run_background_tasks=False)
    try:
        memory = await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="没有向量服务",
            confidence=0.9,
            prompt_version="v1",
        )

        assert await runtime.backfill_embeddings() == 0

        reloaded = await runtime.repository.get_memory(memory.id)
        assert reloaded is not None and reloaded.embedding is None
    finally:
        await runtime.stop()


class _MeteredEmbedding:
    """报告输入 token 的假向量，用来验证补算记账真的落了库。"""

    def __init__(self, *, tokens: int) -> None:
        self.last_input_tokens: int | None = tokens
        self._inner = DeterministicEmbedding(8)

    async def embed(self, texts):  # noqa: ANN001 —— 测试替身，形状对齐协议即可
        return await self._inner.embed(texts)


async def test_embedding_backfill_records_usage_and_cost(
    tmp_path: Path, clock: _Clock
) -> None:
    """补算调向量服务也要进记账：用途为 embed、带模型名与按 embedding_price 估的费用。"""
    runtime = _runtime(
        tmp_path,
        clock,
        embedding_model=_MeteredEmbedding(tokens=1000),
        embedding_price=2.0,
    )
    await runtime.start(run_background_tasks=False)
    try:
        await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="需要补算向量的结论",
            confidence=0.9,
            prompt_version="v1",
        )

        assert await runtime.backfill_embeddings() == 1

        calls = await runtime.repository.list_llm_calls(purpose="embed")
        assert len(calls) == 1
        assert calls[0].model == runtime.settings.embedding_model
        assert calls[0].tokens_in == 1000
        assert calls[0].tokens_out == 0
        assert calls[0].success is True
        assert calls[0].cost == pytest.approx(1000 * 2.0 / 1_000_000)
    finally:
        await runtime.stop()


async def test_embedding_backfill_records_a_failed_call_without_fabricating_usage(
    tmp_path: Path, clock: _Clock
) -> None:
    """服务不可用时也留痕：成功位为假，用量按实际能拿到的 0 记，绝不编造。"""
    runtime = _runtime(
        tmp_path, clock, embedding_model=NullEmbedding(), embedding_price=2.0
    )
    await runtime.start(run_background_tasks=False)
    try:
        await runtime.repository.add_memory(
            group_id=GROUP,
            category=Category.KNOWLEDGE,
            statement="服务不可用",
            confidence=0.9,
            prompt_version="v1",
        )

        assert await runtime.backfill_embeddings() == 0

        calls = await runtime.repository.list_llm_calls(purpose="embed")
        assert len(calls) == 1
        assert calls[0].success is False
        assert calls[0].tokens_in == 0
        assert calls[0].cost == 0.0
        assert calls[0].error is not None
    finally:
        await runtime.stop()


# —— 向量模型账本一致性（同维度换模型也查得出）——


async def _seed_embed_call(tmp_path: Path, *, model: str, success: bool = True) -> None:
    """在与 Runtime 同一个库文件里预置一条 embed 记账。"""
    repo = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    try:
        await repo.add_llm_call(purpose="embed", model=model, success=success)
    finally:
        await repo.close()


def _runtime_with_embedding_name(tmp_path: Path, clock: _Clock, name: str) -> Runtime:
    """与 ``_runtime`` 相同，但显式指定 ``settings.embedding_model`` 这个名字。"""
    return Runtime(
        chat_model=EchoChatModel(),
        embedding_model=DeterministicEmbedding(8),
        settings=_settings(embedding_model=name),
        clock=clock,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}",
        provider="offline",
    )


async def test_a_same_dimension_model_swap_is_reported(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """旧 bge-m3 → 新 Qwen 同为 1024 维：长度检查看不出来，只能靠账本查出。"""
    await _seed_embed_call(tmp_path, model="BAAI/bge-m3")
    runtime = _runtime_with_embedding_name(tmp_path, clock, "Qwen/Qwen3-Embedding-0.6B")

    with caplog.at_level(logging.WARNING, logger="rzyl_core.runtime"):
        await runtime.start(run_background_tasks=False)
        try:
            # 同一进程内再查（后台循环每轮都会走到）也不该重复刷屏。
            assert await runtime.check_embedding_model_ledger() == "BAAI/bge-m3"
            assert await runtime.backfill_embeddings() == 0
        finally:
            await runtime.stop()

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0]
    assert "BAAI/bge-m3" in message
    assert "Qwen/Qwen3-Embedding-0.6B" in message
    assert "reembed" in message


async def test_the_ledger_check_is_silent_without_any_embed_records(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """库里还没有任何 embed 记录：静默通过，不算异常。"""
    runtime = _runtime(tmp_path, clock)

    with caplog.at_level(logging.WARNING, logger="rzyl_core.runtime"):
        await runtime.start(run_background_tasks=False)
        try:
            assert await runtime.check_embedding_model_ledger() is None
        finally:
            await runtime.stop()

    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


async def test_the_ledger_check_is_silent_when_the_model_is_unchanged(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = _runtime(tmp_path, clock)
    await _seed_embed_call(tmp_path, model=runtime.settings.embedding_model)

    with caplog.at_level(logging.WARNING, logger="rzyl_core.runtime"):
        await runtime.start(run_background_tasks=False)
        try:
            assert await runtime.check_embedding_model_ledger() == runtime.settings.embedding_model
        finally:
            await runtime.stop()

    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


# —— 离线假模型 ——


async def test_echo_chat_model_returns_one_valid_entry_for_the_first_message() -> None:
    from rzyl_core.pipeline.extract import parse_extraction

    window = assemble_window(
        group_id=GROUP,
        messages=[
            WindowMessage(
                message_id=101,
                group_id=GROUP,
                user_id=10001,
                text="实验课改到周五下午三点",
                sent_at=BEGIN,
                nickname="小A",
            ),
            WindowMessage(
                message_id=102,
                group_id=GROUP,
                user_id=10002,
                text="好的",
                sent_at=BEGIN + timedelta(seconds=1),
                nickname="小B",
            ),
        ],
    )
    rendered = window.render()

    result = await EchoChatModel().complete(rendered.system, rendered.user)
    items = parse_extraction(result.text)

    assert len(items) == 1
    assert items[0].evidence == [1]
    assert items[0].statement == "实验课改到周五下午三点"
    assert items[0].person_refs[0].user_id == 10001
    assert items[0].person_refs[0].nickname_snapshot == "小A"


# —— 实时链路：采集判定 + 文本化 + 入库（里程碑 2）——


def _segments(*parts: dict[str, object]) -> list[dict[str, object]]:
    return list(parts)


async def test_ingest_message_collects_renders_and_stores_an_allowed_group_message(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, window_message_limit=1, group_whitelist=[GROUP])
    await runtime.start(run_background_tasks=False)
    try:
        result = await runtime.ingest_message(
            group_id=GROUP,
            user_id=USER,
            self_id=999,
            segments=_segments(
                {"type": "text", "data": {"text": "看这张图"}},
                {"type": "image", "data": {"url": "http://cdn/x.png", "file": "x.png"}},
            ),
            sent_at=BEGIN,
            nickname="小A",
            platform_message_id=900001,
        )

        assert result is not None
        assert result.message.text == "看这张图[图片]"
        # 图片的 url / file 进了 segments_json，里程碑 4 的落盘要用。
        assert "http://cdn/x.png" in result.message.segments_json
        assert len(result.outcomes) == 1
    finally:
        await runtime.stop()


async def test_ingest_message_skips_groups_outside_the_whitelist(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, group_whitelist=[GROUP])
    await runtime.start(run_background_tasks=False)
    try:
        result = await runtime.ingest_message(
            group_id=999,
            user_id=USER,
            self_id=999,
            segments=_segments({"type": "text", "data": {"text": "不该收"}}),
            sent_at=BEGIN,
        )

        assert result is None
        assert await runtime.repository.list_messages() == []
    finally:
        await runtime.stop()


async def test_ingest_message_accepts_a_group_enabled_only_at_runtime(
    tmp_path: Path, clock: _Clock
) -> None:
    # 配置白名单为空，运行时开关把群开了——判定取两个来源的并集。
    runtime = _runtime(tmp_path, clock)
    await runtime.start(run_background_tasks=False)
    try:
        await runtime.repository.set_group_enabled(GROUP, True)

        assert await runtime.allowed_groups() == frozenset({GROUP})
        result = await runtime.ingest_message(
            group_id=GROUP,
            user_id=USER,
            self_id=999,
            segments=_segments({"type": "text", "data": {"text": "运行时开的群"}}),
            sent_at=BEGIN,
        )

        assert result is not None
    finally:
        await runtime.stop()


async def test_ingest_message_skips_the_bot_own_messages(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(tmp_path, clock, group_whitelist=[GROUP])
    await runtime.start(run_background_tasks=False)
    try:
        result = await runtime.ingest_message(
            group_id=GROUP,
            user_id=USER,
            self_id=USER,
            segments=_segments({"type": "text", "data": {"text": "我自己发的"}}),
            sent_at=BEGIN,
        )

        assert result is None
    finally:
        await runtime.stop()


# —— 后台任务：死信重试（里程碑 2）——


def _chat_json(text: str) -> ChatResult:
    return ChatResult(text=text, usage=ChatUsage(input_tokens=1, output_tokens=1), model="scripted")


async def test_dead_letter_retry_reprocesses_a_dead_window_to_done(
    tmp_path: Path, clock: _Clock
) -> None:
    chat = FakeChatModel(
        [
            _chat_json("这不是 JSON"),  # 首次尝试 1 失败
            _chat_json("这不是 JSON"),  # 首次尝试 2 失败 → 整窗进死信
            _chat_json("[]"),  # 后台重试这一次成功
        ]
    )
    runtime = _runtime(
        tmp_path,
        clock,
        chat_model=chat,
        window_message_limit=1,
        extract_max_attempts=2,
        dead_letter_max_retries=4,
    )
    await runtime.start(run_background_tasks=False)
    try:
        ingested = await runtime.ingest(group_id=GROUP, user_id=USER, text="会失败", sent_at=BEGIN)
        timeline_window_id = ingested.outcomes[0].window_id
        assert ingested.outcomes[0].status is WindowStatus.DEAD

        retried = await runtime.retry_dead_windows()

        assert retried == 1
        window = await runtime.repository.get_window(timeline_window_id)
        assert window is not None
        assert window.status is WindowStatus.DONE
    finally:
        await runtime.stop()


async def test_dead_letter_retry_stops_at_the_configured_guard(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(
        tmp_path,
        clock,
        chat_model=FakeChatModel([_chat_json("坏") for _ in range(10)]),
        window_message_limit=1,
        extract_max_attempts=2,
        dead_letter_max_retries=2,
    )
    await runtime.start(run_background_tasks=False)
    try:
        ingested = await runtime.ingest(group_id=GROUP, user_id=USER, text="必死", sent_at=BEGIN)
        window_id = ingested.outcomes[0].window_id

        # retry_count 已达护栏（2 不小于 2），后台不再重试，也不再消耗模型脚本。
        assert await runtime.retry_dead_windows() == 0
        window = await runtime.repository.get_window(window_id)
        assert window is not None
        assert window.status is WindowStatus.DEAD
    finally:
        await runtime.stop()


async def test_dead_letter_retry_gives_up_when_the_source_messages_are_gone(
    tmp_path: Path, clock: _Clock
) -> None:
    runtime = _runtime(
        tmp_path,
        clock,
        chat_model=FakeChatModel([_chat_json("坏") for _ in range(4)]),
        window_message_limit=1,
        extract_max_attempts=2,
        dead_letter_max_retries=4,
    )
    await runtime.start(run_background_tasks=False)
    try:
        ingested = await runtime.ingest(group_id=GROUP, user_id=USER, text="必死", sent_at=BEGIN)
        window_id = ingested.outcomes[0].window_id

        # 原文被保留期清掉后无法重建窗口：这一轮判它放弃（计入返回值），并把重试次数推到
        # 上限，避免每轮都白跑一次。
        await runtime.repository.delete_messages_before(BEGIN + timedelta(seconds=1))
        assert await runtime.retry_dead_windows() == 1

        window = await runtime.repository.get_window(window_id)
        assert window is not None
        assert window.status is WindowStatus.DEAD
        assert window.retry_count >= 4

        # 已被推到上限，下一轮不再进入重试集合。
        assert await runtime.retry_dead_windows() == 0
    finally:
        await runtime.stop()


async def test_window_flush_loop_body_closes_a_quiet_window(tmp_path: Path, clock: _Clock) -> None:
    """窗口超时刷新的循环体就是 ``flush_expired``；这里验证它按注入时钟正确收尾。"""
    runtime = _runtime(tmp_path, clock, window_message_limit=30, window_flush_seconds=1)
    await runtime.start(run_background_tasks=False)
    try:
        # 攒够最小条数，再让注入时钟跨过 window_minutes。
        for index in range(5):
            await runtime.ingest(group_id=GROUP, user_id=USER, text=f"第{index}条", sent_at=BEGIN)
        assert await runtime.flush_expired() == ()

        clock.advance(6)
        outcomes = await runtime.flush_expired()

        assert len(outcomes) == 1
        assert outcomes[0].status is WindowStatus.DONE
    finally:
        await runtime.stop()


# —— 模块级 Runtime 注册表（插件取 Runtime 的显式方式）——


def test_module_level_runtime_registry_raises_until_configured() -> None:
    from rzyl_core.runtime import get_runtime, get_runtime_or_none, set_runtime

    set_runtime(None)
    try:
        assert get_runtime_or_none() is None
        with pytest.raises(RuntimeError):
            get_runtime()
    finally:
        set_runtime(None)
