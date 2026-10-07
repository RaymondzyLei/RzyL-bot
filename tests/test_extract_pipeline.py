"""提取、校验、去重、入库的行为测试（工单 #7）。

分两层来测，对应两种 seam：

- **纯函数**（``parse_extraction`` / ``normalize_statement`` / ``cosine_similarity``）
  没有依赖，直接断言输入到输出的映射。
- **管道**（``ExtractionPipeline.process_window``）是这一环对外的 seam：喂进一个组装好的
  窗口，从仓储读 API 断言窗口状态、记忆条目、记账与死信。不断言内部调用顺序，也不给
  内部实现打桩——只换掉注入的假模型 / 假向量 / 时钟。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from rzyl_core.db import (
    Category,
    Memory,
    MemoryStatus,
    Repository,
    WindowStatus,
)
from rzyl_core.llm import (
    ChatModel,
    ChatResult,
    ChatUsage,
    DeterministicEmbedding,
    EmbeddingModel,
    FakeChatModel,
    LLMTimeoutError,
    NullEmbedding,
)
from rzyl_core.pipeline import WindowMessage, assemble_window
from rzyl_core.pipeline.extract import (
    ExtractionParseError,
    cosine_similarity,
    dedupe_hash,
    normalize_statement,
    normalize_supersedes,
    parse_extraction,
)
from rzyl_core.settings import Settings

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)


def _settings(**overrides: object) -> Settings:
    """参数可调、其余用默认值的设置对象；测试只认显式传入的覆盖项。"""
    # ``_env_file`` 是 pydantic-settings 的运行时开关，类型签名里没有，故忽略告警。
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _window(group_id: int = 111, count: int = 3, *, prompt_version: str = "v1"):
    """一个组装好的窗口；``message_id`` 用 100+n，便于断言映射回真实编号。"""
    return assemble_window(
        group_id=group_id,
        messages=[
            WindowMessage(
                message_id=100 + index,
                group_id=group_id,
                user_id=10000 + index,
                text=f"第{index}条",
                sent_at=BEGIN + timedelta(seconds=index),
                nickname=f"用户{index}",
            )
            for index in range(1, count + 1)
        ],
        prompt_version=prompt_version,
    )


def _item(statement: str = "实验课改到周五下午三点", **overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "category": "event",
        "statement": statement,
        "detail": None,
        "confidence": 0.9,
        "evidence": [1, 2],
        "person_refs": [{"user_id": 10001, "nickname_snapshot": "用户1"}],
        "occurred_at": "2026-10-09T15:00",
        "supersedes": [],
    }
    item.update(overrides)
    return item


def _reply(items: list[dict[str, object]], *, model: str = "fake-model") -> ChatResult:
    return ChatResult(
        text=json.dumps(items, ensure_ascii=False),
        usage=ChatUsage(input_tokens=120, output_tokens=30),
        model=model,
    )


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    yield instance
    await instance.close()


# —— 纯函数：校验 ——


def test_parse_accepts_a_well_formed_array() -> None:
    items = parse_extraction(json.dumps([_item()]))

    assert len(items) == 1
    assert items[0].category is Category.EVENT
    assert items[0].evidence == [1, 2]


def test_parse_rejects_broken_json() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction("这不是 JSON")


def test_parse_rejects_a_non_array_top_level() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps({"category": "event"}))


def test_parse_rejects_a_missing_required_field() -> None:
    payload = _item()
    del payload["evidence"]

    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps([payload]))


def test_parse_rejects_an_unknown_category() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps([_item(category="gossip")]))


def test_parse_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps([_item(confidence=1.4)]))


def test_parse_rejects_empty_evidence() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps([_item(evidence=[])]))


def test_parse_rejects_a_non_iso_occurred_at() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(json.dumps([_item(occurred_at="下周五下午")]))


def test_parse_accepts_null_occurred_at_and_detail() -> None:
    item = _item(detail=None, occurred_at=None)

    assert parse_extraction(json.dumps([item]))[0].occurred_at is None


# —— 纯函数：supersedes 归一化 ——
#
# 提示词把已记条目锚点渲染成 ``[#N]``，模型照抄 ``"#2"`` 或包成 ``["#2"]`` 是合理
# 行为；只对这一个字段宽容，避免为格式差异把整窗内容判失败后丢光。


def test_normalize_supersedes_accepts_int_str_hash_and_singleton_array() -> None:
    assert normalize_supersedes(2) == 2
    assert normalize_supersedes("2") == 2
    assert normalize_supersedes("#2") == 2
    assert normalize_supersedes(" #2 ") == 2
    assert normalize_supersedes([2]) == 2
    assert normalize_supersedes(["#2"]) == 2


def test_normalize_supersedes_maps_absent_to_none() -> None:
    assert normalize_supersedes(None) is None
    assert normalize_supersedes([]) is None
    assert normalize_supersedes("") is None


def test_normalize_supersedes_rejects_a_non_numeric_reference() -> None:
    with pytest.raises(ValueError):
        normalize_supersedes("#abc")


def test_parse_accepts_the_hashed_anchor_forms_the_prompt_renders() -> None:
    for value in ("#2", "2", 2, [2], ["#2"]):
        items = parse_extraction(json.dumps([_item(supersedes=value)]))
        assert items[0].supersedes == 2, value


def test_parse_maps_absent_supersedes_to_none() -> None:
    assert parse_extraction(json.dumps([_item(supersedes=None)]))[0].supersedes is None
    assert parse_extraction(json.dumps([_item(supersedes=[])]))[0].supersedes is None


# —— 纯函数：归一化与相似度 ——


def test_normalize_statement_ignores_width_case_and_whitespace() -> None:
    assert normalize_statement(" Ｈｅｌｌｏ  World\n") == normalize_statement("helloworld")
    assert normalize_statement("网盘 链接：abc") == normalize_statement("网盘链接：abc")


def test_dedupe_hash_matches_for_normalized_equals_and_differs_otherwise() -> None:
    assert dedupe_hash("重复  转发") == dedupe_hash("重复转发")
    assert dedupe_hash("一件事") != dedupe_hash("另一件事")


def test_cosine_similarity_is_bounded_and_zero_for_degenerate_input() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


# —— 管道：正常路径 ——


def _pipeline(
    repo: Repository,
    *,
    script: list[ChatResult | Exception] | None = None,
    chat_model: ChatModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    **settings: object,
):
    from rzyl_core.pipeline.extract import ExtractionPipeline

    return ExtractionPipeline(
        repository=repo,
        chat_model=chat_model if chat_model is not None else FakeChatModel(script or []),
        embedding_model=embedding_model if embedding_model is not None else DeterministicEmbedding(8),
        settings=_settings(**settings),
        clock=lambda: BEGIN,
        provider="deepseek",
    )


async def test_normal_reply_persists_a_full_memory_and_finishes_the_window(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_reply([_item()])])

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert len(outcome.memory_ids) == 1
    window = await repo.get_window(outcome.window_id)
    assert window is not None
    assert window.status is WindowStatus.DONE
    assert window.group_id == 111
    assert window.message_count == 3
    assert window.prompt_version == "v1"

    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None
    assert memory.window_id == outcome.window_id
    assert memory.group_id == 111
    assert memory.category is Category.EVENT
    assert memory.statement == "实验课改到周五下午三点"
    assert memory.detail is None
    assert memory.confidence == pytest.approx(0.9)
    # evidence 的窗口内序号 1、2 换回的是真实消息编号 101、102
    assert memory.evidence == [101, 102]
    assert memory.person_refs == [{"user_id": 10001, "nickname_snapshot": "用户1"}]
    assert memory.occurred_at == datetime(2026, 10, 9, 15, 0, tzinfo=CST)
    assert memory.prompt_version == "v1"
    assert memory.dedupe_hash == dedupe_hash("实验课改到周五下午三点")
    assert memory.status is MemoryStatus.ACTIVE
    assert memory.superseded_by is None
    assert memory.model == "fake-model"
    assert memory.tokens_in == 120
    assert memory.tokens_out == 30
    assert memory.embedding is not None


async def test_low_confidence_is_not_gated_on_insert(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_reply([_item(confidence=0.05)])])

    outcome = await pipeline.process_window(_window())

    assert len(outcome.memory_ids) == 1
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.confidence == pytest.approx(0.05)


async def test_empty_array_finishes_the_window_without_entries(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_reply([])])

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert outcome.memory_ids == ()
    assert await repo.list_group_memories(group_id=111) == []


async def test_the_model_call_is_recorded_with_usage_and_estimated_cost(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[_reply([_item()])],
        chat_input_price=2.0,
        chat_output_price=4.0,
    )

    await pipeline.process_window(_window())

    calls = await repo.list_llm_calls(purpose="extract")
    assert len(calls) == 1
    assert calls[0].model == "fake-model"
    assert calls[0].provider == "deepseek"
    assert calls[0].tokens_in == 120
    assert calls[0].tokens_out == 30
    assert calls[0].cost == pytest.approx((120 * 2.0 + 30 * 4.0) / 1_000_000)
    assert calls[0].success is True


# —— 管道：重试与死信 ——


def _broken_reply() -> ChatResult:
    return ChatResult(text="模型今天不想按格式说话", usage=ChatUsage(80, 20), model="fake-model")


async def test_parse_failure_retries_and_then_succeeds(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_broken_reply(), _reply([_item()])])

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert outcome.attempts == 2
    assert len(outcome.memory_ids) == 1
    window = await repo.get_window(outcome.window_id)
    assert window is not None and window.retry_count == 1
    calls = await repo.list_llm_calls(purpose="extract")
    assert [call.success for call in calls] == [True, False]
    # 解析失败的那次同样被计费，token 照记。
    assert calls[1].tokens_in == 80


async def test_out_of_range_evidence_is_a_parse_failure(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_reply([_item(evidence=[99])]), _reply([_item()])])

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert outcome.attempts == 2
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.evidence == [101, 102]


async def test_exhausted_parse_failures_land_in_the_dead_letter(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[_broken_reply(), _broken_reply()],
        extract_max_attempts=2,
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DEAD
    assert outcome.memory_ids == ()
    assert outcome.error is not None and "未能产出可用结果" in outcome.error
    window = await repo.get_window(outcome.window_id)
    assert window is not None
    assert window.status is WindowStatus.DEAD
    assert window.retry_count == 2
    assert window.error is not None and "未能产出可用结果" in window.error
    assert await repo.list_group_memories(group_id=111, include_inactive=True) == []


async def test_timeout_retries_and_then_lands_in_the_dead_letter(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[LLMTimeoutError("超时"), LLMTimeoutError("超时")],
        extract_max_attempts=2,
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DEAD
    window = await repo.get_window(outcome.window_id)
    assert window is not None and window.status is WindowStatus.DEAD and window.retry_count == 2
    calls = await repo.list_llm_calls(purpose="extract")
    assert len(calls) == 2
    assert all(call.success is False and call.tokens_in == 0 for call in calls)


# —— 管道：可观测性（丢了什么要看得见，但不改判定语义）——


def _nonempty_invalid_reply() -> ChatResult:
    """有内容、但格式不合法的输出：数组非空，只是类别不在四类里。"""
    return _reply([_item(category="gossip")])


async def test_parse_failure_logs_the_raw_model_output(
    repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    """校验失败时把模型原始输出记进 WARNING，丢了什么看得见。"""
    pipeline = _pipeline(repo, script=[_broken_reply(), _reply([_item()])])

    with caplog.at_level(logging.WARNING, logger="rzyl_core.pipeline.extract"):
        outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("模型今天不想按格式说话" in message for message in warnings)


async def test_the_logged_raw_output_is_truncated(
    repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    """原始输出可能有几千字，日志里要截断，别把日志刷爆。"""
    long_text = "坏" * 1200
    pipeline = _pipeline(
        repo,
        script=[
            ChatResult(text=long_text, usage=ChatUsage(10, 10), model="fake-model"),
            _reply([_item()]),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="rzyl_core.pipeline.extract"):
        await pipeline.process_window(_window())

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("坏" * 500 in message for message in warnings)
    assert all("坏" * 1200 not in message for message in warnings)


async def test_empty_retry_after_a_content_producing_failure_is_flagged(
    repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    """早先尝试产出过非空内容、最终却以空数组结束：窗口行留说明，状态仍是 done。"""
    pipeline = _pipeline(repo, script=[_nonempty_invalid_reply(), _reply([])])

    with caplog.at_level(logging.WARNING, logger="rzyl_core.pipeline.extract"):
        outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert outcome.memory_ids == ()
    assert outcome.error is not None and "可能有内容被丢弃" in outcome.error
    window = await repo.get_window(outcome.window_id)
    assert window is not None
    # 判定语义不变：空数组仍是正常结束的 done。
    assert window.status is WindowStatus.DONE
    # 但窗口行上要留下一句说明，人工能发现这一窗可能丢了东西。
    assert window.error is not None
    assert "可能有内容被丢弃" in window.error
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("可能有内容被丢弃" in message for message in warnings)


async def test_empty_result_without_a_prior_content_attempt_stays_clean(
    repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    """直接返回空数组是正常结果：不告警、窗口行无 error。"""
    pipeline = _pipeline(repo, script=[_reply([])])

    with caplog.at_level(logging.WARNING, logger="rzyl_core.pipeline.extract"):
        outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    window = await repo.get_window(outcome.window_id)
    assert window is not None and window.status is WindowStatus.DONE and window.error is None
    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


# —— 管道：写库失败保持可重试 ——


class _StorageFailingRepository(Repository):
    """写库必失败的仓储替身：验证管道不把部分结果当成成功。"""

    async def add_window_result(
        self,
        *,
        window_id: int,
        memories: Sequence[Memory] = (),
        supersedings: Sequence[tuple[int, Memory]] = (),
        status: WindowStatus = WindowStatus.DONE,
        retry_count: int | None = None,
        error: str | None = None,
    ) -> list[Memory]:
        raise SQLAlchemyError("模拟写库失败")


async def test_storage_failure_keeps_the_window_retryable(tmp_path: Path) -> None:
    repo = await _StorageFailingRepository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    try:
        pipeline = _pipeline(repo, script=[_reply([_item()])])

        outcome = await pipeline.process_window(_window())

        assert outcome.status is WindowStatus.PENDING
        assert outcome.memory_ids == ()
        assert outcome.error is not None and "写库失败" in outcome.error
        window = await repo.get_window(outcome.window_id)
        assert window is not None and window.status is WindowStatus.PENDING
    finally:
        await repo.close()


async def test_add_window_result_rolls_back_the_whole_window_on_failure(repo: Repository) -> None:
    window = await repo.add_window(group_id=111, started_at=BEGIN, message_count=1)
    good = Memory(
        group_id=111,
        category=Category.EVENT,
        statement="好的条目",
        confidence=0.9,
        evidence=[1],
        person_refs=[],
        prompt_version="v1",
        dedupe_hash="good",
        created_at=BEGIN,
    )
    # created_at 带时区是 TZDateTime 的硬要求；naive 会在写库时炸，用它模拟「写库中途失败」。
    bad = Memory(
        group_id=111,
        category=Category.EVENT,
        statement="坏的条目",
        confidence=0.9,
        evidence=[2],
        person_refs=[],
        prompt_version="v1",
        dedupe_hash="bad",
        created_at=datetime(2026, 10, 7, 9, 0),
    )

    with pytest.raises(SQLAlchemyError):
        await repo.add_window_result(
            window_id=window.id,
            memories=[good, bad],
            status=WindowStatus.DONE,
        )

    reloaded = await repo.get_window(window.id)
    assert reloaded is not None and reloaded.status is WindowStatus.PENDING
    assert await repo.list_group_memories(group_id=111, include_inactive=True) == []


# —— 管道：两层去重 ——


async def test_hash_duplicate_in_the_same_group_is_merged(repo: Repository) -> None:
    existing = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="群友分享了讲义网盘链接",
        confidence=0.9,
        prompt_version="v1",
        dedupe_hash=dedupe_hash("群友分享了讲义网盘链接"),
    )
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(statement="群友分享了讲义网盘  链接")])],
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    assert outcome.memory_ids == ()
    assert outcome.merged_count == 1
    # 已有条目原样保留，不因合并被改写。
    kept = await repo.get_memory(existing.id)
    assert kept is not None and kept.statement == "群友分享了讲义网盘链接"
    assert len(await repo.list_group_memories(group_id=111)) == 1


async def test_hash_dedupe_does_not_cross_groups(repo: Repository) -> None:
    await repo.add_memory(
        group_id=222,
        category=Category.EVENT,
        statement="群友分享了讲义网盘链接",
        confidence=0.9,
        prompt_version="v1",
        dedupe_hash=dedupe_hash("群友分享了讲义网盘链接"),
    )
    pipeline = _pipeline(repo, script=[_reply([_item(statement="群友分享了讲义网盘链接")])])

    outcome = await pipeline.process_window(_window(group_id=111))

    assert len(outcome.memory_ids) == 1
    assert outcome.merged_count == 0


async def test_identical_items_within_one_reply_are_merged(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(statement="同一个链接"), _item(statement="同一个链接")])],
    )

    outcome = await pipeline.process_window(_window())

    assert len(outcome.memory_ids) == 1
    assert outcome.merged_count == 1


class _FixedEmbedding:
    """按文本查表的假向量；查不到给一个正交的默认向量。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._mapping.get(text, [0.0, 1.0]) for text in texts]


class _ExplodingEmbedding:
    """服务不可用：一调就炸。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("向量服务炸了")


async def test_near_duplicate_in_same_group_and_category_is_marked_suspect(repo: Repository) -> None:
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="结论：先跑基线再调参",
        confidence=0.9,
        prompt_version="v1",
        embedding=[1.0, 0.0],
    )
    embedding = _FixedEmbedding({"先去跑个基线再谈调参": [0.99, 0.05]})
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(category="knowledge", statement="先去跑个基线再谈调参")])],
        embedding_model=embedding,
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.suspect_count == 1
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.status is MemoryStatus.SUSPECT_DUPLICATE


async def test_near_duplicate_requires_the_same_category(repo: Repository) -> None:
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="结论：先跑基线再调参",
        confidence=0.9,
        prompt_version="v1",
        embedding=[1.0, 0.0],
    )
    embedding = _FixedEmbedding({"周五下午三点实验课": [0.99, 0.05]})
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(category="event", statement="周五下午三点实验课")])],
        embedding_model=embedding,
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.suspect_count == 0
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.status is MemoryStatus.ACTIVE


async def test_vector_dimension_mismatch_is_reported_as_a_warning(
    repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    """库里留着旧维度的历史行时，近重复判定要出声——不静默算 0。"""
    stored = await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="结论：先跑基线再调参",
        confidence=0.9,
        prompt_version="v1",
        embedding=[1.0, 0.0, 0.0, 0.0],  # 4 维旧向量，模拟换模型前的历史行
    )
    embedding = DeterministicEmbedding(8)  # 新模型产出 8 维
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(category="knowledge", statement="先去跑个基线再谈调参")])],
        embedding_model=embedding,
    )

    with caplog.at_level(logging.WARNING, logger="rzyl_core.pipeline.extract"):
        outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    # 不一致的条目不该被判为重复（安全方向），但必须留下告警。
    assert outcome.suspect_count == 0
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(stored.id) in message
    assert "4" in message and "8" in message


async def test_dedupe_threshold_is_configurable(repo: Repository) -> None:
    await repo.add_memory(
        group_id=111,
        category=Category.KNOWLEDGE,
        statement="结论 A",
        confidence=0.9,
        prompt_version="v1",
        embedding=[1.0, 0.0],
    )
    embedding = _FixedEmbedding({"结论 B": [0.7, 0.714]})  # 余弦约 0.7
    pipeline = _pipeline(
        repo,
        script=[_reply([_item(category="knowledge", statement="结论 B")])],
        embedding_model=embedding,
        dedupe_similarity_threshold=0.6,
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.suspect_count == 1


# —— 管道：向量不可用与 supersede ——


async def test_unavailable_embedding_still_persists_the_memory(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[_reply([_item()])],
        embedding_model=NullEmbedding(),
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None
    assert memory.status is MemoryStatus.ACTIVE
    assert memory.embedding is None


async def test_embedding_service_error_does_not_block_extraction(repo: Repository) -> None:
    pipeline = _pipeline(
        repo,
        script=[_reply([_item()])],
        embedding_model=_ExplodingEmbedding(),
    )

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.embedding is None


async def test_supersede_expires_the_old_memory_and_links_the_new_one(repo: Repository) -> None:
    old = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="实验课改到周三",
        confidence=0.9,
        prompt_version="v1",
        dedupe_hash=dedupe_hash("实验课改到周三"),
    )
    window = assemble_window(
        group_id=111,
        messages=[
            WindowMessage(
                message_id=101,
                group_id=111,
                user_id=10001,
                text="实验课改到周五了",
                sent_at=BEGIN,
                nickname="用户1",
            )
        ],
        remembered=[old],
        prompt_version="v1",
    )
    pipeline = _pipeline(
        repo,
        script=[
            _reply(
                [_item(statement="实验课改到周五下午三点", evidence=[1], supersedes=[old.id])]
            )
        ],
    )

    outcome = await pipeline.process_window(window)

    assert len(outcome.memory_ids) == 1
    new = await repo.get_memory(outcome.memory_ids[0])
    assert new is not None
    assert new.status is MemoryStatus.ACTIVE
    assert new.superseded_by is None

    reloaded_old = await repo.get_memory(old.id)
    assert reloaded_old is not None  # 只标记，不删除
    assert reloaded_old.status is MemoryStatus.EXPIRED
    assert reloaded_old.superseded_by == new.id


async def test_supersede_pointing_at_an_unknown_id_is_ignored(repo: Repository) -> None:
    pipeline = _pipeline(repo, script=[_reply([_item(supersedes=[99999])])])

    outcome = await pipeline.process_window(_window())

    assert outcome.status is WindowStatus.DONE
    memory = await repo.get_memory(outcome.memory_ids[0])
    assert memory is not None and memory.status is MemoryStatus.ACTIVE


async def test_supersede_accepts_the_hashed_reference_the_model_copied(
    repo: Repository,
) -> None:
    """真机失败的原样复现：摘要写 ``[#N]``，模型回 ``["#N"]``，必须照常生效。"""
    old = await repo.add_memory(
        group_id=111,
        category=Category.EVENT,
        statement="实验课改到周三",
        confidence=0.9,
        prompt_version="v1",
        dedupe_hash=dedupe_hash("实验课改到周三"),
    )
    window = assemble_window(
        group_id=111,
        messages=[
            WindowMessage(
                message_id=101,
                group_id=111,
                user_id=10001,
                text="实验课改到周五了",
                sent_at=BEGIN,
                nickname="用户1",
            )
        ],
        remembered=[old],
        prompt_version="v1",
    )
    pipeline = _pipeline(
        repo,
        script=[
            _reply(
                [_item(statement="实验课改到周五下午三点", evidence=[1], supersedes=[f"#{old.id}"])]
            )
        ],
    )

    outcome = await pipeline.process_window(window)

    assert outcome.status is WindowStatus.DONE
    assert len(outcome.memory_ids) == 1
    reloaded_old = await repo.get_memory(old.id)
    assert reloaded_old is not None and reloaded_old.status is MemoryStatus.EXPIRED
    assert reloaded_old.superseded_by == outcome.memory_ids[0]


async def test_process_window_reuses_an_upstream_window_row(repo: Repository) -> None:
    window = await repo.add_window(
        group_id=111,
        started_at=BEGIN,
        ended_at=BEGIN + timedelta(minutes=1),
        message_count=3,
        prompt_version="v1",
    )
    pipeline = _pipeline(repo, script=[_reply([_item()])])

    outcome = await pipeline.process_window(_window(), window_id=window.id)

    assert outcome.window_id == window.id
    assert len(await repo.list_group_memories(group_id=111)) == 1


# —— 在途窗口的登记（重试循环靠它区分「正在处理」与「没处理完」）——


class _BlockingChat:
    """进到 ``complete`` 就挂住，直到测试放行——用来把「正在处理」这个状态钉在半空中。"""

    def __init__(self, result: ChatResult | None = None) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self._result = result or _reply([_item()])

    async def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return self._result


async def test_pipeline_marks_the_window_in_flight_only_while_it_processes_it(
    repo: Repository,
) -> None:
    """窗口在库里的状态是 pending（「没处理完」），而有东西正在处理它——两者要能分开。"""
    chat = _BlockingChat()
    pipeline = _pipeline(repo, chat_model=chat)
    assert pipeline.windows_in_flight == frozenset()

    task = asyncio.create_task(pipeline.process_window(_window()))
    await asyncio.wait_for(chat.entered.wait(), timeout=2)
    try:
        pending = await repo.list_windows_by_status(WindowStatus.PENDING)
        assert len(pending) == 1
        assert pipeline.windows_in_flight == {int(pending[0].id)}
    finally:
        chat.release.set()
    outcome = await asyncio.wait_for(task, timeout=5)

    assert outcome.status is WindowStatus.DONE
    assert pipeline.windows_in_flight == frozenset()


async def test_pipeline_releases_the_window_even_when_processing_blows_up(
    repo: Repository,
) -> None:
    """失败也要放手：否则一个炸掉的窗口会被永远当成「正在处理」，变成新的一种卡死。"""
    pipeline = _pipeline(repo, chat_model=FakeChatModel([]))  # 脚本为空 → 调用即抛

    with pytest.raises(RuntimeError):
        await pipeline.process_window(_window())

    assert pipeline.windows_in_flight == frozenset()
    pending = await repo.list_windows_by_status(WindowStatus.PENDING)
    assert len(pending) == 1
    # 停在 pending 的样子与「正在处理」一模一样，所以原因必须写进窗口行。
    assert pending[0].error is not None and "未预期的异常" in pending[0].error
