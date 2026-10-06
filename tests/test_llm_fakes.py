"""聊天/向量接口的假实现验收测试（工单 #3）。

假实现住在 core 而不是 tests/，因为测试与离线回放脚本要共用同一套。这里断言
外部可观察行为：脚本按顺序吐出结果、四种情况都能模拟、收到的提示词原样留存、
确定性向量同输入同输出、空向量实现返回 None。
"""

import pytest

from rzyl_core.llm import (
    ChatModel,
    ChatResult,
    ChatUsage,
    DeterministicEmbedding,
    EmbeddingModel,
    FakeChatModel,
    NullEmbedding,
)
from rzyl_core.llm.errors import LLMTimeoutError


async def test_fake_chat_model_returns_scripted_results_in_order() -> None:
    script = [
        ChatResult(text='[{"category": "knowledge"}]', usage=ChatUsage(10, 4), model="fake"),
        ChatResult(text="这不是 JSON", usage=ChatUsage(11, 5), model="fake"),
        ChatResult(text="[]", usage=ChatUsage(9, 1), model="fake"),
    ]
    fake = FakeChatModel(script=script)

    first = await fake.complete("系统提示", "第一窗")
    second = await fake.complete("系统提示", "第二窗")
    third = await fake.complete("系统提示", "第三窗")

    assert first.text == '[{"category": "knowledge"}]'
    assert second.text == "这不是 JSON"
    assert third.text == "[]"
    assert third.usage == ChatUsage(input_tokens=9, output_tokens=1)
    assert third.usage.total_tokens == 10


async def test_fake_chat_model_raises_scripted_error() -> None:
    fake = FakeChatModel(script=[LLMTimeoutError("上游超时")])
    with pytest.raises(LLMTimeoutError):
        await fake.complete("系统提示", "用户内容")


async def test_fake_chat_model_keeps_prompts_verbatim_even_on_error() -> None:
    fake = FakeChatModel(script=[LLMTimeoutError("boom"), LLMTimeoutError("boom")])
    with pytest.raises(LLMTimeoutError):
        await fake.complete("系统提示词", "用户内容")
    with pytest.raises(LLMTimeoutError):
        await fake.complete("另一系统", "另一用户")

    assert [p.system for p in fake.received] == ["系统提示词", "另一系统"]
    assert [p.user for p in fake.received] == ["用户内容", "另一用户"]


async def test_fake_chat_model_exhausted_script_fails_clearly() -> None:
    fake = FakeChatModel(script=[])
    with pytest.raises(RuntimeError, match="脚本"):
        await fake.complete("s", "u")


async def test_deterministic_embedding_is_stable_and_dimension_configurable() -> None:
    embedding = DeterministicEmbedding(dimension=8)
    first = await embedding.embed(["论文", "网盘链接"])
    second = await embedding.embed(["论文", "网盘链接"])

    assert first == second  # 同输入永远同输出
    assert first is not None
    assert len(first) == 2
    assert all(len(vector) == 8 for vector in first)
    assert first[0] != first[1]  # 不同输入不同输出


async def test_deterministic_embedding_defaults_to_1024_dims() -> None:
    embedding = DeterministicEmbedding()
    (vector,) = await embedding.embed(["x"])
    assert len(vector) == 1024
    assert embedding.dimension == 1024


def test_deterministic_embedding_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        DeterministicEmbedding(dimension=0)


async def test_null_embedding_reports_unavailable() -> None:
    assert await NullEmbedding().embed(["任何文本"]) is None


def test_fakes_satisfy_the_protocols() -> None:
    assert isinstance(FakeChatModel(script=[]), ChatModel)
    assert isinstance(DeterministicEmbedding(), EmbeddingModel)
    assert isinstance(NullEmbedding(), EmbeddingModel)
