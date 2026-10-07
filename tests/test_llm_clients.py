"""OpenAI 兼容客户端的验收测试（工单 #3）。

真实现只会在参数齐全时才构造；超时、指数退避重试、并发上限都在这里用
``httpx.MockTransport`` 验证——不发任何真实网络请求。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from rzyl_core.llm import ChatModel, EmbeddingModel
from rzyl_core.llm.errors import LLMRequestError, LLMResponseError, LLMTimeoutError
from rzyl_core.llm.openai_compat import (
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
)

Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]
)


def chat_client(handler: Handler, **overrides: object) -> OpenAICompatChatClient:
    params: dict[str, object] = {
        "base_url": "https://api.example.invalid/v1",
        "api_key": "sk-test",
        "model": "test-chat",
        "timeout": 1.0,
        "max_retries": 2,
        "max_concurrency": 4,
        "backoff_base": 0.0,
        "transport": httpx.MockTransport(handler),
    }
    params.update(overrides)
    return OpenAICompatChatClient(**params)  # type: ignore[arg-type]


def embedding_client(handler: Handler, **overrides: object) -> OpenAICompatEmbeddingClient:
    params: dict[str, object] = {
        "base_url": "https://api.example.invalid/v1",
        "api_key": "sk-test",
        "model": "test-embed",
        "timeout": 1.0,
        "max_retries": 2,
        "max_concurrency": 4,
        "backoff_base": 0.0,
        "transport": httpx.MockTransport(handler),
    }
    params.update(overrides)
    return OpenAICompatEmbeddingClient(**params)  # type: ignore[arg-type]


def chat_ok(content: str = "[]", *, input_tokens: int = 12, output_tokens: int = 3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        },
    )


# --- 聊天客户端 ---------------------------------------------------------------


async def test_chat_client_returns_text_and_token_usage() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return chat_ok('[{"category": "resource"}]', input_tokens=21, output_tokens=7)

    client = chat_client(handler)
    async with client:
        result = await client.complete("系统", "用户")

    assert result.text == '[{"category": "resource"}]'
    assert result.usage.input_tokens == 21
    assert result.usage.output_tokens == 7
    assert result.model == "test-chat"
    assert seen_paths == ["/v1/chat/completions"]


async def test_chat_client_retries_then_succeeds_on_rate_limit() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"error": "slow down"})
        return chat_ok()

    client = chat_client(handler, max_retries=3)
    async with client:
        result = await client.complete("s", "u")

    assert result.text == "[]"
    assert calls == 3


async def test_chat_client_retries_timeouts_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("模拟超时", request=request)

    client = chat_client(handler, max_retries=2)
    async with client:
        with pytest.raises(LLMTimeoutError):
            await client.complete("s", "u")

    assert calls == 3  # 首次 + 2 次重试


async def test_chat_client_does_not_retry_client_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = chat_client(handler)
    async with client:
        with pytest.raises(LLMRequestError):
            await client.complete("s", "u")

    assert calls == 1


async def test_chat_client_rejects_malformed_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    client = chat_client(handler)
    async with client:
        with pytest.raises(LLMResponseError):
            await client.complete("s", "u")


async def test_chat_client_requires_usage_for_accounting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    client = chat_client(handler)
    async with client:
        with pytest.raises(LLMResponseError):
            await client.complete("s", "u")


async def test_chat_client_bounds_concurrency() -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return chat_ok()

    client = chat_client(handler, max_concurrency=2)
    async with client:
        await asyncio.gather(*(client.complete("s", f"u{i}") for i in range(6)))

    assert peak <= 2


async def test_chat_client_merges_extra_body_into_the_request() -> None:
    """额外请求体原样合并：模型怪癖（如 enable_thinking）走配置，不改代码。"""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return chat_ok()

    client = chat_client(handler, extra_body={"enable_thinking": False, "top_p": 0.8})
    async with client:
        await client.complete("系统", "用户")

    assert seen["enable_thinking"] is False
    assert seen["top_p"] == 0.8
    # 基本字段仍由客户端生成，不被额外字段挤掉。
    assert seen["model"] == "test-chat"
    assert seen["messages"][0]["role"] == "system"


async def test_chat_client_without_extra_body_sends_only_the_basics() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return chat_ok()

    client = chat_client(handler)
    async with client:
        await client.complete("系统", "用户")

    assert set(seen) == {"model", "messages"}


def test_chat_client_rejects_missing_configuration() -> None:
    transport = httpx.MockTransport(lambda request: chat_ok())
    common = {
        "base_url": "https://api.example.invalid/v1",
        "api_key": "sk-test",
        "model": "test-chat",
        "transport": transport,
    }
    for field, value in (
        ("base_url", "   "),
        ("api_key", ""),
        ("model", ""),
        ("timeout", 0),
        ("max_retries", -1),
        ("max_concurrency", 0),
    ):
        with pytest.raises(ValueError):
            OpenAICompatChatClient(**(common | {field: value}))  # type: ignore[arg-type]


# --- 向量客户端 ---------------------------------------------------------------


async def test_embedding_client_returns_vectors_ordered_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.1, 0.2]},
                    {"index": 0, "embedding": [0.3, 0.4]},
                ]
            },
        )

    client = embedding_client(handler)
    async with client:
        vectors = await client.embed(["甲", "乙"])

    assert vectors == [[0.3, 0.4], [0.1, 0.2]]


async def test_embedding_client_posts_to_embeddings_endpoint() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = embedding_client(handler)
    async with client:
        await client.embed(["x"])

    assert paths == ["/v1/embeddings"]


async def test_embedding_client_retries_server_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = embedding_client(handler)
    async with client:
        await client.embed(["x"])

    assert calls == 2


async def test_embedding_client_rejects_missing_configuration() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    common: dict[str, object] = {
        "base_url": "https://api.example.invalid/v1",
        "api_key": "sk-test",
        "model": "test-embed",
        "transport": transport,
    }
    for field, value in (
        ("base_url", ""),
        ("api_key", "  "),
        ("model", ""),
        ("timeout", -1),
        ("max_concurrency", 0),
    ):
        with pytest.raises(ValueError):
            OpenAICompatEmbeddingClient(**(common | {field: value}))  # type: ignore[arg-type]


def test_real_clients_satisfy_the_protocols() -> None:
    transport = httpx.MockTransport(lambda request: chat_ok())
    async_client = OpenAICompatChatClient(
        base_url="https://api.example.invalid/v1",
        api_key="sk-test",
        model="test-chat",
        transport=transport,
    )
    embedding = OpenAICompatEmbeddingClient(
        base_url="https://api.example.invalid/v1",
        api_key="sk-test",
        model="test-embed",
        transport=transport,
    )
    assert isinstance(async_client, ChatModel)
    assert isinstance(embedding, EmbeddingModel)
