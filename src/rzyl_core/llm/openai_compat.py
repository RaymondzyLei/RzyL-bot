"""OpenAI 兼容的聊天与向量客户端（httpx 手写，不引入 openai SDK）。

两个真实现都只在参数齐全时才构造：基址、密钥、模型名缺失或为空时在构造阶段就
失败，绝不静默把请求发出去。构造函数只接受显式参数（基址、密钥、模型名、超时、
重试次数、并发上限），不认识任何设置对象——把设置注入客户端是装配层（#5）的活，
这样本模块与 #2（设置对象）可以并行开发、互不依赖。

httpx 的传输层可注入（``transport``）：单元测试传 ``httpx.MockTransport`` 即可
验证超时、指数退避重试与并发上限，不需要真实网络。

DeepSeek 官方没有 embedding 接口，所以聊天与向量是两组彼此独立的基址与密钥，
两个客户端各自接受自己那一组。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from rzyl_core.llm.chat import ChatResult, ChatUsage
from rzyl_core.llm.embedding import Embedding
from rzyl_core.llm.errors import LLMError, LLMRequestError, LLMResponseError, LLMTimeoutError

#: 可重试的 HTTP 状态：限流、请求超时与各类服务端错误。
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _require_text(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} 不能为空；客户端只在参数齐全时才构造")
    return value


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMResponseError(f"响应字段 {name} 不是整数：{value!r}")
    return value


class _OpenAICompatClientBase:
    """聊天与向量客户端的公共部分：显式参数校验、httpx 客户端、退避重试与并发闸。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        max_concurrency: int,
        backoff_base: float,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        # 先校验，任何一项不合格都在构造阶段失败——不会有一个半成品客户端溜出去发请求。
        self._base_url = _require_text(base_url, "base_url").rstrip("/") + "/"
        self._model = _require_text(model, "model")
        self._api_key = _require_text(api_key, "api_key")
        if timeout <= 0:
            raise ValueError(f"timeout 必须为正数，收到 {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries 不能为负，收到 {max_retries}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency 至少为 1，收到 {max_concurrency}")
        if backoff_base < 0:
            raise ValueError(f"backoff_base 不能为负，收到 {backoff_base}")

        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._backoff_base = float(backoff_base)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _request_json(
        self, path: str, payload: dict[str, Any], *, operation: str
    ) -> dict[str, Any]:
        """带并发闸与指数退避重试地 POST 一个 JSON 端点，返回解析后的对象。"""
        async with self._semaphore:
            attempt = 0
            while True:
                last_error: LLMError
                try:
                    response = await self._client.post(path, json=payload)
                except httpx.TimeoutException as exc:
                    last_error = LLMTimeoutError(
                        f"{operation} 超时（第 {attempt + 1} 次尝试）：{exc}"
                    )
                except httpx.TransportError as exc:
                    last_error = LLMRequestError(
                        f"{operation} 网络错误（第 {attempt + 1} 次尝试）：{exc}"
                    )
                else:
                    if response.status_code in _RETRYABLE_STATUS:
                        last_error = LLMRequestError(
                            f"{operation} 返回可重试状态 {response.status_code}"
                            f"（第 {attempt + 1} 次尝试）"
                        )
                    elif response.status_code >= 400:
                        # 4xx 是调用方的问题，重试没有意义，直接失败。
                        raise LLMRequestError(
                            f"{operation} 返回 {response.status_code}：{response.text[:200]}"
                        )
                    else:
                        return _parse_json_object(response, operation)

                if attempt >= self._max_retries:
                    raise last_error
                delay = self._backoff_base * (2**attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt += 1


def _parse_json_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        data: Any = response.json()
    except ValueError as exc:
        raise LLMResponseError(f"{operation} 响应不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError(f"{operation} 响应不是 JSON 对象：{type(data).__name__}")
    return data


class OpenAICompatChatClient(_OpenAICompatClientBase):
    """OpenAI 兼容的聊天补全客户端，POST ``{base_url}/chat/completions``。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        backoff_base: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            backoff_base=backoff_base,
            transport=transport,
        )

    async def complete(self, system: str, user: str) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = await self._request_json("chat/completions", payload, operation="聊天补全")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("聊天补全响应缺少非空的 choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseError("聊天补全响应的 choices[0] 不是对象")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("聊天补全响应的 choices[0].message 不是对象")
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMResponseError("聊天补全响应的文本不是字符串")

        usage = data.get("usage")
        if not isinstance(usage, dict):
            # 用量是记账的依据，缺了就不能静默当 0——宁可失败，也不要账目不准。
            raise LLMResponseError("聊天补全响应缺少 usage，无法记账")
        return ChatResult(
            text=content,
            usage=ChatUsage(
                input_tokens=_as_int(usage.get("prompt_tokens"), "usage.prompt_tokens"),
                output_tokens=_as_int(usage.get("completion_tokens"), "usage.completion_tokens"),
            ),
            model=self._model,
        )


class OpenAICompatEmbeddingClient(_OpenAICompatClientBase):
    """OpenAI 兼容的向量客户端，POST ``{base_url}/embeddings``。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        backoff_base: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            backoff_base=backoff_base,
            transport=transport,
        )

    async def embed(self, texts: Sequence[str]) -> list[Embedding] | None:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self._model, "input": list(texts)}
        data = await self._request_json("embeddings", payload, operation="向量嵌入")

        items = data.get("data")
        if not isinstance(items, list):
            raise LLMResponseError("向量嵌入响应缺少 data 列表")
        ordered = sorted(
            items,
            key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
        )
        vectors: list[Embedding] = []
        for item in ordered:
            if not isinstance(item, dict):
                raise LLMResponseError("向量嵌入响应的 data 元素不是对象")
            raw = item.get("embedding")
            if not isinstance(raw, list):
                raise LLMResponseError("向量嵌入响应的 embedding 不是列表")
            vectors.append([float(value) for value in raw])
        if len(vectors) != len(texts):
            raise LLMResponseError(
                f"向量嵌入数量与输入不符：输入 {len(texts)} 条，返回 {len(vectors)} 条"
            )
        return vectors


__all__ = ["OpenAICompatChatClient", "OpenAICompatEmbeddingClient"]
