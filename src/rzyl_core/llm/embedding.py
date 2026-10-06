"""向量嵌入接口与两个不需要网络/密钥的实现。

``embed()`` 返回与输入等长的向量列表；返回 ``None`` 表示"当前不可用"——调用方
应把向量留空、由后台任务补算（见 issue #1 的 embedding 后台补算决策）。第一阶段
没有 embedding key 也能把整条管道跑通，靠的就是这个约定。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable

#: 一条向量就是一组浮点数。维度由具体实现决定（假实现默认 1024）。
Embedding: TypeAlias = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    """可替换的向量嵌入接口。"""

    async def embed(self, texts: Sequence[str]) -> list[Embedding] | None:
        """为一批文本算向量；返回 ``None`` 表示当前不可用。"""
        ...


class DeterministicEmbedding:
    """确定性的假向量：同输入永远同输出，不需要模型文件与网络，维度可配。

    用输入文本的 sha256 作为随机种子，保证跨进程、跨运行稳定（不依赖 Python 的
    加盐 hash），测试与离线回放因此可复现。
    """

    def __init__(self, dimension: int = 1024) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension 必须为正数，收到 {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Embedding:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]


class NullEmbedding:
    """空实现：永远返回 ``None``，表示向量能力当前不可用。"""

    async def embed(self, texts: Sequence[str]) -> None:
        return None


__all__ = ["DeterministicEmbedding", "Embedding", "EmbeddingModel", "NullEmbedding"]
