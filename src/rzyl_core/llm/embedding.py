"""向量嵌入接口与两个不需要网络/密钥的实现。

``embed()`` 返回与输入等长的向量列表；返回 ``None`` 表示"当前不可用"——调用方
应把向量留空、由后台任务补算（见 issue #1 的 embedding 后台补算决策）。第一阶段
没有 embedding key 也能把整条管道跑通，靠的就是这个约定。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

#: 一条向量就是一组浮点数。维度由具体实现决定（假实现默认 1024）。
Embedding: TypeAlias = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    """可替换的向量嵌入接口。"""

    async def embed(self, texts: Sequence[str]) -> list[Embedding] | None:
        """为一批文本算向量；返回 ``None`` 表示当前不可用。"""
        ...


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """一次批量向量调用归一化后的结果。

    - ``vectors`` 为 ``None`` 表示整批不可用（服务返回 ``None``、数量与输入不符或抛异常），
      调用方应把向量留空；
    - 否则它与输入等长，某一条算不出来时对应位置为 ``None``；
    - ``error`` 在整批不可用时说明原因，供记账与排障。
    """

    vectors: list[Embedding | None] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.vectors is not None


async def embed_batch(model: EmbeddingModel, texts: Sequence[str]) -> EmbeddingBatch:
    """批量取向量，把「服务不可用」统一降级为 ``vectors=None``，绝不抛出。

    提取管道与 Runtime 的向量补算共用这段容错：空输入视为成功且返回空列表；服务返回
    ``None``、数量与输入不符或调用抛异常都归一为整批不可用。这样「向量是尽力而为」
    这条约定只实现一次。
    """
    if not texts:
        return EmbeddingBatch(vectors=[])
    try:
        vectors = await model.embed(texts)
    except Exception as exc:  # 向量服务抖动不应阻塞调用方
        return EmbeddingBatch(vectors=None, error=f"向量服务调用失败：{exc}")
    if vectors is None:
        return EmbeddingBatch(vectors=None, error="向量服务不可用（返回 None）")
    if len(vectors) != len(texts):
        return EmbeddingBatch(
            vectors=None,
            error=f"向量数量与输入不符：返回 {len(vectors)} 条，输入 {len(texts)} 条",
        )
    return EmbeddingBatch(vectors=[list(vector) if vector else None for vector in vectors])


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


__all__ = [
    "DeterministicEmbedding",
    "Embedding",
    "EmbeddingBatch",
    "EmbeddingModel",
    "NullEmbedding",
    "embed_batch",
]
