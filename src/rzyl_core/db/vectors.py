"""向量的 BLOB 编解码。

向量以 float32 小端序存进 ``memory.embedding``。查询侧读进内存做暴力余弦——这个
量级不需要向量索引，日后换 sqlite-vec / FAISS 也不用改数据模型。
"""

from __future__ import annotations

import struct


def encode_vector(vector: list[float]) -> bytes:
    """把浮点列表编成 float32 小端序字节串；空向量编成空字节串。"""
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes | None) -> list[float]:
    """把字节串解回浮点列表；``None`` 或空字节串解成空列表。"""
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))
