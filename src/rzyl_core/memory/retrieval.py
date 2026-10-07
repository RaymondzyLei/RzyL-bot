"""混合检索：关键词一路 + 向量一路，用 RRF（倒数排名融合）合成一个名次。

为什么用 RRF 而不是给两路加权求和：两路的分数**量纲完全不同**（FTS5 是 bm25 的负值，
余弦是 -1..1），凑权重等于引入一组只在部分样本上才好的参数，还要人维护；RRF 只用名次，
不需要调参，任一路没命中也能出结果（见 issue #1「检索」一节）。

这一层是**注入依赖的函数**，不持有状态：仓储、向量实现都由调用方给。Runtime 只是把它
接上去（``Runtime.search_memories``），命令与测试可以直接调用。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from rzyl_core.db import Category, Memory, Repository
from rzyl_core.llm import EmbeddingModel, embed_batch
from rzyl_core.pipeline.extract import cosine_similarity

logger = logging.getLogger(__name__)

#: RRF 的平滑常数。60 是原论文（Cormack 等，2009）用的值，也让名次差异不会被放大。
RRF_K = 60

#: 每一路各取多少名候选进融合。融合前多取一些，融合后再砍到用户要的条数——
#: 两路各自的第一名未必是融合后的第一名，候选取太少会把「另一路很靠前」的条目漏掉。
KEYWORD_CANDIDATES = 50
VECTOR_CANDIDATES = 50

#: 用户要检索结果时的默认条数。
DEFAULT_SEARCH_LIMIT = 10


def rrf_fuse(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K
) -> list[tuple[int, float]]:
    """把若干路名次融合成 ``[(编号, 分数), ...]``，分数降序、并列按编号升序。

    ``1 起``的名次，每条得分是 ``Σ 1/(k + 名次)``。同一路里重复出现的编号只算第一次
    （名次表本来就不该有重复，这只是防御）。并列时按编号升序是为了**可复现**：
    分数相同的两条谁在前不该由字典顺序决定。
    """
    if k < 0:
        raise ValueError(f"RRF 的 k 不能为负，收到 {k}")
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, memory_id in enumerate(ranking, start=1):
            if memory_id in seen:
                continue
            seen.add(memory_id)
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """一条命中：条目本身、融合分、以及它在两路里的名次（用来解释「为什么是它」）。"""

    memory: Memory
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None

    @property
    def matched_by_vector_only(self) -> bool:
        """只有语义这一路命中——纯关键词检索找不到它，是「语义检索真的在工作」的证据。"""
        return self.vector_rank is not None and self.keyword_rank is None


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """一次混合检索的结果，含两路各自的命中数目与语义是否可用。"""

    hits: tuple[RetrievalHit, ...]
    keyword_hits: int
    vector_hits: int
    semantic_available: bool
    semantic_error: str | None = None
    """语义那一路不可用的原因（向量服务未配置或调用失败）；可用时为 ``None``。"""

    @property
    def total(self) -> int:
        return len(self.hits)


async def hybrid_search(
    *,
    repository: Repository,
    embedding: EmbeddingModel,
    query: str,
    min_similarity: float,
    category: Category | None = None,
    group_id: int | None = None,
    person_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> RetrievalOutcome:
    """关键词 + 语义两路检索，RRF 融合后取前 ``limit`` 条。

    两路**用同一组前置过滤**（类别 / 群 / 相关人 / 时间），否则会从「另一路」里捞回
    本该被过滤掉的条目。

    ``min_similarity`` 是语义那一路的**相似度下限**（必填，取自
    ``settings.retrieval_min_similarity``，只有一个默认值来源）：余弦低于它的候选不参与
    融合。没有这条下限，向量那一路会永远排出一串候选，于是「搜什么都能搜到东西」，
    用户无法用搜索结果判断「我到底有没有记过这件事」——那检索就白做了。

    向量服务不可用时**不报错、也不假装**：退化成纯关键词检索，并在结果里留下原因
    （``semantic_available=False`` / ``semantic_error``），让上层能如实告诉用户
    「这次只用了关键词」。
    """
    if not query.strip():
        return RetrievalOutcome(
            hits=(), keyword_hits=0, vector_hits=0, semantic_available=False
        )

    keyword = await repository.search_memories(
        query,
        category=category,
        group_id=group_id,
        person_id=person_id,
        since=since,
        until=until,
        limit=KEYWORD_CANDIDATES,
    )
    keyword_ids = [int(memory.id) for memory in keyword]

    vector_ids, semantic_error = await _vector_ranking(
        repository=repository,
        embedding=embedding,
        query=query,
        min_similarity=min_similarity,
        category=category,
        group_id=group_id,
        person_id=person_id,
        since=since,
        until=until,
    )

    fused = rrf_fuse([keyword_ids, vector_ids])[: max(0, limit)]
    keyword_ranks = {memory_id: rank for rank, memory_id in enumerate(keyword_ids, start=1)}
    vector_ranks = {memory_id: rank for rank, memory_id in enumerate(vector_ids, start=1)}
    memories = await repository.list_memories_by_id([memory_id for memory_id, _ in fused])
    score_by_id = dict(fused)
    hits = tuple(
        RetrievalHit(
            memory=memory,
            score=score_by_id[int(memory.id)],
            keyword_rank=keyword_ranks.get(int(memory.id)),
            vector_rank=vector_ranks.get(int(memory.id)),
        )
        for memory in memories
    )
    return RetrievalOutcome(
        hits=hits,
        keyword_hits=len(keyword_ids),
        vector_hits=len(vector_ids),
        semantic_available=semantic_error is None,
        semantic_error=semantic_error,
    )


async def _vector_ranking(
    *,
    repository: Repository,
    embedding: EmbeddingModel,
    query: str,
    min_similarity: float,
    category: Category | None,
    group_id: int | None,
    person_id: int | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[int], str | None]:
    """语义那一路的名次：``(编号降序, 不可用原因)``。

    向量全部读进内存做暴力余弦——这个量级（连一万条都很难到）不需要向量索引，日后换
    sqlite-vec 也不用改这一层的调用约定（见 issue #1）。低于 ``min_similarity`` 的候选
    直接丢掉：留下它们只会让「搜什么都有结果」，把检索变成噪声。

    库里的向量长度与本条查询向量长度不符时**跳过该条并记一条 WARNING**：那是「换过向量
    模型但没重算」的信号，静默当成不相似正是本项目最不能接受的失败方式。
    """
    batch = await embed_batch(embedding, [query])
    if batch.vectors is None:
        return [], batch.error or "向量服务不可用"
    query_vector = batch.vectors[0] if batch.vectors else None
    if not query_vector:
        return [], "向量服务未配置或未返回查询向量"

    pairs = await repository.list_embeddings(
        category=category, group_id=group_id, person_id=person_id, since=since, until=until
    )
    scored: list[tuple[int, float]] = []
    mismatched = 0
    for memory_id, vector in pairs:
        if len(vector) != len(query_vector):
            mismatched += 1
            continue
        similarity = cosine_similarity(query_vector, vector)
        if similarity >= min_similarity:
            scored.append((memory_id, similarity))
    if mismatched:
        logger.warning(
            "语义检索跳过 %d 条向量维度不一致的条目（查询向量 %d 维）："
            "疑似换过向量模型但未重算向量",
            mismatched,
            len(query_vector),
        )
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [memory_id for memory_id, _ in scored[:VECTOR_CANDIDATES]], None


__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "KEYWORD_CANDIDATES",
    "RRF_K",
    "VECTOR_CANDIDATES",
    "RetrievalHit",
    "RetrievalOutcome",
    "hybrid_search",
    "rrf_fuse",
]
