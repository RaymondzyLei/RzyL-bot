"""混合检索：RRF 融合与两路的配合（故事 26-29）。

两个层次分别测：

- :func:`rrf_fuse` 是纯函数，直接断言名次算术与并列时的稳定顺序；
- :func:`hybrid_search` 用真库（临时 SQLite）+ 一个**可控的假向量**，验证「关键词找不到、
  语义找得到」这件事真的会发生，以及向量服务不可用时如实降级。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import pytest

from rzyl_core.db import Category, Repository
from rzyl_core.llm import NullEmbedding, embed_batch
from rzyl_core.memory.retrieval import (
    RRF_K,
    RetrievalOutcome,
    hybrid_search,
    rrf_fuse,
)


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncGenerator[Repository, None]:
    instance = await Repository.create(f"sqlite+aiosqlite:///{tmp_path / 'rzyl.db'}")
    yield instance
    await instance.close()


class TopicEmbedding:
    """可控的假向量：按出现的词给方向，让「语义」这一路的行为可预测。

    维度是 5；没命中任何词的文本得到零向量（余弦恒为 0），正好用来验证下限确实在起作用。
    """

    def __init__(self, topics: dict[str, list[float]] | None = None) -> None:
        self._topics = topics or {
            "论文": [1.0, 0.0, 0.0],
            "文献": [0.98, 0.02, 0.0],
            "网盘": [0.0, 1.0, 0.0],
        }

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        for keyword, vector in self._topics.items():
            if keyword in text:
                return list(vector)
        return [0.0, 0.0, 1.0]


# —— RRF 本身 ——


def test_rrf_scores_are_the_reciprocal_of_rank() -> None:
    fused = rrf_fuse([[7, 8], [8]])

    assert dict(fused)[7] == pytest.approx(1 / (RRF_K + 1))
    # 8 在第二路是第一、在第一路是第二，两路相加。
    assert dict(fused)[8] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert [memory_id for memory_id, _ in fused] == [8, 7]


def test_rrf_keeps_items_found_by_only_one_arm() -> None:
    """任一路没命中也能出结果——这正是选 RRF 而不是加权求和的原因之一。"""
    fused = rrf_fuse([[1], [2]])

    assert {memory_id for memory_id, _ in fused} == {1, 2}


def test_rrf_is_stable_on_ties() -> None:
    first = rrf_fuse([[5], [4]])
    second = rrf_fuse([[4], [5]])

    # 分数相同（都是一次第一名），顺序由编号定，不由字典顺序定。
    assert first == second
    assert [memory_id for memory_id, _ in first] == [4, 5]


def test_rrf_ignores_repeats_inside_one_ranking_and_empty_input() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    assert [memory_id for memory_id, _ in rrf_fuse([[3, 3, 3]])] == [3]


def test_rrf_rejects_a_negative_k() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([[1]], k=-1)


def test_rrf_k_shifts_the_weighting_but_not_the_order() -> None:
    assert [memory_id for memory_id, _ in rrf_fuse([[1, 2], [2]], k=0)] == [2, 1]


# —— 混合检索 ——


async def _add(
    repo: Repository,
    *,
    statement: str,
    group_id: int = 111,
    vector: list[float] | None = None,
    category: Category = Category.KNOWLEDGE,
    person_refs: Sequence[dict[str, object]] = (),
) -> int:
    memory = await repo.add_memory(
        group_id=group_id,
        category=category,
        statement=statement,
        confidence=0.9,
        prompt_version="v2",
        embedding=vector or TopicEmbedding()._vector(statement),
        person_refs=person_refs,  # pyright: ignore[reportArgumentType]
    )
    return int(memory.id)


async def test_keyword_arm_finds_the_literal_word(repo: Repository) -> None:
    await _add(repo, statement="这篇论文给了新的证明")
    await _add(repo, statement="网盘链接在这里")

    outcome = await hybrid_search(
        repository=repo,
        embedding=TopicEmbedding(),
        query="论文",
        min_similarity=0.5,
    )

    assert [hit.memory.statement for hit in outcome.hits] == ["这篇论文给了新的证明"]
    assert outcome.hits[0].keyword_rank is not None
    assert outcome.semantic_available


async def test_semantic_arm_finds_a_synonym_the_keyword_arm_misses(repo: Repository) -> None:
    """故事 27：搜「文献」要能找到记成「论文」的那条。"""
    await _add(repo, statement="这篇论文给了新的证明")

    keyword_only = await repo.search_memories("文献")
    outcome = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="文献", min_similarity=0.5
    )

    assert keyword_only == []  # 关键词这一路确实找不到
    assert len(outcome.hits) == 1
    assert outcome.hits[0].matched_by_vector_only
    assert outcome.vector_hits == 1


async def test_min_similarity_drops_unrelated_candidates(repo: Repository) -> None:
    """没有下限的话「搜什么都能搜到东西」，检索就不再能回答「我记过这事吗」。"""
    await _add(repo, statement="网盘链接在这里")

    strict = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="论文", min_similarity=0.5
    )
    loose = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="论文", min_similarity=0.0
    )

    assert strict.hits == ()
    assert len(loose.hits) == 1


async def test_vector_arm_can_be_switched_off_with_a_floor_above_one(repo: Repository) -> None:
    await _add(repo, statement="这篇论文给了新的证明")

    outcome = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="论文", min_similarity=1.1
    )

    assert outcome.vector_hits == 0
    assert outcome.keyword_hits == 1
    assert outcome.semantic_available  # 服务可用，只是没有候选达标——两件事要分清


async def test_hybrid_search_degrades_to_keywords_when_vectors_are_unavailable(
    repo: Repository,
) -> None:
    await _add(repo, statement="这篇论文给了新的证明")

    outcome = await hybrid_search(
        repository=repo, embedding=NullEmbedding(), query="论文", min_similarity=0.5
    )

    assert outcome.semantic_available is False
    assert outcome.semantic_error is not None
    assert [hit.memory.statement for hit in outcome.hits] == ["这篇论文给了新的证明"]


async def test_hybrid_search_applies_the_same_filters_to_both_arms(repo: Repository) -> None:
    await _add(repo, statement="这篇论文给了新的证明", group_id=111)
    await _add(repo, statement="另一个群的论文结论", group_id=222)

    outcome = await hybrid_search(
        repository=repo,
        embedding=TopicEmbedding(),
        query="论文",
        min_similarity=0.5,
        group_id=111,
    )

    assert [hit.memory.group_id for hit in outcome.hits] == [111]
    assert outcome.vector_hits == 1  # 向量那一路也被同一个条件约束过


async def test_hybrid_search_filters_by_person(repo: Repository) -> None:
    """故事 29：按相关人过滤。相关人存的是 JSON 快照，所以这也是在验 ``json_each`` 那条路。"""
    with_person = await _add(
        repo,
        statement="论文 A 在这",
        person_refs=[{"user_id": 10001, "nickname_snapshot": "小明"}],
    )
    await _add(repo, statement="论文 B 在那")

    outcome = await hybrid_search(
        repository=repo,
        embedding=TopicEmbedding(),
        query="论文",
        min_similarity=0.5,
        person_id=10001,
    )

    assert [hit.memory.id for hit in outcome.hits] == [with_person]
    assert outcome.vector_hits == 1  # 向量那一路也被同一个条件约束过


async def test_hybrid_search_respects_the_limit(repo: Repository) -> None:
    for index in range(5):
        await _add(repo, statement=f"论文里第 {index} 个结论")

    outcome = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="论文", min_similarity=0.5, limit=2
    )

    assert len(outcome.hits) == 2
    assert outcome.keyword_hits == 5  # 候选数照实报，只是最后只给出前 limit 条


async def test_empty_query_returns_nothing_without_calling_anything(repo: Repository) -> None:
    await _add(repo, statement="这篇论文给了新的证明")

    outcome = await hybrid_search(
        repository=repo, embedding=TopicEmbedding(), query="   ", min_similarity=0.5
    )

    assert outcome == RetrievalOutcome(
        hits=(), keyword_hits=0, vector_hits=0, semantic_available=False
    )


async def test_embed_batch_still_reports_ok_for_a_working_fake() -> None:
    batch = await embed_batch(TopicEmbedding(), ["论文"])
    assert batch.ok and batch.vectors is not None and len(batch.vectors) == 1
