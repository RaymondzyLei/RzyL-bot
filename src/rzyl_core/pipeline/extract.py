"""提取：把一个组装好的窗口喂进模型，校验、去重、入库。

这是工单 #7 的全部职责，也是这一环对外的契约所在：

    pipeline = ExtractionPipeline(
        repository=repo, chat_model=chat, embedding_model=embedding,
        settings=settings, clock=clock, provider="deepseek",
    )
    outcome = await pipeline.process_window(window)

``window`` 是 :class:`~rzyl_core.pipeline.window.AssembledWindow`（由 #4 组装、#5 装配时
喂进来）。本模块负责：为本窗口建 ``window`` 行、渲染提示词、调模型、严格校验、两层去重、
写 ``memory`` 行、写 ``llm_call`` 记账，并把窗口状态推进到 ``done`` 或 ``dead``。

几个刻意的选择：

- **校验即失败**：JSON 坏了、字段缺失、类别不在四类、``evidence`` 越界，一律算解析失败，
  按窗口重试；重试用尽整窗进死信，绝不把半成品静默写库。
- **空数组是正常结果**：窗口结束为 ``done``，不产生条目，也不算失败。
- **丢了什么要看得见**：解析失败时把模型的原始输出截断后记进 ``WARNING``；若早先某次
  尝试产出过非空内容、最终却以空数组结束，则额外告警并在窗口行 ``error`` 留一句说明
  （状态仍是 ``done``）——只加可观测性，不改判定语义。
- **入库不设置信度门槛**：只要归入四类之一就写库（门槛只在推送时用）。
- **supersede 只标记不删除**：旧条目置为 ``expired`` 并让它的 ``superseded_by`` 指向新条目，
  新条目不做任何反向修改。
- **向量是尽力而为**：算不出来（空实现或服务异常）条目照常入库、向量列留空。
- **写库是一个事务**：一个窗口的条目、supersede 标记与窗口状态一起提交，任一步失败整体
  回滚，窗口保持可重试。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from sqlalchemy.exc import SQLAlchemyError

from rzyl_core.db import (
    Category,
    Memory,
    MemoryStatus,
    PersonRef,
    Repository,
    WindowStatus,
    decode_vector,
    encode_vector,
)
from rzyl_core.llm import (
    ChatModel,
    ChatUsage,
    EmbeddingModel,
    LLMError,
    embed_batch,
)
from rzyl_core.llm.prompts import RenderedPrompt
from rzyl_core.pipeline.window import AssembledWindow, UnknownSequenceError
from rzyl_core.settings import Settings
from rzyl_core.timeutil import resolve_timezone

#: 聊天调用写进 ``llm_call.purpose`` 的用途标签。
LLM_PURPOSE = "extract"

#: 校验失败时，模型的原始输出写进 WARNING 日志前截断到这个长度（字符数）。
#: 原始输出可能上千字，截断是为了「丢了什么看得见」而不是「把日志刷爆」。
RAW_OUTPUT_LOG_LIMIT = 500

logger = logging.getLogger(__name__)


class ExtractionParseError(ValueError):
    """模型输出无法变成合法条目：JSON 坏了、字段缺失、evidence 越界等。"""


def _coerce_supersede(candidate: object) -> int | None:
    """把单个候选值变成编号；``None`` / 空串到 ``None``，其余不合法就抛 ``ValueError``。"""
    if candidate is None:
        return None
    if isinstance(candidate, bool):
        # bool 是 int 的子类，但把它当编号是错的（True 会悄悄变成 1）。
        raise ValueError(f"supersedes 不接受布尔值：{candidate!r}")
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, float):
        # JSON 里的整数有时会写成 2.0；只接受整值浮点，小数一律算格式错误。
        if candidate.is_integer():
            return int(candidate)
        raise ValueError(f"supersedes 不是整数编号：{candidate!r}")
    if isinstance(candidate, str):
        text = candidate.strip().lstrip("#").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"supersedes 不是可解析的编号：{candidate!r}") from exc
    raise ValueError(f"supersedes 不是编号：{candidate!r}")


def normalize_supersedes(value: object) -> int | None:
    """把一个 ``supersedes`` 值归一化成 ``int | None``。

    **只有这一个字段做归一化**，理由：提示词把「已记条目摘要」的锚点渲染成 ``[#编号]``
    （见 :func:`~rzyl_core.pipeline.window.summarize_memories`），模型照着写 ``"#2"``、
    或按 JSON 习惯包成 ``["#2"]`` 都是合理行为。若判成格式错误，轻则白白多花一次调用，
    重则重试拿到空数组、整窗内容静默丢失——为一种写法差异丢一窗内容是得不偿失的。

    接受：``2``、``"2"``、``"#2"``、``[2]``、``["#2"]``，以及表示「没有」的 ``null`` /
    ``[]`` / 空串，统一成 ``int | None``。多元素数组不在提示词契约里（v2 明确要求单个
    标量编号），取第一个可用元素：宁可少挂一条 supersede 链接，也不把整条记忆丢掉。
    真的解析不出编号（如 ``"#abc"``）仍抛 ``ValueError``，交给上层按解析失败处理。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _coerce_supersede(value[0])
    return _coerce_supersede(value)


class ExtractedPersonRef(BaseModel):
    """模型输出里的一个相关人；形状与提示词声明的一致。"""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    nickname_snapshot: str


class ExtractedMemory(BaseModel):
    """模型输出的单条记忆，字段与 ``llm/prompts/v2.*.md`` 的约定逐一对齐。

    ``extra="forbid"``：多出来的字段视为格式错误。``detail`` / ``occurred_at`` 允许为
    ``null`` 但**必须出现**——提示词要求每条都带这两个键，缺键就是没按格式来。

    ``supersedes`` 是**唯一**宽容的字段：提示词把「已记条目摘要」的锚点渲染成
    ``[#编号]``，模型照抄 ``"#2"``、或按 JSON 习惯包成 ``["#2"]`` 是合理行为；
    为此把整窗判成解析失败、重试又拿到空数组而静默丢光内容，得不偿失。所以这里先经
    :func:`normalize_supersedes` 归一化成 ``int | None``，其余一律保持严格。
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    statement: str = Field(min_length=1)
    detail: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[int] = Field(min_length=1)
    person_refs: list[ExtractedPersonRef]
    occurred_at: str | None
    supersedes: int | None

    @field_validator("supersedes", mode="before")
    @classmethod
    def _normalize_supersedes(cls, value: object) -> int | None:
        """把模型给的锚点写法归一化成 ``int | None``，见 :func:`normalize_supersedes`。"""
        return normalize_supersedes(value)

    @field_validator("occurred_at")
    @classmethod
    def _occurred_at_must_be_iso(cls, value: str | None) -> str | None:
        """只校验可解析性；锚定时区是入库时才做的（时区来自设置）。"""
        if value is None:
            return None
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"occurred_at 不是合法 ISO 8601 字符串：{value!r}") from exc
        return value


_EXTRACTED_LIST = TypeAdapter(list[ExtractedMemory])


def parse_extraction(text: str) -> list[ExtractedMemory]:
    """把模型返回的文本严格解析成条目列表；任何不合格式都抛 :class:`ExtractionParseError`。

    只 strip 首尾空白后当 JSON 数组解析，不做代码围栏之类的宽容处理——提示词明确要求
    只输出裸 JSON 数组，这里放过围栏只会掩盖提示词退化。唯一的例外是 ``supersedes``
    字段：它经 :func:`normalize_supersedes` 归一出 ``int | None``（见那里的理由），
    其余字段一律严格。
    """
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionParseError(f"模型输出不是合法 JSON：{exc}") from exc
    if not isinstance(payload, list):
        raise ExtractionParseError(f"模型输出不是 JSON 数组，而是 {type(payload).__name__}")
    try:
        return _EXTRACTED_LIST.validate_python(payload)
    except ValidationError as exc:
        raise ExtractionParseError(f"模型输出不符合条目格式：{exc}") from exc


def _has_nonempty_candidates(text: str) -> bool:
    """尽力判断一次**未通过校验**的原始输出里是否真有候选条目（非空 JSON 数组）。

    只服务于可观测性：用来识别「这次尝试其实产出了内容、只是格式不对」。判不出来
    （不是 JSON、不是数组、或本来就是空数组）就当没有，免得把纯噪声也算成「丢了内容」
    而虚报。它**不参与**任何判定——空数组依然是正常结果。
    """
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(payload, list) and len(payload) > 0


def normalize_statement(statement: str) -> str:
    """归一化一句话陈述，作为文本去重的比较基准。

    规则（这是以后调参的依据，改动等于改去重行为）：NFKC 统一宽度 → casefold 折叠大小写
    → 去掉全部空白字符。标点保留：合并「同一件事」靠语义相似度那一层，而不是把标点也抹掉、
    冒着脸把两件不同的事合成一件的风险。
    """
    folded = unicodedata.normalize("NFKC", statement).casefold()
    return "".join(char for char in folded if not char.isspace())


def dedupe_hash(statement: str) -> str:
    """一句话陈述的归一化 hash，直接落 ``memory.dedupe_hash``，供同群文本去重比对。"""
    return hashlib.sha256(normalize_statement(statement).encode("utf-8")).hexdigest()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """两个等长向量的余弦相似度；长度不符或任一为零向量时返回 0。

    这是**纯函数**：长度不符时只返回 0，不抛异常、不记日志。发现并上报这种不一致
    （例如库里留着旧维度的历史行）是**调用方**的责任——见
    :meth:`ExtractionPipeline._is_near_duplicate`，它会在比较前显式检查长度并记
    ``WARNING``，免得维度不一致被误当成「不相似」而静默流失召回。
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class _ResolvedItem:
    """一条已通过校验、evidence 已换回真实编号、时间已锚定时区的条目。"""

    category: Category
    statement: str
    detail: str | None
    confidence: float
    evidence: list[int]
    person_refs: list[PersonRef]
    occurred_at: datetime | None
    supersedes: list[int]
    dedupe_hash: str


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """一次 ``process_window`` 的结果，也是给 #5（Runtime / 回放）的返回值。

    - ``status`` 为 ``done``：正常结束（含模型返回空数组）；
    - ``status`` 为 ``dead``：模型连续失败，整窗进了死信，``error`` 是最后一次错误；
    - ``status`` 为 ``pending``：写库失败，事务已回滚，窗口保持可重试；
    - ``memory_ids`` 是本窗口**新增**条目的编号（顺序与模型输出一致），空数组时为空元组；
    - ``merged_count`` 是归一化文本 hash 撞上已有条目、被合并掉（未新增）的条数；
    - ``suspect_count`` 是向量判为疑似重复的条数；
    - ``error`` 是失败原因；``done`` 时也可能带一句说明（例如「本窗口可能有内容被
      丢弃」），此时它只作可观测性提示，不代表窗口失败。
    """

    window_id: int
    status: WindowStatus
    memory_ids: tuple[int, ...] = ()
    attempts: int = 1
    merged_count: int = 0
    suspect_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PreviewOutcome:
    """一次 :meth:`ExtractionPipeline.preview_window` 的结果：``--dry-run`` 的产物。

    - ``rendered`` 是**完整提示词**（版本号 + 系统提示词 + 用户内容），原样可打印；
    - ``raw_output`` 是模型的**原始输出文本**（未做任何清洗）；调用失败时为 ``None``；
    - ``items`` 是校验通过的条目（与正常模式同一套 pydantic 校验与依据换号）；
    - ``error`` 是调用失败或格式错误的原因，正常时为 ``None``。

    这个结果**不代表任何库写入**——preview 全程不落库，所以库不会出现窗口行或条目。
    """

    rendered: RenderedPrompt
    raw_output: str | None
    items: tuple[ExtractedMemory, ...]
    error: str | None
    usage: ChatUsage | None = None
    model: str | None = None


class ExtractionPipeline:
    """提取入库管道：组装好的窗口进来，记忆条目、窗口状态与记账出去。

    注入四项外部依赖——仓储、聊天模型、向量模型、设置，外加时钟与（可选）服务商名。
    时间一律从 ``clock`` 取；聊天调用失败 / 输出不合格式都按 ``settings.extract_max_attempts``
    重试，用尽则整窗进死信。
    """

    def __init__(
        self,
        *,
        repository: Repository,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        settings: Settings,
        clock: Callable[[], datetime],
        provider: str | None = None,
    ) -> None:
        self._repository = repository
        self._chat = chat_model
        self._embedding = embedding_model
        self._settings = settings
        self._clock = clock
        self._provider = provider
        self._timezone = resolve_timezone(settings.timezone)

    async def process_window(
        self, window: AssembledWindow, *, window_id: int | None = None
    ) -> ExtractionOutcome:
        """处理一个组装好的窗口，返回本次结果。

        ``window_id`` 缺省时为本窗口新建 ``window`` 行；传已存在的编号则复用它（上游若已
        建好窗口就不再重复建）。
        """
        target_window_id = window_id if window_id is not None else await self._ensure_window(window)
        attempts = 0
        last_error: str | None = None
        #: 早先某次尝试产出过非空候选内容的次数（1 起）；只用于「内容可能被丢弃」的可观测性。
        content_attempt: int | None = None

        while attempts < max(1, self._settings.extract_max_attempts):
            attempts += 1
            rendered = window.render()
            started = self._clock()
            try:
                result = await self._chat.complete(rendered.system, rendered.user)
            except LLMError as exc:
                last_error = f"模型调用失败：{exc}"
                await self._record_call(usage=None, model=None, latency_ms=None, success=False, error=last_error)
                await self._mark_retryable(target_window_id, attempts, last_error)
                continue

            latency_ms = max(0, int((self._clock() - started).total_seconds() * 1000))
            try:
                items = self._resolve(window, result.text)
            except ExtractionParseError as exc:
                last_error = str(exc)
                # 校验失败时把原始输出截断后记进 WARNING：丢了什么看得见，而不是只剩一句
                # 「格式不对」。调用成功、token 已被计费，只是输出不可用——照记 token。
                logger.warning(
                    "窗口 %s 第 %d 次提取输出不合格式，原始输出（至多 %d 字）：%s",
                    target_window_id,
                    attempts,
                    RAW_OUTPUT_LOG_LIMIT,
                    result.text[:RAW_OUTPUT_LOG_LIMIT],
                )
                if content_attempt is None and _has_nonempty_candidates(result.text):
                    content_attempt = attempts
                await self._record_call(
                    usage=result.usage,
                    model=result.model,
                    latency_ms=latency_ms,
                    success=False,
                    error=last_error,
                )
                await self._mark_retryable(target_window_id, attempts, last_error)
                continue

            await self._record_call(
                usage=result.usage,
                model=result.model,
                latency_ms=latency_ms,
                success=True,
                error=None,
            )
            # 空数组本身是正常结果（判定语义不动）；但若早先某次尝试产出过非空内容，
            # 说明这一窗可能有东西被丢掉了——只加可观测性，不改判定。
            drop_note: str | None = None
            if not items and content_attempt is not None:
                drop_note = (
                    f"第 {content_attempt} 次尝试产出过非空内容，最终却以空数组结束："
                    "本窗口可能有内容被丢弃，建议人工复核原文"
                )
                logger.warning(
                    "窗口 %s 可能有内容被丢弃：第 %d 次尝试产出过非空候选，"
                    "最终以空数组结束（本窗口 %d 条消息）",
                    target_window_id,
                    content_attempt,
                    window.message_count,
                )
            try:
                return await self._persist(
                    window=window,
                    window_id=target_window_id,
                    items=items,
                    model=result.model,
                    usage=result.usage,
                    attempts=attempts,
                    error=drop_note,
                )
            except SQLAlchemyError as exc:
                # 事务已整体回滚：#5 可以把窗口重排一次，库不会留下半个窗口的条目。
                storage_error = f"写库失败：{exc}"
                await self._mark_retryable(target_window_id, attempts, storage_error)
                return ExtractionOutcome(
                    window_id=target_window_id,
                    status=WindowStatus.PENDING,
                    attempts=attempts,
                    error=storage_error,
                )

        dead_error = f"模型连续 {attempts} 次未能产出可用结果：{last_error}"
        await self._repository.update_window_status(
            target_window_id, WindowStatus.DEAD, retry_count=attempts, error=dead_error
        )
        return ExtractionOutcome(
            window_id=target_window_id,
            status=WindowStatus.DEAD,
            attempts=attempts,
            error=dead_error,
        )

    async def preview_window(self, window: AssembledWindow) -> PreviewOutcome:
        """渲染提示词、调一次模型、校验输出，但**全程不碰任何库**（``--dry-run``）。

        与 :meth:`process_window` 共用 ``window.render()`` 与 :meth:`_resolve` 这套渲染 /
        校验代码，差别只有两点：只调模型一次、不重试；以及不建窗口行、不写条目、不记账。
        因此它既不会把半个窗口的条目写进库，也不会注册任何 llm_call。

        模型调用失败时 ``raw_output`` 为 ``None``；输出不合格式时 ``raw_output`` 是原始
        文本、``error`` 是原因——两种情况都原样返回，不抛异常，方便回放脚本打印。
        """
        rendered = window.render()
        try:
            result = await self._chat.complete(rendered.system, rendered.user)
        except LLMError as exc:
            return PreviewOutcome(
                rendered=rendered,
                raw_output=None,
                items=(),
                error=f"模型调用失败：{exc}",
            )

        try:
            # 先按正常模式的校验走一遍（含 evidence 越界检查），失败就只报告原因。
            self._resolve(window, result.text)
        except ExtractionParseError as exc:
            return PreviewOutcome(
                rendered=rendered,
                raw_output=result.text,
                items=(),
                error=str(exc),
                usage=result.usage,
                model=result.model,
            )

        return PreviewOutcome(
            rendered=rendered,
            raw_output=result.text,
            items=tuple(parse_extraction(result.text)),
            error=None,
            usage=result.usage,
            model=result.model,
        )

    async def _ensure_window(self, window: AssembledWindow) -> int:
        """为窗口建一行 ``pending`` 记录，返回它的编号。"""
        created = await self._repository.add_window(
            group_id=window.group_id,
            started_at=window.started_at,
            ended_at=window.ended_at,
            message_count=window.message_count,
            status=WindowStatus.PENDING,
            prompt_version=window.prompt_version,
        )
        return int(created.id)

    async def _mark_retryable(self, window_id: int, attempts: int, error: str) -> None:
        """把窗口保持为 ``pending``（可重试），并记下出错次数与错误信息。"""
        await self._repository.update_window_status(
            window_id, WindowStatus.PENDING, retry_count=attempts, error=error
        )

    async def _record_call(
        self,
        *,
        usage: ChatUsage | None,
        model: str | None,
        latency_ms: int | None,
        success: bool,
        error: str | None,
    ) -> None:
        tokens_in = usage.input_tokens if usage is not None else 0
        tokens_out = usage.output_tokens if usage is not None else 0
        await self._repository.add_llm_call(
            purpose=LLM_PURPOSE,
            provider=self._provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=self._settings.estimate_chat_cost(tokens_in, tokens_out),
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

    def _resolve(self, window: AssembledWindow, text: str) -> list[_ResolvedItem]:
        """校验模型输出、把 evidence 序号换回真实编号、锚定 occurred_at 时区。"""
        raw_items = parse_extraction(text)
        resolved: list[_ResolvedItem] = []
        for raw in raw_items:
            try:
                evidence = window.resolve_evidence(raw.evidence)
            except UnknownSequenceError as exc:
                raise ExtractionParseError(f"evidence 越界：{exc}") from exc
            resolved.append(
                _ResolvedItem(
                    category=raw.category,
                    statement=raw.statement,
                    detail=raw.detail,
                    confidence=raw.confidence,
                    evidence=evidence,
                    person_refs=[
                        PersonRef(user_id=person.user_id, nickname_snapshot=person.nickname_snapshot)
                        for person in raw.person_refs
                    ],
                    occurred_at=self._anchor_occurred_at(raw.occurred_at),
                    # 校验层已把锚点归一化成单个编号；下游仍按列表处理（兼容多引用）。
                    supersedes=[raw.supersedes] if raw.supersedes is not None else [],
                    dedupe_hash=dedupe_hash(raw.statement),
                )
            )
        return resolved

    def _anchor_occurred_at(self, value: str | None) -> datetime | None:
        """把模型给的 ISO 时间锚到设置时区；已经带时区的原样保留。"""
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ExtractionParseError(f"occurred_at 不是合法 ISO 8601 字符串：{value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self._timezone)
        return parsed

    async def _persist(
        self,
        *,
        window: AssembledWindow,
        window_id: int,
        items: list[_ResolvedItem],
        model: str,
        usage: ChatUsage,
        attempts: int,
        error: str | None = None,
    ) -> ExtractionOutcome:
        """两层去重、算向量、组装 ``Memory``，最后一个事务写库。

        ``error`` 只在「窗口正常结束、但有一段说明要留在窗口行上」时使用（例如早先尝试
        有内容、最终却是空数组，可能丢了东西），**不影响** ``status``——空数组仍是 ``done``。
        """
        candidates = await self._repository.list_group_memories(
            group_id=window.group_id, include_inactive=True
        )
        existing_by_id = {int(candidate.id): candidate for candidate in candidates}
        blocked_hashes = {
            candidate.dedupe_hash
            for candidate in candidates
            if candidate.status is not MemoryStatus.EXPIRED and candidate.dedupe_hash
        }

        survivors: list[_ResolvedItem] = []
        merged_count = 0
        for item in items:
            if item.dedupe_hash in blocked_hashes:
                # 归一化文本与已有条目（或本批已保留的条目）完全相同：合并，不新增。
                merged_count += 1
                continue
            blocked_hashes.add(item.dedupe_hash)
            survivors.append(item)

        vectors = await self._embed([item.statement for item in survivors])
        threshold = self._settings.dedupe_similarity_threshold
        batch_vectors: list[tuple[Category, list[float]]] = []
        memories: list[Memory] = []
        supersedings: list[tuple[int, Memory]] = []
        suspect_count = 0
        for item, vector in zip(survivors, vectors):
            status = MemoryStatus.ACTIVE
            if vector is not None and self._is_near_duplicate(
                item, vector, candidates, batch_vectors, threshold
            ):
                status = MemoryStatus.SUSPECT_DUPLICATE
                suspect_count += 1
            if vector is not None:
                batch_vectors.append((item.category, vector))
            memory = self._build_memory(
                window=window,
                window_id=window_id,
                item=item,
                status=status,
                vector=vector,
                model=model,
                usage=usage,
            )
            memories.append(memory)
            for old_id in item.supersedes:
                if old_id in existing_by_id and old_id != 0:
                    supersedings.append((old_id, memory))

        created = await self._repository.add_window_result(
            window_id=window_id,
            memories=memories,
            supersedings=supersedings,
            status=WindowStatus.DONE,
            retry_count=max(0, attempts - 1),
            error=error,
        )
        return ExtractionOutcome(
            window_id=window_id,
            status=WindowStatus.DONE,
            memory_ids=tuple(int(memory.id) for memory in created),
            attempts=attempts,
            merged_count=merged_count,
            suspect_count=suspect_count,
            error=error,
        )

    def _is_near_duplicate(
        self,
        item: _ResolvedItem,
        vector: list[float],
        candidates: Sequence[Memory],
        batch_vectors: Sequence[tuple[Category, list[float]]],
        threshold: float,
    ) -> bool:
        """同群、同类别、且与已有向量或本批已保留向量余弦不低于阈值。

        比较已有条目的向量前先看长度：库里可能留着换模型前、另一种维度的历史向量，
        长度不符时 :func:`cosine_similarity` 只会返回 0。这里显式检查并记一条
        ``WARNING``（带上条目编号与两个长度），以便发现「换模型后没重算向量」——
        否则近重复认不出来却毫无提示，正是本项目最不能接受的静默召回流失。
        """
        for candidate in candidates:
            if candidate.status is MemoryStatus.EXPIRED:
                continue
            if candidate.category is not item.category:
                continue
            other = decode_vector(candidate.embedding)
            if not other:
                continue
            if len(other) != len(vector):
                logger.warning(
                    "向量维度不一致：条目 #%s 是 %d 维，本批新向量是 %d 维——"
                    "疑似换过向量模型但未重算向量，本次跳过该条目的近重复比较",
                    candidate.id,
                    len(other),
                    len(vector),
                )
                continue
            if cosine_similarity(vector, other) >= threshold:
                return True
        for category, other in batch_vectors:
            if category is item.category and cosine_similarity(vector, other) >= threshold:
                return True
        return False

    def _build_memory(
        self,
        *,
        window: AssembledWindow,
        window_id: int,
        item: _ResolvedItem,
        status: MemoryStatus,
        vector: list[float] | None,
        model: str,
        usage: ChatUsage,
    ) -> Memory:
        """组装一条尚未入库的 ``Memory``；事务写入由仓储的 ``add_window_result`` 负责。"""
        return Memory(
            window_id=window_id,
            group_id=window.group_id,
            category=item.category,
            statement=item.statement,
            detail=item.detail,
            confidence=item.confidence,
            evidence=item.evidence,
            person_refs=list(item.person_refs),
            occurred_at=item.occurred_at,
            prompt_version=window.prompt_version,
            dedupe_hash=item.dedupe_hash,
            embedding=encode_vector(vector) if vector else None,
            status=status,
            model=model,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost=self._settings.estimate_chat_cost(usage.input_tokens, usage.output_tokens),
            created_at=self._clock(),
        )

    async def _embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        """批量算向量；拿不到（空实现、服务异常、数量不符）就整批留空，绝不因此失败。

        容错逻辑与 Runtime 的向量补算共用 :func:`rzyl_core.llm.embed_batch`。
        """
        if not texts:
            return []
        batch = await embed_batch(self._embedding, texts)
        if batch.vectors is None:
            return [None] * len(texts)
        return batch.vectors


__all__ = [
    "LLM_PURPOSE",
    "RAW_OUTPUT_LOG_LIMIT",
    "ExtractedMemory",
    "ExtractedPersonRef",
    "ExtractionOutcome",
    "ExtractionParseError",
    "ExtractionPipeline",
    "PreviewOutcome",
    "cosine_similarity",
    "dedupe_hash",
    "normalize_statement",
    "normalize_supersedes",
    "parse_extraction",
]
