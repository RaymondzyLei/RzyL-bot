#!/usr/bin/env python
"""离线回放：把某个群的一段历史消息喂进整条管道（工单 #5）。

这是里程碑 1 的日常入口——调提示词、验管道、看落库，全程**不需要 API key**。脚本
不依赖 NoneBot，也不构造任何插件：它只构造 ``Runtime``，把历史来源接上去然后跑。

    # 对固定样本文件回放（不需要 QQ、不需要网络、不需要任何 API key）
    uv run python scripts/replay.py --sample tests/fixtures/sample_history.json --group 100200300

    # 只看提示词与模型原始输出，不写库
    uv run python scripts/replay.py --sample tests/fixtures/sample_history.json --group 100200300 --dry-run

    # 从 OneBot（NapCat）按时间戳向历史翻页拉真实消息
    RZYL_ONEBOT_API_ROOT=http://127.0.0.1:3000 RZYL_ONEBOT_ACCESS_TOKEN=xxx \\
      uv run python scripts/replay.py --onebot --group 100200300 --limit 200

模型与向量的选择全看配置，代码不动：设了 ``RZYL_CHAT_API_KEY`` 就用真的 OpenAI 兼容
客户端，没设就用离线假模型 ``EchoChatModel``（把窗口第一条消息回声成一条条目，足以让
校验、去重、入库整条链路跑通）；向量同理，没 key 时用确定性的假向量。

``--dry-run`` 与正常模式的差别只有一处：每个窗口只调一次模型、只打印**完整提示词**
与**模型原始输出**，并且**不往配置的库写任何东西**（消息落在一个临时库，用完即删）。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from datetime import datetime

from rzyl_core.llm import (
    ChatModel,
    DeterministicEmbedding,
    EchoChatModel,
    EmbeddingModel,
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
)
from rzyl_core.pipeline import HistorySource, OneBotHistorySource, SampleHistorySource
from rzyl_core.runtime import ReplayReport, Runtime
from rzyl_core.settings import Settings

RULE = "=" * 72


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay.py",
        description="对一段群历史消息跑整条记忆管道（离线可用）。",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", metavar="PATH", help="固定样本文件（JSON 数组）")
    source.add_argument("--onebot", action="store_true", help="从 OneBot HTTP 接口拉历史")
    parser.add_argument("--group", type=int, required=True, help="群号")
    parser.add_argument("--since", help="只取该 ISO 8601 时刻之后的消息，如 2026-10-07T09:00:00+08:00")
    parser.add_argument("--limit", type=int, help="最多取最新的多少条")
    parser.add_argument("--dry-run", action="store_true", help="只打印提示词与原始输出，不写库")
    parser.add_argument("--database-url", help="覆盖 RZYL_DATABASE_URL")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="强制用离线假模型，即使配置了 API key",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="OneBot / 模型请求超时（秒）")
    return parser.parse_args(argv)


def _parse_since(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"--since 不是合法 ISO 8601 时间：{value!r}")
    if parsed.tzinfo is None:
        raise SystemExit("--since 必须带时区，例如 2026-10-07T09:00:00+08:00")
    return parsed


def _build_chat_model(settings: Settings, *, force_fake: bool, timeout: float) -> ChatModel:
    """有 key 且没强制假模型就用真客户端，否则用离线假模型——只改配置不改代码。"""
    if force_fake or not settings.chat_api_key:
        return EchoChatModel()
    return OpenAICompatChatClient(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.chat_model,
        timeout=timeout,
    )


def _build_embedding_model(settings: Settings, *, timeout: float) -> EmbeddingModel:
    """没有向量 key 就用确定性的假向量，管道里向量这层照样有值可测。"""
    if not settings.embedding_api_key:
        return DeterministicEmbedding(settings.embedding_dim)
    return OpenAICompatEmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        timeout=timeout,
        # 与 bot.py 一致：写入侧守住维度，不和配置一致就在调用时抛错。
        expected_dimension=settings.embedding_dim,
    )


def _build_source(settings: Settings, args: argparse.Namespace) -> HistorySource:
    if args.onebot:
        return OneBotHistorySource.from_settings(settings, timeout=args.timeout)
    return SampleHistorySource(args.sample)


def _print_previews(report: ReplayReport) -> None:
    for index, preview in enumerate(report.previews, start=1):
        print(RULE)
        print(f"[窗口 {index}] 提示词版本：{preview.rendered.version}")
        print(RULE)
        print("----- 系统提示词 -----")
        print(preview.rendered.system)
        print("----- 用户内容 -----")
        print(preview.rendered.user)
        print("----- 模型原始输出 -----")
        print(preview.raw_output if preview.raw_output is not None else "(模型调用失败，无输出)")
        if preview.error is not None:
            print(f"----- 校验/调用错误 -----\n{preview.error}")
        print()


async def _print_persisted(runtime: Runtime, group_id: int) -> None:
    memories = await runtime.repository.list_group_memories(
        group_id=group_id, include_inactive=True
    )
    print(RULE)
    print(f"落库条目（群 {group_id}，共 {len(memories)} 条）")
    print(RULE)
    for memory in reversed(memories):
        print(
            f"[#{memory.id}] {memory.category.value} conf={memory.confidence:.2f} "
            f"evidence={list(memory.evidence)}"
        )
        print(f"    {memory.statement}")


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.database_url:
        settings = settings.model_copy(update={"database_url": args.database_url})

    chat_model = _build_chat_model(settings, force_fake=args.fake, timeout=args.timeout)
    embedding_model = _build_embedding_model(settings, timeout=args.timeout)
    provider = "offline" if isinstance(chat_model, EchoChatModel) else settings.chat_model
    source = _build_source(settings, args)
    since = _parse_since(args.since)

    runtime = Runtime(
        chat_model=chat_model,
        embedding_model=embedding_model,
        settings=settings,
        provider=provider,
    )
    print(
        f"回放：群 {args.group}，来源 {type(source).__name__}，"
        f"模式 {'dry-run（不写库）' if args.dry_run else '正常（写库）'}，"
        f"库 {'不写（dry-run）' if args.dry_run else settings.database_url}，"
        f"模型 {getattr(chat_model, 'model_name', provider)}"
    )
    if not args.dry_run:
        await runtime.start(run_background_tasks=False)
    try:
        report = await runtime.replay(
            source=source, group_id=args.group, since=since, limit=args.limit, dry_run=args.dry_run
        )
        if args.dry_run:
            _print_previews(report)
        else:
            for index, outcome in enumerate(report.outcomes, start=1):
                print(
                    f"[窗口 {index}] 状态 {outcome.status.value}，尝试 {outcome.attempts} 次，"
                    f"新增 {len(outcome.memory_ids)} 条，合并 {outcome.merged_count} 条，"
                    f"疑似重复 {outcome.suspect_count} 条"
                    + (f"，错误：{outcome.error}" if outcome.error else "")
                )
            await _print_persisted(runtime, args.group)
        print(RULE)
        print(
            f"消息 {report.message_count} 条，窗口 {report.window_count} 个，"
            f"新增条目 {len(report.memory_ids)} 条"
            + ("（dry-run 未写库）" if report.dry_run else "")
        )
    finally:
        await runtime.stop()
        aclose = getattr(source, "aclose", None)
        if callable(aclose):
            maybe = aclose()
            if inspect.isawaitable(maybe):
                await maybe
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
