"""日报投递里那段纯逻辑：按行切段。

项目的测试约定是「不测 NoneBot 插件层」——插件只做搬运、没有业务判定。但
``split_for_send`` 是个**纯函数**，而且切错了的后果很具体：切漏一个字就丢了记忆内容，
切长了整份日报发不出去。所以这里破例测它一个函数，用 ``importlib`` 直接按文件加载，
不触发包 ``__init__``（那会连整个 NoneBot 插件一起拉起来）。

投递本身（``deliver_daily_report``）不测：它只调 ``bot.call_api``，属于「与 NapCat 的
交互」，由真机验收覆盖。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PLUGIN_PUSH = (
    Path(__file__).resolve().parent.parent / "plugins" / "memory_bridge" / "push.py"
)


def _load_plugin_push() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_rzyl_plugin_push_under_test", PLUGIN_PUSH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def push() -> ModuleType:
    return _load_plugin_push()


def test_short_text_goes_out_as_one_message(push: ModuleType) -> None:
    assert push.split_for_send("一行字", limit=100) == ["一行字"]
    assert push.split_for_send("", limit=100) == [""]


def test_text_is_split_on_line_boundaries_and_nothing_is_lost(push: ModuleType) -> None:
    text = "\n".join(f"第 {index} 条记忆" for index in range(1, 21))

    chunks = push.split_for_send(text, limit=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert "\n".join(chunks) == text  # 一个字都没丢，也没多出来
    # 切在行边界上：没有哪一段以半行开头或结尾。
    assert all(chunk.startswith("第 ") for chunk in chunks)


def test_a_single_oversized_line_is_hard_split_rather_than_left_to_fail(
    push: ModuleType,
) -> None:
    """硬切难看，但比「整份日报投递不出去、重试到放弃」好得多。"""
    text = "长" * 25

    chunks = push.split_for_send(text, limit=10)

    assert [len(chunk) for chunk in chunks] == [10, 10, 5]
    assert "".join(chunks) == text


def test_every_chunk_respects_the_limit_even_with_mixed_line_lengths(
    push: ModuleType,
) -> None:
    text = "\n".join(["短", "中" * 15, "很长的" * 40, "尾"])

    chunks = push.split_for_send(text, limit=30)

    assert all(len(chunk) <= 30 for chunk in chunks)
    assert sum(len(chunk.replace("\n", "")) for chunk in chunks) == len(
        text.replace("\n", "")
    )


def test_a_non_positive_limit_is_a_programming_error(push: ModuleType) -> None:
    with pytest.raises(ValueError):
        push.split_for_send("任意", limit=0)


def test_the_default_limit_is_conservative_enough_for_qq(push: ModuleType) -> None:
    assert 0 < push.MAX_PRIVATE_MESSAGE_CHARS <= 2000
