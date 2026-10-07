"""采集判定与消息文本化的行为测试（里程碑 2）。

seam 是 ``rzyl_core.pipeline.collector`` 的四个纯函数：白名单合并、段落渲染、
接收判定、以及「判定 + 渲染」的组合。它们没有依赖（不读库、不联网），所以直接断言；
插件只负责把 OneBot 事件的几样字段拆进来。
"""

from __future__ import annotations

import json
from typing import Any

from rzyl_core.pipeline.collector import (
    RenderedSegments,
    collect,
    merge_allowed_groups,
    render_segments,
    should_collect,
)

# —— 白名单合并 ——


def test_merge_allowed_groups_is_the_union_of_config_and_runtime_switch() -> None:
    # 配置里有的群默认收，运行时开过的群也收——两者取并集。
    assert merge_allowed_groups([100, 200], [200, 300]) == frozenset({100, 200, 300})


def test_merge_allowed_groups_drops_duplicates_and_accepts_empty() -> None:
    assert merge_allowed_groups([], []) == frozenset()
    assert merge_allowed_groups([100, 100], []) == frozenset({100})


def test_pausing_a_group_masks_the_configured_whitelist() -> None:
    """里程碑 3 的关键语义：数据库里有行的群以那一行为准。

    不这样，``记忆 暂停 <群号>`` 对一个写在 ``RZYL_GROUP_WHITELIST`` 里的群就是空操作，
    而「暂停不生效」比「没有这个命令」更糟。默认 ``disabled`` 为空，所以旧的并集语义
    没有变。
    """
    assert merge_allowed_groups([100, 200], [], [200]) == frozenset({100})
    assert merge_allowed_groups([100], [200], [200]) == frozenset({100})
    assert merge_allowed_groups([], [200], [200, 300]) == frozenset()
    # 同一个群在库里只有一行，所以 enabled 与 disabled 不会同时含它——`− disabled` 只可能
    # 摘掉来自配置白名单的群。
    assert merge_allowed_groups([100], [], [100, 999]) == frozenset()


# —— 段落文本化 ——


def test_render_segments_joins_text_and_placeholders_in_order() -> None:
    segments: list[dict[str, Any]] = [
        {"type": "text", "data": {"text": "看这张图"}},
        {"type": "image", "data": {"url": "http://cdn.example/x.png", "file": "x.png"}},
        {"type": "text", "data": {"text": "和这个文件"}},
        {"type": "file", "data": {"name": "讲义.pdf"}},
        {"type": "forward", "data": {}},
        {"type": "record", "data": {"file": "a.silk"}},
        {"type": "video", "data": {"file": "a.mp4"}},
    ]

    rendered = render_segments(segments)

    assert rendered.text == "看这张图[图片]和这个文件[文件: 讲义.pdf][合并转发][语音][视频]"


def test_render_segments_keeps_raw_image_url_and_file_in_segments_json() -> None:
    segments: list[dict[str, Any]] = [
        {"type": "text", "data": {"text": "图"}},
        {"type": "image", "data": {"url": "http://cdn.example/sig.png", "file": "s.png"}},
    ]

    rendered = render_segments(segments)

    payload = json.loads(rendered.segments_json)
    assert payload[1]["type"] == "image"
    # 图片的 url 与 file 必须原样保留：里程碑 4 的视觉理解与本地落盘要用。
    assert payload[1]["data"]["url"] == "http://cdn.example/sig.png"
    assert payload[1]["data"]["file"] == "s.png"


def test_render_segments_falls_back_to_file_name_when_name_is_absent() -> None:
    rendered = render_segments([{"type": "file", "data": {"file": "成绩单.xlsx"}}])

    assert rendered.text == "[文件: 成绩单.xlsx]"


def test_render_segments_renders_at_as_text_not_cq_code() -> None:
    segments: list[dict[str, Any]] = [
        {"type": "at", "data": {"qq": "12345", "name": "小明"}},
        {"type": "text", "data": {"text": " 记得交作业"}},
    ]

    rendered = render_segments(segments)

    assert rendered.text == "@小明 记得交作业"
    assert "[CQ:" not in rendered.text


def test_render_segments_at_without_name_falls_back_to_qq_or_all() -> None:
    assert render_segments([{"type": "at", "data": {"qq": "12345"}}]).text == "@12345"
    assert render_segments([{"type": "at", "data": {"qq": "all"}}]).text == "@全体成员"


def test_render_segments_uses_bracketed_type_for_unknown_segments() -> None:
    rendered = render_segments([{"type": "weather", "data": {}}])

    assert rendered.text == "[weather]"


def test_render_segments_of_empty_input_is_blank_and_json_is_an_array() -> None:
    rendered = render_segments([])

    assert rendered.text == ""
    assert rendered.segments_json == "[]"
    assert rendered.has_non_text is False


# —— 接收判定 ——


def _rendered(text: str = "你好") -> RenderedSegments:
    return RenderedSegments(text=text, segments_json="[]", has_non_text=False)


def test_should_collect_requires_the_group_to_be_in_the_merged_whitelist() -> None:
    allowed = merge_allowed_groups([100], [])

    assert should_collect(
        group_id=100, user_id=1, self_id=9, rendered=_rendered(), allowed_groups=allowed
    )
    assert not should_collect(
        group_id=999, user_id=1, self_id=9, rendered=_rendered(), allowed_groups=allowed
    )


def test_should_collect_includes_a_group_enabled_only_at_runtime() -> None:
    # 配置白名单为空，但运行时开关把这个群开了——仍应接收。
    allowed = merge_allowed_groups([], [200])

    assert should_collect(
        group_id=200, user_id=1, self_id=9, rendered=_rendered(), allowed_groups=allowed
    )


def test_should_collect_skips_the_bot_own_messages() -> None:
    allowed = merge_allowed_groups([100], [])

    assert not should_collect(
        group_id=100, user_id=9, self_id=9, rendered=_rendered(), allowed_groups=allowed
    )


def test_should_collect_skips_blank_text_without_any_non_text_segment() -> None:
    allowed = merge_allowed_groups([100], [])

    assert not should_collect(
        group_id=100, user_id=1, self_id=9, rendered=_rendered("   \n "), allowed_groups=allowed
    )


def test_should_collect_keeps_a_blank_message_that_has_a_non_text_segment() -> None:
    allowed = merge_allowed_groups([100], [])
    rendered = RenderedSegments(text="", segments_json="[]", has_non_text=True)

    assert should_collect(
        group_id=100, user_id=1, self_id=9, rendered=rendered, allowed_groups=allowed
    )


# —— 判定 + 渲染 ——


def test_collect_returns_rendered_message_for_an_allowed_group() -> None:
    allowed = merge_allowed_groups([100], [])

    rendered = collect(
        group_id=100,
        user_id=1,
        self_id=9,
        segments=[{"type": "text", "data": {"text": "实验课改到周五"}}],
        allowed_groups=allowed,
    )

    assert rendered is not None
    assert rendered.text == "实验课改到周五"


def test_collect_returns_none_for_a_disallowed_group() -> None:
    rendered = collect(
        group_id=999,
        user_id=1,
        self_id=9,
        segments=[{"type": "text", "data": {"text": "不该收"}}],
        allowed_groups=merge_allowed_groups([100], []),
    )

    assert rendered is None
