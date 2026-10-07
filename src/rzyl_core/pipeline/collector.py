"""采集判定与消息文本化：把「这条群消息该不该收、收进来长什么样」变成纯函数。

插件只负责把 OneBot 事件的几样东西拆出来（群号、发送者、机器人自己的号、消息段列表），
判定与文本化全在这里——插件里没有业务逻辑（见 issue #1「插件保持薄」）。

三件事：

- :func:`merge_allowed_groups` —— 白名单来源是**两个**：设置里的 ``group_whitelist``
  （配置的初始值）与数据库里的每群开关（``Repository.enabled_groups`` /
  ``disabled_groups``，里程碑 3 的命令会写它）。**数据库里有行的群以那一行为准**：
  显式开启的群加入，显式关掉的群从配置白名单里**摘掉**。这一条不能少——不然
  ``记忆 暂停 <群号>`` 对一个写在 ``RZYL_GROUP_WHITELIST`` 里的群就是个空操作，
  而「暂停不生效」比「没有这个命令」更糟。
- :func:`render_segments` —— OB11 消息段数组 → ``(text, segments_json)``。文本段拼成
  文本；``image`` / ``record`` / ``video`` / ``file`` / ``forward`` 等非文本段在文本里
  落成可读占位符，保证「你们看这张图」后面不断层；``at`` 渲染成 ``@某人``，绝不把 CQ 码
  原样塞进文本。原始段落序列化进 ``segments_json``——**图片的 url 与 file 原样保留**，
  里程碑 4 的本地落盘与视觉理解要用。
- :func:`should_collect` / :func:`collect` —— 接收判定：只收白名单里的群，跳过机器人
  自己发的（``user_id == self_id``），跳过纯空白且没有任何非文本段的消息。

与 :mod:`rzyl_core.pipeline.history` 的 ``render_message_segments`` 的差别只有一处：
历史渲染把手里的文件段一律写成 ``[文件]``，这里会带上文件名（``[文件: 讲义.pdf]``），
因为实时链路拿到的是完整段落。历史那条路的行为属于既有契约，不在这里动。
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: 非文本段落 → 占位符。未知类型退成 ``[类型名]``，绝不吞掉。
SEGMENT_PLACEHOLDERS: dict[str, str] = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[合并转发]",
    "json": "[卡片]",
    "xml": "[卡片]",
    "poke": "[戳一戳]",
    "music": "[音乐]",
}


@dataclass(frozen=True, slots=True)
class RenderedSegments:
    """一次段落文本化的结果。

    ``text`` 是给模型与人看的纯文本；``segments_json`` 是原始段落的 JSON 串（图片的
    ``url`` / ``file`` 都在里面）；``has_non_text`` 记录是否存在非文本段，供「纯空白
    但有图」的消息不被误丢。
    """

    text: str
    segments_json: str
    has_non_text: bool


def merge_allowed_groups(
    configured: Collection[int],
    enabled: Collection[int],
    disabled: Collection[int] = (),
) -> frozenset[int]:
    """把配置白名单与运行时开关合并成一份群号集合。

    规则：``(配置 − 显式关掉) ∪ 显式开启``。``disabled`` 缺省为空，所以「还没有任何
    运行时开关」时它退化成「配置白名单 ∪ 运行时开过的群」，与里程碑 2 的行为一致。
    """
    return (frozenset(configured) | frozenset(enabled)) - frozenset(disabled)


def _at_target(data: dict[str, Any]) -> str:
    """``at`` 段落的显示名：优先群名片 / 昵称，其次 QQ 号，``all`` 特判。"""
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    qq = data.get("qq")
    if qq == "all":
        return "全体成员"
    if isinstance(qq, int) and not isinstance(qq, bool):
        return str(qq)
    if isinstance(qq, str) and qq.strip():
        return qq.strip()
    return ""


def _file_placeholder(data: dict[str, Any]) -> str:
    """文件段的占位符，尽量带上文件名：``[文件: 讲义.pdf]``。"""
    for key in ("name", "file"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return f"[文件: {value.strip()}]"
    return SEGMENT_PLACEHOLDERS["file"]


def _normalized_segments(segments: Sequence[Any]) -> list[dict[str, Any]]:
    """只保留形状正确的段落（``{"type": str, "data": dict}``），保证可 JSON 序列化。"""
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("type")
        if not isinstance(kind, str):
            continue
        data = segment.get("data")
        normalized.append({"type": kind, "data": data if isinstance(data, dict) else {}})
    return normalized


def render_segments(segments: Sequence[Any]) -> RenderedSegments:
    """把 OB11 消息段数组渲染成 ``(text, segments_json)``。

    非文本段按原位置落占位符，保持对话顺序；``segments_json`` 保存归一化后的原始段落。
    """
    normalized = _normalized_segments(segments)
    parts: list[str] = []
    has_non_text = False
    for segment in normalized:
        kind = segment["type"]
        data = segment["data"]
        if kind == "text":
            parts.append(str(data.get("text", "")))
            continue
        has_non_text = True
        if kind == "at":
            parts.append(f"@{_at_target(data)}")
        elif kind == "file":
            parts.append(_file_placeholder(data))
        else:
            parts.append(SEGMENT_PLACEHOLDERS.get(kind, f"[{kind}]"))
    return RenderedSegments(
        text="".join(parts),
        segments_json=json.dumps(normalized, ensure_ascii=False),
        has_non_text=has_non_text,
    )


def should_collect(
    *,
    group_id: int,
    user_id: int,
    self_id: int | None,
    rendered: RenderedSegments,
    allowed_groups: Collection[int],
) -> bool:
    """这条消息该不该收：在白名单里、不是机器人自己发的、不是纯空白噪音。"""
    if group_id not in allowed_groups:
        return False
    if self_id is not None and user_id == self_id:
        return False
    if not rendered.text.strip() and not rendered.has_non_text:
        return False
    return True


def collect(
    *,
    group_id: int,
    user_id: int,
    self_id: int | None,
    segments: Sequence[Any],
    allowed_groups: Collection[int],
) -> RenderedSegments | None:
    """判定并文本化一条群消息；不该收时返回 ``None``。"""
    rendered = render_segments(segments)
    if not should_collect(
        group_id=group_id,
        user_id=user_id,
        self_id=self_id,
        rendered=rendered,
        allowed_groups=allowed_groups,
    ):
        return None
    return rendered


__all__: Iterable[str] = [
    "SEGMENT_PLACEHOLDERS",
    "RenderedSegments",
    "collect",
    "merge_allowed_groups",
    "render_segments",
    "should_collect",
]
