"""窗口渲染的行为测试（工单 #4）。

断言三段内容确实落在提示词的正确位置：上一窗口尾部在开头、标注为仅上下文且不编号；
已记条目摘要进「已记条目摘要」段；本窗口消息带窗口内序号，供 ``evidence`` 引用。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rzyl_core.llm.prompts import RenderedPrompt
from rzyl_core.pipeline import (
    PREVIOUS_TAIL_HEADER,
    SequencedMessage,
    WindowMessage,
    assemble_window,
)

CST = timezone(timedelta(hours=8))
BEGIN = datetime(2026, 10, 7, 9, 0, tzinfo=CST)


def _message(message_id: int, text: str) -> WindowMessage:
    return WindowMessage(
        message_id=message_id,
        group_id=111,
        user_id=10000 + message_id,
        text=text,
        sent_at=BEGIN,
        nickname=f"用户{message_id}",
    )


def _tail(sequence: int, message_id: int, text: str) -> SequencedMessage:
    return SequencedMessage(
        sequence=sequence,
        message_id=message_id,
        group_id=111,
        user_id=10000 + message_id,
        sender=f"用户{message_id}",
        text=text,
        sent_at=BEGIN,
    )


def test_render_places_previous_tail_remembered_and_current_window() -> None:
    window = assemble_window(
        group_id=111,
        messages=[_message(101, "实验改到周三"), _message(102, "收到")],
        previous_tail=[_tail(1, 91, "上周那份讲义我看过了")],
        remembered_summary="[#7] event 实验原本定在周五",
    )

    rendered = window.render()

    assert isinstance(rendered, RenderedPrompt)
    assert rendered.version == "v1"
    # 三段都在，且边界可见
    assert PREVIOUS_TAIL_HEADER in rendered.user
    assert "上周那份讲义我看过了" in rendered.user
    assert "[#7] event 实验原本定在周五" in rendered.user
    assert "1：用户101(10101) 实验改到周三" in rendered.user
    assert "2：用户102(10102) 收到" in rendered.user
    assert "{{" not in rendered.user


def test_previous_tail_is_marked_context_only_and_not_numbered() -> None:
    window = assemble_window(
        group_id=111,
        messages=[_message(101, "本窗口的话")],
        previous_tail=[_tail(1, 91, "上一窗口的话")],
    )

    user = window.render().user

    assert "仅作上下文" in user
    assert "不要重复提取" in user
    assert "- 用户91(10091) 上一窗口的话" in user
    # 尾部不参与编号：不应以「1：」这种可被 evidence 引用的形式出现
    assert "1：用户91" not in user
    assert user.index("上一窗口的话") < user.index("本窗口的话")


def test_first_window_without_tail_or_remembered_still_renders() -> None:
    window = assemble_window(group_id=111, messages=[_message(101, "唯一一条")])

    user = window.render().user

    assert window.previous_tail == ()
    assert window.remembered_summary == ""
    assert PREVIOUS_TAIL_HEADER not in user
    assert "1：用户101(10101) 唯一一条" in user
    assert "{{" not in user


def test_render_is_deterministic() -> None:
    window = assemble_window(group_id=111, messages=[_message(101, "甲")])

    assert window.render().user == window.render().user
