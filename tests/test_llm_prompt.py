"""提示词模板的验收测试（工单 #3）。

模板外置在包内并带版本号，运行时能同时取到文本与版本号；模板里要能填入
本窗口消息与「已记条目摘要」。这里只断言外部可观察行为：版本可取、
占位符可填、未知版本明确失败、v1 文本里确实写进了那几条判据。
"""

import pytest

from rzyl_core.llm.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    RenderedPrompt,
    get_prompt_template,
)


def test_default_version_is_available_and_versioned() -> None:
    template = get_prompt_template()
    assert template.version == DEFAULT_PROMPT_VERSION
    assert template.version in PROMPT_VERSIONS
    assert template.version == "v1"


def test_template_exposes_text_and_placeholders() -> None:
    template = get_prompt_template("v1")
    assert template.system.strip()
    assert "{{window_messages}}" in template.user_template
    assert "{{remembered_summary}}" in template.user_template


def test_render_fills_window_messages_and_summary() -> None:
    template = get_prompt_template("v1")
    rendered = template.render(
        window_messages="1: 小A(10001) 这篇论文不错 https://example.invalid/p.pdf",
        remembered_summary="（上一窗口没有值得记的）",
    )
    assert isinstance(rendered, RenderedPrompt)
    assert rendered.version == "v1"
    assert "小A(10001)" in rendered.user
    assert "（上一窗口没有值得记的）" in rendered.user
    assert "{{window_messages}}" not in rendered.user
    assert "{{remembered_summary}}" not in rendered.user


def test_render_leaves_json_braces_untouched() -> None:
    """模板里要放 JSON 示例，渲染不能用 str.format，否则单花括号会被吃掉。"""
    template = get_prompt_template("v1")
    rendered = template.render(window_messages="(空)", remembered_summary="(空)")
    assert "{" in rendered.system
    assert "}" in rendered.system


def test_unknown_version_fails_clearly() -> None:
    with pytest.raises(KeyError):
        get_prompt_template("v999")


def test_v1_states_the_judgement_rules() -> None:
    """v1 必须落实工单列的判据，回放与 #7 都依赖这些措辞。"""
    template = get_prompt_template("v1")
    text = template.system + template.user_template

    for phrase in (
        "可复用",  # 只记可复用的
        "此刻",  # 不记此刻发生了什么
        "resource",
        "knowledge",
        "event",
        "request",
        "没有「其他」",  # 不设其他类
        "confidence",  # 置信度
        "evidence",  # 原文依据
        "supersedes",  # 允许指向已记条目编号
        "空数组",  # 没有值得记的返回空数组
    ):
        assert phrase in text, f"提示词 v1 缺少判据措辞：{phrase}"
