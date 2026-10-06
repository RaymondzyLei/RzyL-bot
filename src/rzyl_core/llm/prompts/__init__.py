"""外置的、带版本号的提示词模板。

模板文本不写死在 Python 字符串里，而是随包存放成 ``v<版本>.system.md`` 与
``v<版本>.user.md`` 两个文件——改措辞不动代码，版本号又会被窗口和记忆条目记下，
这样"改了提示词之后变好还是变坏"可回答、可回滚（见 issue #1）。

占位符用 ``{{window_messages}}`` / ``{{remembered_summary}}`` 这种双花括号形式，
渲染时做纯文本替换而不是 ``str.format``：模板里要给 JSON 示例，含大量单花括号，
用 format 会逼着模板到处转义，反而更容易出错。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

#: 本包内可用的全部提示词版本。新增模板文件时同步登记到这里。
PROMPT_VERSIONS: tuple[str, ...] = ("v1",)

#: 未显式指定版本时使用的版本。
DEFAULT_PROMPT_VERSION = "v1"

#: 用户内容模板里的两个占位符。
WINDOW_MESSAGES_PLACEHOLDER = "{{window_messages}}"
REMEMBERED_SUMMARY_PLACEHOLDER = "{{remembered_summary}}"

_PACKAGE = "rzyl_core.llm.prompts"


class PromptTemplateNotFoundError(KeyError):
    """请求了不存在的提示词版本。"""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """渲染好的提示词：版本号 + 系统提示词 + 用户内容。

    直接把 ``system`` 与 ``user`` 交给 ``ChatModel.complete()`` 即可；``version``
    随窗口与条目一并落库。
    """

    version: str
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """一个版本的提示词模板。文本与版本号同时可见，供装配层与回放脚本取用。"""

    version: str
    system: str
    user_template: str

    def render(self, *, window_messages: str, remembered_summary: str) -> RenderedPrompt:
        """把本窗口消息与已记条目摘要填进模板。

        用 replace 而非 format：模板里的 JSON 示例含单花括号，不会成为转义目标。
        """
        user = self.user_template.replace(
            WINDOW_MESSAGES_PLACEHOLDER, window_messages
        ).replace(REMEMBERED_SUMMARY_PLACEHOLDER, remembered_summary)
        return RenderedPrompt(version=self.version, system=self.system, user=user)


def get_prompt_template(version: str = DEFAULT_PROMPT_VERSION) -> PromptTemplate:
    """按版本号取模板；未知版本明确失败，绝不回退到别的版本。"""
    if version not in PROMPT_VERSIONS:
        raise PromptTemplateNotFoundError(
            f"未知的提示词版本 {version!r}；可用版本：{PROMPT_VERSIONS}"
        )
    return PromptTemplate(
        version=version,
        system=_read_template_part(version, "system"),
        user_template=_read_template_part(version, "user"),
    )


def current_prompt_version() -> str:
    """当前默认提示词版本，供只需要版本号的调用方（如窗口记账）取用。"""
    return DEFAULT_PROMPT_VERSION


def _read_template_part(version: str, part: str) -> str:
    asset = resources.files(_PACKAGE) / f"{version}.{part}.md"
    return asset.read_text(encoding="utf-8")


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "PromptTemplate",
    "PromptTemplateNotFoundError",
    "REMEMBERED_SUMMARY_PLACEHOLDER",
    "RenderedPrompt",
    "WINDOW_MESSAGES_PLACEHOLDER",
    "current_prompt_version",
    "get_prompt_template",
]
