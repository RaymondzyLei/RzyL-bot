"""大模型能力层：聊天模型、向量嵌入、外置提示词模板。

对外提供两组可替换的接口（``ChatModel`` / ``EmbeddingModel``）、各配假实现，
以及两个 OpenAI 兼容的真客户端和带版本号的提示词模板。这一层不依赖数据库、
不依赖设置对象、更不依赖 NoneBot——换服务商只改装配层注入的实现。

给后续工单的接口约定见各模块 docstring；要点：
- 用量随 ``ChatResult.usage`` 返回，记账落库由 #7 负责；
- 假实现 ``FakeChatModel`` 按脚本吐结果并把提示词原样留存，供测试与回放共用；
- 提示词用 ``get_prompt_template().render(...)`` 取，版本号随结果一并用。
"""

from __future__ import annotations

from rzyl_core.llm.chat import ChatModel, ChatResult, ChatUsage
from rzyl_core.llm.embedding import (
    DeterministicEmbedding,
    Embedding,
    EmbeddingModel,
    NullEmbedding,
)
from rzyl_core.llm.errors import (
    LLMError,
    LLMRequestError,
    LLMResponseError,
    LLMTimeoutError,
)
from rzyl_core.llm.fakes import EchoChatModel, FakeChatModel, ReceivedPrompt
from rzyl_core.llm.openai_compat import (
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
)
from rzyl_core.llm.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    PromptTemplate,
    PromptTemplateNotFoundError,
    RenderedPrompt,
    current_prompt_version,
    get_prompt_template,
)

__all__ = [
    "ChatModel",
    "ChatResult",
    "ChatUsage",
    "DEFAULT_PROMPT_VERSION",
    "DeterministicEmbedding",
    "EchoChatModel",
    "Embedding",
    "EmbeddingModel",
    "FakeChatModel",
    "LLMError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMTimeoutError",
    "NullEmbedding",
    "OpenAICompatChatClient",
    "OpenAICompatEmbeddingClient",
    "PROMPT_VERSIONS",
    "PromptTemplate",
    "PromptTemplateNotFoundError",
    "ReceivedPrompt",
    "RenderedPrompt",
    "current_prompt_version",
    "get_prompt_template",
]
