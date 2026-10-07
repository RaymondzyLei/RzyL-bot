import logging
import os
from pathlib import Path

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from rzyl_core.llm import (
    EmbeddingModel,
    NullEmbedding,
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
)
from rzyl_core.runtime import Runtime, set_runtime
from rzyl_core.settings import Settings

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

# 本地开发时默认再启一个 console 适配器，方便不接 QQ 直接在终端里聊天测试。
# 容器里没有 TTY，用 ENABLE_CONSOLE_ADAPTER=false 关掉。
if os.getenv("ENABLE_CONSOLE_ADAPTER", "true").lower() not in {"0", "false", "no"}:
    from nonebot.adapters.console import Adapter as ConsoleAdapter

    driver.register_adapter(ConsoleAdapter)

nonebot.load_plugin("nonebot_plugin_docs")  # 挂载离线文档到 /website/
nonebot.load_builtin_plugins("echo")
nonebot.load_plugins(str(Path(__file__).parent / "plugins"))

logger = logging.getLogger("rzyl.bot")


def _build_embedding_model(settings: Settings) -> EmbeddingModel:
    """有向量密钥就用真客户端；没有就用空实现——条目向量留空，交给后台补算。

    这里**刻意不造假向量**：``NullEmbedding`` 是设计好的「当前不可用」，条目的向量列
    留空，进程内的补算循环会在有向量服务时补齐；而聊天模型缺密钥是另一回事（见下）。
    """
    if not settings.embedding_api_key:
        logger.warning(
            "未配置向量密钥（RZYL_EMBEDDING_API_KEY），条目向量留空由后台补算；"
            "语义检索与向量去重暂不可用。"
        )
        return NullEmbedding()
    return OpenAICompatEmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )


def _build_runtime(settings: Settings) -> Runtime | None:
    """装配 Runtime；聊天模型缺密钥时**不静默降级**，明确报错并返回 ``None``。

    返回 ``None`` 表示记忆功能未启用：机器人照常连接 QQ、echo 等插件照常工作，但没有
    Runtime——bridge 插件取到 ``None`` 会静默跳过，不会采集或落库任何群消息。
    """
    if not settings.chat_api_key:
        logger.error(
            "未配置聊天模型密钥（RZYL_CHAT_API_KEY），记忆功能未启用："
            "机器人照常连接 QQ，但不会采集、分析或入库任何群消息。"
        )
        return None
    chat_model = OpenAICompatChatClient(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.chat_model,
        timeout=settings.chat_timeout,
        # 额外请求体原样进请求 JSON：模型怪癖（如 enable_thinking）走配置，不改代码。
        extra_body=settings.chat_extra_body,
    )
    return Runtime(
        chat_model=chat_model,
        embedding_model=_build_embedding_model(settings),
        settings=settings,
        provider=settings.chat_model,
    )


_settings = Settings()
_runtime = _build_runtime(_settings)

# 插件通过 rzyl_core.runtime.get_runtime_or_none() 取这里注册的 Runtime。
set_runtime(_runtime)

if _runtime is not None:
    _active_runtime = _runtime

    @driver.on_startup
    async def _start_memory_runtime() -> None:
        await _active_runtime.start()
        logger.info(
            "记忆管道已启动：库 %s，配置白名单 %s",
            _settings.database_url,
            _settings.group_whitelist or "（空）",
        )

    @driver.on_shutdown
    async def _stop_memory_runtime() -> None:
        await _active_runtime.stop()
        logger.info("记忆管道已停止")


if __name__ == "__main__":
    nonebot.run()
