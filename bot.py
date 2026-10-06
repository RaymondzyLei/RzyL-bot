import os
from pathlib import Path

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

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

if __name__ == "__main__":
    nonebot.run()
