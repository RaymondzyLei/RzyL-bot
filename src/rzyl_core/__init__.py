"""RzyL-bot 的平台无关核心包。

这个包**不依赖 NoneBot**，也不应该依赖任何 QQ 协议适配器。它装的是所有与
"消息从哪来、怎么发出去"无关的能力：配置、数据库、大模型客户端、窗口调度、
记忆存储与检索、推送。

依赖方向是硬的：

    plugins/  →  rzyl_core          （业务插件只做薄适配）
    rzyl_core →  （只依赖标准库与第三方库，不含 nonebot）

这里出现 ``import nonebot`` 就是设计错误。这样约束是为了两件事：核心逻辑能脱离
机器人单测，以及离线回放脚本能复用同一套代码与同一份配置。

设计见 https://github.com/RaymondzyLei/RzyL-bot/issues/1
"""

from __future__ import annotations

from rzyl_core.runtime import IngestResult, ReplayReport, Runtime

__all__ = ["IngestResult", "ReplayReport", "Runtime"]
