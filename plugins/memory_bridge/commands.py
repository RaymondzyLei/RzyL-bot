"""私聊命令：``记忆 …`` 的入口。

三条边界在这里写死，别处不再重复：

- **只在私聊**：规则要求 ``PrivateMessageEvent``。群里发「记忆 xxx」不会有任何回显——
  记忆内容是别人的聊天记录，在群里回显性质就变了（issue #1「群内永不回显」）。
  注意这条必须靠**运行时的 isinstance 判定**，不能只靠类型标注：``on_message`` 什么
  消息事件都会收到，标注挡不住群消息，而漏掉这一步的后果是群里回显。
- **只有超管**：``permission=SUPERUSER``。非超管发来连一句「无权限」都没有，
  不确认命令的存在。
- **插件只做搬运**：解析、执行、渲染全在 ``rzyl_core.memory``；这里只把文本递给
  Runtime、把回复发回私聊。判定用「解析得出命令」而不是字符串前缀匹配，这样
  「记忆」这个前缀的语义只有一处定义。
"""

from __future__ import annotations

import logging

from nonebot import on_message
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import Bot as OneBotV11Bot
from nonebot.adapters.onebot.v11 import PrivateMessageEvent
from nonebot.permission import SUPERUSER

from rzyl_core.memory import parse_memory_command
from rzyl_core.runtime import get_runtime_or_none

logger = logging.getLogger("rzyl.plugin.memory_bridge.commands")

#: 未启用记忆功能时的回复。说清是「没配密钥」而不是「命令写错了」。
DISABLED_REPLY = "记忆功能未启用（未配置聊天模型密钥），命令暂时不可用。"


def looks_like_memory_command(event: Event) -> bool:
    """这条**私聊**消息是不是记忆命令。

    两道判定缺一不可：先挡住群消息（群内永不回显），再问解析器。解析器是命令前缀语义的
    唯一来源。
    """
    if not isinstance(event, PrivateMessageEvent):
        return False
    return parse_memory_command(event.get_plaintext()) is not None


matcher = on_message(
    rule=looks_like_memory_command,
    permission=SUPERUSER,
    priority=5,
    # block=True：命令已经被我们接管，不让 echo 之类更低的响应器再看到它。
    block=True,
)


@matcher.handle()
async def _handle_memory_command(bot: OneBotV11Bot, event: PrivateMessageEvent) -> None:
    """解析并执行一条记忆命令，把结果发回私聊。"""
    runtime = get_runtime_or_none()
    if runtime is None:
        await matcher.finish(DISABLED_REPLY)
    command = parse_memory_command(event.get_plaintext())
    if command is None:  # 规则已经挡过一次，这里只是兜底
        return
    try:
        result = await runtime.execute_command(command)
    except Exception:
        # 只记命令种类，不记命令全文与回复内容：回复里是群聊原文。
        logger.exception("执行记忆命令失败：%s", command.kind.value)
        await matcher.finish("执行这条命令时出错了，细节见容器日志。")
    logger.info("记忆命令 %s 执行完成（ok=%s）", command.kind.value, result.ok)
    await matcher.finish(result.text)
