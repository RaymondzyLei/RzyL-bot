"""全局设置对象。

从环境变量读全部可调项（前缀 ``RZYL_``），密钥也从环境变量读——代码里不出现任何
密钥。每项都有默认值，所以**不配任何东西也能构造**，这是第一阶段"不需要 API key"
的前提。构造过程只读环境变量，不发起任何网络请求。

环境变量写法：

- 普通标量：``RZYL_WINDOW_MESSAGE_LIMIT=50``
- 群白名单：``RZYL_GROUP_WHITELIST=123456,789012`` 或 JSON 数组 ``[123456, 789012]``

时间一律用带时区的 ``datetime``；"今天"按 ``timezone``（默认 Asia/Shanghai）的自然日
00:00 起算。
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """所有可调项的单一来源。"""

    model_config = SettingsConfigDict(
        env_prefix="RZYL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # —— 窗口组装 ——
    window_message_limit: int = 30
    """一个窗口最多攒多少条消息。"""

    window_minutes: int = 5
    """一个窗口最多横跨多少分钟。条数与分钟数先到者成窗。"""

    # —— 保留与推送 ——
    retention_days: int = 30
    """原文（群消息）滚动保留天数；记忆条目永久保留。"""

    push_hour: int = 22
    """每日推送时刻的小时（0-23），按 ``timezone`` 计。"""

    push_minute: int = 0
    """每日推送时刻的分钟（0-59）。"""

    confidence_threshold: float = 0.7
    """推送时只列不低于该置信度的条目；入库不设门槛。"""

    # —— 提取与去重 ——
    extract_max_attempts: int = 3
    """一个窗口最多尝试提取几次（含首次）；用尽仍失败则整窗进死信。"""

    dedupe_similarity_threshold: float = 0.92
    """向量余弦去重阈值：同群同类别且相似度不低于它，新条目标为疑似重复。"""

    timezone: str = "Asia/Shanghai"
    """自然日与推送时刻所依据的时区。"""

    group_whitelist: Annotated[list[int], NoDecode] = []
    """初始监听的群号；默认空列表，表示一个群都不监听。"""

    # —— 存储 ——
    database_url: str = "sqlite+aiosqlite:///data/rzyl.db"
    """SQLite 连接串。默认落在宿主机的 ``data/`` 目录，便于备份与手工查询。"""

    # —— OneBot 历史拉取（回放 / 里程碑 2 的掉线回补用）——
    onebot_api_root: str = ""
    """OneBot（NapCat）的 HTTP API 根地址，例如 ``http://127.0.0.1:3000``。

    默认空，表示本机没有可用的 OneBot HTTP 端点——此时只有样本文件那条回放路径可用。
    """

    onebot_access_token: str = ""
    """OneBot HTTP API 的 Bearer token；默认空。"""

    # —— 聊天模型 ——
    chat_base_url: str = "https://api.deepseek.com/v1"
    chat_model: str = "deepseek-chat"
    chat_api_key: str = ""
    chat_input_price: float = 0.0
    """每百万输入 token 的单价，用于记账估费；未配置按 0 估算。"""
    chat_output_price: float = 0.0
    """每百万输出 token 的单价。"""

    # —— 向量服务 ——
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_api_key: str = ""
    embedding_dim: int = 1024
    embedding_price: float = 0.0
    """每百万 token 的单价（向量服务通常按输入 token 计费）。"""

    @field_validator("group_whitelist", mode="before")
    @classmethod
    def _split_group_whitelist(cls, value: object) -> object:
        """允许逗号分隔的朴素写法，也接受 JSON 数组写法。"""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return json.loads(text)
            return [int(item.strip()) for item in text.split(",") if item.strip()]
        return value

    def estimate_chat_cost(self, tokens_in: int, tokens_out: int) -> float:
        """按配置单价估算一次聊天调用的费用（价格单位：每百万 token）。"""
        return (tokens_in * self.chat_input_price + tokens_out * self.chat_output_price) / 1_000_000
