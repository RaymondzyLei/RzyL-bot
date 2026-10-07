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
from typing import Annotated, Any

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
    """一个窗口最多攒多少条消息。到了就**无条件**成窗，不受 ``window_min_messages`` 影响。"""

    window_minutes: int = 5
    """一个窗口最多横跨多少分钟。

    到点**且**缓冲里条数不少于 ``window_min_messages`` 时才成窗；条数不够就继续攒着。
    """

    window_min_messages: int = 5
    """时间窗触发成窗时，缓冲里至少要有的消息条数。

    条数到了 ``window_minutes`` 想成窗，还要求缓冲里至少有这么多条；不够就**继续攒着**，
    等后续消息把条数补上。稀疏流量下 5 分钟的时间规则会把只有一两条消息的缓冲也收掉，
    白调一次模型却几乎提取不出东西——这条下限就是为它加的。

    只约束「时间到点」这条规则：``window_message_limit``（条数上限）不受它影响，条数一到
    照样无条件成窗。
    """

    window_max_minutes: int = 60
    """兜底上限：缓冲里最老那条消息的年龄达到这么多分钟，就**无条件**关窗（不看条数）。

    为什么需要兜底：只加 ``window_min_messages`` 下限的话，一个只发了一句话就安静下来的
    群，那条消息会无限期躺在内存缓冲里——消息本身已经落库没丢，但**永远不会被提取**，
    日报也就永远看不到它。这条兜底保证最多延迟这么久就会被处理掉。
    """

    # —— 保留与推送 ——
    retention_days: int = 30
    """原文（群消息）滚动保留天数；记忆条目永久保留。"""

    push_hour: int = 22
    """每日推送时刻的小时（0-23），按 ``timezone`` 计。"""

    push_minute: int = 0
    """每日推送时刻的分钟（0-59）。"""

    push_enabled: bool = True
    """是否启用每日定时推送。关掉后只剩手动查询与命令可用。"""

    push_max_entries: int = 50
    """一条日报最多列多少条；超出的会在末尾说明，不静默截断。"""

    confidence_threshold: float = 0.7
    """推送时只列不低于该置信度的条目；入库不设门槛。"""

    # —— 提取与去重 ——
    extract_max_attempts: int = 3
    """一个窗口最多尝试提取几次（含首次）；用尽仍失败则整窗进死信。"""

    dedupe_similarity_threshold: float = 0.92
    """向量余弦去重阈值：同群同类别且相似度不低于它，新条目标为疑似重复。"""

    retrieval_min_similarity: float = 0.5
    """语义检索的相似度下限：余弦低于它的候选不参与 RRF 融合。

    与 ``dedupe_similarity_threshold`` 是**两个不同的旋钮**，别混：去重那个是在问
    「这两条是不是同一件事」（要严，0.92），这个是在问「这条跟查询词有没有关系」（要松）。

    没有下限的话，向量那一路永远排得出一串候选，「搜什么都能搜到东西」，用户就没法用
    检索结果判断「我到底记没记过这件事」。

    **0.5 是量出来的，不是拍的**（2026-10-07，Qwen3-Embedding-0.6B，15 条真实条目）：

    - 无关对的上限 0.46（``论文`` ↔ 库里最像的那条 0.463，无意义查询最高 0.39）；
    - 明显相关的下限 0.56（``论文``↔``文献`` 0.83、``三文鱼``↔``虹鳟`` 0.71、
      ``考试改时间了吗``↔``实验改到周五下午三点`` 0.66、``论文``↔``这篇文献给了新的证明`` 0.556）。

    0.5 落在这道缝里，两边各留 0.04~0.06 的余量。**代价要说清楚**：模型对「问句 vs 陈述句」
    的打分整体偏低（``有没有网盘链接``↔``这个链接里有讲义`` 只有 0.448），落进了无关区间、
    没法与噪声分开，所以这类松散的改写问法会被漏掉——它们该用具体关键词再问一次
    （关键词那一路是字面匹配，不受这个下限影响）。想让语义这一路救回它们，得改成
    「关键词无命中时才降下限」这类两档策略，那需要另一批数据来验证，见 issue #1。
    """

    timezone: str = "Asia/Shanghai"
    """自然日与推送时刻所依据的时区。"""

    group_whitelist: Annotated[list[int], NoDecode] = []
    """初始监听的群号；默认空列表，表示一个群都不监听。"""

    # —— 实时链路的后台循环（里程碑 2）——
    window_flush_seconds: int = 60
    """窗口超时刷新循环的间隔（秒）：群里没人说话时，靠它把未满的缓冲按时成窗。"""

    window_retry_seconds: int = 300
    """窗口重试循环的间隔（秒）：去捡没处理成的窗口（死信、以及停在 pending 的）。"""

    window_retry_max_attempts: int = 5
    """一个没处理成的窗口被后台重试到 ``retry_count`` 达到该值就放弃。

    ``retry_count`` 由首次失败时的尝试次数起算，之后每被后台重试一次加一：默认
    ``extract_max_attempts=3`` 配 ``window_retry_max_attempts=5`` 大致是「首次失败后
    再重试两轮」。达到上限就**不再重试**，避免无限重试同一个窗口。

    「没处理成」有两种：``dead``（模型连续失败）与 ``pending``（写库失败，或者处理时撞上
    没被分类捕获的异常）。后者以前没人管——窗口停在 ``pending``，对账又因为「时间区间已被
    这个窗口盖住」而检测不到，那段内容就永远不再被提取。见
    :meth:`rzyl_core.runtime.Runtime.retry_unfinished_windows`。
    """

    # —— 掉线回补（里程碑 2）——
    backfill_max_hours: int = 24
    """回补的时间护栏：至多回补最近这么多小时，更早的放弃并记一条告警。"""

    backfill_max_messages: int = 500
    """回补的条数护栏：单个群一次最多回补这么多条，达到上限记一条告警。"""

    backfill_page_size: int = 20
    """回补时 ``get_group_msg_history`` 每页的条数。"""

    # —— 启动对账（里程碑 2）——
    reconcile_on_startup: bool = True
    """是否在 ``Runtime.start()`` 时自动跑一次「未覆盖消息对账」。

    窗口缓冲只在内存里，进程重启时未满的窗口会丢；那些消息还在 ``message`` 表里，却因为
    掉线回补的锚点是「最后一条已存消息的时间」而永远不会被提取。打开这一项，启动时用
    一次性后台任务把它们重新成窗、提取、入库。任务不阻塞启动（可能调几十次模型），
    ``run_background_tasks=False`` 时（测试与脚本）不自动跑，方法仍可手动调用。
    """

    reconcile_max_messages: int = 200
    """对账时**每个群**最多处理多少条未覆盖的消息。

    到上限即停止该群、记一条 ``WARNING``，剩下的留到下次启动或手动调用；这是防止对账
    一次拉太多进内存、调太多次模型的护栏。
    """

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
    chat_timeout: float = 120.0
    """单次聊天请求的超时（秒）。给足：实测 Qwen3.5-4B 这类模型单次可到 50 秒以上。"""
    chat_extra_body: Annotated[dict[str, Any], NoDecode] = {}
    """原样合并进聊天请求 JSON 的额外字段（JSON 对象）。

    用途是让**模型怪癖不必改代码**：例如硅基流动的 ``Qwen/Qwen3.5-4B`` 默认进思考模式
    会把请求挂死，必须带 ``{"enable_thinking": false}`` 才正常返回，配置写成
    ``RZYL_CHAT_EXTRA_BODY={"enable_thinking": false}`` 即可。默认空对象，不代表任何
    服务商专有参数。
    """
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

    @field_validator("chat_extra_body", mode="before")
    @classmethod
    def _parse_chat_extra_body(cls, value: object) -> object:
        """接受 JSON 对象字符串；空串视为空对象。非对象一律报错，不静默丢弃。"""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("RZYL_CHAT_EXTRA_BODY 必须是 JSON 对象")
            return parsed
        return value

    def estimate_chat_cost(self, tokens_in: int, tokens_out: int) -> float:
        """按配置单价估算一次聊天调用的费用（价格单位：每百万 token）。"""
        return (tokens_in * self.chat_input_price + tokens_out * self.chat_output_price) / 1_000_000

    def estimate_embedding_cost(self, tokens_in: int) -> float:
        """按配置单价估算一次向量调用的费用（``embedding_price`` 是每百万输入 token 的单价）。

        向量服务通常只按输入 token 计费；调用方拿不到用量时传 0，费用自然为 0。
        """
        return tokens_in * self.embedding_price / 1_000_000
