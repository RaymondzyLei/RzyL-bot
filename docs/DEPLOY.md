# 部署与运维

面向「把这套东西在一台机器上跑起来、并且之后能自己查、能重来」的操作手册。设计决策与
规格在 [issue #1](https://github.com/RaymondzyLei/RzyL-bot/issues/1)，这里只讲怎么装、怎么验、
怎么排障。**本文件会进公开仓库：只写占位符，绝不写真实密钥、QQ 号、群号。**

---

## 1. 拓扑与「什么落在哪里」

两个容器，一条反向 WebSocket：

```
QQ 服务器
   ↑
 napcat 容器（QQ 协议端，含登录态）      ── 主动连 ──→  nonebot 容器（本项目）
                                                      ws://nonebot:8080/onebot/v11/ws
```

- **服务名必须是 `nonebot`、端口必须是 `8080`**：NapCat 镜像内置的 `nonebot` 模板里写死了
  `ws://nonebot:8080/onebot/v11/ws`，容器启动时由它生成 `napcat/config/onebot11.json`，
  改了就接不上。
- 落盘的三处，各自的意义不同：

| 位置 | 内容 | 丢了会怎样 |
|---|---|---|
| `./data/` | SQLite 记忆库（原文、窗口、条目、记账） | 记忆全丢，可从 NapCat 历史回补最近几小时 |
| `./napcat/config/` | NapCat 配置（反向 WS 模板等） | 重建容器时会用模板重新生成 |
| 命名卷 `rzyl-bot_napcat-qq` | QQ 登录态（约 360 MB） | **要重新扫码登录** |

`./data` 与 `./napcat/config` 是 bind mount（宿主机能直接看、能备份），登录态用命名卷是
为了避开宿主机属主问题。

---

## 2. 前置条件

- Docker 与 compose 插件：`docker compose version`
- 一个 QQ 号（机器人自己用的那个，建议不要用主号）
- **两个模型服务的 key**：聊天模型 + 向量模型。两家可以是同一家，但注意
  **DeepSeek 官方 API 没有 embedding 接口**，所以「DeepSeek 提取 + 向量检索」必须再配一家
  向量服务（硅基流动、阿里百炼等都行，只要兼容 OpenAI 的 `/embeddings`）。
- 只在本地跑测试/回放时才需要 uv（Python 3.14）：

```bash
uv sync
uv run pytest        # 全量测试
uv run pyright       # 类型检查
```

---

## 3. 配置：`.env`

```bash
cp .env.example .env
```

`.env.example` 里有各项配置的注释与默认值，下面是**必须改**的几项：

| 键 | 说明 |
|---|---|
| `WEBUI_TOKEN` | NapCat WebUI 的登录令牌，改成自己的 |
| `SUPERUSERS` | 超级管理员的 QQ 号。**必须是 JSON 数组**，见下方陷阱 ② |
| `RZYL_CHAT_*` | 聊天模型：`BASE_URL` / `MODEL` / `API_KEY` / `TIMEOUT` / `EXTRA_BODY` / 单价 |
| `RZYL_EMBEDDING_*` | 向量服务：`BASE_URL` / `MODEL` / `API_KEY` / `DIM` / 单价 |
| `RZYL_GROUP_WHITELIST` | 要监听的群号，逗号分隔（运行时也能用 `记忆 开启 <群号>` 加） |

两点提醒：

- **`RZYL_CHAT_API_KEY` 缺失时记忆功能会明确关闭**（日志一条 ERROR），机器人照样连 QQ、
  照样跑 echo 之类已有插件，但一条群消息都不采集。这是有意的：不静默降级成一个「看起来
  在跑、其实什么都没记」的状态。
- 价格两项（`RZYL_CHAT_INPUT_PRICE` / `OUTPUT_PRICE`）只用于记账估费，填错不影响功能，
  只影响 `llm_call` 表里的估算金额。

### 一个够用的最小配置（占位符）

```dotenv
WEBUI_TOKEN=改成一个随机串
SUPERUSERS=["123456789"]
RZYL_CHAT_BASE_URL=https://api.siliconflow.cn/v1
RZYL_CHAT_MODEL=Qwen/Qwen3.5-35B-A3B
RZYL_CHAT_API_KEY=sk-xxxx
# 硅基流动的 Qwen3.5 默认进思考模式会把请求挂到超时，必须显式关掉
RZYL_CHAT_EXTRA_BODY={"enable_thinking": false}
RZYL_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
RZYL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RZYL_EMBEDDING_API_KEY=sk-xxxx
RZYL_EMBEDDING_DIM=1024
RZYL_GROUP_WHITELIST=123456789,987654321
```

---

## 4. 首次部署

```bash
docker compose up -d --build
docker compose logs napcat | grep -i webui   # WebUI 地址与 token（它在日志四十多行处，别用 head -40 截掉）
```

浏览器打开 `http://127.0.0.1:6099/webui`（只绑本机），用 `WEBUI_TOKEN` 登录，**扫码登录
机器人那个 QQ**。登录成功后 NapCat 会按模板自动建立反向 WS，机器人日志里出现连接记录。

登录之后验证采集是否真的通了：往白名单群里发一条消息，机器人日志应出现
`Matcher(...) running complete`；等窗口攒满或超时后，数据库里会多出 `message` / `window` 行。

---

## 5. 怎么算「部署成功」：日志逐条对照

启动后 `docker logs rzyl-bot --tail 60` 应当能看到（顺序大致如下）：

```
记忆管道已启动：库 sqlite+aiosqlite:///data/rzyl.db，配置白名单 [123456789]
启动对账完成：0 个群、0 条消息、0 个窗口、0 条新增条目
记忆采集白名单（配置 ∪ 运行时开关）：[123456789]；机器人当前在的群：[123456789, 987654321]
掉线回补完成：本次共灌入 0 条消息
每日推送已排定：下一次 2026-01-01T22:00:00+08:00（Asia/Shanghai）
```

每一条都在回答一个具体问题：

| 日志行 | 它证明什么 |
|---|---|
| `记忆管道已启动` | Runtime 装配成功、库能打开 |
| `启动对账完成` | 「重启丢掉的内存缓冲」已被捡回（0 条就是没有东西要捡） |
| `记忆采集白名单` | 配置白名单与「机器人实际在的群」各是什么——**两个列表不一致时最容易漏事** |
| `掉线回补完成：本次共灌入 N 条消息` | 回补真的跑过；**0 条也会打这一行**，否则「跑了但没拉到」与「根本没跑」长得一样 |
| `每日推送已排定` | 推送循环起来了，且下一次时刻符合预期 |

**红旗**（看到就要查，别当噪声）：

- `未配置 SUPERUSERS：…` —— 命令不会响应任何人、日报没有收件人
- `未配置聊天模型密钥…记忆功能未启用` —— 采集被明确关掉了
- `未配置向量密钥…语义检索与向量去重暂不可用` —— 能用，但检索只有关键词那一路
- 任何 `Traceback` —— 逐条看；`uvicorn.error` 只是 logger 名字，不是错误

---

## 6. 日常运维

### 日报与命令

- 每天 **22:00**（`RZYL_PUSH_HOUR` / `RZYL_PUSH_MINUTE`，按 `RZYL_TIMEZONE`）给每个超管发
  一条私聊日报，按 事务/请求/资源/知识 分节；低于 `RZYL_CONFIDENCE_THRESHOLD` 的只报数、
  不列出。推送前会**先强制关掉未满的窗口**，否则最近一两个小时的对话会漏出当天日报。
- 命令**只在私聊、只对 `SUPERUSERS`**，群内永不回显。私聊发 `记忆` 看完整用法。

```
记忆 <关键词>            混合检索（关键词 + 语义，RRF 融合）
记忆 今天                今天新增的条目（含低置信度的）
记忆 群 <群号>           某个群的条目
记忆 来源 <编号>         这条记忆的原文窗口（能看出哪几条是依据）
记忆 误报 <编号> [备注]   标为记错：进错例集，并移出推送与检索
记忆 恢复 <编号>         撤回一次标记
记忆 漏了 [群号] [备注]   记一条「本该记但没记」
记忆 开启|暂停 <群号>     开关某个群的采集（暂停会压过配置白名单）
记忆 清空 <编号>|群 <群号>|全部   彻底删除，不可逆
```

### 查库

库就在宿主机上，直接用 sqlite3 打开即可（WAL 模式下允许一边写一边读）：

```bash
sqlite3 data/rzyl.db "SELECT category, COUNT(*) FROM memory GROUP BY category;"
sqlite3 data/rzyl.db "SELECT id, confidence, statement FROM memory ORDER BY id DESC LIMIT 10;"
# 有没有消息没被任何窗口覆盖（正常基线是「群在说话时略大于 0」，安静下来会回到 0）
sqlite3 data/rzyl.db "SELECT COUNT(*) FROM message m WHERE NOT EXISTS (
  SELECT 1 FROM window w WHERE w.group_id=m.group_id AND w.started_at IS NOT NULL
    AND w.ended_at IS NOT NULL AND m.sent_at>=w.started_at AND m.sent_at<=w.ended_at);"
```

### 改配置 vs 改代码

| 改了什么 | 怎么做 |
|---|---|
| `.env`（密钥、白名单、阈值、推送时刻） | `docker compose up -d` —— 会重建容器以重新注入环境变量 |
| 代码（`src/`、`plugins/`、`bot.py`） | `docker compose up -d --build` |
| `docker-compose.yml` | `docker compose up -d --build` |

### 保留与备份

- **原文滚动保留 30 天**（`RZYL_RETENTION_DAYS`），**记忆条目永久**。所以库会稳定在几 MB。
- 备份就是拷 `data/`；库里开着 WAL，热拷请用 `sqlite3 data/rzyl.db ".backup /tmp/x.db"`，
  或用 `/tmp` 里那种「先停容器再拷」的老办法。

---

## 7. 清空重来

```bash
docker compose down
cp data/rzyl.db /tmp/rzyl-$(date +%F).db     # 想留个底就先备份
rm -f data/rzyl.db data/rzyl.db-wal data/rzyl.db-shm
docker compose up -d
```

`down` **不会**删命名卷，所以 QQ 登录态留着（但见陷阱 ⑦：它不保证还有效）。空库启动后，
掉线回补会按 `RZYL_BACKFILL_MAX_HOURS`（默认 24 小时）把最近的群历史重新灌一遍，记忆是
**从这批历史重建**的，不是凭空恢复——所以清了就是清了。

要把 QQ 登录一起清掉做一次真正的「从零」：

```bash
docker compose down
docker volume rm rzyl-bot_napcat-qq     # 之后必须重新扫码登录
```

想连镜像一起验证（确认 Dockerfile 从零也能构建）：

```bash
docker compose build --no-cache
```

---

## 8. 踩过的坑（现象 → 原因 → 做法）

① **构建直接失败，报解析 `docker/dockerfile:1` 超时。**
原因：Dockerfile 顶部写 `# syntax=docker/dockerfile:1` 会让每次构建都去 Docker Hub 拉前端
镜像，而国内网络经常连不上 `registry-1.docker.io`。做法：**不写那行**，本文件只用经典指令，
内置前端够用。

② **容器反复重启，日志里 `ValidationError: superusers Input should be a valid set`。**
原因：`SUPERUSERS=123456789`（裸数字）会被 pydantic 先按 JSON 解析成 `int`，再往 `set[str]`
上转就炸——**这一刻整条记忆管道是停的**。做法：写成 JSON 数组 `SUPERUSERS=["123456789"]`，
多人用 `["123456789","987654321"]`。

③ **模型请求挂到超时。**
原因：硅基流动的 `Qwen/Qwen3.5-*` 默认进思考模式，短提示词也会让请求长时间不返回。
做法：`RZYL_CHAT_EXTRA_BODY={"enable_thinking": false}`。这个字段原样进请求体，所以换模型
换了怪癖时改配置就行，不必改代码。

④ **群里有人发「记忆 xxx」，机器人把记忆内容回显到了群里。**
原因：命令的判定如果只靠类型标注而没有运行时的 `isinstance`，`on_message` 收下的群消息
也会被当成命令——这条曾在开发中真实发生过。做法：判定必须是「事件是私聊 **且** 能解析成
命令」的运行时判断（`tests/test_plugin_commands.py` 把这条隐私边界钉死了）。**群内永不回显
是硬边界**：那是别人的聊天记录。

⑤ **`docker logs --since 21:21` 报 `failed to parse value as time or duration`，而错误被
管道吞掉，看起来像「没有日志」。**
原因：`--since` 只接受时长（`10m`）或完整时间戳。做法：用 `--since 10m` 或
`--since 2026-01-01T21:00:00`。**更要注意**：容器每次重建都会开一段新的日志，旧实例的日志
`docker logs` 看不到了——排查时先说清「这是哪一次启动的日志」。

⑥ **单行 shell 的退出码骗人。**
`grep -c ... && echo ok || echo fail` 这类写法里，`grep` 没匹配到会返回 1，于是 `||` 分支
被执行，看起来像「命令失败」；反过来 `cmd | head` 的退出码是 `head` 的。做法：要判断成败
就把输出写文件再看退出码（`cmd > f 2>&1; echo $?`），别把判断藏在管道里。

⑦ **重建容器后 NapCat 又要扫码，尽管登录态在命名卷里（约 360 MB 状态都在）。**
原因：QQ 侧的会话有时效，命名卷只保证「状态没丢」，不保证「腾讯还认这个会话」。
做法：把「重建后可能要重新扫码」当成常态而不是异常；WebUI 在 `http://127.0.0.1:6099/webui`。
真正的症状是**机器人侧毫无信号**：NapCat 没登录 → 反向 WS 不建立 → `on_bot_connect` 不触发
→ 采集与回补都不跑，而机器人自己的日志一切正常。所以「群里没消息进库」时的第一件事是
看 napcat 容器日志有没有在要二维码。

⑧ **`/onebot/v11/ws` 没有鉴权，所以 8080 故意不对宿主机发布。**
反向 WS 端点目前不做 token 校验，只让容器网络内可达。要在宿主机调试（例如看离线文档
`/website/`）时再临时打开 `docker-compose.yml` 里注释掉的那两行 `ports`，用完关掉。

⑨ **`.env` 与 `napcat/config/` 绝不提交。**
仓库是**公开**的：密钥、token、QQ 号、群号一律不进仓库、不进 issue、不进提交信息。
`.gitignore` 已经挡住了它们，但**写测试与文档时也要自觉**——这条踩过一次：真实群号曾被写
进测试用例，后来改成了虚构号码。

---

## 9. 本机开发（不接 QQ）

```bash
uv sync
uv run python bot.py            # 默认额外启用 console 适配器，可在终端里直接聊天测试
uv run pytest -q
uv run pyright
```

离线回放一段历史——它和线上跑的是同一套 `Runtime` 与同一份配置，所以调提示词、复现某个
窗口的提取结果都从这条路径走。**但有两个开关必须显式给，否则会踩到线上**：

```bash
# 用固定样本 + 离线假模型，写进一个临时库（不碰线上库、不联网、不花钱）
uv run python scripts/replay.py --sample tests/fixtures/sample_history.json --group 100200300 \
  --fake --database-url "sqlite+aiosqlite:////tmp/rzyl-replay.db"

# 只看完整提示词与模型原始输出，一个字都不写库
uv run python scripts/replay.py --sample tests/fixtures/sample_history.json --group 100200300 \
  --dry-run --fake
```

- **`--fake`**：不加就用配置里的真模型（联网、花钱）。只要 `.env` 里配了 key 就是真的，
  配置齐全的机器上「离线回放」这句话是假的。
- **`--database-url`**：不加就写进 `RZYL_DATABASE_URL`，也就是**线上那个库**——拿样本文件
  跑一次就会往记忆库里灌一批虚构消息和条目。`--dry-run` 不受影响（它压根不写库）。

脚本开头会打印三行「群 / 模式 / 库 / 模型」，跑之前照它核对一眼，这是最省事的自检。
