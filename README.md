# RzyL-bot

基于 **NoneBot2** 的 QQ 机器人，通过 **NapCatQQ**（OneBot v11 反向 WebSocket）接入 QQ。

## 技术栈

| | |
|---|---|
| 框架 | NoneBot2 2.5.0（fastapi 驱动） |
| 协议 | OneBot v11（`nonebot-adapter-onebot`） |
| QQ 侧 | NapCatQQ（Docker 镜像 `mlikiowa/napcat-docker`） |
| 依赖管理 | uv（Python 3.14） |

## 快速开始（Docker，推荐）

```bash
cp .env.example .env        # 改掉 WEBUI_TOKEN / 模型 key / SUPERUSERS / 群白名单
docker compose up -d --build
docker compose logs napcat  # 查看 WebUI 登录信息
```

浏览器打开 <http://127.0.0.1:6099/webui> 扫码登录 QQ。登录后 NapCat 会按镜像内置的
`nonebot` 模板自动建立反向 WebSocket；机器人日志出现 `OneBot V11 | Bot <QQ> connected`
即接入成功。

QQ 登录态保存在命名卷 `rzyl-bot_napcat-qq` 中，**通常**能跨重启；但腾讯侧会话有时效，
重建容器后仍可能要求重新扫码（实测过）。**NapCat 没登录时机器人自己毫无信号**——不报错、
只是什么都不采集，所以「群里没消息进库」时先看 `docker compose logs napcat` 有没有在要二维码。

配置项、启动日志逐条对照、日常运维、清空重来的步骤、以及踩过的坑，见
**[docs/DEPLOY.md](docs/DEPLOY.md)**。

## 本地开发（不用 Docker）

```bash
uv sync
uv run python bot.py     # 默认额外启用 console 适配器，可在终端直接聊天测试
```

机器人启动后访问 <http://localhost:8080/website/> 可查看离线 NoneBot 文档
（由 `nonebot-plugin-docs` 提供）。

## 项目结构

```
├── bot.py                # 入口：注册适配器、加载插件
├── docker-compose.yml    # nonebot + napcat 两个服务
├── Dockerfile            # uv 多阶段构建（python:3.14-slim）
├── .env.example          # 配置模板（复制为 .env）
├── docs/DEPLOY.md        # 部署与运维手册（含排障）
├── plugins/              # 本地插件目录
└── napcat/config/        # NapCat 配置（运行时生成，未纳入版本控制）
```

## 两个容易踩的点

- 机器人服务名必须叫 **`nonebot`**、端口必须是 **8080**：NapCat 镜像内置模板里写死了
  `ws://nonebot:8080/onebot/v11/ws`。
- `.env` 与 `napcat/config/` 不提交（含 token 与 QQ 登录信息）；`/onebot/v11/ws` 目前
  未开鉴权，所以 compose 没有对外发布 8080 端口，只让容器网络内可达。
