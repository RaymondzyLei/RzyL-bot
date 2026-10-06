# syntax=docker/dockerfile:1

# ---------- 构建阶段：用 uv 装依赖 ----------
FROM python:3.14-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app

# 先只复制依赖清单，让依赖层能被缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# ---------- 运行阶段：只带运行期需要的东西 ----------
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# uid 1000 与宿主机用户一致，便于将来绑定挂载目录时的权限
RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

USER app
EXPOSE 8080

CMD ["python", "bot.py"]
