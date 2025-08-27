# 使用官方Python基础镜像
FROM python:3.12.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# 安装系统依赖和uv
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 复制项目配置文件
COPY pyproject.toml uv.lock* ./
COPY README.md LICENSE MANIFEST.in ./

# 使用uv安装依赖
RUN uv sync --frozen --no-dev

# 复制源代码
COPY vit_pytorch/ ./vit_pytorch/
COPY tests/ ./tests/

# 复制主要脚本
COPY *.py ./

# 复制入口脚本
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 创建数据和输出目录
RUN mkdir -p /app/data /app/outputs /app/logs

# 设置用户权限
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD uv run python -c "from vit_pytorch import FractalHilbertTokenizer; print('Health check passed')" || exit 1

# 暴露端口（如果需要Web服务）
EXPOSE 8000

# 默认命令
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uv", "run", "python", "debug_tokenizer.py"]
