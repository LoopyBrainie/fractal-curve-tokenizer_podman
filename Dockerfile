# 基于官方 PyTorch CUDA 镜像
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 设置环境变量，避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装 git, curl 等基础工具
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 将 uv 添加到 PATH
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 克隆仓库
# 使用 --depth 1 进行浅克隆以减小镜像体积
RUN git clone --depth 1 https://github.com/LoopyBrainie/fractal-curve-tokenizer.git .

# 使用 uv 同步环境 (安装依赖)
# 这会根据仓库中的 pyproject.toml 或 uv.lock 创建虚拟环境
RUN uv sync

# 复制并设置入口脚本
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]

# 默认启动 bash，让用户自行决定执行什么命令
CMD ["/bin/bash"]