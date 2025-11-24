# Fractal Curve Tokenizer (Podman/Docker Container)

[English](README.md) | [中文](README_zh.md)

这是一个用于运行 `fractal-curve-tokenizer` 训练任务的容器化环境。它会自动从 GitHub 拉取最新代码，并配置好 CUDA 加速的 PyTorch 环境。

## 前置要求

- **Podman** (或 Docker)
- **NVIDIA GPU** 及驱动
- **NVIDIA Container Toolkit** (用于容器内 GPU 支持)

## 快速开始

1. **构建镜像**:

    ```bash
    podman build -t fractal-curve-tokenizer .
    ```

2. **运行训练并同步结果**:

    使用以下命令运行容器，它会将训练结果（日志、检查点、可视化）同步到宿主机的 `outputs/experiments` 目录：

    ```bash
    # 创建本地输出目录
    mkdir -p outputs/experiments

    # 运行容器
    # -v 挂载卷以持久化保存实验结果
    # :Z 选项用于处理 SELinux 权限 (Fedora/RHEL/CentOS)
    podman run --rm -it --gpus all \
      -v "$(pwd)/outputs/experiments:/app/experiments:rw,Z" \
      fractal-curve-tokenizer \
      uv run python examples/training/train_fractal_vit.py --quick-test --epochs 2 --batch-size 8
    ```

## 使用 Docker Compose (可选)

如果你更喜欢使用 Compose：

```bash
# 使用 Podman Compose
podman-compose up --build
```

或者

```bash
docker-compose up --build
```

## 项目结构

- `Dockerfile`: 定义训练环境，包含 CUDA、uv 和自动拉取的源码。
- `docker-compose.yml`: 定义服务运行方式，配置 GPU 和数据卷挂载。
- `outputs/`: (自动创建) 用于存放训练输出文件。
