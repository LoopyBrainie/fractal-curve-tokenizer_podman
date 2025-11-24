# Fractal Curve Tokenizer (Podman/Docker Container)

[English](README.md) | [中文](README_zh.md)

This is a containerized environment for running `fractal-curve-tokenizer` training tasks. It automatically pulls the latest code from GitHub and configures a CUDA-accelerated PyTorch environment.

## Prerequisites

- **Podman** (or Docker)
- **NVIDIA GPU** and drivers
- **NVIDIA Container Toolkit** (for GPU support inside containers)

## Quick Start

1. **Build the Image**:

    ```bash
    podman build -t fractal-curve-tokenizer .
    ```

2. **Run Training and Sync Results**:

    Use the following command to run the container. It will sync training results (logs, checkpoints, visualizations) to the `outputs/experiments` directory on your host machine:

    ```bash
    # Create local output directory
    mkdir -p outputs/experiments

    # Run container
    # -v mounts the volume to persist experiment results
    # :Z option handles SELinux permissions (Fedora/RHEL/CentOS)
    podman run --rm -it --gpus all \
      -v "$(pwd)/outputs/experiments:/app/experiments:rw,Z" \
      fractal-curve-tokenizer \
      uv run python examples/training/train_fractal_vit.py --quick-test --epochs 2 --batch-size 8
    ```

## Using Docker Compose (Optional)

If you prefer using Compose:

```bash
# Using Podman Compose
podman-compose up --build
```

Or

```bash
docker-compose up --build
```

## Project Structure

- `Dockerfile`: Defines the training environment, including CUDA, uv, and automatically pulled source code.
- `docker-compose.yml`: Defines how the service runs, configuring GPU and volume mounts.
- `outputs/`: (Automatically created) Stores training output files.
