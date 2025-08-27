# 分形曲线Tokenizer - Podman容器化部署指南

## 🐳 容器化概述

本项目已完全容器化，支持使用Podman进行部署和运行。容器化后的项目具有以下特点：

- ✅ **完全隔离的运行环境**
- ✅ **一键部署和启动**
- ✅ **支持GPU加速**（如果可用）
- ✅ **数据卷持久化**
- ✅ **多服务编排支持**
- ✅ **健康检查和自动重启**

## 📋 前置要求

### 1. 安装Podman

**Windows：**
```powershell
# 使用Winget安装
winget install RedHat.Podman

# 或者下载安装包
# https://podman.io/getting-started/installation
```

**Linux：**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install podman

# CentOS/RHEL/Fedora
sudo dnf install podman
```

### 2. 验证安装
```bash
podman --version
podman info
```

## 🚀 快速启动

### 方法1：使用管理脚本（推荐）

**Windows PowerShell：**
```powershell
# 构建并启动容器
.\podman-manage.ps1 run

# 启动Jupyter Lab
.\podman-manage.ps1 jupyter

# 查看容器状态
.\podman-manage.ps1 status
```

**Linux/macOS Bash：**
```bash
# 给脚本执行权限
chmod +x podman-manage.sh

# 构建并启动容器
./podman-manage.sh run

# 启动Jupyter Lab  
./podman-manage.sh jupyter
```

### 方法2：手动命令

```bash
# 1. 构建镜像
podman build -t fractal-curve-tokenizer:latest .

# 2. 创建卷
podman volume create fractal-tokenizer-data
podman volume create fractal-tokenizer-outputs
podman volume create fractal-tokenizer-logs

# 3. 运行容器
podman run -d \
    --name fractal-tokenizer-container \
    -p 8000:8000 \
    -v fractal-tokenizer-data:/app/data \
    -v fractal-tokenizer-outputs:/app/outputs \
    -v fractal-tokenizer-logs:/app/logs \
    --env PYTHONPATH=/app \
    fractal-curve-tokenizer:latest
```

### 方法3：使用Docker Compose

```bash
# 启动所有服务
podman-compose up -d

# 只启动主服务
podman-compose up -d fractal-tokenizer

# 启动包括Jupyter的所有服务
podman-compose --profile jupyter up -d
```

## 📁 目录结构说明

```
fractal-curve-tokenizer_podman/
├── 📄 Dockerfile                 # 主容器构建文件
├── 📄 Dockerfile.jupyter          # Jupyter容器构建文件
├── 📄 docker-compose.yml          # 容器编排配置
├── 📄 .dockerignore               # Docker忽略文件
├── 📄 .env.example                # 环境变量模板
├── 📄 entrypoint.sh               # 容器入口脚本
├── 📄 podman-manage.sh            # Linux管理脚本
├── 📄 podman-manage.ps1           # Windows管理脚本
├── 📄 CONTAINER_GUIDE.md          # 本文档
│
├── 📂 vit_pytorch/                # 核心代码
├── 📂 tests/                      # 测试套件
├── 📄 *.py                        # Python脚本
└── 📄 pyproject.toml              # 项目配置
```

## 🎛️ 管理脚本使用

### 基本命令

| 命令 | 功能 | Windows | Linux |
|------|------|---------|--------|
| `build` | 构建镜像 | `.\podman-manage.ps1 build` | `./podman-manage.sh build` |
| `run` | 启动容器 | `.\podman-manage.ps1 run` | `./podman-manage.sh run` |
| `stop` | 停止容器 | `.\podman-manage.ps1 stop` | `./podman-manage.sh stop` |
| `exec` | 进入容器 | `.\podman-manage.ps1 exec` | `./podman-manage.sh exec` |
| `logs` | 查看日志 | `.\podman-manage.ps1 logs` | `./podman-manage.sh logs` |
| `test` | 运行测试 | `.\podman-manage.ps1 test` | `./podman-manage.sh test` |
| `status` | 查看状态 | `.\podman-manage.ps1 status` | `./podman-manage.sh status` |
| `clean` | 清理资源 | `.\podman-manage.ps1 clean` | `./podman-manage.sh clean` |

### 运行特定脚本

```bash
# 运行调试脚本
./podman-manage.sh script debug_tokenizer.py

# 运行训练脚本
./podman-manage.sh script train_fractal_vs_standard_cifar10.py

# 运行实验脚本
./podman-manage.sh script run_experiments.py
```

## 🔧 高级配置

### 1. 环境变量配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env  # 或使用其他编辑器
```

主要配置项：
- `WANDB_API_KEY`: Weights & Biases API密钥
- `CUDA_VISIBLE_DEVICES`: GPU设备ID
- `DEFAULT_BATCH_SIZE`: 默认批量大小
- `DEFAULT_MAX_LEVEL`: 默认最大分形层级

### 2. 数据卷挂载

```bash
# 挂载本地数据目录
podman run -d \
    --name fractal-tokenizer \
    -v /path/to/your/data:/app/data:ro \
    -v /path/to/outputs:/app/outputs \
    fractal-curve-tokenizer:latest
```

### 3. GPU支持

```bash
# 启用GPU支持（需要nvidia-container-toolkit）
podman run -d \
    --name fractal-tokenizer \
    --device nvidia.com/gpu=all \
    fractal-curve-tokenizer:latest
```

### 4. 资源限制

```bash
podman run -d \
    --name fractal-tokenizer \
    --memory=8g \
    --cpus=4 \
    fractal-curve-tokenizer:latest
```

## 🧪 开发环境

### 使用Jupyter Lab

```bash
# 启动Jupyter服务
./podman-manage.sh jupyter

# 访问 http://localhost:8888
# 默认无密码，可在容器中设置
```

### 挂载开发代码

```bash
# 开发模式：挂载源代码目录
podman run -d \
    --name fractal-dev \
    -v $(pwd):/app/workspace \
    -p 8888:8888 \
    fractal-curve-tokenizer-jupyter:latest
```

## 📊 监控和日志

### 查看实时日志
```bash
podman logs -f fractal-tokenizer-container
```

### 健康检查
```bash
podman healthcheck run fractal-tokenizer-container
```

### 资源使用情况
```bash
podman stats fractal-tokenizer-container
```

## 🐛 故障排除

### 常见问题

1. **容器启动失败**
   ```bash
   # 检查日志
   podman logs fractal-tokenizer-container
   
   # 检查镜像
   podman images
   ```

2. **端口占用**
   ```bash
   # 查看端口使用
   netstat -tulpn | grep :8000
   
   # 更换端口
   podman run -p 8001:8000 ...
   ```

3. **权限问题**
   ```bash
   # 检查容器用户
   podman exec fractal-tokenizer-container whoami
   
   # 检查文件权限
   podman exec fractal-tokenizer-container ls -la /app
   ```

4. **依赖问题**
   ```bash
   # 重新构建镜像
   podman build --no-cache -t fractal-curve-tokenizer:latest .
   ```

### 调试模式

```bash
# 以交互模式启动容器
podman run -it --rm fractal-curve-tokenizer:latest /bin/bash

# 或进入运行中的容器
podman exec -it fractal-tokenizer-container /bin/bash
```

## 🚀 生产部署

### 多实例部署

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  fractal-tokenizer:
    image: fractal-curve-tokenizer:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### 负载均衡

```bash
# 使用nginx或traefik进行负载均衡
# 详细配置请参考相关文档
```

## 📝 最佳实践

1. **定期备份数据卷**
   ```bash
   podman run --rm -v fractal-tokenizer-outputs:/data -v $(pwd):/backup alpine tar czf /backup/outputs-$(date +%Y%m%d).tar.gz /data
   ```

2. **定期更新镜像**
   ```bash
   ./podman-manage.sh clean
   ./podman-manage.sh build
   ```

3. **监控资源使用**
   ```bash
   # 设置资源警报
   podman run --memory=4g --memory-reservation=2g ...
   ```

4. **使用健康检查**
   ```bash
   # 容器会自动重启失败的实例
   ```

## 📚 相关资源

- [Podman官方文档](https://podman.io/docs)
- [容器最佳实践](https://docs.podman.io/en/latest/markdown/podman-run.1.html)
- [分形ViT论文](https://arxiv.org/abs/2010.11929)

---

**需要帮助？** 请查看项目的GitHub Issues或联系维护者。
