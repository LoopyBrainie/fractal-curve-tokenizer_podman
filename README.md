# Fractal Curve Tokenizer

This repository contains the fractal curve tokenizer implementation and related Vision Transformer (ViT) components.

## ⚡ 快速开始 (使用uv - 推荐)

项目已完全支持uv包管理器，提供更快的依赖解析和安装体验：

**Windows用户：**
```powershell
# 运行快速开始脚本
.\quickstart-uv.ps1
```

**Linux/macOS用户：**
```bash
# 给脚本执行权限并运行
chmod +x quickstart-uv.sh
./quickstart-uv.sh
```

**手动设置：**
```bash
# 1. 安装uv (如果没有安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境和安装依赖
uv venv
uv sync

# 3. 运行脚本
uv run python debug_tokenizer.py
```

## 🐳 容器化部署

项目已完全容器化，支持Podman/Docker一键部署：

**快速启动：**
```bash
# Windows PowerShell
.\podman-manage.ps1 run

# Linux/macOS
./podman-manage.sh run
```

**启动Jupyter开发环境：**
```bash
# Windows PowerShell
.\podman-manage.ps1 jupyter

# Linux/macOS  
./podman-manage.sh jupyter
# 访问 http://localhost:8888
```

详细的容器化使用指南请查看 [CONTAINER_GUIDE.md](./CONTAINER_GUIDE.md)

## 📦 传统安装方式

如果您偏好传统的Python环境安装：

```bash
pip install -e .
```

## Core Components

### 1. Fractal Curve Tokenizer
- `vit_pytorch/fractal_curve_tokenizer.py`: Main fractal Hilbert curve tokenizer implementation
- `vit_pytorch/fractal_curve_tokenizer_backup.py`: Backup version for compatibility

### 2. Fractal ViT Architecture
- `vit_pytorch/fractal_vit.py`: Next-generation fractal ViT implementation with advanced features:
  - Hilbert-aware multi-scale attention
  - Enhanced fractal token processing
  - Adaptive fractal feedforward networks

## Features

- **Fractal Hilbert Tokenization**: Hierarchical image tokenization using fractal Hilbert curves
- **Learnable Split Decision**: Adaptive patch splitting based on content complexity
- **Multi-level Attention**: Attention mechanisms aware of fractal hierarchy
- **Enhanced Position Encoding**: Advanced position encoding for fractal tokens

## Testing

### Unit Tests
- `tests/unit_tests/test_fractal_hilbert_tokenizer.py`: Core tokenizer tests
- `tests/unit_tests/test_hilbert_*.py`: Hilbert curve specific tests

### Integration Tests
- `tests/integration_tests/test_enhanced_fractal_vit.py`: Full model integration tests
- `tests/integration_tests/test_fractal_zigzag_manim.py`: Visualization tests

### Benchmarks
- `tests/benchmarks/benchmark_fractal_vit.py`: Performance benchmarks
- `tests/benchmarks/compare_fractal_vs_standard.py`: Comparison with standard ViT

### Training Scripts
- `tests/training_scripts/train_fractal_vit_complete.py`: Complete training pipeline
- `tests/training_scripts/train_enhanced_fractal_vit.py`: Enhanced model training
- `tests/training_scripts/train_fractal_cifar10.py`: CIFAR-10 specific training

## Main Scripts

- `train_fractal_vs_standard_cifar10.py`: Comparison training script
- `run_experiments.py`: Automated experiment runner
- `debug_tokenizer.py`: Tokenizer debugging utilities

## Installation

```bash
pip install -e .
```

## Usage

```python
from vit_pytorch import FractalHilbertTokenizer, NextGenerationFractalViT

# Initialize tokenizer
tokenizer = FractalHilbertTokenizer(
    min_patch_size=(4, 4),
    max_level=3,
    learnable_split=True
)

# Initialize fractal ViT
model = NextGenerationFractalViT(
    image_size=224,
    min_patch_size=(4, 4),
    num_classes=1000,
    dim=512,
    depth=12,
    heads=8
)
```

## License

See LICENSE file for details.
