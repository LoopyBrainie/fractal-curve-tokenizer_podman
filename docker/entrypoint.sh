#!/bin/bash
# 容器启动入口脚本 - 使用uv

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== 分形曲线Tokenizer容器启动 (使用uv) ===${NC}"

# 检查uv和Python环境
echo -e "${BLUE}检查uv和Python环境...${NC}"
uv --version
uv run python --version
echo -e "${GREEN}✓ uv和Python环境正常${NC}"

# 检查依赖
echo -e "${BLUE}检查依赖包...${NC}"
uv run python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
uv run python -c "from vit_pytorch import FractalHilbertTokenizer; print('✓ 分形Tokenizer导入成功')"
echo -e "${GREEN}✓ 依赖检查完成${NC}"

# 创建必要目录
mkdir -p /app/data /app/outputs /app/logs

# 设置权限
chown -R app:app /app/outputs /app/logs 2>/dev/null || true

echo -e "${BLUE}环境变量:${NC}"
echo "PYTHONPATH: $PYTHONPATH"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "当前用户: $(whoami)"
echo "工作目录: $(pwd)"

# 如果有参数，执行指定命令
if [ $# -gt 0 ]; then
    echo -e "${BLUE}执行命令: $@${NC}"
    exec "$@"
else
    echo -e "${BLUE}运行默认命令...${NC}"
    # 默认运行调试脚本
    uv run python debug_tokenizer.py
fi
