#!/bin/bash
# uv项目快速开始脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 分形曲线Tokenizer - uv快速开始${NC}"
echo ""

# 检查uv是否安装
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv未安装${NC}"
        echo -e "${YELLOW}请运行以下命令安装uv:${NC}"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "或访问: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo -e "${GREEN}✅ uv已安装: $(uv --version)${NC}"
}

# 初始化项目
init_project() {
    echo -e "${BLUE}📦 初始化项目依赖...${NC}"
    
    # 创建虚拟环境
    if [ ! -d ".venv" ]; then
        echo "创建虚拟环境..."
        uv venv
    fi
    
    # 同步依赖
    echo "同步项目依赖..."
    uv sync
    
    echo -e "${GREEN}✅ 项目初始化完成${NC}"
}

# 验证安装
verify_installation() {
    echo -e "${BLUE}🔍 验证安装...${NC}"
    
    # 测试核心导入
    uv run python -c "
import torch
print(f'✅ PyTorch版本: {torch.__version__}')

try:
    from vit_pytorch import FractalHilbertTokenizer
    print('✅ 分形Tokenizer导入成功')
except ImportError as e:
    print(f'❌ 分形Tokenizer导入失败: {e}')
    exit(1)

try:
    import einops
    print(f'✅ einops版本: {einops.__version__}')
except ImportError:
    print('❌ einops导入失败')
"
}

# 运行测试
run_tests() {
    echo -e "${BLUE}🧪 运行基本测试...${NC}"
    
    # 运行核心测试
    uv run python tests/unit_tests/test_fractal_hilbert_tokenizer.py || {
        echo -e "${YELLOW}⚠️ 部分测试可能需要数据文件${NC}"
    }
}

# 显示使用说明
show_usage() {
    echo -e "${GREEN}🎉 项目设置完成！${NC}"
    echo ""
    echo -e "${YELLOW}📚 使用方法:${NC}"
    echo ""
    echo -e "${BLUE}本地开发:${NC}"
    echo "  # 激活环境"
    echo "  source .venv/bin/activate"
    echo ""
    echo "  # 或者使用uv run直接运行命令"
    echo "  uv run python debug_tokenizer.py"
    echo "  uv run python train_fractal_vs_standard_cifar10.py"
    echo ""
    echo -e "${BLUE}容器化部署:${NC}"
    echo "  # Linux/macOS"
    echo "  ./podman-manage.sh init    # 初始化uv项目"
    echo "  ./podman-manage.sh run     # 构建并运行容器"
    echo "  ./podman-manage.sh jupyter # 启动Jupyter Lab"
    echo ""
    echo "  # Windows PowerShell"
    echo "  .\\podman-manage.ps1 init"
    echo "  .\\podman-manage.ps1 run"
    echo "  .\\podman-manage.ps1 jupyter"
    echo ""
    echo -e "${BLUE}常用命令:${NC}"
    echo "  uv add <package>          # 添加依赖"
    echo "  uv remove <package>       # 移除依赖"
    echo "  uv sync                   # 同步依赖"
    echo "  uv run pytest tests/     # 运行测试"
    echo "  uv run jupyter lab        # 启动Jupyter"
    echo ""
    echo -e "${BLUE}开发工具:${NC}"
    echo "  uv run black .            # 代码格式化"
    echo "  uv run isort .            # 导入排序"
    echo "  uv run flake8             # 代码检查"
    echo "  uv run mypy vit_pytorch/  # 类型检查"
    echo ""
    echo -e "${GREEN}📖 更多信息请查看 README.md 和 CONTAINER_GUIDE.md${NC}"
}

# 主函数
main() {
    check_uv
    init_project
    verify_installation
    # run_tests  # 可选，因为可能需要特定数据
    show_usage
}

# 运行主函数
main "$@"
