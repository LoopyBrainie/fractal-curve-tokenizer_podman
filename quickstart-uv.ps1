# uv项目快速开始脚本 - Windows PowerShell版本

$ErrorActionPreference = "Stop"

Write-Host "🚀 分形曲线Tokenizer - uv快速开始" -ForegroundColor Blue
Write-Host ""

# 检查uv是否安装
function Test-UvInstallation {
    try {
        $uvVersion = uv --version
        Write-Host "✅ uv已安装: $uvVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ uv未安装" -ForegroundColor Red
        Write-Host "请运行以下命令安装uv:" -ForegroundColor Yellow
        Write-Host "powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
        Write-Host "或访问: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    }
}

# 初始化项目
function Initialize-Project {
    Write-Host "📦 初始化项目依赖..." -ForegroundColor Blue
    
    # 创建虚拟环境
    if (-not (Test-Path ".venv")) {
        Write-Host "创建虚拟环境..."
        uv venv
    }
    
    # 同步依赖
    Write-Host "同步项目依赖..."
    uv sync
    
    Write-Host "✅ 项目初始化完成" -ForegroundColor Green
}

# 验证安装
function Test-Installation {
    Write-Host "🔍 验证安装..." -ForegroundColor Blue
    
    # 测试核心导入
    $testScript = @"
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
"@
    
    uv run python -c $testScript
}

# 运行测试
function Invoke-BasicTests {
    Write-Host "🧪 运行基本测试..." -ForegroundColor Blue
    
    try {
        uv run python tests/unit_tests/test_fractal_hilbert_tokenizer.py
    }
    catch {
        Write-Host "⚠️ 部分测试可能需要数据文件" -ForegroundColor Yellow
    }
}

# 显示使用说明
function Show-Usage {
    Write-Host "🎉 项目设置完成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "📚 使用方法:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "本地开发:" -ForegroundColor Blue
    Write-Host "  # 激活环境"
    Write-Host "  .venv\Scripts\activate"
    Write-Host ""
    Write-Host "  # 或者使用uv run直接运行命令"
    Write-Host "  uv run python debug_tokenizer.py"
    Write-Host "  uv run python train_fractal_vs_standard_cifar10.py"
    Write-Host ""
    Write-Host "容器化部署:" -ForegroundColor Blue
    Write-Host "  # Windows PowerShell"
    Write-Host "  .\podman-manage.ps1 init    # 初始化uv项目"
    Write-Host "  .\podman-manage.ps1 run     # 构建并运行容器"
    Write-Host "  .\podman-manage.ps1 jupyter # 启动Jupyter Lab"
    Write-Host ""
    Write-Host "常用命令:" -ForegroundColor Blue
    Write-Host "  uv add <package>          # 添加依赖"
    Write-Host "  uv remove <package>       # 移除依赖"
    Write-Host "  uv sync                   # 同步依赖"
    Write-Host "  uv run pytest tests/     # 运行测试"
    Write-Host "  uv run jupyter lab        # 启动Jupyter"
    Write-Host ""
    Write-Host "开发工具:" -ForegroundColor Blue
    Write-Host "  uv run black .            # 代码格式化"
    Write-Host "  uv run isort .            # 导入排序"
    Write-Host "  uv run flake8             # 代码检查"
    Write-Host "  uv run mypy vit_pytorch/  # 类型检查"
    Write-Host ""
    Write-Host "📖 更多信息请查看 README.md 和 CONTAINER_GUIDE.md" -ForegroundColor Green
}

# 主函数
function Main {
    if (-not (Test-UvInstallation)) {
        exit 1
    }
    
    Initialize-Project
    Test-Installation
    # Invoke-BasicTests  # 可选，因为可能需要特定数据
    Show-Usage
}

# 运行主函数
Main
