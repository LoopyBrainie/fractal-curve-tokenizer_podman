# Podman容器管理脚本 - PowerShell版本 (使用uv)
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    [Parameter(Position=1)]
    [string]$ScriptName
)

$ProjectName = "fractal-tokenizer"
$ImageName = "fractal-curve-tokenizer"
$ContainerName = "fractal-tokenizer-container"

# 颜色输出函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 检查Podman是否安装
function Test-Podman {
    try {
        $version = podman --version
        Write-Info "Podman版本: $version"
        return $true
    }
    catch {
        Write-Error-Custom "Podman未安装，请先安装Podman"
        return $false
    }
}

# 初始化uv项目
function Initialize-UvProject {
    Write-Info "初始化uv项目..."
    
    if (-not (Test-Path "uv.lock")) {
        Write-Info "创建uv.lock文件..."
        uv lock
    }
    
    Write-Success "uv项目初始化完成"
}

# 构建镜像
function Build-Image {
    Write-Info "使用uv构建Docker镜像..."
    
    # 确保uv.lock存在
    Initialize-UvProject
    
    podman build -f docker/Dockerfile -t "${ImageName}:latest" .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "镜像构建完成: ${ImageName}:latest (使用uv)"
    } else {
        Write-Error-Custom "镜像构建失败"
        exit 1
    }
}

# 运行容器
function Start-Container {
    Write-Info "启动容器 (使用uv运行时)..."
    
    # 停止并删除已存在的容器
    if (podman container exists $ContainerName) {
        Write-Warning "容器已存在，正在停止..."
        podman stop $ContainerName 2>$null
        podman rm $ContainerName 2>$null
    }
    
    # 创建卷
    podman volume create "$ProjectName-data" 2>$null
    podman volume create "$ProjectName-outputs" 2>$null
    podman volume create "$ProjectName-logs" 2>$null
    podman volume create "$ProjectName-uv-cache" 2>$null
    
    # 运行容器
    podman run -d `
        --name $ContainerName `
        --hostname fractal-tokenizer `
        -p 8000:8000 `
        -v "$ProjectName-data:/app/data" `
        -v "$ProjectName-outputs:/app/outputs" `
        -v "$ProjectName-logs:/app/logs" `
        -v "$ProjectName-uv-cache:/tmp/uv-cache" `
        -v "$(Get-Location)/data:/app/host-data:ro" `
        --env PYTHONPATH=/app `
        --env UV_CACHE_DIR=/tmp/uv-cache `
        --restart unless-stopped `
        "${ImageName}:latest"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "容器启动成功: $ContainerName (使用uv)"
    } else {
        Write-Error-Custom "容器启动失败"
    }
}

# 启动Jupyter服务
function Start-Jupyter {
    Write-Info "启动Jupyter Lab服务 (使用uv)..."
    
    $JupyterContainer = "$ProjectName-jupyter"
    
    if (podman container exists $JupyterContainer) {
        Write-Warning "Jupyter容器已存在，正在停止..."
        podman stop $JupyterContainer 2>$null
        podman rm $JupyterContainer 2>$null
    }
    
    # 确保uv.lock存在
    Initialize-UvProject
    
    # 构建Jupyter镜像
    podman build -f docker/Dockerfile.jupyter -t "${ImageName}-jupyter:latest" .
    
    # 运行Jupyter容器
    podman run -d `
        --name $JupyterContainer `
        -p 8888:8888 `
        -v "$(Get-Location):/app/workspace" `
        -v "$ProjectName-outputs:/app/outputs" `
        -v "$ProjectName-uv-cache:/tmp/uv-cache" `
        --env PYTHONPATH=/app/workspace `
        --env UV_CACHE_DIR=/tmp/uv-cache `
        "${ImageName}-jupyter:latest"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Jupyter Lab已启动，访问地址: http://localhost:8888"
        Write-Info "使用uv管理的Jupyter环境"
    }
}

# 本地开发环境设置
function Setup-LocalDev {
    Write-Info "设置本地uv开发环境..."
    
    # 检查uv是否安装
    try {
        $uvVersion = uv --version
        Write-Info "uv版本: $uvVersion"
    }
    catch {
        Write-Error-Custom "uv未安装，请先安装uv: https://docs.astral.sh/uv/"
        return
    }
    
    # 创建和同步环境
    if (-not (Test-Path ".venv")) {
        Write-Info "创建Python虚拟环境..."
        uv venv
    }
    
    Write-Info "同步依赖..."
    uv sync
    
    Write-Success "本地uv开发环境设置完成"
    Write-Info "激活环境: .venv\Scripts\activate"
    Write-Info "或使用: uv run <command>"
}

# 运行脚本使用uv
function Invoke-Script {
    param([string]$Script)
    if ([string]::IsNullOrEmpty($Script)) {
        Write-Error-Custom "请指定要运行的脚本名称"
        return
    }
    
    Write-Info "使用uv运行脚本: $Script"
    podman exec $ContainerName uv run python $Script
}

# 运行测试使用uv
function Invoke-Tests {
    Write-Info "使用uv运行测试..."
    podman exec $ContainerName uv run pytest tests/ -v
}

# 进入容器
function Enter-Container {
    Write-Info "进入容器..."
    podman exec -it $ContainerName /bin/bash
}

# 查看日志
function Show-Logs {
    Write-Info "显示容器日志..."
    podman logs -f $ContainerName
}

# 停止容器
function Stop-Container {
    Write-Info "停止容器..."
    podman stop $ContainerName 2>$null
    Write-Success "容器已停止"
}

# 清理资源
function Remove-All {
    Write-Info "清理容器和镜像..."
    podman stop $ContainerName 2>$null
    podman rm $ContainerName 2>$null
    podman stop "$ProjectName-jupyter" 2>$null
    podman rm "$ProjectName-jupyter" 2>$null
    podman rmi "${ImageName}:latest" 2>$null
    podman rmi "${ImageName}-jupyter:latest" 2>$null
    podman volume rm "$ProjectName-uv-cache" 2>$null
    Write-Success "清理完成"
}

# 显示状态
function Show-Status {
    Write-Info "容器状态:"
    podman ps -a --filter "name=$ProjectName"
    Write-Host ""
    Write-Info "镜像信息:"
    podman images --filter "reference=$ImageName"
    Write-Host ""
    Write-Info "卷信息:"
    podman volume ls --filter "name=$ProjectName"
    Write-Host ""
    
    # 显示uv环境信息
    if (Test-Path "uv.lock") {
        Write-Info "uv项目状态: ✅ 已初始化"
        try {
            $pythonVersion = uv run python --version
            Write-Info "Python版本: $pythonVersion"
        }
        catch {
            Write-Warning "无法获取Python版本信息"
        }
    } else {
        Write-Warning "uv项目状态: ❌ 未初始化"
    }
}

# 显示帮助
function Show-Help {
    Write-Host "Fractal Curve Tokenizer - Podman管理脚本 (使用uv) (PowerShell版本)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法: .\podman-manage.ps1 <command> [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "命令:" -ForegroundColor Yellow
    Write-Host "  init        初始化uv项目和依赖"
    Write-Host "  dev         设置本地uv开发环境"
    Write-Host "  build       构建Docker镜像"
    Write-Host "  run         运行主容器"
    Write-Host "  jupyter     启动Jupyter Lab服务"
    Write-Host "  exec        进入容器"
    Write-Host "  logs        查看容器日志"
    Write-Host "  stop        停止容器"
    Write-Host "  clean       清理所有容器和镜像"
    Write-Host "  test        运行测试套件"
    Write-Host "  script <name> 运行指定脚本"
    Write-Host "  status      查看容器状态"
    Write-Host "  help        显示此帮助信息"
    Write-Host ""
    Write-Host "uv特性:" -ForegroundColor Green
    Write-Host "  - 快速依赖解析和安装"
    Write-Host "  - 锁定文件确保一致性"
    Write-Host "  - 内置虚拟环境管理"
    Write-Host "  - 更好的缓存机制"
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Green
    Write-Host "  .\podman-manage.ps1 init                     # 初始化uv项目"
    Write-Host "  .\podman-manage.ps1 dev                      # 设置本地开发环境"
    Write-Host "  .\podman-manage.ps1 build                    # 构建镜像"
    Write-Host "  .\podman-manage.ps1 run                      # 启动容器"
    Write-Host "  .\podman-manage.ps1 script debug_tokenizer.py # 运行调试脚本"
}

# 主逻辑
if (-not (Test-Podman)) {
    exit 1
}

switch ($Command.ToLower()) {
    "init" { Initialize-UvProject }
    "dev" { Setup-LocalDev }
    "build" { Build-Image }
    "run" { Build-Image; Start-Container }
    "jupyter" { Start-Jupyter }
    "exec" { Enter-Container }
    "logs" { Show-Logs }
    "stop" { Stop-Container }
    "clean" { Remove-All }
    "test" { Invoke-Tests }
    "script" { Invoke-Script $ScriptName }
    "status" { Show-Status }
    default { Show-Help }
}
