#!/bin/bash
# Podman容器管理脚本 - 使用uv

set -e

PROJECT_NAME="fractal-tokenizer"
IMAGE_NAME="fractal-curve-tokenizer"
CONTAINER_NAME="fractal-tokenizer-container"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Podman是否安装
check_podman() {
    if ! command -v podman &> /dev/null; then
        log_error "Podman未安装，请先安装Podman"
        exit 1
    fi
    log_info "Podman版本: $(podman --version)"
}

# 检查uv是否安装
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_warning "uv未安装，将尝试在容器中使用"
        return 1
    fi
    log_info "uv版本: $(uv --version)"
    return 0
}

# 初始化uv项目
initialize_uv_project() {
    log_info "初始化uv项目..."
    
    if [ ! -f "uv.lock" ]; then
        if check_uv; then
            log_info "创建uv.lock文件..."
            uv lock
        else
            log_warning "本地没有uv，将在容器构建时创建锁定文件"
        fi
    fi
    
    log_success "uv项目初始化完成"
}

# 本地开发环境设置
setup_local_dev() {
    log_info "设置本地uv开发环境..."
    
    if ! check_uv; then
        log_error "请先安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    
    # 创建和同步环境
    if [ ! -d ".venv" ]; then
        log_info "创建Python虚拟环境..."
        uv venv
    fi
    
    log_info "同步依赖..."
    uv sync
    
    log_success "本地uv开发环境设置完成"
    log_info "激活环境: source .venv/bin/activate"
    log_info "或使用: uv run <command>"
}

# 构建镜像
build_image() {
    log_info "使用uv构建Docker镜像..."
    
    # 确保uv.lock存在
    initialize_uv_project
    
    podman build -t ${IMAGE_NAME}:latest .
    log_success "镜像构建完成: ${IMAGE_NAME}:latest (使用uv)"
}

# 运行容器
run_container() {
    log_info "启动容器 (使用uv运行时)..."
    
    # 停止并删除已存在的容器
    if podman container exists ${CONTAINER_NAME}; then
        log_warning "容器已存在，正在停止..."
        podman stop ${CONTAINER_NAME} || true
        podman rm ${CONTAINER_NAME} || true
    fi
    
    # 创建卷
    podman volume create ${PROJECT_NAME}-data || true
    podman volume create ${PROJECT_NAME}-outputs || true
    podman volume create ${PROJECT_NAME}-logs || true
    podman volume create ${PROJECT_NAME}-uv-cache || true
    
    # 运行容器
    podman run -d \
        --name ${CONTAINER_NAME} \
        --hostname fractal-tokenizer \
        -p 8000:8000 \
        -v ${PROJECT_NAME}-data:/app/data \
        -v ${PROJECT_NAME}-outputs:/app/outputs \
        -v ${PROJECT_NAME}-logs:/app/logs \
        -v ${PROJECT_NAME}-uv-cache:/tmp/uv-cache \
        -v $(pwd)/data:/app/host-data:ro \
        --env PYTHONPATH=/app \
        --env UV_CACHE_DIR=/tmp/uv-cache \
        --restart unless-stopped \
        ${IMAGE_NAME}:latest
    
    log_success "容器启动成功: ${CONTAINER_NAME} (使用uv)"
}

# 运行Jupyter服务
run_jupyter() {
    log_info "启动Jupyter Lab服务 (使用uv)..."
    
    JUPYTER_CONTAINER="${PROJECT_NAME}-jupyter"
    
    if podman container exists ${JUPYTER_CONTAINER}; then
        log_warning "Jupyter容器已存在，正在停止..."
        podman stop ${JUPYTER_CONTAINER} || true
        podman rm ${JUPYTER_CONTAINER} || true
    fi
    
    # 确保uv.lock存在
    initialize_uv_project
    
    # 构建Jupyter镜像
    podman build -f Dockerfile.jupyter -t ${IMAGE_NAME}-jupyter:latest .
    
    # 运行Jupyter容器
    podman run -d \
        --name ${JUPYTER_CONTAINER} \
        -p 8888:8888 \
        -v $(pwd):/app/workspace \
        -v ${PROJECT_NAME}-outputs:/app/outputs \
        -v ${PROJECT_NAME}-uv-cache:/tmp/uv-cache \
        --env PYTHONPATH=/app/workspace \
        --env UV_CACHE_DIR=/tmp/uv-cache \
        ${IMAGE_NAME}-jupyter:latest
    
    log_success "Jupyter Lab已启动，访问地址: http://localhost:8888"
    log_info "使用uv管理的Jupyter环境"
}

# 进入容器
exec_container() {
    log_info "进入容器..."
    podman exec -it ${CONTAINER_NAME} /bin/bash
}

# 查看日志
show_logs() {
    log_info "显示容器日志..."
    podman logs -f ${CONTAINER_NAME}
}

# 停止容器
stop_container() {
    log_info "停止容器..."
    podman stop ${CONTAINER_NAME} || true
    log_success "容器已停止"
}

# 清理资源
clean() {
    log_info "清理容器和镜像..."
    podman stop ${CONTAINER_NAME} || true
    podman rm ${CONTAINER_NAME} || true
    podman stop ${PROJECT_NAME}-jupyter || true
    podman rm ${PROJECT_NAME}-jupyter || true
    podman rmi ${IMAGE_NAME}:latest || true
    podman rmi ${IMAGE_NAME}-jupyter:latest || true
    podman volume rm ${PROJECT_NAME}-uv-cache || true
    log_success "清理完成"
}

# 运行测试
run_tests() {
    log_info "使用uv运行测试..."
    podman exec ${CONTAINER_NAME} uv run pytest tests/ -v
}

# 运行特定脚本
run_script() {
    local script_name=$1
    if [ -z "$script_name" ]; then
        log_error "请指定要运行的脚本名称"
        exit 1
    fi
    
    log_info "使用uv运行脚本: $script_name"
    podman exec ${CONTAINER_NAME} uv run python $script_name
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}Fractal Curve Tokenizer - Podman管理脚本 (使用uv)${NC}"
    echo ""
    echo "用法: $0 <command> [options]"
    echo ""
    echo -e "${YELLOW}命令:${NC}"
    echo "  init        初始化uv项目和依赖"
    echo "  dev         设置本地uv开发环境"
    echo "  build       构建Docker镜像"
    echo "  run         运行主容器"
    echo "  jupyter     启动Jupyter Lab服务"
    echo "  exec        进入容器"
    echo "  logs        查看容器日志"
    echo "  stop        停止容器"
    echo "  clean       清理所有容器和镜像"
    echo "  test        运行测试套件"
    echo "  script <name> 运行指定脚本"
    echo "  status      查看容器状态"
    echo "  help        显示此帮助信息"
    echo ""
    echo -e "${GREEN}uv特性:${NC}"
    echo "  - 快速依赖解析和安装"
    echo "  - 锁定文件确保一致性"
    echo "  - 内置虚拟环境管理"
    echo "  - 更好的缓存机制"
    echo ""
    echo -e "${GREEN}示例:${NC}"
    echo "  $0 init                    # 初始化uv项目"
    echo "  $0 dev                     # 设置本地开发环境"
    echo "  $0 build                   # 构建镜像"
    echo "  $0 run                     # 启动容器"
    echo "  $0 script debug_tokenizer.py # 运行调试脚本"
}

# 查看状态
show_status() {
    log_info "容器状态:"
    podman ps -a --filter name=${PROJECT_NAME}
    echo ""
    log_info "镜像信息:"
    podman images --filter reference=${IMAGE_NAME}
    echo ""
    log_info "卷信息:"
    podman volume ls --filter name=${PROJECT_NAME}
    echo ""
    
    # 显示uv环境信息
    if [ -f "uv.lock" ]; then
        log_info "uv项目状态: ✅ 已初始化"
        if check_uv; then
            log_info "Python版本: $(uv run python --version 2>/dev/null || echo '需要运行 uv sync')"
        fi
    else
        log_warning "uv项目状态: ❌ 未初始化"
    fi
}

# 主函数
main() {
    check_podman
    
    case "${1:-help}" in
        init)
            initialize_uv_project
            ;;
        dev)
            setup_local_dev
            ;;
        build)
            build_image
            ;;
        run)
            build_image
            run_container
            ;;
        jupyter)
            run_jupyter
            ;;
        exec)
            exec_container
            ;;
        logs)
            show_logs
            ;;
        stop)
            stop_container
            ;;
        clean)
            clean
            ;;
        test)
            run_tests
            ;;
        script)
            run_script $2
            ;;
        status)
            show_status
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
