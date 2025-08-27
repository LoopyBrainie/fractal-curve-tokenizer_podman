#!/bin/bash
# 项目管理入口脚本 - Linux/macOS

# 检查脚本是否存在并执行
if [ -f "scripts/podman-manage.sh" ]; then
    chmod +x scripts/podman-manage.sh
    exec scripts/podman-manage.sh "$@"
else
    echo "错误: 找不到 scripts/podman-manage.sh"
    echo "请确保您在项目根目录中运行此脚本"
    exit 1
fi
