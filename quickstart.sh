#!/bin/bash
# uv快速开始入口脚本 - Linux/macOS

# 检查脚本是否存在并执行
if [ -f "scripts/quickstart-uv.sh" ]; then
    chmod +x scripts/quickstart-uv.sh
    exec scripts/quickstart-uv.sh "$@"
else
    echo "错误: 找不到 scripts/quickstart-uv.sh"
    echo "请确保您在项目根目录中运行此脚本"
    exit 1
fi
