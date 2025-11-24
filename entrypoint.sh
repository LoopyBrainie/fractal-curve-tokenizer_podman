#!/bin/bash
set -e

# 如果存在 .venv，则激活它
if [ -d "/app/.venv" ]; then
    source /app/.venv/bin/activate
fi

# 执行传入的命令
exec "$@"
