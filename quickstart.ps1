# uv快速开始入口脚本 - Windows PowerShell

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

# 检查脚本是否存在并执行
$ScriptPath = "scripts\quickstart-uv.ps1"

if (Test-Path $ScriptPath) {
    & $ScriptPath @RemainingArgs
} else {
    Write-Host "错误: 找不到 $ScriptPath" -ForegroundColor Red
    Write-Host "请确保您在项目根目录中运行此脚本" -ForegroundColor Yellow
    exit 1
}
