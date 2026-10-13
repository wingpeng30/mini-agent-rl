param(
    [string]$Python = "D:\anaconda3\envs\mini-agent-rl\python.exe",
    [switch]$NoBrowser,
    [switch]$ReplayOnly
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeFolder = "E:\AgentLightningAssets\runs\web-server"
if (!(Test-Path -LiteralPath $Python)) { throw "找不到项目Python：$Python。请用 -Python 指定GPU环境解释器。" }
New-Item -ItemType Directory -Path $runtimeFolder -Force | Out-Null
$env:PYTHONPATH = "$(Join-Path $projectRoot 'src');$(Join-Path $projectRoot 'scripts')"
$url = "http://127.0.0.1:8765"
$alreadyRunning = $false
try {
    $status = Invoke-RestMethod "$url/api/status" -TimeoutSec 2
    if ($status.application -ne "mini-agent-final-web-v1") { throw "端口8765被其他服务占用" }
    $alreadyRunning = $true
} catch {
    if ([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() | Where-Object { $_.Port -eq 8765 }) { throw "端口8765已有服务，请先手动处理；不会终止其他进程。" }
}
if (!$alreadyRunning) {
    $arguments = @("-m", "mini_agent_rl.web.server")
    if ($ReplayOnly) { $arguments += "--no-autoload" }
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeFolder "server.stdout.log") -RedirectStandardError (Join-Path $runtimeFolder "server.stderr.log")
    $process.Id | Set-Content (Join-Path $runtimeFolder "server.pid")
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Milliseconds 500
        if ($process.HasExited) { throw "服务启动失败，请查看E:\AgentLightningAssets\runs\web-server\server.stderr.log" }
        try { $status = Invoke-RestMethod "$url/api/status" -TimeoutSec 2; $alreadyRunning = $true } catch {}
    } while (!$alreadyRunning -and (Get-Date) -lt $deadline)
    if (!$alreadyRunning) { throw "服务未就绪，请查看.demo-runtime日志；未自动重试。" }
}
Write-Host "本地搜索工作台：$url"
Write-Host "网页启动后自动加载AM（当前实测最佳研究候选）。等待页面显示模型已就绪，再输入问题并点击运行搜索。"
if (!$NoBrowser) { Start-Process $url }
