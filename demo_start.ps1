# 电商售后智能客服 Agent 演示一键启动
# 用法：右键"使用 PowerShell 运行"，默认启动 Web 演示（浏览器聊天页）
#       加参数 -Mode cli 可启动命令行交互模式
# 功能：检测服务是否已启动 -> 未启动则启动 -> 健康检查 -> 打开聊天页

param(
    [ValidateSet("web", "cli")]
    [string]$Mode = "web"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$CondaPython = Join-Path $env:USERPROFILE '.conda\envs\agent-service\python.exe'
$Python = if (Test-Path -LiteralPath $CondaPython -PathType Leaf) {
    $CondaPython
} else {
    (Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
}
$Port = 8765
$Url = "http://127.0.0.1:$Port"

Write-Host ""
Write-Host "==== 电商售后智能客服 Agent 演示启动 ($Mode) ====" -ForegroundColor Cyan

# ---- 1. 环境自检 ----
if (-not $Python) {
    Write-Host "[X] 未找到 Python。请安装 Python 或先激活项目环境。" -ForegroundColor Red
    Read-Host "按回车退出"; exit 1
}
if ($Mode -eq "web" -and -not (Test-Path "$Root\webapp\server.py")) {
    Write-Host "[X] 未找到 Web 入口: $Root\webapp\server.py" -ForegroundColor Red
    Read-Host "按回车退出"; exit 1
}
# ---- 2. 使用本地配置或模板 ----
$ConfigPath = Join-Path $Root 'config.json'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    $ConfigPath = Join-Path $Root 'config.example.json'
}
$Config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$hasKey = -not [string]::IsNullOrWhiteSpace($Config.api_key)
if ($hasKey) {
    Write-Host "[OK] 检测到 API Key" -ForegroundColor Green
} else {
    Write-Host "[!] 未配置 API Key，将使用 mock 演示。" -ForegroundColor Yellow
}

# ---- 3. Web 模式：检测服务是否已在运行 ----
function Test-Service {
    try {
        $resp = Invoke-WebRequest -Uri "$Url/api/health" -TimeoutSec 2 -UseBasicParsing
        return ($resp.StatusCode -eq 200)
    } catch { return $false }
}

if ($Mode -eq "web") {
    if (Test-Service) {
        Write-Host "[OK] 服务已在运行，无需重复启动" -ForegroundColor Green
    } else {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            $owner = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            Write-Host "[X] 端口 $Port 被其他进程占用: $($owner.ProcessName) (PID $($conn.OwningProcess))" -ForegroundColor Red
            Read-Host "按回车退出"; exit 1
        }

        Write-Host "[..] 正在启动 Web 服务（独立窗口运行）..." -ForegroundColor Yellow
        Start-Process -FilePath $Python `
            -ArgumentList "webapp\server.py", "--port", "$Port" `
            -WorkingDirectory $Root

        # ---- 4. 健康检查轮询（最多 20 秒）----
        $ok = $false
        foreach ($i in 1..10) {
            Start-Sleep -Seconds 2
            if (Test-Service) { $ok = $true; break }
            Write-Host "    等待服务就绪... ($($i*2)s)" -ForegroundColor DarkGray
        }
        if (-not $ok) {
            Write-Host "[X] 20 秒内服务未就绪，请查看弹出的启动窗口报错信息" -ForegroundColor Red
            Read-Host "按回车退出"; exit 1
        }
        Write-Host "[OK] 服务启动成功" -ForegroundColor Green
    }

    # ---- 5. 打开聊天页 ----
    Write-Host "[..] 打开聊天页: $Url"
    Start-Process $Url

    Write-Host ""
    Write-Host "==== 就绪。演示完成后关闭弹出的服务窗口即可停止 ====" -ForegroundColor Cyan
    Write-Host "提示：会话面板默认选择离线演示" -ForegroundColor DarkGray
    Read-Host "按回车关闭本窗口"
    exit 0
}

# ---- CLI 模式：从项目根目录进入交互循环 ----
Write-Host "[..] 进入命令行交互（输入 exit 退出）..." -ForegroundColor Yellow
$CliOptions = @('--verbose', '--memory')
if (-not $hasKey) {
    $CliOptions += '--mock'
}
Push-Location -LiteralPath $Root
try {
    & $Python main.py @CliOptions
    $CliExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $CliExitCode
