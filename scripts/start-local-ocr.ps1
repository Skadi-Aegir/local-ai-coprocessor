<#
.SYNOPSIS
  start-local-ocr.ps1 - start the local-ocr service on 127.0.0.1:8116.

.DESCRIPTION
  Reads config\local-ocr.yaml, checks the venv, server and BOTH model directories exist, refuses
  to start if the port is taken or a live PID is already recorded, then waits for /health before
  reporting success.

  CPU only: this service occupies no VRAM, so there is no VRAM preflight and no GPU check. It is
  started as a detached process with its own pid file. Loopback only; no firewall rule is created
  and no autostart is installed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-local-ocr.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start-local-ocr.ps1 -Tier server
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-ocr.yaml",
  [int]$Port,
  [string]$Tier,
  [int]$Threads,
  [int]$WaitSeconds = 240
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyCfg = Join-Path $ScriptDir 'lf_config.py'
$Py = 'python'

function Cfg([string]$key, $default = $null) {
  $out = & $Py $PyCfg $key --config $Config 2>$null
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) { return $default }
  return ($out | Select-Object -First 1).ToString().Trim()
}
function Write-Step($m) { Write-Host "[start-local-ocr] $m" }
function Write-Fail($m) { Write-Host "[start-local-ocr] ERROR: $m" -ForegroundColor Red }

$venvPy    = Cfg 'paths.venv_python'
$server    = Cfg 'paths.server'
$modelRoot = Cfg 'paths.model_root'
$pidFile   = Cfg 'paths.pid_file'
$stdout    = Cfg 'paths.stdout_log'
$stderr    = Cfg 'paths.stderr_log'
$Host_     = Cfg 'runtime.host' '127.0.0.1'
$Port      = if ($Port)    { $Port }    else { [int](Cfg 'runtime.port' 8116) }
$Tier      = if ($Tier)    { $Tier }    else { Cfg 'tier' 'mobile' }
$Threads   = if ($Threads) { $Threads } else { [int](Cfg 'runtime.threads' 8) }

Write-Step "config    : $Config"
Write-Step "venv      : $venvPy"
Write-Step "model root: $modelRoot"
Write-Step "endpoint  : http://${Host_}:$Port  (tier: $Tier, device cpu, threads $Threads)"

if (-not (Test-Path $venvPy)) { Write-Fail "venv python not found: $venvPy"; exit 2 }
if (-not (Test-Path $server)) { Write-Fail "server not found: $server"; exit 2 }

# both model directories for the selected tier must be present, or the service cannot load
$detDir = Join-Path $modelRoot "PP-OCRv5_${Tier}_det"
$recDir = Join-Path $modelRoot "PP-OCRv5_${Tier}_rec"
foreach ($d in @($detDir, $recDir)) {
  if (-not (Test-Path $d)) {
    Write-Fail "model directory not found: $d"
    Write-Fail "run: python benchmarks\ocr\b1_models.py   (models are not in git)"
    exit 2
  }
  if (-not (Test-Path (Join-Path $d 'inference.pdiparams'))) {
    Write-Fail "model weights missing in $d (inference.pdiparams)"
    exit 2
  }
}

if (Test-Path $pidFile) {
  $old = (Get-Content $pidFile -Raw).Trim()
  if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
    Write-Fail "already running with PID $old (use stop-local-ocr.ps1)"; exit 3
  }
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Fail "port $Port already in use by PID $($busy[0].OwningProcess)"; exit 3 }

$logsDir = Split-Path -Parent $stdout
if ($logsDir -and -not (Test-Path $logsDir)) { New-Item -ItemType Directory -Force -Path $logsDir | Out-Null }

$proc = Start-Process -FilePath $venvPy -ArgumentList @(
    $server, '--host', $Host_, '--port', $Port,
    '--config', $Config, '--tier', $Tier, '--threads', $Threads
  ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
Write-Step "pid       : $($proc.Id)"

$ready = $false
for ($i = 0; $i -lt $WaitSeconds; $i++) {
  Start-Sleep -Seconds 1
  if ($proc.HasExited) {
    Write-Fail "service exited early (code $($proc.ExitCode)). stderr tail:"
    if (Test-Path $stderr) { Get-Content $stderr -Tail 20 | ForEach-Object { Write-Host "    $_" } }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    exit 5
  }
  try {
    $h = Invoke-RestMethod -Uri "http://${Host_}:$Port/health" -TimeoutSec 3
    if ($h.status -eq 'ok') { $ready = $true; break }
  } catch { }
}

if (-not $ready) { Write-Fail "health did not become ready within $WaitSeconds s"; exit 6 }

$st = Invoke-RestMethod -Uri "http://${Host_}:$Port/stats" -TimeoutSec 15
$hh = Invoke-RestMethod -Uri "http://${Host_}:$Port/health" -TimeoutSec 15
Write-Step "health    : ok after ~$i s"
Write-Step "model     : $($hh.model)   device: $($hh.device)   onednn: $($hh.onednn)"
Write-Step "load      : $($hh.model_load_s) s   rss: $($st.rss_mb) MB"
Write-Step "READY - http://${Host_}:$Port"
exit 0
