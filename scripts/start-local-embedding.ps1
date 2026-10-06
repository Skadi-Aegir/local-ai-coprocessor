<#
.SYNOPSIS
  start-local-embedding.ps1 - start the local-embedding service on 127.0.0.1:8113.

.DESCRIPTION
  Reads config\local-embedding.yaml, checks the model and venv exist, refuses to start
  if the port is taken, then waits for /health before reporting success.

  Unlike the llama.cpp workers this service loads a PyTorch model, so it is started as a
  detached process with its own pid file. Loopback only; no firewall rule is created and
  no autostart is installed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-local-embedding.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start-local-embedding.ps1 -Dimension 512
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-embedding.yaml",
  [int]$Port,
  [int]$Dimension,
  [int]$BatchSize,
  [int]$WaitSeconds = 240
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyCfg = Join-Path $ScriptDir 'lf_config.py'
$Py = "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = 'python' }

function Cfg([string]$key, $default = $null) {
  $out = & $Py $PyCfg $key --config $Config 2>$null
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) { return $default }
  return ($out | Select-Object -First 1).ToString().Trim()
}
function Write-Step($m) { Write-Host "[start-local-embedding] $m" }
function Write-Fail($m) { Write-Host "[start-local-embedding] ERROR: $m" -ForegroundColor Red }

$venvPy   = Cfg 'paths.venv_python'
$server   = Cfg 'paths.server'
$indexRoot= Cfg 'paths.index_root'
$pidFile  = Cfg 'paths.pid_file'
$stdout   = Cfg 'paths.stdout_log'
$stderr   = Cfg 'paths.stderr_log'
$project  = Cfg 'retrieval.default_project' 'local-ai'
$Port     = if ($Port)      { $Port }      else { [int](Cfg 'runtime.port' 8113) }
$Dimension= if ($Dimension) { $Dimension } else { [int](Cfg 'runtime.dimension' 1024) }
$BatchSize= if ($BatchSize) { $BatchSize } else { [int](Cfg 'runtime.batch_size' 32) }
$MaxSeq   = [int](Cfg 'runtime.max_seq_length' 8192)
$Host_    = Cfg 'runtime.host' '127.0.0.1'
$ModelPath= Cfg 'model_path'

Write-Step "config    : $Config"
Write-Step "venv      : $venvPy"
Write-Step "index root: $indexRoot"
Write-Step "endpoint  : http://${Host_}:$Port  (project: $project, dim $Dimension)"

if (-not (Test-Path $venvPy)) { Write-Fail "venv python not found: $venvPy"; exit 2 }
if (-not (Test-Path $server)) { Write-Fail "server not found: $server"; exit 2 }
if (-not (Test-Path $ModelPath)) { Write-Fail "model not found: $ModelPath"; exit 2 }

if (Test-Path $pidFile) {
  $old = (Get-Content $pidFile -Raw).Trim()
  if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
    Write-Fail "already running with PID $old (use stop-local-embedding.ps1)"; exit 3
  }
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Fail "port $Port already in use by PID $($busy[0].OwningProcess)"; exit 3 }

$logsDir = Split-Path -Parent $stdout
if ($logsDir -and -not (Test-Path $logsDir)) { New-Item -ItemType Directory -Force -Path $logsDir | Out-Null }

$proc = Start-Process -FilePath $venvPy -ArgumentList @(
    $server, '--host', $Host_, '--port', $Port,
    '--index-root', $indexRoot, '--dimension', $Dimension,
    '--batch-size', $BatchSize, '--max-seq-length', $MaxSeq,
    '--project', "$project=$Dimension"
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
Write-Step "health    : ok after ~$i s"
Write-Step "device    : $($st.device)   VRAM: $($st.vram_mb) MiB"
foreach ($p in $st.projects.PSObject.Properties) {
  Write-Step "project   : $($p.Name)  vectors=$($p.Value.vectors_in_index) chunks=$($p.Value.chunks_in_metadata) consistent=$($p.Value.consistent)"
}
Write-Step "READY - http://${Host_}:$Port"
exit 0
