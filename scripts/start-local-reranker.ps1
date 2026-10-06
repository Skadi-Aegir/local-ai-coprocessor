<#
.SYNOPSIS
  start-local-reranker.ps1 - start the local-reranker service on 127.0.0.1:8114.

.DESCRIPTION
  Reads config\local-reranker.yaml, checks the model, the venv and the recorded SHA-256
  of the weights, refuses to start if the port is taken, then waits for /health.

  The SHA-256 check matters here: a hash that is transcribed rather than computed is
  worse than no hash, and this project has already had the reranker's config carry the
  EMBEDDING model's weight hash once. Skipping it is possible with -SkipHashCheck.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-local-reranker.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start-local-reranker.ps1 -MaxLength 1024
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-reranker.yaml",
  [int]$Port,
  [int]$MaxLength,
  [int]$BatchSize,
  [string]$InstructionType = 'default',
  [int]$WaitSeconds = 240,
  [switch]$SkipHashCheck
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
function Write-Step($m) { Write-Host "[start-local-reranker] $m" }
function Write-Fail($m) { Write-Host "[start-local-reranker] ERROR: $m" -ForegroundColor Red }

$venvPy    = Cfg 'paths.venv_python'
$server    = Cfg 'paths.server'
$pidFile   = Cfg 'paths.pid_file'
$stdout    = Cfg 'paths.stdout_log'
$stderr    = Cfg 'paths.stderr_log'
$ModelPath = Cfg 'model_path'
$wantHash  = Cfg 'model_sha256'
$Port      = if ($Port)      { $Port }      else { [int](Cfg 'runtime.port' 8114) }
$MaxLength = if ($MaxLength) { $MaxLength } else { [int](Cfg 'runtime.max_length' 2048) }
$BatchSize = if ($BatchSize) { $BatchSize } else { [int](Cfg 'runtime.batch_size' 20) }
$MaxBatch  = [int](Cfg 'runtime.max_batch_tokens' 8192)
$Dtype     = Cfg 'runtime.dtype' 'bfloat16'
$Host_     = Cfg 'runtime.host' '127.0.0.1'

Write-Step "config    : $Config"
Write-Step "venv      : $venvPy"
Write-Step "endpoint  : http://${Host_}:$Port  (max_length $MaxLength, instr '$InstructionType')"

if (-not (Test-Path $venvPy)) { Write-Fail "venv python not found: $venvPy"; exit 2 }
if (-not (Test-Path $server)) { Write-Fail "server not found: $server"; exit 2 }
if (-not (Test-Path $ModelPath)) { Write-Fail "model not found: $ModelPath"; exit 2 }

if (-not $SkipHashCheck -and $wantHash) {
  $w = Join-Path $ModelPath 'model.safetensors'
  if (Test-Path $w) {
    Write-Step "verifying weights sha256 ..."
    $got = (Get-FileHash $w -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $wantHash.ToLower()) {
      Write-Fail "weights sha256 mismatch`n    expected $($wantHash.ToLower())`n    actual   $got"
      exit 4
    }
    Write-Step "sha256 OK"
  }
}

if (Test-Path $pidFile) {
  $old = (Get-Content $pidFile -Raw).Trim()
  $parsed = 0
  if ($old -and [int]::TryParse($old, [ref]$parsed) -and (Get-Process -Id $parsed -ErrorAction SilentlyContinue)) {
    Write-Fail "already running with PID $parsed (use stop-local-reranker.ps1)"; exit 3
  }
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Fail "port $Port already in use by PID $($busy[0].OwningProcess)"; exit 3 }

$logsDir = Split-Path -Parent $stdout
if ($logsDir -and -not (Test-Path $logsDir)) { New-Item -ItemType Directory -Force -Path $logsDir | Out-Null }

$proc = Start-Process -FilePath $venvPy -ArgumentList @(
    $server, '--host', $Host_, '--port', $Port,
    '--model-path', $ModelPath, '--dtype', $Dtype,
    '--max-length', $MaxLength, '--batch-size', $BatchSize,
    '--max-batch-tokens', $MaxBatch,
    '--instruction-type', $InstructionType
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

$st = Invoke-RestMethod -Uri "http://${Host_}:$Port/stats" -TimeoutSec 20
Write-Step "health    : ok after ~$i s"
Write-Step "device    : $($st.device) $($st.dtype)  VRAM $($st.vram_mb) MiB  load $($st.load_s)s"
Write-Step "scoring   : yes=$($st.yes_token_id) no=$($st.no_token_id) body_budget=$($st.body_budget)"
Write-Step "READY - http://${Host_}:$Port"
exit 0
