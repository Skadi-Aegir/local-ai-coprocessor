<#
.SYNOPSIS
  check-local-fast.ps1 - health and capability check for the local-fast worker.

.DESCRIPTION
  Verifies, in order:
    1. process alive + PID file
    2. GET /health
    3. GET /v1/models contains the logical alias `local-fast`
    4. GET /props reports the expected context size
    5. POST /v1/chat/completions answers correctly (smoke test)
    6. VRAM headroom after load
    7. listening address is loopback only (Phase 2 section 14)
  Exit codes: 0 = healthy, 1 = degraded, 2 = down.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\check-local-fast.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-fast.yaml",
  [switch]$Quick          # skip the inference smoke test
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyCfg     = Join-Path $ScriptDir 'lf_config.py'
$Py        = if (Test-Path "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe") { "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe" } else { 'python' }

function Cfg([string]$key, $default = $null) {
  $out = & $Py $PyCfg $key --config $Config 2>$null
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) { return $default }
  return ($out | Select-Object -First 1).ToString().Trim()
}

$Port      = [int](Cfg 'runtime.port' 8111)
$Host_     = Cfg 'runtime.host' '127.0.0.1'
$Alias     = Cfg 'source.logical_name' 'local-fast'
$ExpCtx    = [int](Cfg 'runtime.context' 8192)
$PidFile   = Cfg 'paths.pid_file'
$ExpectVram= [int](Cfg 'runtime.expected_vram_mb' 4500)
$MinFree   = [int](Cfg 'runtime.min_free_vram_mb' 4096)

$base = "http://${Host_}:$Port"
$problems = @()
$warnings = @()

Write-Host "=== check-local-fast ===" -ForegroundColor Cyan
Write-Host "endpoint : $base"
Write-Host "alias    : $Alias"
Write-Host "pid file : $PidFile"

# -- 1. process -------------------------------------------------------------
$pid_val = $null
if (Test-Path $PidFile) { $pid_val = (Get-Content $PidFile -Raw).Trim() }
$proc = if ($pid_val) { Get-Process -Id $pid_val -ErrorAction SilentlyContinue } else { $null }
if ($proc) {
  Write-Host "process  : OK pid=$pid_val ($($proc.ProcessName))" -ForegroundColor Green
} else {
  Write-Host "process  : NOT RUNNING" -ForegroundColor Red
  $problems += "server process not running"
}

# -- 2. health --------------------------------------------------------------
$healthOk = $false
try {
  $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5
  if ($h.status -eq 'ok') { $healthOk = $true; Write-Host "health   : OK" -ForegroundColor Green }
  else { Write-Host "health   : UNEXPECTED $($h | ConvertTo-Json -Compress)" -ForegroundColor Red; $problems += "health status not ok" }
} catch {
  Write-Host "health   : UNREACHABLE ($($_.Exception.Message))" -ForegroundColor Red
  $problems += "health endpoint unreachable"
}

if (-not $healthOk) {
  Write-Host "`nRESULT: DOWN" -ForegroundColor Red
  exit 2
}

# -- 3. models / alias ------------------------------------------------------
try {
  $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
  $ids = @($models.data | ForEach-Object { $_.id })
  Write-Host "models   : $($ids -join ', ')"
  if ($ids -contains $Alias) {
    Write-Host "alias    : OK '$Alias' served" -ForegroundColor Green
  } else {
    Write-Host "alias    : MISSING '$Alias'" -ForegroundColor Red
    $problems += "logical alias '$Alias' not served"
  }
} catch { Write-Host "models   : ERROR $($_.Exception.Message)" -ForegroundColor Red; $problems += "could not read /v1/models" }

# -- 4. context -------------------------------------------------------------
try {
  $props = Invoke-RestMethod -Uri "$base/props" -TimeoutSec 10
  $ctx = $props.default_generation_settings.n_ctx
  Write-Host "context  : $ctx (expected $ExpCtx)"
  if ($ctx -ne $ExpCtx) { $warnings += "context is $ctx, config says $ExpCtx" }
} catch { $warnings += "could not read /props" }

# -- 5. inference smoke test ------------------------------------------------
if (-not $Quick) {
  $body = @{
    model    = $Alias
    messages = @(
      @{ role = 'system'; content = 'Answer with a single word.' },
      @{ role = 'user';   content = 'What is the capital of France?' }
    )
    max_tokens = 16
    temperature = 0
    chat_template_kwargs = @{ enable_thinking = $false }
  } | ConvertTo-Json -Depth 8

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120
    $sw.Stop()
    $txt = $r.choices[0].message.content
    Write-Host "inference: OK  $([math]::Round($sw.Elapsed.TotalSeconds,2))s -> '$($txt.Trim())'" -ForegroundColor Green
    if ($r.timings) {
      Write-Host ("timings  : prompt {0:N0} t/s, gen {1:N1} t/s" -f $r.timings.prompt_per_second, $r.timings.predicted_per_second)
    }
    if ($txt -notmatch 'Paris') { $warnings += "smoke-test answer unexpected: '$txt'" }
  } catch {
    Write-Host "inference: FAILED $($_.Exception.Message)" -ForegroundColor Red
    $problems += "chat completion failed"
  }
}

# -- 6. VRAM ----------------------------------------------------------------
try {
  $row = (& nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits) -split ',\s*'
  $total = [int]$row[0]; $used = [int]$row[1]; $free = [int]$row[2]
  Write-Host "vram     : total=$total used=$used free=$free MiB"
  if ($free -lt $MinFree) {
    Write-Host "vram     : BELOW safety margin ($free < $MinFree MiB)" -ForegroundColor Red
    $problems += "VRAM headroom below configured margin"
  } else {
    Write-Host "vram     : OK headroom $free MiB (>= $MinFree)" -ForegroundColor Green
  }
} catch { $warnings += "nvidia-smi unavailable" }

# -- 7. bind address --------------------------------------------------------
$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listeners) {
  $addrs = @($listeners | ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
  Write-Host "bind     : $($addrs -join ', ')"
  $nonLoop = @($addrs | Where-Object { $_ -notin @('127.0.0.1', '::1') })
  if ($nonLoop.Count -gt 0) {
    Write-Host "bind     : EXPOSED on $($nonLoop -join ', ') - must be loopback only" -ForegroundColor Red
    $problems += "service bound to non-loopback address"
  } else {
    Write-Host "bind     : OK loopback only" -ForegroundColor Green
  }
} else { $problems += "no listener on port $Port" }

# -- verdict ----------------------------------------------------------------
Write-Host ""
if ($problems.Count -eq 0 -and $warnings.Count -eq 0) {
  Write-Host "RESULT: HEALTHY" -ForegroundColor Green
  exit 0
} elseif ($problems.Count -eq 0) {
  Write-Host "RESULT: DEGRADED" -ForegroundColor Yellow
  $warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
  exit 1
} else {
  Write-Host "RESULT: UNHEALTHY" -ForegroundColor Red
  $problems | ForEach-Object { Write-Host "  ! $_" -ForegroundColor Red }
  $warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
  exit 1
}
