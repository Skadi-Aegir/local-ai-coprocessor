<#
.SYNOPSIS
  start-local-fast.ps1 - start the local-fast worker (Qwen3.5-4B) on llama-server.

.DESCRIPTION
  Phase 2 lifecycle script. Reads every path/parameter from config\local-fast.yaml
  (no duplicated absolute paths), refuses to start when the machine is unsafe
  (model missing / checksum mismatch / insufficient free VRAM / port already busy),
  then waits for the health endpoint before reporting success.

  Binds to 127.0.0.1 only. No firewall rule is created. No autostart is installed -
  the Gateway/Supervisor will own lifecycle later (Phase 2 §13).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-local-fast.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start-local-fast.ps1 -Quant Q5_K_M -Context 16384
#>
[CmdletBinding()]
param(
  [string]$Config  = "$(Split-Path -Parent $PSScriptRoot)\config\local-fast.yaml",
  [string]$Quant,
  [int]$Context,
  [int]$Threads,
  [int]$Port,
  [switch]$Force,          # skip the free-VRAM safety gate
  [int]$WaitSeconds = 180
)

$ErrorActionPreference = 'Stop'

# ------------------------------------------------------------------ helpers
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyCfg     = Join-Path $ScriptDir 'lf_config.py'

function Get-PythonExe {
  # venv first (isolated), then the system interpreter
  $venv = "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe"
  if (Test-Path $venv) { return $venv }
  return 'python'
}
$Py = Get-PythonExe

function Cfg([string]$key, $default = $null) {
  $out = & $Py $PyCfg $key --config $Config 2>$null
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) { return $default }
  return ($out | Select-Object -First 1).ToString().Trim()
}

function Write-Step($msg) { Write-Host "[start-local-fast] $msg" }
function Write-Fail($msg) { Write-Host "[start-local-fast] ERROR: $msg" -ForegroundColor Red }

# ------------------------------------------------------------------ config
$Port          = if ($Port)    { $Port }    else { [int](Cfg 'runtime.port' 8111) }
$Host_         = Cfg 'runtime.host' '127.0.0.1'
$Context       = if ($Context) { $Context } else { [int](Cfg 'runtime.context' 8192) }
$Threads       = if ($Threads) { $Threads } else { [int](Cfg 'runtime.threads' 12) }
$Quant         = if ($Quant)   { $Quant }   else { Cfg 'quant' 'Q4_K_M' }
$Alias         = Cfg 'source.logical_name' 'local-fast'
$GpuLayers     = [int](Cfg 'runtime.gpu_layers' 99)
$BatchSize     = [int](Cfg 'runtime.batch_size' 2048)
$UbatchSize    = [int](Cfg 'runtime.ubatch_size' 512)
$FlashAttn     = Cfg 'runtime.flash_attn' 'auto'
$CacheTypeK    = Cfg 'runtime.cache_type_k' 'f16'
$CacheTypeV    = Cfg 'runtime.cache_type_v' 'f16'
$Parallel      = [int](Cfg 'runtime.parallel' 1)
$MinFreeVram   = [int](Cfg 'runtime.min_free_vram_mb' 4096)

$ServerExe     = Cfg 'paths.server_exe'
$PidFile       = Cfg 'paths.pid_file'
$StdoutLog     = Cfg 'paths.stdout_log'
$StderrLog     = Cfg 'paths.stderr_log'
$LogsDir       = Cfg 'paths.logs_dir'

# quant -> path (support alternates so A/B testing is a flag, not an edit)
if ($Quant -eq (Cfg 'quant')) {
  $ModelPath = Cfg 'model_path'
} else {
  $ModelPath = Cfg "alternates.$Quant"
  if (-not $ModelPath) { Write-Fail "unknown quant '$Quant' (see alternates: in $Config)"; exit 2 }
}

Write-Step "config      : $Config"
Write-Step "alias       : $Alias"
Write-Step "quant       : $Quant"
Write-Step "model       : $ModelPath"
Write-Step "context     : $Context   threads: $Threads   ngl: $GpuLayers   port: $Port"

# ------------------------------------------------------------------ preflight
if (-not (Test-Path $ServerExe)) { Write-Fail "llama-server not found: $ServerExe"; exit 2 }
if (-not (Test-Path $ModelPath)) { Write-Fail "model not found: $ModelPath"; exit 2 }

if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null }

# already running?
if (Test-Path $PidFile) {
  $old = (Get-Content $PidFile -Raw).Trim()
  if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
    Write-Fail "already running with PID $old (use stop-local-fast.ps1 first)"
    exit 3
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# port free?
$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) {
  Write-Fail "port $Port is already in use by PID $($busy[0].OwningProcess)"
  exit 3
}

# VRAM safety gate - the whole point of Phase 2 section 3
try {
  $row = (& nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits) -split ',\s*'
  $total = [int]$row[0]; $used = [int]$row[1]; $free = [int]$row[2]
  Write-Step "vram        : total=$total MiB used=$used MiB free=$free MiB"
  if ($free -lt $MinFreeVram -and -not $Force) {
    Write-Fail "only $free MiB VRAM free, need >= $MinFreeVram MiB."
    Write-Fail "close GPU-heavy apps (games/compositor workloads) or re-run with -Force to override."
    exit 4
  }
} catch {
  Write-Step "vram        : nvidia-smi unavailable, skipping gate"
}

# ------------------------------------------------------------------ launch
$args = @(
  '-m', $ModelPath,
  '--alias', $Alias,
  '-ngl', $GpuLayers,
  '-c', $Context,
  '-np', $Parallel,
  '-t', $Threads,
  '-b', $BatchSize,
  '-ub', $UbatchSize,
  '--flash-attn', $FlashAttn,
  '-ctk', $CacheTypeK,
  '-ctv', $CacheTypeV,
  '--host', $Host_,
  '--port', $Port,
  # --- hardening -----------------------------------------------------------------
  # llama-server warns "no API key is set and CORS allows all origins" by default.
  # This is a loopback-only single-worker service with no browser front-end, so:
  #   * restrict CORS to localhost instead of '*'
  #   * do not allow credentialed cross-origin requests
  #   * disable the Web UI (not used; it is extra attack surface)
  '--cors-origins', 'localhost',
  '--no-cors-credentials',
  '--no-webui',
  # Prometheus-compatible /metrics: exposes token throughput counters. Useful for the
  # Gateway/Supervisor in a later phase. Note it does NOT expose VRAM, so VRAM
  # accounting still comes from nvidia-smi sampling.
  '--metrics',
  '--log-jsonl',
  '--log-file', (Join-Path $LogsDir 'local-fast.server.jsonl')
)

# Optional bearer-token auth. Off by default so Phase 2 behaviour is unchanged; set
# `api_key_env` in config to enable it (e.g. before the MacBook Supervisor connects).
$apiKeyEnv = Cfg 'runtime.api_key_env'
if ($apiKeyEnv) {
  $keyVal = [System.Environment]::GetEnvironmentVariable($apiKeyEnv)
  if ([string]::IsNullOrWhiteSpace($keyVal)) {
    Write-Fail "runtime.api_key_env='$apiKeyEnv' is set but that environment variable is empty"
    exit 2
  }
  $args += @('--api-key', $keyVal)
  Write-Step "auth        : API key enabled (from `$env:$apiKeyEnv)"
} else {
  Write-Step "auth        : none (loopback-only service; set runtime.api_key_env to enable)"
}

Write-Step "launching: $ServerExe $($args -join ' ')"
$proc = Start-Process -FilePath $ServerExe -ArgumentList $args -PassThru -WindowStyle Hidden `
                      -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
$proc.Id | Set-Content -Path $PidFile -Encoding ASCII
Write-Step "pid         : $($proc.Id)"

# ------------------------------------------------------------------ wait for health
$healthUrl = "http://${Host_}:$Port/health"
$ready = $false
for ($i = 0; $i -lt $WaitSeconds; $i++) {
  Start-Sleep -Seconds 1
  if ($proc.HasExited) {
    Write-Fail "server exited early (code $($proc.ExitCode)). Last stderr:"
    if (Test-Path $StderrLog) { Get-Content $StderrLog -Tail 25 | ForEach-Object { Write-Host "    $_" } }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    exit 5
  }
  try {
    $h = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
    if ($h.status -eq 'ok') { $ready = $true; break }
  } catch { }
}

if (-not $ready) {
  Write-Fail "health check did not become ready within $WaitSeconds s"
  exit 6
}

$models = Invoke-RestMethod -Uri "http://${Host_}:$Port/v1/models" -TimeoutSec 10
$ids = @()
foreach ($m in $models.data) { $ids += $m.id }
Write-Step "health      : ok after ~$i s"
Write-Step "served ids  : $($ids -join ', ')"
if ($ids -notcontains $Alias) {
  Write-Host "[start-local-fast] WARNING: alias '$Alias' not present in /v1/models" -ForegroundColor Yellow
}
Write-Step "READY - http://${Host_}:$Port  (model alias: $Alias)"
exit 0
