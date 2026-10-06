<#
.SYNOPSIS
  check-local-embedding.ps1 - health and capability check for local-embedding.

.DESCRIPTION
  Verifies, in order:
    1. process alive + pid file
    2. GET /health
    3. GET /v1/models has the logical alias `local-embedding`
    4. GET /stats shows a consistent index (vectors == chunks)
    5. POST /v1/embeddings returns unit-length vectors of the configured dimension
    6. POST /v1/retrieve returns scored results
    7. listening address is loopback only
  Exit codes: 0 healthy, 1 degraded/unhealthy, 2 down.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\check-local-embedding.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-embedding.yaml",
  [switch]$Quick
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

$Port    = [int](Cfg 'runtime.port' 8113)
$Host_   = Cfg 'runtime.host' '127.0.0.1'
$Alias   = Cfg 'model' 'local-embedding'
$Dim     = [int](Cfg 'runtime.dimension' 1024)
$Project = Cfg 'retrieval.default_project' 'local-ai'
$pidFile = Cfg 'paths.pid_file'

$base = "http://${Host_}:$Port"
$problems = @(); $warnings = @()

Write-Host "=== check-local-embedding ===" -ForegroundColor Cyan
Write-Host "endpoint : $base"
Write-Host "alias    : $Alias   dimension: $Dim   project: $Project"

$pidVal = $null
if (Test-Path $pidFile) { $pidVal = (Get-Content $pidFile -Raw).Trim() }
$proc = if ($pidVal) { Get-Process -Id $pidVal -ErrorAction SilentlyContinue } else { $null }
if ($proc) { Write-Host "process  : OK pid=$pidVal ($($proc.ProcessName))" -ForegroundColor Green }
else { Write-Host "process  : NOT RUNNING" -ForegroundColor Red; $problems += "service process not running" }

$healthOk = $false
try {
  $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5
  if ($h.status -eq 'ok') {
    $healthOk = $true
    Write-Host "health   : OK  device=$($h.device) dim=$($h.dimension)" -ForegroundColor Green
  } else { Write-Host "health   : UNEXPECTED" -ForegroundColor Red; $problems += "health not ok" }
} catch {
  Write-Host "health   : UNREACHABLE ($($_.Exception.Message))" -ForegroundColor Red
  $problems += "health endpoint unreachable"
}
if (-not $healthOk) { Write-Host "`nRESULT: DOWN" -ForegroundColor Red; exit 2 }

try {
  $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
  $ids = @($models.data | ForEach-Object { $_.id })
  if ($ids -contains $Alias) { Write-Host "models   : OK '$Alias'" -ForegroundColor Green }
  else { Write-Host "models   : MISSING '$Alias' ($($ids -join ','))" -ForegroundColor Red; $problems += "alias not served" }
} catch { $problems += "could not read /v1/models" }

try {
  $st = Invoke-RestMethod -Uri "$base/stats" -TimeoutSec 15
  Write-Host "device   : $($st.device)  vram=$($st.vram_mb) MiB"
  foreach ($p in $st.projects.PSObject.Properties) {
    $v = $p.Value
    if ($v.error) { Write-Host "index    : $($p.Name) ERROR $($v.error)" -ForegroundColor Red; $problems += "index error for $($p.Name)" }
    elseif ($v.consistent) { Write-Host "index    : $($p.Name) OK vectors=$($v.vectors_in_index) chunks=$($v.chunks_in_metadata)" -ForegroundColor Green }
    else { Write-Host "index    : $($p.Name) INCONSISTENT $($v | ConvertTo-Json -Compress)" -ForegroundColor Red; $problems += "index inconsistent for $($p.Name)" }
  }
} catch { $warnings += "could not read /stats" }

if (-not $Quick) {
  # embeddings: verify dimension AND unit length
  try {
    $body = @{ model = $Alias; input = @('a short probe sentence') } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$base/v1/embeddings" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120
    $vec = $r.data[0].embedding
    $norm = [math]::Sqrt(($vec | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
    Write-Host "embed    : dim=$($vec.Count) norm=$([math]::Round($norm,6))"
    if ($vec.Count -ne $Dim) { $problems += "embedding dim $($vec.Count) != configured $Dim" }
    if ([math]::Abs($norm - 1.0) -gt 0.01) { $problems += "embedding not unit length (norm $([math]::Round($norm,4)))" }
    if ($vec.Count -eq $Dim -and [math]::Abs($norm - 1.0) -le 0.01) {
      Write-Host "embed    : OK dimension and normalisation" -ForegroundColor Green
    }
  } catch { Write-Host "embed    : FAILED $($_.Exception.Message)" -ForegroundColor Red; $problems += "embeddings call failed" }

  # retrieval
  try {
    $body = @{ project = $Project; query = 'what port does the fast worker use'; top_k = 3 } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$base/v1/retrieve" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120
    Write-Host "retrieve : $($r.results.Count) results in $($r.seconds)s"
    if ($r.results.Count -eq 0) { $warnings += "retrieve returned no results" }
    else {
      $top = $r.results[0]
      Write-Host "           top: $($top.source_path):$($top.start_line) score=$([math]::Round($top.score,4))" -ForegroundColor Green
    }
  } catch { Write-Host "retrieve : FAILED $($_.Exception.Message)" -ForegroundColor Red; $problems += "retrieve call failed" }
}

$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listeners) {
  $addrs = @($listeners | ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
  $nonLoop = @($addrs | Where-Object { $_ -notin @('127.0.0.1', '::1') })
  Write-Host "bind     : $($addrs -join ', ')"
  if ($nonLoop.Count) { Write-Host "bind     : EXPOSED on $($nonLoop -join ', ')" -ForegroundColor Red; $problems += "non-loopback bind" }
  else { Write-Host "bind     : OK loopback only" -ForegroundColor Green }
} else { $problems += "no listener on port $Port" }

Write-Host ""
if (-not $problems.Count -and -not $warnings.Count) { Write-Host "RESULT: HEALTHY" -ForegroundColor Green; exit 0 }
if (-not $problems.Count) {
  Write-Host "RESULT: DEGRADED" -ForegroundColor Yellow
  $warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
  exit 1
}
Write-Host "RESULT: UNHEALTHY" -ForegroundColor Red
$problems | ForEach-Object { Write-Host "  ! $_" -ForegroundColor Red }
$warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
exit 1
