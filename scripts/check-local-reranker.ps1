<#
.SYNOPSIS
  check-local-reranker.ps1 - health and capability check for local-reranker.

.DESCRIPTION
  Verifies, in order:
    1. process alive + pid file
    2. GET /health
    3. GET /v1/models serves the `local-reranker` alias and the official scoring contract
    4. recorded model sha256 still matches the weights on disk
    5. POST /v1/rerank ORDERS a known relevant document above a known irrelevant one,
       which is the property the whole phase depends on
    6. returned indices refer to positions in the submitted array
    7. top_n truncates correctly
    8. listening address is loopback only
  Exit codes: 0 healthy, 1 degraded/unhealthy, 2 down.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\check-local-reranker.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-reranker.yaml",
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

$Port    = [int](Cfg 'runtime.port' 8114)
$Host_   = Cfg 'runtime.host' '127.0.0.1'
$Alias   = Cfg 'model' 'local-reranker'
$pidFile = Cfg 'paths.pid_file'
$ModelPath = Cfg 'model_path'
$wantHash  = Cfg 'model_sha256'

$base = "http://${Host_}:$Port"
$problems = @(); $warnings = @()

Write-Host "=== check-local-reranker ===" -ForegroundColor Cyan
Write-Host "endpoint : $base"
Write-Host "alias    : $Alias"

$pidVal = $null
if (Test-Path $pidFile) { $pidVal = (Get-Content $pidFile -Raw).Trim() }
$proc = $null
$parsed = 0
if ($pidVal -and [int]::TryParse($pidVal, [ref]$parsed)) {
  $proc = Get-Process -Id $parsed -ErrorAction SilentlyContinue
}
if ($proc) { Write-Host "process  : OK pid=$pidVal ($($proc.ProcessName))" -ForegroundColor Green }
else { Write-Host "process  : NOT RUNNING" -ForegroundColor Red; $problems += "service process not running" }

try {
  $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5
  if ($h.status -eq 'ok') {
    Write-Host "health   : OK  device=$($h.device) dtype=$($h.dtype) max_length=$($h.max_length)" -ForegroundColor Green
  } else { $problems += "health not ok" }
} catch {
  Write-Host "health   : UNREACHABLE ($($_.Exception.Message))" -ForegroundColor Red
  Write-Host "`nRESULT: DOWN" -ForegroundColor Red
  exit 2
}

try {
  $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
  $m = $models.data[0]
  if ($m.id -eq $Alias) { Write-Host "models   : OK '$Alias'" -ForegroundColor Green }
  else { $problems += "alias mismatch: $($m.id)" }
  Write-Host "scoring  : $($m.meta.scoring)"
  Write-Host "tokens   : yes=$($m.meta.yes_token_id) no=$($m.meta.no_token_id) body_budget=$($m.meta.body_budget)"
} catch { $problems += "could not read /v1/models" }

# recorded hash must still match the weights actually on disk
if (-not $Quick -and $wantHash -and $ModelPath) {
  $w = Join-Path $ModelPath 'model.safetensors'
  if (Test-Path $w) {
    $got = (Get-FileHash $w -Algorithm SHA256).Hash.ToLower()
    if ($got -eq $wantHash.ToLower()) { Write-Host "weights  : sha256 OK" -ForegroundColor Green }
    else { Write-Host "weights  : sha256 MISMATCH" -ForegroundColor Red; $problems += "weights hash mismatch" }
  } else { $warnings += "weights file not found for hashing" }
}

if (-not $Quick) {
  # The capability that matters: does it rank a relevant document above an irrelevant one?
  try {
    $body = @{
      model = $Alias
      query = 'What is the capital of China?'
      documents = @(
        'Gravity is a force that attracts two bodies towards each other.',
        'The Pacific Ocean is the largest and deepest of Earth''s oceanic divisions.',
        'The capital of China is Beijing.'
      )
      top_n = 3
    } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$base/v1/rerank" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 180
    Write-Host "rerank   : $($r.count) results in $($r.seconds)s (instr=$($r.instruction_type))"
    $top = $r.results[0]
    if ($top.index -eq 2) {
      Write-Host "ordering : OK relevant document (index 2) ranked first, score $([math]::Round($top.score,4))" -ForegroundColor Green
    } else {
      Write-Host "ordering : WRONG top index $($top.index) (expected 2)" -ForegroundColor Red
      $problems += "reranker failed to rank the relevant document first"
    }
    $idxs = @($r.results | ForEach-Object { $_.index })
    if (($idxs | Sort-Object) -join ',' -eq '0,1,2') { Write-Host "indices  : OK permutation of the input array" -ForegroundColor Green }
    else { $problems += "indices are not a permutation: $($idxs -join ',')" }
    $scores = @($r.results | ForEach-Object { $_.score })
    $desc = $true
    for ($i = 1; $i -lt $scores.Count; $i++) { if ($scores[$i] -gt $scores[$i-1]) { $desc = $false } }
    if ($desc) { Write-Host "scores   : OK non-increasing" -ForegroundColor Green }
    else { $problems += "scores not in descending order" }
  } catch {
    Write-Host "rerank   : FAILED $($_.Exception.Message)" -ForegroundColor Red
    $problems += "rerank call failed"
  }

  # top_n must truncate
  try {
    $body = @{ query = 'capital of China'; documents = @('a','b','c','d','e'); top_n = 2 } | ConvertTo-Json
    $r2 = Invoke-RestMethod -Uri "$base/v1/rerank" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 180
    if ($r2.count -eq 2) { Write-Host "top_n    : OK truncated 5 -> 2" -ForegroundColor Green }
    else { $problems += "top_n did not truncate (got $($r2.count))" }
  } catch { $problems += "top_n call failed" }

  # invalid requests must be rejected, not silently accepted
  try {
    $body = @{ query = ''; documents = @('x') } | ConvertTo-Json
    Invoke-RestMethod -Uri "$base/v1/rerank" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 30 | Out-Null
    $problems += "empty query was accepted"
  } catch { Write-Host "validation: OK empty query rejected" -ForegroundColor Green }
}

$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listeners) {
  $addrs = @($listeners | ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
  $nonLoop = @($addrs | Where-Object { $_ -notin @('127.0.0.1', '::1') })
  if ($nonLoop.Count) { Write-Host "bind     : EXPOSED on $($nonLoop -join ', ')" -ForegroundColor Red; $problems += "non-loopback bind" }
  else { Write-Host "bind     : OK loopback only ($($addrs -join ', '))" -ForegroundColor Green }
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
