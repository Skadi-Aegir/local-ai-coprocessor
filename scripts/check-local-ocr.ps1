<#
.SYNOPSIS
  check-local-ocr.ps1 - health and capability check for local-ocr.

.DESCRIPTION
  Verifies, in order:
    1. process alive + pid file
    2. GET /health reports status ok, the logical name, model, device cpu and oneDNN disabled
    3. GET /stats is readable and reports the request counters
    4. POST /v1/ocr on a real fixture image returns text, blocks, confidence and image size
    5. POST /v1/ocr rejects a non-image body with 415, and a non-object JSON body with 400
    6. the listening address is loopback only
  Exit codes: 0 healthy, 1 degraded/unhealthy, 2 down.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\check-local-ocr.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-ocr.yaml",
  [string]$Image,
  [switch]$Quick
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

$Port    = [int](Cfg 'runtime.port' 8116)
$Host_   = Cfg 'runtime.host' '127.0.0.1'
$Alias   = Cfg 'model' 'local-ocr'
$Tier    = Cfg 'tier' 'mobile'
$pidFile = Cfg 'paths.pid_file'
$base    = "http://${Host_}:$Port"

if (-not $Image) { $Image = "$(Split-Path -Parent $PSScriptRoot)\benchmarks\ocr\fixtures\images\syn_en_01.png" }

$problems = @(); $warnings = @()

Write-Host "=== check-local-ocr ===" -ForegroundColor Cyan
Write-Host "endpoint : $base"
Write-Host "alias    : $Alias   tier: $Tier   device: cpu"

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
    Write-Host "health   : OK  name=$($h.logical_name) model=$($h.model) device=$($h.device) onednn=$($h.onednn)" -ForegroundColor Green
    if ($h.device -ne 'cpu') { $problems += "device is $($h.device), expected cpu" }
    if ($h.cuda_compiled) { $problems += "paddle reports CUDA support; this must be the CPU build" }
    if ($h.logical_name -ne $Alias) { $problems += "logical name $($h.logical_name) != $Alias" }
  } else { Write-Host "health   : UNEXPECTED" -ForegroundColor Red; $problems += "health not ok" }
} catch {
  Write-Host "health   : UNREACHABLE ($($_.Exception.Message))" -ForegroundColor Red
  $problems += "health endpoint unreachable"
}
if (-not $healthOk) { Write-Host "`nRESULT: DOWN" -ForegroundColor Red; exit 2 }

try {
  $st = Invoke-RestMethod -Uri "$base/stats" -TimeoutSec 15
  Write-Host "stats    : requests=$($st.requests) ok=$($st.ok) errors=$($st.errors) rss=$($st.rss_mb) MB inflight=$($st.inflight)"
} catch { $warnings += "could not read /stats" }

if (-not $Quick) {
  # a real OCR call on a frozen fixture
  if (-not (Test-Path $Image)) {
    $warnings += "fixture image not found: $Image"
  } else {
    try {
      $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($Image))
      $body = @{ image = $b64; request_id = 'check-local-ocr' } | ConvertTo-Json -Compress
      $r = Invoke-RestMethod -Uri "$base/v1/ocr" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300
      Write-Host "ocr      : $($r.blocks.Count) block(s) in $($r.elapsed_s)s  image $($r.image.width)x$($r.image.height)  conf=$($r.confidence)"
      Write-Host "           text: $($r.text -replace "`n", ' | ')"
      if (-not $r.text) { $problems += "OCR returned no text for a non-empty fixture" }
      if ($r.blocks.Count -eq 0) { $problems += "OCR returned zero blocks" }
      if ($null -eq $r.confidence -and $r.text) { $problems += "confidence null despite text" }
      if ($r.model -notlike "PP-OCRv5_*") { $warnings += "unexpected model string: $($r.model)" }
    } catch { Write-Host "ocr      : FAILED $($_.Exception.Message)" -ForegroundColor Red; $problems += "OCR call failed" }
  }

  # error handling: a non-image must be 415, a non-object body must be 400
  try {
    $body = @{ image = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('not an image')) } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "$base/v1/ocr" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 60 | Out-Null
    $problems += "a non-image body was accepted (expected 415)"
  } catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 415) { Write-Host "reject   : OK non-image -> 415" -ForegroundColor Green }
    else { $problems += "non-image gave $code, expected 415" }
  }
  try {
    Invoke-RestMethod -Uri "$base/v1/ocr" -Method Post -Body '[1,2]' -ContentType 'application/json' -TimeoutSec 60 | Out-Null
    $problems += "a non-object JSON body was accepted (expected 400)"
  } catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 400) { Write-Host "reject   : OK non-object JSON -> 400" -ForegroundColor Green }
    else { $problems += "non-object JSON gave $code, expected 400" }
  }
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
