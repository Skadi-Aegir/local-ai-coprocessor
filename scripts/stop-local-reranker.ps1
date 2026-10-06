<#
.SYNOPSIS
  stop-local-reranker.ps1 - stop the local-reranker service cleanly.

.DESCRIPTION
  Reads the pid file, terminates the service, waits for the port to be released and
  reports the resulting VRAM. Safe to run when nothing is running.

  The pid file is validated as numeric before being handed to Get-Process: Get-Process
  throws a parameter-binding error on non-numeric input that $ErrorActionPreference='Stop'
  does not catch, which previously crashed the Phase 2 stop script.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\stop-local-reranker.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-reranker.yaml",
  [int]$TimeoutSeconds = 30
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
function Write-Step($m) { Write-Host "[stop-local-reranker] $m" }

$pidFile = Cfg 'paths.pid_file'
$Port    = [int](Cfg 'runtime.port' 8114)

$target = $null
if (Test-Path $pidFile) {
  $raw = (Get-Content $pidFile -Raw -ErrorAction SilentlyContinue)
  if ($raw) {
    $raw = $raw.Trim()
    $parsed = 0
    if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -gt 0) { $target = $parsed }
    else { Write-Step "pid file contains a non-numeric value ('$raw'); ignoring it" }
  }
}

if (-not $target -or -not (Get-Process -Id $target -ErrorAction SilentlyContinue)) {
  $owner = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty OwningProcess
  if ($owner) {
    $target = $owner
    Write-Step "pid file stale/absent; found listener on port $Port (pid $target)"
  }
}

if (-not $target) {
  Write-Step "not running (no pid file, nothing listening on $Port)"
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  exit 0
}

$proc = Get-Process -Id $target -ErrorAction SilentlyContinue
if (-not $proc) {
  Write-Step "pid $target no longer exists"
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  exit 0
}

Write-Step "stopping pid $target ($($proc.ProcessName))"
Stop-Process -Id $target -Force -ErrorAction SilentlyContinue
for ($i = 0; $i -lt $TimeoutSeconds; $i++) {
  Start-Sleep -Seconds 1
  if (-not (Get-Process -Id $target -ErrorAction SilentlyContinue)) { break }
}
if (Get-Process -Id $target -ErrorAction SilentlyContinue) {
  Write-Step "WARNING: pid $target still alive after ${TimeoutSeconds}s"
} else {
  Write-Step "process exited after ~$i s"
}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

for ($i = 0; $i -lt 10; $i++) {
  if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Milliseconds 500
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
  Write-Step "WARNING: port $Port still listening"; exit 7
}
Write-Step "port $Port released"

try { Write-Step "vram      : $(& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)" } catch { }
Write-Step "STOPPED"
exit 0
