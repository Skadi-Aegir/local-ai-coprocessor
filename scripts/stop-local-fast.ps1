<#
.SYNOPSIS
  stop-local-fast.ps1 - stop the local-fast worker cleanly.

.DESCRIPTION
  Reads the PID file and terminates the server, then waits for the port to be
  released and reports the resulting VRAM state. Safe to run when nothing is
  running (reports and exits 0).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\stop-local-fast.ps1
#>
[CmdletBinding()]
param(
  [string]$Config = "$(Split-Path -Parent $PSScriptRoot)\config\local-fast.yaml",
  [int]$TimeoutSeconds = 30
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
function Write-Step($m) { Write-Host "[stop-local-fast] $m" }

$PidFile = Cfg 'paths.pid_file'
$Port    = [int](Cfg 'runtime.port' 8111)

$target = $null
if (Test-Path $PidFile) {
  $raw = (Get-Content $PidFile -Raw -ErrorAction SilentlyContinue)
  if ($raw) {
    $raw = $raw.Trim()
    # The pid file can be truncated or corrupted (crash mid-write, manual edit).
    # Validate it is a positive integer BEFORE handing it to Get-Process, which
    # throws a parameter-binding error on non-numeric input. That error is not
    # catchable via $ErrorActionPreference and would abort the script with a
    # stack trace instead of a clean "nothing to stop".
    $parsedPid = 0
    if ([int]::TryParse($raw, [ref]$parsedPid) -and $parsedPid -gt 0) {
      $target = $parsedPid
    } else {
      Write-Step "pid file contains a non-numeric value ('$raw'); ignoring it"
    }
  }
}

# fall back to whatever owns the port, in case the pid file was lost
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
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  exit 0
}

$proc = if ($target) { Get-Process -Id $target -ErrorAction SilentlyContinue } else { $null }
if (-not $proc) {
  if ($target) { Write-Step "pid $target no longer exists" }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  exit 0
}

Write-Step "stopping pid $target ($($proc.ProcessName))"
Stop-Process -Id $target -Force -ErrorAction SilentlyContinue

for ($i = 0; $i -lt $TimeoutSeconds; $i++) {
  Start-Sleep -Seconds 1
  if (-not (Get-Process -Id $target -ErrorAction SilentlyContinue)) { break }
}
$still = Get-Process -Id $target -ErrorAction SilentlyContinue
if ($still) { Write-Step "WARNING: pid $target still alive after ${TimeoutSeconds}s" }
else        { Write-Step "process exited after ~$i s" }

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue

# confirm the port is free again
for ($i = 0; $i -lt 10; $i++) {
  $busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
  if (-not $busy) { break }
  Start-Sleep -Milliseconds 500
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
  Write-Step "WARNING: port $Port still listening"
  exit 7
}
Write-Step "port $Port released"

try {
  $v = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  Write-Step "vram        : $v"
} catch { }
Write-Step "STOPPED"
exit 0
