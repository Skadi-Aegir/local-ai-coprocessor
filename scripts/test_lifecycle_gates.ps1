<#
.SYNOPSIS
  test_lifecycle_gates.ps1 - regression tests for the local-fast lifecycle scripts.

.DESCRIPTION
  Verifies every refusal path and edge case in start/stop/check-local-fast WITHOUT
  needing the GPU or a running model, so the safety gates are proven rather than
  assumed.

  Three testing traps this harness is built to avoid (all three bit us during
  development):

  1. Do NOT pipe the scripts into Out-Null. $LASTEXITCODE then reflects the
     pipeline, not the script, and an unhandled terminating error still looks like
     success. That produced a "passing" test for stop-local-fast while it was
     actually throwing a ParameterBindingException on a corrupt pid file.
  2. Do NOT use Start-Process -Wait on start-local-fast. It launches llama-server
     as a detached child, and -Wait blocks on that grandchild too, so the harness
     hangs forever even after the script itself has exited.
  3. Use a unique stdout/stderr file per test. A shared file gets overwritten and
     you cannot tell which test produced which output.

  Each start test therefore runs under a hard timeout. If the script launched a
  server (the gate FAILED to refuse), the timeout expires, we detect the stray
  process, kill it, and record a failure. Refusal paths finish in well under a
  second, so a timeout unambiguously means "it tried to start".

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\test_lifecycle_gates.ps1
#>

$ErrorActionPreference = 'Continue'
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Cfg        = "$(Split-Path -Parent $PSScriptRoot)\config\local-fast.yaml"
$PidFile    = "$(Split-Path -Parent $PSScriptRoot)\logs\local-fast.pid"
$TmpDir     = "$(Split-Path -Parent $PSScriptRoot)\logs\_gate_tests"
$ServerExe  = 'llama-server'

New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
Get-ChildItem $TmpDir -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$Script:Results = @()

function Clear-Strays {
  <# Kill anything a leaking test may have started, and clear the pid file. #>
  $strays = @(Get-Process -Name $ServerExe -ErrorAction SilentlyContinue)
  foreach ($s in $strays) { Stop-Process -Id $s.Id -Force -ErrorAction SilentlyContinue }
  if ($strays.Count) {
    Write-Host "      (cleaned up $($strays.Count) stray $ServerExe process)" -ForegroundColor Yellow
    Start-Sleep -Milliseconds 800
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  return $strays.Count
}

function Invoke-Script {
  <#
    Run a lifecycle script under a hard timeout, capturing the exit code and stderr.
    Returns ExitCode = -1 when it had to be killed (i.e. it did not refuse).
  #>
  param(
    [string]$Path,
    [string[]]$Arguments = @(),
    [int]$TimeoutSeconds = 20,
    [string]$Tag = 'test'
  )
  $out = Join-Path $TmpDir "$Tag.out.txt"
  $err = Join-Path $TmpDir "$Tag.err.txt"

  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = Join-Path $PSHOME 'powershell.exe'
  $q = ($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
  $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Path`" $q"
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true

  $p = [System.Diagnostics.Process]::Start($psi)
  $finished = $p.WaitForExit($TimeoutSeconds * 1000)

  if (-not $finished) {
    # it did not refuse: kill the process tree (script + any detached server)
    try { $p.Kill() } catch { }
    Clear-Strays | Out-Null
    return [pscustomobject]@{ ExitCode = -1; Stderr = ''; Stdout = ''; TimedOut = $true }
  }

  $stdout = $p.StandardOutput.ReadToEnd()
  $stderr = $p.StandardError.ReadToEnd()
  if (Test-Path $out) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
  if (Test-Path $err) { Remove-Item $err -Force -ErrorAction SilentlyContinue }
  $stdout | Set-Content $out -Encoding UTF8
  $stderr | Set-Content $err -Encoding UTF8
  return [pscustomobject]@{
    ExitCode = $p.ExitCode; Stderr = $stderr.Trim(); Stdout = $stdout.Trim(); TimedOut = $false
  }
}

function Test-Gate {
  param([string]$Name, [int]$Expect, $Result, [switch]$ExpectClean)
  $codeOk = ($Result.ExitCode -eq $Expect)
  $cleanOk = $true
  if ($ExpectClean) { $cleanOk = [string]::IsNullOrWhiteSpace($Result.Stderr) }
  $pass = $codeOk -and $cleanOk
  $Script:Results += [pscustomobject]@{
    Test = $Name; Expect = $Expect; Actual = $Result.ExitCode
    StderrClean = [string]::IsNullOrWhiteSpace($Result.Stderr); Pass = $pass
  }
  $detail = ''
  if ($Result.TimedOut) { $detail = '  TIMED OUT - gate did not refuse' }
  elseif (-not [string]::IsNullOrWhiteSpace($Result.Stderr)) {
    $detail = '  STDERR: ' + ($Result.Stderr -split "`r?`n")[0].Trim()
  }
  Write-Host ("{0} {1,-46} exit={2} (want {3}){4}" -f `
    $(if ($pass) { 'OK  ' } else { 'FAIL' }), $Name, $Result.ExitCode, $Expect, $detail)
}

Write-Host "=== start-local-fast.ps1 refusal paths ===" -ForegroundColor Cyan

Test-Gate 'unknown quant rejected' 2 `
  (Invoke-Script (Join-Path $ScriptsDir 'start-local-fast.ps1') @('-Quant', 'Q9_FAKE') -Tag 'unknownquant') -ExpectClean

$tmpVram = Join-Path $TmpDir 'gate_vram.yaml'
(Get-Content $Cfg -Raw) -replace 'min_free_vram_mb: 4096', 'min_free_vram_mb: 99000' |
  Set-Content $tmpVram -Encoding UTF8
Test-Gate 'VRAM gate refuses below threshold' 4 `
  (Invoke-Script (Join-Path $ScriptsDir 'start-local-fast.ps1') @('-Config', $tmpVram) -Tag 'vram') -ExpectClean

$tmpModel = Join-Path $TmpDir 'gate_model.yaml'
(Get-Content $Cfg -Raw) -replace [regex]::Escape('Qwen3.5-4B-Q4_K_M.gguf'), 'NOPE.gguf' |
  Set-Content $tmpModel -Encoding UTF8
Test-Gate 'missing model rejected' 2 `
  (Invoke-Script (Join-Path $ScriptsDir 'start-local-fast.ps1') @('-Config', $tmpModel) -Tag 'model') -ExpectClean

$tmpExe = Join-Path $TmpDir 'gate_exe.yaml'
(Get-Content $Cfg -Raw) -replace [regex]::Escape('llama-server.exe'), 'nope-server.exe' |
  Set-Content $tmpExe -Encoding UTF8
Test-Gate 'missing server binary rejected' 2 `
  (Invoke-Script (Join-Path $ScriptsDir 'start-local-fast.ps1') @('-Config', $tmpExe) -Tag 'exe') -ExpectClean

# busy port: hold 8111, the gate must refuse
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8111)
$listener.Start()
try {
  Test-Gate 'busy port rejected' 3 `
    (Invoke-Script (Join-Path $ScriptsDir 'start-local-fast.ps1') @() -Tag 'busyport') -ExpectClean
} finally { $listener.Stop() }
Clear-Strays | Out-Null

Write-Host "`n=== stop-local-fast.ps1 edge cases ===" -ForegroundColor Cyan

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
Test-Gate 'nothing running -> clean exit' 0 `
  (Invoke-Script (Join-Path $ScriptsDir 'stop-local-fast.ps1') -Tag 'stop_none') -ExpectClean

'not-a-pid' | Set-Content $PidFile -Encoding ASCII
Test-Gate 'corrupt pid file -> clean exit, no stderr' 0 `
  (Invoke-Script (Join-Path $ScriptsDir 'stop-local-fast.ps1') -Tag 'stop_corrupt') -ExpectClean
Write-Host ("{0} corrupt pid file removed" -f $(if (Test-Path $PidFile) { 'FAIL' } else { 'OK  ' }))

'' | Set-Content $PidFile -Encoding ASCII
Test-Gate 'empty pid file -> clean exit' 0 `
  (Invoke-Script (Join-Path $ScriptsDir 'stop-local-fast.ps1') -Tag 'stop_empty') -ExpectClean

"`n  999999  `n" | Set-Content $PidFile -Encoding ASCII
Test-Gate 'whitespace-padded dead pid -> clean exit' 0 `
  (Invoke-Script (Join-Path $ScriptsDir 'stop-local-fast.ps1') -Tag 'stop_ws') -ExpectClean
Remove-Item $PidFile -Force -ErrorAction SilentlyContinue

Write-Host "`n=== check-local-fast.ps1 edge cases ===" -ForegroundColor Cyan
Test-Gate 'check reports DOWN when stopped' 2 `
  (Invoke-Script (Join-Path $ScriptsDir 'check-local-fast.ps1') -Tag 'check_down')

Write-Host "`n===== SUMMARY ====="
$Script:Results | Format-Table -AutoSize
$fails = @($Script:Results | Where-Object { -not $_.Pass })
Write-Host "passed: $($Script:Results.Count - $fails.Count)/$($Script:Results.Count)"
$Script:Results | Format-Table -AutoSize | Out-String | Set-Content (Join-Path $TmpDir 'summary.txt') -Encoding UTF8
if ($fails.Count) {
  Write-Host "FAILURES:" -ForegroundColor Red
  $fails | Format-Table -AutoSize
  exit 1
}
Write-Host "all lifecycle gate tests passed" -ForegroundColor Green
exit 0
