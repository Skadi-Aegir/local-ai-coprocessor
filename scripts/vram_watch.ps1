# vram_watch.ps1 - sample GPU VRAM/utilisation while a workload runs.
# Phase 1/2 helper for the Local AI Coprocessor project.
#
# Writes each sample to the CSV immediately rather than buffering until the end.
# That matters: an earlier version accumulated rows and only wrote on completion, so
# when the watcher was stopped early (or the parent was killed) the whole series was
# lost and a stability run had no VRAM evidence at all.
#
# Usage: pwsh -File vram_watch.ps1 -Seconds 30 -IntervalMs 500 -OutFile logs\vram.csv

param(
  [int]$Seconds = 30,
  [int]$IntervalMs = 500,
  [string]$OutFile = ''
)

$deadline = (Get-Date).AddSeconds($Seconds)
$first = $true
$count = 0
$peakMem = 0; $peakUtil = 0; $sumUtil = 0; $peakPwr = 0.0; $peakClk = 0
$writer = $null

if ($OutFile) {
  $dir = Split-Path -Parent $OutFile
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  # truncate and write the header up front so the file is valid even if we are killed
  'time,gpu_util,mem_used_mb,mem_total_mb,temp_c,power_w,sm_clock_mhz' |
    Set-Content -Path $OutFile -Encoding UTF8
}

while ((Get-Date) -lt $deadline) {
  $line = & nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>$null
  if ($line) {
    $p = $line -split ',\s*'
    $util = [int]$p[1]; $mem = [int]$p[2]; $pwr = [double]$p[5]; $clk = [int]$p[6]
    $count++
    if ($mem -gt $peakMem) { $peakMem = $mem }
    if ($util -gt $peakUtil) { $peakUtil = $util }
    if ($pwr -gt $peakPwr) { $peakPwr = $pwr }
    if ($clk -gt $peakClk) { $peakClk = $clk }
    $sumUtil += $util

    if ($OutFile) {
      # append immediately, flushing every sample
      "$($p[0]),$util,$mem,$([int]$p[3]),$([int]$p[4]),$pwr,$clk" |
        Add-Content -Path $OutFile -Encoding UTF8
    }
  }
  Start-Sleep -Milliseconds $IntervalMs
}

if ($count -gt 0) {
  Write-Output "samples            : $count"
  Write-Output "peak VRAM used     : $peakMem MiB"
  Write-Output "peak GPU util      : $peakUtil %"
  Write-Output "avg GPU util       : $([math]::Round($sumUtil/$count,1)) %"
  Write-Output "peak power         : $peakPwr W"
  Write-Output "peak SM clock      : $peakClk MHz"
  if ($OutFile) { Write-Output "raw samples        : $OutFile" }
} else {
  Write-Output "no samples collected"
}
