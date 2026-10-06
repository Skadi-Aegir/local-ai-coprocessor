# llama_bench_suite.ps1 - Phase 1 acceptance benchmark for the Local AI Coprocessor.
#
# Purpose:
#   * prove CUDA GPU offload is actually used (not silent CPU fallback)
#   * measure CPU-only vs GPU-offloaded throughput
#   * record VRAM / utilisation while loaded
#
# Usage:
#   pwsh -File llama_bench_suite.ps1 -Model <repo>\models\llm\Qwen3-0.6B-Q8_0.gguf

param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$LlamaDir = "$(Split-Path -Parent $PSScriptRoot)\tools\llama.cpp\b11254-cuda13.4",
  [string]$ResultDir = "$(Split-Path -Parent $PSScriptRoot)\benchmarks",
  [int]$GpuLayersAll = 99,
  [int]$GpuLayersSplit = 14,
  [string]$Tag = 'qwen3-0.6b-q8_0',
  [int]$Threads = 12
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$bench = Join-Path $LlamaDir 'llama-bench.exe'
$cli   = Join-Path $LlamaDir 'llama-cli.exe'
$watch = "$(Split-Path -Parent $PSScriptRoot)\scripts\vram_watch.ps1"

if (-not (Test-Path $Model)) { throw "model not found: $Model" }
if (-not (Test-Path $bench)) { throw "llama-bench not found: $bench" }

function Get-IdleVram {
  $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  return [int]($v -replace '\s','')
}

$modelMB = [math]::Round((Get-Item $Model).Length / 1MB, 1)
Write-Output "=============================================================="
Write-Output " llama.cpp Phase 1 acceptance benchmark"
Write-Output "=============================================================="
Write-Output "model            : $Model"
Write-Output "model size       : $modelMB MB"
Write-Output "llama dir        : $LlamaDir"
Write-Output "threads          : $Threads"
Write-Output ""

# ---------------------------------------------------------------- device + harness
Write-Output "--- device enumeration -------------------------------------------------"
& (Join-Path $LlamaDir 'llama-cli.exe') --list-devices 2>&1
Write-Output ""

$idleVram = Get-IdleVram
Write-Output "idle VRAM before load : $idleVram MiB"
Write-Output ""

# ---------------------------------------------------------------- benchmark runs
# -ngl 0        -> pure CPU (reference baseline)
# -ngl 14       -> partial offload (hybrid)
# -ngl 99       -> full offload (all layers on GPU)
$runs = @(
  @{ label = 'gpu_full_99';     ngl = $GpuLayersAll;   secs = 100 },
  @{ label = 'gpu_hybrid_14';   ngl = $GpuLayersSplit; secs = 130 },
  @{ label = 'cpu_only';        ngl = 0;               secs = 260 }
)

$csvFiles = @()
foreach ($r in $runs) {
  $csv = Join-Path $ResultDir ("llama-bench_{0}_{1}.csv" -f $Tag, $r.label)
  Remove-Item $csv -ErrorAction SilentlyContinue

  Write-Output "--- run: $($r.label)  (-ngl $($r.ngl), -t $Threads) ------------------"

  # start a VRAM/utilisation sampler covering this run
  $samples = Join-Path $ResultDir ("vram_{0}_{1}.csv" -f $Tag, $r.label)
  $job = Start-Job -ScriptBlock {
    param($w, $sec, $out)
    & (Join-Path $PSHOME 'powershell.exe') -NoProfile -ExecutionPolicy Bypass -File $w -Seconds $sec -IntervalMs 300 -OutFile $out
  } -ArgumentList $watch, $r.secs, $samples

  # llama-bench prints its metrics table to stdout; capture verbatim for the record
  & $bench -m $Model -ngl $r.ngl -t $Threads -p 512 -n 128 -r 3 -o csv 2>$null |
    Tee-Object -FilePath $csv | ForEach-Object { Write-Output "    $_" }

  Stop-Job $job -ErrorAction SilentlyContinue
  Receive-Job $job -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "    $_" }
  Remove-Job $job -Force -ErrorAction SilentlyContinue

  Write-Output ""
  if (Test-Path $csv) { $csvFiles += $csv }
}

# ---------------------------------------------------------------- summarise
Write-Output "--- summary ------------------------------------------------------------"
foreach ($csv in $csvFiles) {
  Write-Output ""
  Write-Output "### $(Split-Path -Leaf $csv)"
  $data = Import-Csv $csv
  $data | Format-Table -AutoSize | Out-String -Width 250 | Write-Output
}

Write-Output ""
Write-Output "--- VRAM peaks per configuration --------------------------------------"
Get-ChildItem $ResultDir -Filter "vram_$Tag*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
  $d = Import-Csv $_.FullName
  if ($d) {
    $peak = ($d | Measure-Object mem_used_mb -Maximum).Maximum
    $min  = ($d | Measure-Object mem_used_mb -Minimum).Minimum
    $util = ($d | Measure-Object gpu_util -Maximum).Maximum
    $avgU = [math]::Round(($d | Measure-Object gpu_util -Average).Average,1)
    $pwr  = ($d | Measure-Object power_w -Maximum).Maximum
    [pscustomobject]@{
      config     = $_.BaseName
      min_mem_MB = $min
      peak_mem_MB= $peak
      delta_MB   = $peak - $idleVram
      peak_util  = $util
      avg_util   = $avgU
      peak_W     = $pwr
    }
  }
} | Format-Table -AutoSize

Write-Output ""
Write-Output "idle baseline VRAM was $idleVram MiB"
Write-Output "artifacts in $ResultDir"
