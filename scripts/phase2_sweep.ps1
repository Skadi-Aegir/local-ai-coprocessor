# =============================================================================
# phase2_sweep.ps1 - Phase 2 measurement driver
#
# Runs the quantisation sweep and the context-length sweep against llama-bench
# using the Phase 1 build, sampling VRAM throughout. Results land in
# benchmarks/local-fast/results/.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\phase2_sweep.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\phase2_sweep.ps1 -Only quant
# =============================================================================

param(
  [ValidateSet('all','quant','context')][string]$Only = 'all',
  [int]$Reps = 3,
  [int]$Threads = 12,
  [int]$PromptTokens = 512,
  [int]$GenTokens = 128
)

$ErrorActionPreference = 'Stop'
$LlamaDir  = "$(Split-Path -Parent $PSScriptRoot)\tools\llama.cpp\b11254-cuda13.4"
$Bench     = Join-Path $LlamaDir 'llama-bench.exe'
$Models    = "$(Split-Path -Parent $PSScriptRoot)\models\llm"
$Results   = "$(Split-Path -Parent $PSScriptRoot)\benchmarks\local-fast\results"
$Logs      = "$(Split-Path -Parent $PSScriptRoot)\logs"
$Watch     = "$(Split-Path -Parent $PSScriptRoot)\scripts\vram_watch.ps1"

New-Item -ItemType Directory -Force -Path $Results | Out-Null
$csvOut = Join-Path $Results 'phase2_sweep.csv'
'test,quant_or_ctx,param,file_mb,prompt_tps,gen_tps,ttft_ms,peak_vram_mb,mean_util,peak_power_w' |
  Set-Content $csvOut -Encoding UTF8

function Get-Vram([string]$which) {
  $v = & nvidia-smi --query-gpu=memory.$which --format=csv,noheader,nounits
  return [int]($v -replace '\s','')
}

function Invoke-Bench {
  param([string]$Model, [int]$Ngl, [int]$Ctx, [int]$Secs)
  $safe = [System.IO.Path]::GetFileNameWithoutExtension($Model)
  $vramCsv = Join-Path $Results ("vram_{0}_ngl{1}_ctx{2}.csv" -f $safe, $Ngl, $Ctx)

  $argList = @('-m', $Model, '-ngl', $Ngl, '-c', $Ctx, '-t', $Threads,
               '-p', $PromptTokens, '-n', $GenTokens, '-r', $Reps, '-o', 'csv')
  $proc = Start-Process -FilePath $Bench -ArgumentList $argList -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $Logs 'sweep_bench.out') `
            -RedirectStandardError  (Join-Path $Logs 'sweep_bench.err')

  # sample VRAM until the run finishes
  & $Watch -Seconds $Secs -IntervalMs 300 -OutFile $vramCsv | Out-Null
  $proc.WaitForExit(600000) | Out-Null
  if (-not $proc.HasExited) { $proc.Kill() }

  $raw = Get-Content (Join-Path $Logs 'sweep_bench.out') -ErrorAction SilentlyContinue
  $rows = $raw | Where-Object { $_ -match '^\s*"' } | Select-Object -Skip 0
  $pp = $null; $tg = $null
  foreach ($r in $rows) {
    $f = $r -split ','
    $nPrompt = ($f[-4] -replace '"','').Trim()
    $ts      = ($f[-2] -replace '"','').Trim()
    if ($nPrompt -eq "$PromptTokens") { $pp = [double]$ts }
    elseif ($nPrompt -eq '0')          { $tg = [double]$ts }
  }

  $peak = $null; $util = $null; $pwr = $null
  if (Test-Path $vramCsv) {
    $d = Import-Csv $vramCsv
    if ($d) {
      $peak = ($d | Measure-Object mem_used_mb -Maximum).Maximum
      $util = [math]::Round(($d | Measure-Object gpu_util -Average).Average,1)
      $pwr  = ($d | Measure-Object power_w -Maximum).Maximum
    }
  }
  return [pscustomobject]@{ pp=$pp; tg=$tg; peak=$peak; util=$util; pwr=$pwr }
}

# ---------------------------------------------------------------- quant sweep
if ($Only -in @('all','quant')) {
  Write-Host "`n===== QUANT SWEEP (ctx 8192) =====" -ForegroundColor Cyan
  foreach ($q in @('Q4_K_M','Q5_K_M','Q6_K')) {
    $model = Join-Path $Models "Qwen3.5-4B-$q.gguf"
    if (-not (Test-Path $model)) { Write-Host "skip $q (missing)"; continue }
    $sizeMb = [math]::Round((Get-Item $model).Length/1MB,1)
    Write-Host "--- $q  ($sizeMb MB) ---"
    $idle = Get-Vram 'used'
    $r = Invoke-Bench -Model $model -Ngl 99 -Ctx 8192 -Secs 75
    $delta = if ($r.peak) { $r.peak - $idle } else { '' }
    Write-Host ("    prompt {0:N0} t/s | gen {1:N1} t/s | peak VRAM {2} MiB (delta {3}) | util {4}% | {5} W" -f $r.pp, $r.tg, $r.peak, $delta, $r.util, $r.pwr)
    "quant,$q,ngl99_c8192,$sizeMb,$([math]::Round($r.pp,1)),$([math]::Round($r.tg,1)),,$($r.peak),$($r.util),$($r.pwr)" |
      Add-Content $csvOut -Encoding UTF8
  }
}

# ---------------------------------------------------------------- context sweep
if ($Only -in @('all','context')) {
  Write-Host "`n===== CONTEXT SWEEP (Q4_K_M) =====" -ForegroundColor Cyan
  $model = Join-Path $Models 'Qwen3.5-4B-Q4_K_M.gguf'
  $sizeMb = [math]::Round((Get-Item $model).Length/1MB,1)
  foreach ($ctx in @(4096, 8192, 16384)) {
    Write-Host "--- ctx $ctx ---"
    $idle = Get-Vram 'used'
    $r = Invoke-Bench -Model $model -Ngl 99 -Ctx $ctx -Secs 75
    $delta = if ($r.peak) { $r.peak - $idle } else { '' }
    Write-Host ("    prompt {0:N0} t/s | gen {1:N1} t/s | peak VRAM {2} MiB (delta {3}) | util {4}% | {5} W" -f $r.pp, $r.tg, $r.peak, $delta, $r.util, $r.pwr)
    "context,$ctx,ngl99,$sizeMb,$([math]::Round($r.pp,1)),$([math]::Round($r.tg,1)),,$($r.peak),$($r.util),$($r.pwr)" |
      Add-Content $csvOut -Encoding UTF8
  }
}

Write-Host "`n===== SWEEP COMPLETE =====" -ForegroundColor Green
Get-Content $csvOut | Format-Table -AutoSize
