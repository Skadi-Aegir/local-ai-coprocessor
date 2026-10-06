# =============================================================================
# coresidency.ps1 - Phase 3 core experiment
#
# Answers: should local-fast and local-smart be RESIDENT together on a 12 GB card,
# or does local-smart need EXCLUSIVE residency? Measures the three strategies:
#
#   Strategy 1  unload/reload        (one model resident at a time)
#   Strategy 2  dual resident, serialized inference
#   Strategy 3  dual resident, concurrent inference
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\coresidency.ps1 -Mode alone9b
#   powershell -ExecutionPolicy Bypass -File scripts\coresidency.ps1 -Mode both
#   powershell -ExecutionPolicy Bypass -File scripts\coresidency.ps1 -Mode reload
# =============================================================================

param(
  [ValidateSet('alone4b','alone9b','both','reload','concurrent')][string]$Mode = 'alone9b',
  [string]$Quant9B = 'Q4_K_M',
  [int]$Ctx9B = 8192,
  [int]$Ctx4B = 8192,
  [int]$Reps = 3
)

$ErrorActionPreference = 'Continue'
$Llama   = "$(Split-Path -Parent $PSScriptRoot)\tools\llama.cpp\b11254-cuda13.4"
$Bench   = Join-Path $Llama 'llama-bench.exe'
$Server  = Join-Path $Llama 'llama-server.exe'
$Models  = "$(Split-Path -Parent $PSScriptRoot)\models\llm"
$Results = "$(Split-Path -Parent $PSScriptRoot)\benchmarks\local-smart\results"
$Logs    = "$(Split-Path -Parent $PSScriptRoot)\logs"
New-Item -ItemType Directory -Force -Path $Results | Out-Null

$m4b = Join-Path $Models 'Qwen3.5-4B-Q4_K_M.gguf'
$m9b = Join-Path $Models "Qwen3.5-9B-$Quant9B.gguf"

function Vram([string]$which) {
  return [int]((& nvidia-smi --query-gpu=memory.$which --format=csv,noheader,nounits) -replace '\s','')
}
function PsLine { param([string]$tag) [pscustomobject]@{ tag=$tag; used=Vram 'used'; free=Vram 'free' } }

function BenchOne {
  param([string]$Model, [string]$Tag, [int]$Ctx = 0, [int]$Secs = 60)
  $args = @('-m', $Model, '-ngl', '99', '-t', '12', '-p', '512', '-n', '128', '-r', $Reps, '-o', 'csv')
  $of = Join-Path $Logs "co_$Tag.out"; $ef = Join-Path $Logs "co_$Tag.err"
  $vw = Join-Path $Results "vram_co_$Tag.csv"
  $p = Start-Process -FilePath $Bench -ArgumentList $args -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput $of -RedirectStandardError $ef
  & "$(Split-Path -Parent $PSScriptRoot)\scripts\vram_watch.ps1" -Seconds $Secs -IntervalMs 400 -OutFile $vw | Out-Null
  $p.WaitForExit(900000) | Out-Null
  if (-not $p.HasExited) { $p.Kill() }

  $rows = Get-Content $of -ErrorAction SilentlyContinue | Where-Object { $_.StartsWith('"') }
  $pp = $null; $tg = $null
  foreach ($r in $rows) {
    $f = $r -split ','
    $np = ($f[34] -replace '"','').Trim()
    $ts = [double](($f[40] -replace '"','').Trim())
    if ($np -ne '0') { $pp = $ts } else { $tg = $ts }
  }
  $peak = $null; $util = $null; $pwr = $null
  if (Test-Path $vw) {
    $d = Import-Csv $vw
    if ($d) {
      $peak = ($d | Measure-Object mem_used_mb -Maximum).Maximum
      $util = [math]::Round(($d | Measure-Object gpu_util -Average).Average,1)
      $pwr  = ($d | Measure-Object power_w -Maximum).Maximum
    }
  }
  return [pscustomobject]@{ tag=$Tag; prompt_tps=$pp; gen_tps=$tg; peak_vram=$peak; util=$util; power=$pwr }
}

function StartServer {
  param([string]$Model, [string]$Alias, [int]$Port, [int]$Ctx, [string]$Tag)
  $args = @('-m',$Model,'--alias',$Alias,'-ngl','99','-c',$Ctx,'-np','1','-t','12',
            '-b','2048','-ub','512','--flash-attn','auto','-ctk','f16','-ctv','f16',
            '--host','127.0.0.1','--port',$Port,'--cors-origins','localhost',
            '--no-cors-credentials','--no-webui','--log-jsonl',
            '--log-file',(Join-Path $Logs "$Tag.server.jsonl"))
  $p = Start-Process -FilePath $Server -ArgumentList $args -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput (Join-Path $Logs "$Tag.out") -RedirectStandardError (Join-Path $Logs "$Tag.err")
  for ($i=0; $i -lt 180; $i++) {
    Start-Sleep -Milliseconds 1000
    try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
          if ($h.status -eq 'ok') { return $p } } catch {}
    if ($p.HasExited) { return $null }
  }
  return $null
}

function StopAny {
  Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
}

function ChatProbe {
  param([int]$Port, [string]$Alias, [string]$Prompt, [int]$MaxTok = 160)
  $py = "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe"
  $code = @"
import json, sys, time, urllib.request
port, alias, prompt, mt = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
body = {"model": alias, "messages": [{"role":"user","content":prompt}],
        "max_tokens": mt, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=600))
dt = time.time() - t0
t = d.get("timings", {})
print(json.dumps({"wall_s": round(dt,3),
                  "prompt_tps": round(t.get("prompt_per_second",0),1),
                  "gen_tps": round(t.get("predicted_per_second",0),2),
                  "completion_tokens": t.get("predicted_n",0)}))
"@
  $tmp = Join-Path $Logs '_probe.py'
  Set-Content -Path $tmp -Value $code -Encoding UTF8
  $out = & $py $tmp $Port $Alias $Prompt $MaxTok 2>&1
  return ($out | Select-Object -Last 1)
}

$report = @()
$base = PsLine 'baseline_desktop'
Write-Host "baseline (no model loaded): used=$($base.used) free=$($base.free) MiB" -ForegroundColor Cyan

switch ($Mode) {

  'alone4b' {
    Write-Host "`n=== 4B alone ===" -ForegroundColor Cyan
    $r = BenchOne -Model $m4b -Tag 'alone_4b' -Secs 70
    $report += $r
    Write-Host ("  4B alone: prompt {0:N0} t/s gen {1:N1} t/s peak VRAM {2} MiB util {3}%" -f $r.prompt_tps,$r.gen_tps,$r.peak_vram,$r.util)
  }

  'alone9b' {
    Write-Host "`n=== 9B ($Quant9B) alone ===" -ForegroundColor Cyan
    $r = BenchOne -Model $m9b -Tag "alone_9b_$Quant9B" -Secs 90
    $report += $r
    Write-Host ("  9B alone: prompt {0:N0} t/s gen {1:N1} t/s peak VRAM {2} MiB util {3}%" -f $r.prompt_tps,$r.gen_tps,$r.peak_vram,$r.util)
  }

  'both' {
    Write-Host "`n=== Strategy 2: dual resident ===" -ForegroundColor Cyan
    StopAny
    Write-Host "starting local-fast (8111, 4B)..."
    $p4 = StartServer -Model $m4b -Alias 'local-fast' -Port 8111 -Ctx $Ctx4B -Tag 'co_fast'
    if (-not $p4) { Write-Host "FAILED to start 4B"; exit 1 }
    $afterFast = PsLine 'after_4b'
    Write-Host "  after 4B: used=$($afterFast.used) free=$($afterFast.free) MiB"
    Write-Host "starting local-smart (8112, 9B $Quant9B)..."
    $p9 = StartServer -Model $m9b -Alias 'local-smart' -Port 8112 -Ctx $Ctx9B -Tag 'co_smart'
    if (-not $p9) {
      Write-Host "FAILED to start 9B alongside 4B - co-residency NOT possible at this config" -ForegroundColor Red
      StopAny; exit 2
    }
    Start-Sleep -Seconds 3
    $both = PsLine 'both_loaded'
    Write-Host "  BOTH loaded: used=$($both.used) free=$($both.free) MiB" -ForegroundColor Green

    # shared-memory spill check
    $q = & nvidia-smi -q 2>&1
    $sharedUsed = ($q | Select-String -Pattern 'Shared' -Context 0,3 | Out-String)
    Write-Host "  --- shared memory section ---"
    ($q | Select-String -Pattern 'BAR1|Shared' | Select-Object -First 4) | ForEach-Object { Write-Host "    $($_.Line.Trim())" }

    # sequential probes
    Write-Host "`n  -- fast alone (smart idle) --"
    $fastIdle = ChatProbe -Port 8111 -Alias 'local-fast' -Prompt 'Summarise: the scheduler moved from a Redis queue to three tiers.'
    Write-Host "    $fastIdle"
    Write-Host "  -- smart alone (fast idle) --"
    $smartIdle = ChatProbe -Port 8112 -Alias 'local-smart' -Prompt 'Explain why an off-by-one in 1-based pagination returns the wrong page.'
    Write-Host "    $smartIdle"

    Write-Host "`n  -- concurrent: both models inferring at once --"
    $job1 = Start-Job -ScriptBlock { param($s,$port,$alias,$prompt)
      & powershell -NoProfile -File $s $port $alias $prompt 160 } -ArgumentList (Join-Path $Logs '_probe.py'),8111,'local-fast','Extract JSON: task T-1 is pending and needs a GPU.'
    $job2 = Start-Job -ScriptBlock { param($s,$port,$alias,$prompt)
      & powershell -NoProfile -File $s $port $alias $prompt 300 } -ArgumentList (Join-Path $Logs '_probe.py'),8112,'local-smart','Write a pytest test for a function that parses duration strings.'
    Wait-Job $job1,$job2 -Timeout 600 | Out-Null
    $c1 = Receive-Job $job1; $c2 = Receive-Job $job2
    Remove-Job $job1,$job2 -Force -ErrorAction SilentlyContinue
    Write-Host "    fast : $($c1 | Select-Object -Last 1)"
    Write-Host "    smart: $($c2 | Select-Object -Last 1)"
    $afterLoad = PsLine 'after_concurrent'

    $report += [pscustomobject]@{
      tag='coresidency'; baseline=$base.used; after_4b=$afterFast.used
      both=$both.used; free_when_both=$both.free
      fast_seq=$fastIdle; smart_seq=$smartIdle; fast_conc=($c1|Select-Object -Last 1); smart_conc=($c2|Select-Object -Last 1)
    }
    StopAny
  }

  'reload' {
    Write-Host "`n=== Strategy 1: reload costs ===" -ForegroundColor Cyan
    foreach ($cfg in @(@{n='4b';m=$m4b;a='local-fast';p=8111;c=$Ctx4B},
                       @{n='9b';m=$m9b;a='local-smart';p=8112;c=$Ctx9B})) {
      StopAny
      $sw = [System.Diagnostics.Stopwatch]::StartNew()
      $p = StartServer -Model $cfg.m -Alias $cfg.a -Port $cfg.p -Ctx $cfg.c -Tag "reload_$($cfg.n)"
      $sw.Stop()
      $loadS = [math]::Round($sw.Elapsed.TotalSeconds,1)
      Write-Host ("  cold load {0}: {1}s" -f $cfg.n, $loadS)
      $sup = PsLine 'loaded'
      $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
      StopAny
      $sw2.Stop()
      Write-Host ("  unload {0}: {1}s   (vram after: used={2})" -f $cfg.n, [math]::Round($sw2.Elapsed.TotalSeconds,1), (Vram 'used'))
      $report += [pscustomobject]@{ model=$cfg.n; cold_load_s=$loadS; unload_s=[math]::Round($sw2.Elapsed.TotalSeconds,1); loaded_vram=$sup.used }
    }
  }

  'concurrent' {
    Write-Host "concurrent mode is covered by -Mode both" -ForegroundColor Yellow
  }
}

$out = Join-Path $Results "coresidency_$Mode.json"
$report | ConvertTo-Json -Depth 6 | Set-Content $out -Encoding UTF8
Write-Host "`nwrote $out" -ForegroundColor Green
$report | Format-List
