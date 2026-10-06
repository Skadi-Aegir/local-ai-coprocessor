# =============================================================================
# phase3_coresidency.ps1 - Phase 3 core experiment, written as a script file
# (long inline command blocks proved unreliable once the harness runner was
#  quarantined mid-run).
#
# Measures:
#   * 4B alone VRAM / throughput
#   * 9B alone VRAM / throughput
#   * BOTH resident: total VRAM, remaining headroom, spill, sequential and
#     CONCURRENT inference
#   * reload costs for each model
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\phase3_coresidency.ps1
# =============================================================================

$ErrorActionPreference = 'Continue'
$Llama   = "$(Split-Path -Parent $PSScriptRoot)\tools\llama.cpp\b11254-cuda13.4"
$Server  = Join-Path $Llama 'llama-server.exe'
$Bench   = Join-Path $Llama 'llama-bench.exe'
$Models  = "$(Split-Path -Parent $PSScriptRoot)\models\llm"
$Results = "$(Split-Path -Parent $PSScriptRoot)\benchmarks\local-smart\results"
$Logs    = "$(Split-Path -Parent $PSScriptRoot)\logs"
$Py      = "$(Split-Path -Parent $PSScriptRoot)\services\llm\venv\Scripts\python.exe"
$Probe   = "$(Split-Path -Parent $PSScriptRoot)\benchmarks\local-fast\ctx_probe.py"
New-Item -ItemType Directory -Force -Path $Results | Out-Null

$m4 = Join-Path $Models 'Qwen3.5-4B-Q4_K_M.gguf'
$m9 = Join-Path $Models 'Qwen3.5-9B-Q4_K_M.gguf'

function GpuUsed { [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -replace '\s','') }
function GpuFree { [int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -replace '\s','') }
function KillServers {
  Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
}

function StartSrv {
  param([string]$Model,[string]$Alias,[int]$Port,[int]$Ctx,[string]$Tag,[int]$TimeoutSec = 240)
  Remove-Item (Join-Path $Logs "$Tag.server.jsonl") -Force -ErrorAction SilentlyContinue
  $proc = Start-Process -FilePath $Server -ArgumentList @(
      '-m',$Model,'--alias',$Alias,'-ngl','99','-c',$Ctx,'-np','1','-t','12',
      '-b','2048','-ub','512','--flash-attn','auto','-ctk','f16','-ctv','f16',
      '--host','127.0.0.1','--port',$Port,'--no-webui','--log-jsonl',
      '--log-file',(Join-Path $Logs "$Tag.server.jsonl")
    ) -PassThru -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $Logs "$Tag.out") `
      -RedirectStandardError  (Join-Path $Logs "$Tag.err")
  for ($i=0; $i -lt $TimeoutSec; $i++) {
    Start-Sleep -Milliseconds 1000
    if ($proc.HasExited) { return $null }
    try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
          if ($h.status -eq 'ok') { return $proc } } catch {}
  }
  return $null
}

function Breakdown([string]$Tag) {
  $f = Join-Path $Logs "$Tag.server.jsonl"
  if (-not (Test-Path $f)) { return $null }
  $line = Get-Content $f | Select-String 'fit_memory_breakdown' | Select-Object -Last 1
  if (-not $line) { return $null }
  $json = ($line.Line -replace '^.*?(\{"type")','$1')
  try { return ($json | ConvertFrom-Json).data.rows | Where-Object { $_.kind -eq 'device' } } catch { return $null }
}

function Probe {
  param([int]$Port,[string]$Prompt,[int]$MaxTok = 200)
  $tmp = Join-Path $Logs '_p3probe.py'
  $code = @'
import json, sys, time, urllib.request
port, prompt, mt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
body = {"model":"x","messages":[{"role":"user","content":prompt}],
        "max_tokens":mt,"temperature":0.0,
        "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0=time.time()
d=json.load(urllib.request.urlopen(req, timeout=900))
dt=time.time()-t0
t=d.get("timings",{})
print(json.dumps({"wall_s":round(dt,2),
                  "gen_tps":round(t.get("predicted_per_second",0),2),
                  "prompt_tps":round(t.get("prompt_per_second",0),1),
                  "completion_tokens":t.get("predicted_n",0)}))
'@
  Set-Content -Path $tmp -Value $code -Encoding UTF8
  $out = & $Py $tmp $Port $Prompt $MaxTok 2>&1
  return ($out | Select-Object -Last 1)
}

$summary = [ordered]@{}

# ---------------------------------------------------------------- 1. 4B alone
Write-Host "`n[1/5] 4B alone" -ForegroundColor Cyan
KillServers
$base = GpuUsed
Write-Host "  desktop baseline: $base MiB"
$p4 = StartSrv -Model $m4 -Alias 'local-fast' -Port 8111 -Ctx 8192 -Tag 'co4'
if ($p4) {
  $bd = Breakdown 'co4'
  $summary['fourb_alone'] = [ordered]@{
    self_MiB = $bd.self; model_MiB = $bd.model; context_MiB = $bd.context
    gpu_used = (GpuUsed); gpu_free = (GpuFree)
    load_s = 'see reload section'
  }
  Write-Host ("  self={0} model={1} ctx={2} | gpu used={3} free={4}" -f $bd.self,$bd.model,$bd.context,(GpuUsed),(GpuFree))
  $summary['fourb_alone']['probe'] = (Probe -Port 8111 -Prompt 'Write a haiku about garbage collection.')
  Write-Host "  probe: $($summary['fourb_alone']['probe'])"
} else { Write-Host "  4B FAILED to start" -ForegroundColor Red }

# ---------------------------------------------------------------- 2. CO-RESIDENT
Write-Host "`n[2/5] BOTH resident (4B on 8111 + 9B on 8112, 4B already up)" -ForegroundColor Cyan
$before9 = GpuUsed
Write-Host "  gpu used with 4B only: $before9 MiB ; starting 9B..."
$p9 = StartSrv -Model $m9 -Alias 'local-smart' -Port 8112 -Ctx 16384 -Tag 'co9'
if ($p9) {
  $bd9 = Breakdown 'co9'
  $bothUsed = GpuUsed; $bothFree = GpuFree
  Write-Host ("  BOTH RESIDENT OK: 9B self={0} | gpu used={1} free={2}" -f $bd9.self,$bothUsed,$bothFree) -ForegroundColor Green
  $summary['coresident'] = [ordered]@{
    ok = $true
    nineb_self_MiB = $bd9.self; nineb_model_MiB = $bd9.model; nineb_context_MiB = $bd9.context
    gpu_used = $bothUsed; gpu_free = $bothFree
    fourb_self_MiB = $summary['fourb_alone'].self_MiB
    combined_model_MiB = $summary['fourb_alone'].self_MiB + $bd9.self
  }
  # shared memory / spill check
  Write-Host "  --- shared GPU memory ---"
  $q = & nvidia-smi -q 2>&1
  ($q | Select-String -Pattern 'Shared' | Select-Object -First 2) | ForEach-Object { Write-Host "    $($_.Line.Trim())" }

  # Scenario A: fast active, smart idle
  Write-Host "`n  Scenario A: fast ACTIVE, smart idle"
  $a = Probe -Port 8111 -Prompt 'Summarise in 3 bullets: the scheduler moved from a Redis queue to three tiers.'
  Write-Host "    fast  : $a"
  $summary['scenario_A_fast_active'] = $a

  # Scenario B: smart active, fast idle
  Write-Host "  Scenario B: smart ACTIVE, fast idle"
  $b = Probe -Port 8112 -Prompt 'Explain the root cause of an off-by-one error in 1-based pagination.'
  Write-Host "    smart : $b"
  $summary['scenario_B_smart_active'] = $b

  # Scenario C: both concurrent
  Write-Host "  Scenario C: BOTH inferring concurrently"
  # NOTE: invoke python directly with the script path. Using
  #   powershell -File <something>.py
  # fails ("does not have a '.ps1' extension"), which silently produced empty
  # results in the first run of this experiment.
  $j1 = Start-Job -ScriptBlock {
    param($py,$s,$port,$prompt,$mt)
    & $py $s $port $prompt $mt
  } -ArgumentList $Py,(Join-Path $Logs '_p3probe.py'),8111,'Extract JSON: task T-9 is pending, needs GPU, owned by local-smart.',160
  $j2 = Start-Job -ScriptBlock {
    param($py,$s,$port,$prompt,$mt)
    & $py $s $port $prompt $mt
  } -ArgumentList $Py,(Join-Path $Logs '_p3probe.py'),8112,'Write a pytest test for a function that parses duration strings like 1h30m.',300
  Wait-Job $j1,$j2 -Timeout 900 | Out-Null
  $c1 = (Receive-Job $j1 | Where-Object { $_ -match '^\s*\{' } | Select-Object -Last 1)
  $c2 = (Receive-Job $j2 | Where-Object { $_ -match '^\s*\{' } | Select-Object -Last 1)
  Remove-Job $j1,$j2 -Force -ErrorAction SilentlyContinue
  Write-Host "    fast  concurrent: $c1"
  Write-Host "    smart concurrent: $c2"
  $summary['scenario_C_concurrent'] = [ordered]@{ fast=$c1; smart=$c2 }
  $summary['scenario_C'] = [ordered]@{ gpu_after = (GpuUsed) }

  # Scenario D: sustained alternating
  Write-Host "  Scenario D: 8 alternating requests"
  $lat = @()
  for ($k=0; $k -lt 8; $k++) {
    $port = if ($k % 2 -eq 0) { 8111 } else { 8112 }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Probe -Port $port -Prompt "Request $k : reply with one short sentence about testing."
    $sw.Stop()
    $lat += [pscustomobject]@{ n=$k; port=$port; wall_s=[math]::Round($sw.Elapsed.TotalSeconds,2); result=$r }
    Write-Host ("    #{0} port {1}  {2}s  {3}" -f $k,$port,[math]::Round($sw.Elapsed.TotalSeconds,2),$r)
  }
  $summary['scenario_D_alternating'] = $lat
} else {
  Write-Host "  9B COULD NOT START ALONGSIDE 4B" -ForegroundColor Red
  Write-Host "  --- stderr tail ---"
  Get-Content (Join-Path $Logs 'co9.err') -Tail 8 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
  $summary['coresident'] = [ordered]@{ ok = $false; gpu_used_before = $before9; gpu_free_before = (GpuFree) }
}
KillServers

# ---------------------------------------------------------------- 3. 9B alone
Write-Host "`n[3/5] 9B alone" -ForegroundColor Cyan
$p9b = StartSrv -Model $m9 -Alias 'local-smart' -Port 8112 -Ctx 16384 -Tag 'co9solo'
if ($p9b) {
  $bd = Breakdown 'co9solo'
  $summary['nineb_alone'] = [ordered]@{
    self_MiB = $bd.self; model_MiB = $bd.model; context_MiB = $bd.context
    gpu_used = (GpuUsed); gpu_free = (GpuFree)
  }
  Write-Host ("  self={0} model={1} ctx={2} | gpu used={3} free={4}" -f $bd.self,$bd.model,$bd.context,(GpuUsed),(GpuFree))
  $summary['nineb_alone']['probe'] = (Probe -Port 8112 -Prompt 'Write a haiku about garbage collection.', 200)
  Write-Host "  probe: $($summary['nineb_alone']['probe'])"
}
KillServers

# ---------------------------------------------------------------- 4. reload costs
Write-Host "`n[4/5] reload costs" -ForegroundColor Cyan
$reload = @()
foreach ($cfg in @(@{n='4B';m=$m4;a='local-fast';p=8111;c=8192;t='rl4'},
                   @{n='9B';m=$m9;a='local-smart';p=8112;c=16384;t='rl9'})) {
  KillServers
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = StartSrv -Model $cfg.m -Alias $cfg.a -Port $cfg.p -Ctx $cfg.c -Tag $cfg.t -TimeoutSec 300
  $sw.Stop()
  if (-not $p) { Write-Host "  $($cfg.n) failed to load"; continue }
  $load = [math]::Round($sw.Elapsed.TotalSeconds,1)
  $usedNow = GpuUsed
  $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
  KillServers
  $sw2.Stop()
  $unload = [math]::Round($sw2.Elapsed.TotalSeconds,1)
  Write-Host ("  {0}: cold load {1}s | unload {2}s | resident VRAM {3} MiB" -f $cfg.n,$load,$unload,$usedNow)
  $reload += [pscustomobject]@{ model=$cfg.n; cold_load_s=$load; unload_s=$unload; resident_MiB=$usedNow }
}
$summary['reload'] = $reload

# ---------------------------------------------------------------- 5. file
$out = Join-Path $Results 'coresidency.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content $out -Encoding UTF8
Write-Host "`n[5/5] wrote $out" -ForegroundColor Green
Write-Host "`n===== SUMMARY ====="
$summary | ConvertTo-Json -Depth 8
