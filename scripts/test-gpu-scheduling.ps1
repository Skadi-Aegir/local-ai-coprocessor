<#
.SYNOPSIS
  test-gpu-scheduling.ps1 - regression test for the GPU scheduler prototype.

.DESCRIPTION
  Runs, in order:
    1. environment check (python, nvidia-smi, disk, protected paths intact)
    2. the scheduler policy self-test (24 assertions; does NOT start or stop services)
    3. a live admission plan for each job class, against real current memory
    4. the safety property that matters most: a request that cannot satisfy the 2 GiB
       margin is REFUSED with an explanation, not silently attempted
    5. confirmation that Qwen-Image is not in the registry and therefore cannot be unloaded

  Deliberately does NOT start or stop any service: that is what `--live-demo` and
  `--compare` are for, and a regression test must be safe to run at any time, including
  while real work is in progress.

  Exit codes: 0 all passed, 1 assertion failed, 2 environment problem.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\test-gpu-scheduling.ps1
#>
[CmdletBinding()]
param(
  [string]$RepoRoot = "$(Split-Path -Parent $PSScriptRoot)",
  [switch]$IncludeLiveCompare
)

$ErrorActionPreference = 'Stop'
$fails = @()

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "  OK   $m" -ForegroundColor Green }
function Write-Bad($m)  { Write-Host "  FAIL $m" -ForegroundColor Red; $script:fails += $m }
function Test-Assert($name, $cond, $detail = '') {
  if ($cond) { Write-Ok "$name$(if ($detail) { " :: $detail" })" }
  else { Write-Bad "$name$(if ($detail) { " :: $detail" })" }
}

# ---------------------------------------------------------------- environment
Write-Step 'environment'
$py = Join-Path $RepoRoot 'services\retrieval\venv\Scripts\python.exe'
Test-Assert 'retrieval venv python exists' (Test-Path $py) $py
if (-not (Test-Path $py)) { Write-Host 'cannot continue without python' -ForegroundColor Red; exit 2 }

$cli = Join-Path $RepoRoot 'services\gpu_scheduler\scheduler_cli.py'
$mod = Join-Path $RepoRoot 'services\gpu_scheduler\gpu_scheduler.py'
Test-Assert 'scheduler module present' (Test-Path $mod) $mod
Test-Assert 'scheduler CLI present' (Test-Path $cli) $cli
if (-not (Test-Path $cli)) { exit 2 }

try {
  $smi = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1
  Test-Assert 'nvidia-smi responds' ($LASTEXITCODE -eq 0) $smi
} catch {
  Test-Assert 'nvidia-smi responds' $false $_.Exception.Message
}

# protected environments must still exist
foreach ($p in @('D:\DSH\computer_science\qwen-image-deploy',
                 (Join-Path $RepoRoot 'tools\llama.cpp\b11254-cuda13.4\llama-server.exe'),
                 (Join-Path $RepoRoot 'services\retrieval\venv\Scripts\python.exe'),
                 (Join-Path $RepoRoot 'services\reranker\venv\Scripts\python.exe'))) {
  Test-Assert "protected path intact: $(Split-Path -Leaf $p)" (Test-Path $p)
}

# ---------------------------------------------------------------- policy self-test
Write-Step 'scheduler policy self-test (no services touched)'
$out = & $py $cli --selftest 2>&1
$code = $LASTEXITCODE
$okCount = ($out | Where-Object { $_ -match '^OK ' }).Count
$failLines = @($out | Where-Object { $_ -match '^FAIL' })
Test-Assert 'policy self-test exits 0' ($code -eq 0) "exit=$code"
Test-Assert 'policy self-test reports at least 20 passing assertions' ($okCount -ge 20) "passed=$okCount"
Test-Assert 'policy self-test reports no failures' ($failLines.Count -eq 0) ($failLines -join '; ')

# ---------------------------------------------------------------- live admission plans
Write-Step 'live admission plans against real memory'
$free = [int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits).Trim())
Write-Host "  current free VRAM: $free MiB"
foreach ($cls in @('interactive', 'background', 'exclusive')) {
  $plan = & $py $cli --plan $cls --json 2>&1 | ConvertFrom-Json
  Write-Host ("  {0,-12} allowed={1,-6} required={2,6} MiB  free={3,6} MiB" -f `
    $cls, $plan.allowed, $plan.required_mb, $plan.free_mb)
  Test-Assert "plan for '$cls' reports a reason" ([bool]$plan.reason) $plan.reason
  Test-Assert "plan for '$cls' states the margin" ($plan.margin_mb -eq 2048) "margin=$($plan.margin_mb)"
  # the decision must be consistent with its own arithmetic
  if ($plan.allowed) {
    Test-Assert "plan for '$cls' only allows when the margin is satisfied" `
      (($plan.free_mb - $plan.margin_mb) -ge $plan.required_mb -or $plan.unload.Count -gt 0) `
      "usable=$($plan.free_mb - $plan.margin_mb) required=$($plan.required_mb) unload=$($plan.unload -join ',')"
  } else {
    Test-Assert "refusal for '$cls' explains itself" `
      ($plan.reason -match 'margin|cannot satisfy|does not permit') $plan.reason
  }
}

# ---------------------------------------------------------------- margin is enforced
Write-Step 'safety: the 2 GiB margin is enforced, not advisory'
# an exclusive job needs 2000 MiB of headroom. If free memory cannot cover the margin plus
# that budget, the scheduler must refuse - and the refusal must say why.
$plan = & $py $cli --plan exclusive --json 2>&1 | ConvertFrom-Json
$usable = $plan.free_mb - $plan.margin_mb
if ($usable -lt $plan.required_mb) {
  Test-Assert 'an unsatisfiable exclusive request is refused' (-not $plan.allowed) `
    "usable=$usable required=$($plan.required_mb)"
  Test-Assert 'the refusal names the shortfall' `
    ($plan.reason -match 'short|insufficient margin|cannot satisfy') $plan.reason
} else {
  Write-Host "  (enough memory right now: usable=$usable >= required=$($plan.required_mb);" -ForegroundColor Yellow
  Write-Host "   the refusal path is covered by the self-test's stubbed-memory cases)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- Qwen-Image protection
Write-Step 'safety: Qwen-Image cannot be unloaded by this prototype'
$status = & $py $cli --json 2>&1 | ConvertFrom-Json
$regNames = @($status.registry.PSObject.Properties.Name)
$qi = @($regNames | Where-Object { $_ -match 'qwen.*image|image.*qwen' })
Test-Assert 'no Qwen-Image entry in the scheduler registry' ($qi.Count -eq 0) ($qi -join ',')
Test-Assert 'Qwen-Image is listed as protected/not registered' `
  (@($status.protected_not_registered).Count -ge 1) (($status.protected_not_registered) -join '; ')
Test-Assert 'local-fast is essential and therefore unloadable-proof' `
  ($status.registry.'local-fast'.essential -eq $true)

# ---------------------------------------------------------------- essential guard
Write-Step 'safety: essential services refuse unloading'
$guard = & $py -c @"
import sys
sys.path.insert(0, r'$($RepoRoot)\services\gpu_scheduler')
from gpu_scheduler import GpuScheduler
s = GpuScheduler(strategy='unload', verbose=False, dry_run=True)
print('essential_unload_refused', s._unload_service('local-fast') is False)
"@ 2>&1
Test-Assert 'a direct unload request for local-fast is refused' `
  ($guard -match 'essential_unload_refused True') ($guard -join ' ')

# ---------------------------------------------------------------- optional live compare
if ($IncludeLiveCompare) {
  Write-Step 'live comparison of the three strategies on a fixed task set'
  Write-Host '  this starts real work and takes a few minutes' -ForegroundColor Yellow
  & $py $cli --compare 2>&1 | ForEach-Object { Write-Host "  $_" }
  Test-Assert 'live compare completed' ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"
} else {
  Write-Host "`n(live strategy comparison skipped; pass -IncludeLiveCompare to run it)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- summary
Write-Host ''
if ($fails.Count -eq 0) {
  Write-Host 'RESULT: PASS - GPU scheduler regression test' -ForegroundColor Green
  exit 0
}
Write-Host "RESULT: FAIL ($($fails.Count) problems)" -ForegroundColor Red
$fails | ForEach-Object { Write-Host "  ! $_" -ForegroundColor Red }
exit 1
