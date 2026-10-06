# GPU Scheduling — Phase 5.5

**Scope:** a GPU admission-control prototype. **Not a Gateway.** It decides *when* work may
run and can unload a non-essential service to make room; it does not proxy inference traffic.

**Code:** [`services/gpu_scheduler/`](../services/gpu_scheduler/)
**Test:** [`scripts/test-gpu-scheduling.ps1`](../scripts/test-gpu-scheduling.ps1)

---

## 1. Why this exists

Phase 4 and Phase 5 measured that concurrent GPU work is not free:

| finding | measurement |
|---|---|
| two LLMs inferring together | generation −62 to −65 % |
| embedding under concurrent load | −29 % |
| serialized vs concurrent (embedding + rerank) | total wall **−14.4 %**, but each operation **+45 to +69 %** slower |

The last row is the important one: **concurrency is not uniformly good or bad.** It reduces
total wall time for independent work while making every individual operation slower. A
blanket "GPU semaphore = 1" is therefore wrong, and a scheduler needs to know what kind of
job it is admitting.

## 2. Three strategies

| strategy | behaviour |
|---|---|
| `serial` | one GPU job at a time, always |
| `conditional` | interactive jobs run alone; background jobs may overlap **each other** (at most `max_background_units`), never an interactive job, and none is admitted while an interactive request waits |
| `unload` | `conditional`, plus a non-essential idle service may be unloaded to satisfy a request that would otherwise be refused |

The lock inside `GpuScheduler.acquire()` is the serialisation mechanism. `serial` is therefore
trivially correct, and `conditional` is a **policy change rather than a concurrency
rewrite** — which is why the three can be compared on the same task set.

> **Corrected 2026-10-05.** The `conditional` row used to say background jobs "may overlap an interactive job" - the self-contradiction P55R-01 removed from the code (AD-9). See §3.

## 3. Three job classes

Priority alone cannot express the required policy, so jobs are classified:

| class | who | policy |
|---|---|---|
| `interactive` | a user is waiting | runs **alone** (not even beside another interactive request); waits for background units that are **already running**, and while it waits **no new background unit is admitted** |
| `background` | indexing / batch | may overlap other background units within budget, at most `max_background_units` at once; never admitted while an interactive request runs or waits |
| `exclusive` | needs a dedicated budget | **never shares the GPU at all** |

An interactive request therefore never queues behind background work that arrives after it, and
its wait for work already running is bounded by `max_background_units` × the unit duration
(measured at 0.207 s against a 0.600 s bound). "Interactive never waits for background" is
**not** promised: without preemption it cannot be achieved.

> **Corrected 2026-10-05.** This table used to say interactive work "never waits behind
> background work; never shares with it; **may** share with another interactive request", and
> that it "always gets the machine to itself". P55R-01 replaced that policy, which contradicted
> itself, and the code changed with it; this page did not. The policy as implemented is the
> CONCURRENCY POLICY block of `services/gpu_scheduler/gpu_scheduler.py`; the decision is
> ARCHITECTURE_DECISIONS AD-9.

## 4. Safety properties

These are enforced in code and asserted by the self-test:

1. **2 GiB VRAM margin before every admission**, measured with `nvidia-smi` rather than
   estimated from a table. A request that would breach it is refused, not queued forever.
2. **Only non-essential services may be unloaded.** `local-fast` is registered
   `essential=True` and a request to unload it is refused by an explicit test.
3. **Qwen-Image is deliberately absent from the registry**, so this prototype *cannot*
   unload it. A test asserts it is not present. The Qwen-Image deployment lives outside this
   project's lifecycle and is protected by the phase instructions.
4. **A running task is never force-killed to make room.** Unloading happens only between
   jobs, through the service's own stop script, after the scheduler has decided it is safe.
5. **Refusal is explicit and reported.** When the margin cannot be met even after unloading,
   the decision names the shortfall (`free X + reclaimable Y − margin Z is still short`) and
   refuses. The phase instruction is to unload and *report* rather than keep loading.
6. **Unload is verified.** The scheduler waits for the port to free and re-measures memory
   afterwards, so a failed unload is visible instead of assumed to have worked.

### Queued and in-flight requests

`acquire()` **blocks** while another job merely holds the GPU — waiting is the correct
behaviour for `serial` and `conditional` — but it **refuses immediately** when the memory
situation cannot be satisfied, and times out rather than waiting forever. Queue depth per
class is tracked and reported in `status()`. Unloading is only ever attempted from within
`acquire()`, i.e. between jobs, never against a job in flight.

## 5. Job profiles

| class | services needed | extra headroom |
|---|---|---|
| `interactive` | `local-fast` | 900 MiB |
| `background` | — | 600 MiB |
| `exclusive` | — | 2000 MiB |

A class that may run alongside others must also cover their **peak**, not just their steady
footprint: the reranker's transient spike is larger than its resting size, and Phase 5
measured that the spike is what runs out of memory.

## 6. Measured co-residency (context for the policy)

Peak VRAM under concurrent load, sampled every 150 ms while all services actually worked:

| configuration | peak MiB | free at peak | ≥ 2 GB |
|---|---:|---:|---|
| reranker alone | 5248 | 6979 | yes |
| embedding + reranker | 6649 | 5578 | yes |
| local-fast + reranker | 8482 | 3745 | yes |
| **all three** | **9865** | **2362** | **yes** |

All three can be resident and still hold the margin. **Fitting in memory is not the same as
running fast** — see §1 — so residency and scheduling are separate decisions.

> **Noted 2026-10-05.** Holding the margin while the three sit idle is not the same as admitting work. With all three resident, ~152 MiB was usable above the margin and every normal request was refused - 0 of 10 completed in the live validation (OPEN_ISSUES P1-6, ARCHITECTURE_DECISIONS AD-12).

## 7. Verification

`python scheduler_cli.py --selftest` → **24 checks, all passing.** It exercises the policy
only; no service is started or stopped. Covered: serial/conditional overlap rules, exclusive
isolation, the essential-service guard, Qwen-Image absence, margin enforcement against real
shortfalls, unload candidate selection (cheapest-to-reload first), explicit refusal, and
serialised execution under a threaded workload.

`powershell -File scripts\test-gpu-scheduling.ps1` runs the same self-test plus environment
checks and prints a live admission plan for each job class.

### A test-design correction worth recording

Four checks failed on the first run: *"background may join background"*, *"interactive may
join interactive"*, *"exclusive is allowed when the GPU is idle"*. Reading the failure
reasons showed the **policy was fine and the test was wrong** — every case was refused for
**memory**, because the machine had only ~3.5 GiB free at the time.

The tests were conflating two questions: *"is this overlap permitted?"* and *"does it fit?"*
They now stub free memory so the overlap policy is tested in isolation, with separate checks
covering margin enforcement against real shortfalls. **Testing a scheduling policy against a
saturated GPU measures the GPU, not the policy.**

> **Noted 2026-10-05.** Under the current policy (AD-9) "interactive may join interactive" is refused by design. The lesson recorded here - test the overlap policy with memory stubbed - stands.

## 8. CLI

```powershell
python services/gpu_scheduler/scheduler_cli.py --status
python services/gpu_scheduler/scheduler_cli.py --plan interactive
python services/gpu_scheduler/scheduler_cli.py --strategy unload --plan background
python services/gpu_scheduler/scheduler_cli.py --selftest
python services/gpu_scheduler/scheduler_cli.py --compare      # fixed task set, 3 strategies
```

`--compare` runs a fixed four-task set (two interactive, two background) under each strategy
and reports wall time and refusals. It is the honest comparison the phase asks for; it is run
manually rather than in the test script because it needs all services up and takes minutes.

## 9. What this deliberately is NOT

* **Not a Gateway.** No request routing, no auth, no protocol translation.
* **No GPU sharing via MPS or time-slicing.** The margin and the measured contention say
  serialising is the right default for latency-sensitive work.
* **No automatic service start.** `start()` exists and is used by the live demo, but the
  scheduler never starts a model by itself; that stays an operator decision.
* **No eviction of running work.** Ever.
* **No control over Qwen-Image.**

## 10. Recommendation

| workload | policy |
|---|---|
| interactive request | **runs alone** — waits only for background units already running; no new one is admitted while it waits |
| background batch (indexing, bulk re-ranking) | **overlap permitted** with other background units, never with an interactive job |
| exclusive job needing a private budget | **dedicated**, refuse if the budget cannot be met |

Default strategy: **`conditional`**. Use `unload` only where a non-essential service is
genuinely disposable, because loading it back costs 1.2–1.7 s.

> **Corrected 2026-10-05.** The background row used to permit overlap "including with an interactive job"; that policy was replaced in P55R-01 (AD-9). See §3.
