# OPEN_ISSUES P1-11 — embedding service memory retention: cause, fix, measurement

**Date:** 2026-10-05. **Status: fix implemented and measured in isolation and on a service started
from this tree. NOT yet verified on the deployed 8113**, which still runs the old code until it is
restarted from a tree that carries the fix.

## The problem as recorded

Phase A and Phase B each saw the embedding service's `/stats vram_mb` go from 2360.6 to ~7750 MiB
after a 520-chunk build, free VRAM fall below the 2048 MiB margin, and only a restart of 8113
give the memory back.

## Cause

PyTorch's caching allocator. A document request pads every text to the longest one, so a request
of long chunks needs activations far larger than the weights; when it finishes, the allocator keeps
those blocks reserved for reuse. The process therefore grows to its high-water mark and stays there.
`services/retrieval/embedder.py` never handed the cache back.

## Fix

* `Embedder._release_cache()` (`services/retrieval/embedder.py`): after each encode, if the
  allocator holds more than `release_cache_mb` (default 256) MiB of cache above what is allocated,
  `torch.cuda.empty_cache()` returns it. `server.py --release-cache-mb 0` restores the old
  behaviour.
* `/stats` now reports the service's own allocator under `memory` (`torch_reserved_mb`,
  `torch_allocated_mb`, `torch_peak_reserved_mb`, `cache_releases`). `vram_mb` is kept for the
  scripts that print it - see "What /stats vram_mb is" below.
* The index builder sends 4 texts per request instead of 16, refuses to start embedding when free
  VRAM minus the measured transient would breach the margin, samples free VRAM during the build,
  and reads the service's retention after it (`benchmarks/phaseA/a_build_index.py`).

## Measurement

`p1_11_embed_retention.py`: the model loaded **in an isolated process**; the script refuses to run
while any project service is listening; workload = the exact Phase A corpus (the 48 manifest files
read from commit `854fabc`, each checked against the manifest hash, chunked as the builder does:
520 chunks), sent through `encode_documents` - the path `/v1/embeddings` takes for documents.

| run | texts / request | release | time | peak above weights | **left behind above weights** | device free min (nvidia-smi) |
|---|---:|---|---:|---:|---:|---:|
| `rs16b` | 16 | off (Phase A/B behaviour) | 22.18 s | 5061.6 MiB | **5053.5 MiB** | 2479 MiB |
| `rs16b` | 16 | on | 22.49 s | 3751.5 MiB | **15.5 MiB** | 3776 MiB |
| `rs4` | 4 | off | 13.34 s | 1017.6 MiB | **1009.5 MiB** | 6476 MiB |
| `rs4` | 4 | on | 13.09 s | 993.5 MiB | **35.5 MiB** | 6498 MiB |

* The release removes the retention: 5053.5 -> 15.5 MiB at 16 per request. The 5053.5 MiB matches
  the ~5.4 GB growth Phase A and B recorded.
* It costs no measurable time, and the vectors with and without it are **bit-identical**
  (`max_abs_diff 0.0` in every run).
* It also lowers the peak (5062 -> 3752 MiB at 16 per request), because blocks cached from one
  request no longer fragment the next.
* 4 texts per request needs a quarter of the transient and is faster than 16 (less padding).

**The transient is a margin risk of its own.** At 16 per request a build adds ~3.75 GB inside the
service. With local-fast resident that cannot stay above the 2048 MiB margin, release or not. Hence
the builder's 4 per request (~1 GB) and its preflight.

**Request size changes the vectors - slightly, and not in favour of any size.** Against the
unpadded single-text vectors (`rs1`): mean row cosine 0.99985 at 2 and 4 per request, 0.99984 at 8
and 16; nearest-neighbour agreement 90-94 %, because near-tied neighbours swap. A rebuild with the
new request size is therefore a new index, not a byte-for-byte reproduction of the Phase A one,
and must be validated as such (OPEN_ISSUES P2-9).

**Service level** (`p1_11_service_smoke.py --start --port 8199`, the modified `server.py` started
from this tree): 8 longest chunks through `/v1/embeddings`, peak 1893.6 MiB above the weights,
**0.0 MiB left behind**, one cache release; stopped afterwards, port free.

## What `/stats vram_mb` is

`total - free` from cudaMemGetInfo. On this Windows (WDDM) machine it is neither the device-wide
figure nor the service's footprint: with only a CUDA context in the process it read **1232.6 MiB
while nvidia-smi reported 3438-3469 MiB used**, and it then rose one-for-one with the process's own
allocations (2390.6 after the 1160 MiB of weights; 2360.6 at a service start). That is why it
tracked the retention, and why PROJECT_STATE's "use /stats, not the startup line" was a misreading:
both print the same function. The margin is defined on nvidia-smi; the footprint is
`memory.torch_reserved_mb`.

## A mistake in my own first run, recorded

Run `rs16` capped the experiment's allocator from cudaMemGetInfo. For the reason above that cap
(8690 MiB) did not protect the margin; the margin held anyway (lowest sampled free 2269 MiB). The
script now derives the cap from nvidia-smi, and `rs16` is kept but superseded by `rs16b`. It also
read the corpus from the working tree, which already carried this fix: 522 chunks instead of 520.

## What closes P1-11

Restart the deployed 8113 from a tree carrying the fix, then:

```bash
python benchmarks/p1_11/p1_11_service_smoke.py --port 8113 --run-id deployed-1
```

must report `released`, and the next index build must report a retention below 512 MiB and a
held margin in its `build_vram` / `embedding_service_memory` blocks.
