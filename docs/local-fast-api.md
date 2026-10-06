# local-fast — upstream integration guide

**Audience:** Codex/Astra, Claude Code, DeepSeek Harness (and later the MacBook Supervisor).
**Subject:** how to call the Local Fast Worker.
**Status:** Phase 2. Service verified to start, serve the alias and answer requests; the measured
quality/throughput numbers land in `../../benchmarks/local-fast-results.md`.

---

## 1. The one rule

Call the logical name **`local-fast`**. Never a GGUF filename, never a HuggingFace repo id,
never a quantisation label.

```jsonc
{ "model": "local-fast", ... }   // correct
{ "model": "Qwen3.5-4B-Q4_K_M.gguf" }   // wrong - breaks when the weights change
```

The whole point of the alias is that the quantisation underneath can change (Q4_K_M → Q5_K_M →
something newer in 2027) without any upstream agent noticing. Anything that hardcodes the file
name defeats that.

---

## 2. Endpoint

```
POST http://127.0.0.1:8111/v1/chat/completions
GET  http://127.0.0.1:8111/health
GET  http://127.0.0.1:8111/v1/models
```

OpenAI-compatible. **Loopback only** — `127.0.0.1`. It is not reachable from the LAN by design;
a MacBook Supervisor will need a deliberate access layer (Tailscale, mTLS/API key or an SSH
tunnel) built in a later phase, not a firewall hole.

---

## 3. Mandatory parameter: disable thinking

Qwen3.5 defaults to emitting a `<think>` reasoning block **before** its answer. For a fast
worker handling mechanical tasks that is pure latency and it breaks structured output.

Always send:

```jsonc
"chat_template_kwargs": { "enable_thinking": false }
```

If you omit it you will get prose before your JSON, inflated token counts and much higher
latency. The benchmark suite detects this condition explicitly and reports it as
`thinking_leaks_total`; treat any non-zero value as a configuration bug, not a model weakness.

---

## 4. Two presets

| Preset | Use for | temperature | top_p | top_k | min_p |
|---|---|---|---|---|---|
| **deterministic** | JSON extraction, routing, classification, validation, any answer a machine will parse | `0.0` | `1.0` | `20` | `0.0` |
| **general** | summaries, simple code, translation, handoff writing | `0.3` | `0.9` | `40` | `0.05` |

Both pass `enable_thinking: false`. Exact values live in
[`config/local-fast.yaml`](../../config/local-fast.yaml); they are provisional until the Phase 2
benchmark confirms them.

Rule of thumb: **if a program will parse it, use deterministic.** If a human will read it, general
is fine.

---

## 5. Examples

### Structured extraction (deterministic)

```bash
curl -s http://127.0.0.1:8111/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-fast",
    "messages": [
      {"role": "system", "content": "You extract structured data. Reply with ONLY a JSON object, no prose, no code fences."},
      {"role": "user",   "content": "任务 T-204 属于 eye-tracking 项目，优先级 80，需要 GPU，预计由 local-smart 处理，目前状态 pending。"}
    ],
    "temperature": 0.0, "top_p": 1.0, "top_k": 20, "max_tokens": 200,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected shape:

```json
{"task_id":"T-204","project":"eye-tracking","priority":80,
 "requires_gpu":true,"worker":"local-smart","status":"pending"}
```

When the answer must be machine-parseable and cannot be retried, constrain it server-side rather
than hoping. `llama-server` accepts either a JSON schema or a GBNF grammar:

```jsonc
"response_format": {
  "type": "json_schema",
  "json_schema": { "name": "result", "schema": { /* JSON Schema */ } }
}
// or
"grammar": "root ::= \"{\" ws ... \"}\""
```

The benchmark suite measures adherence both with and without these constraints; use whichever the
results show is needed.

### Routing

```jsonc
{"route": "<one of local-fast|local-smart|deepseek|claude|codex|local-ocr|local-vision|local-image|human>",
 "reason": "<short reason>", "confidence": 0.0}
```

Ask for `local-fast`'s *opinion* on routing; do not let it dispatch. In Phase 2 it only classifies
— it calls nothing.

---

## 6. What to send it, and what never to send it

**Good fits** (spec §2.1): classification, routing assistance, JSON/structured extraction, log and
file summarisation, commit messages, simple translation, prompt/context compression, traceback
triage, simple Python/Shell/SQL, simple test generation, test-result interpretation, small change
suggestions, handoff assembly.

**Do not send:** complex system architecture, security-critical changes, complex mathematical
reasoning, large-scale refactors, or any decision that cannot be verified. It will produce
confident-looking output for those and it will be wrong often enough to be dangerous. Escalate per
spec §9.

**Context:** 8192 tokens by default, with the full window available to a single request
(`parallel: 1`). A 16K configuration is affordable on this card if a specific task needs it, but
8K is the default and 32K+ is out of scope for this phase — do not assume the model's advertised
262K context is usable here.

---

## 7. Operational notes

* **Lifecycle is not automatic.** Phase 2 deliberately installs no autostart. Start with
  `scripts/start-local-fast.ps1`, stop with `scripts/stop-local-fast.ps1`, verify with
  `scripts/check-local-fast.ps1`. The Gateway/Supervisor will own lifecycle in a later phase.
* **It is not always resident.** The VRAM plan is that EXCLUSIVE-class models (9B, VL, image) get
  loaded on demand and unloaded when idle. `local-fast` is MEDIUM-class, so treat "connection
  refused" as *not currently loaded*, and retry after starting it rather than treating it as a
  hard failure.
* **`check-local-fast.ps1` exit codes:** `0` healthy, `1` degraded or unhealthy, `2` down. Use
  these rather than parsing output.
* **No API key today** (loopback only). Set `runtime.api_key_env` before exposing it beyond
  localhost.
* **No prompt content is logged** by the worker beyond what `llama-server` writes; the server log
  is JSONL at `logs/local-fast.server.jsonl`. Do not put secrets in prompts.

---

## 8. Handling failure correctly

| Symptom | Likely cause | Action |
|---|---|---|
| connection refused | worker not started | run `scripts/start-local-fast.ps1` |
| `model` id not found | alias mismatch | send `"model": "local-fast"` exactly |
| JSON parse failures | thinking mode leaked, or no constraint | add `enable_thinking:false`; consider `grammar`/`json_schema` |
| very slow | another GPU-heavy process is resident | check `nvidia-smi`; the worker shares the card by design |
| HTTP 500 / OOM | insufficient free VRAM | free VRAM, or lower context / use Q4_K_M |

Do **not** respond to failures by re-downloading models or reinstalling the runtime. The Phase 1/2
environments are verified and pinned; see `../../docs/models/local-fast.md` for the exact revisions.
