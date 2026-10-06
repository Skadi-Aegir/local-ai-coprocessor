"""
mock_server.py - a fake OpenAI-compatible endpoint used to test the Phase 2 harness.

Why this exists: the real benchmark run needs an idle GPU, and a plumbing bug found
mid-run would waste that window. This mock speaks the same wire format as
llama-server (SSE deltas plus a final chunk carrying `timings`) so the whole
pipeline - HTTP client, TTFT measurement, JSON extraction, schema grading,
routing grading, code extraction and execution - can be exercised without a GPU.

It is a TEST FIXTURE. It never runs as part of the local-fast service.

Usage:
  python mock_server.py --port 8199
  python run_bench.py --base-url http://127.0.0.1:8199 --out results/_mock --tag mock
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "local-fast"

# --------------------------------------------------------------------------
# canned behaviour per benchmark test
# --------------------------------------------------------------------------

TRACEBACK_REPLY = """Exception: json.decoder.JSONDecodeError
Location: /srv/app/config.py:27
Cause: the config file read at that path is empty or is not valid JSON, so the
parser failed on the very first character.
Fix: guard the load and surface a useful error instead of letting JSONDecodeError escape:

    text = Path(path).read_text()
    if not text.strip():
        raise ValueError(f"config file is empty: {path}")
    return json.loads(text)"""

HANDOFF_REPLY = """## Goal
Rebuild the eye-tracking pipeline so one command processes a day of data reproducibly.

## Current state
Pipeline runs end to end but has not met the byte-identical acceptance criterion.

## Completed
Preprocessing (sync, calibrate, face crop), scheduler with retry, quality report.

## Failed
IDT detection (sampling-rate coupled), GPU face detection (not worth it),
hash-based incremental execution (hashing 180 GB cost more than rerunning).

## Important files
- src/eyetrack/cli.py
- src/eyetrack/scheduler.py
- src/eyetrack/detect/ivt.py
- src/eyetrack/preprocess/calibrate.py
- config/pipeline.yaml

## Important decisions
Parquet intermediates (no float round-trip), no GPU, explicit seeding (seed 20260401),
no global state.

## Tests
test_sync.py and test_calibrate.py pass; test_ivt.py is flaky (~1 in 20).

## Open questions
Where the byte difference originates; whether the unstable sort key and the
scheduler write order are the same bug; whether rotated calibration must be supported.

## Next step
Make the scheduler write in subject_id order and add a secondary sort key in
detect/ivt.py, then run the same 5 subjects three times and diff the output."""

SQL_REPLY = """```sql
SELECT project, COUNT(*) AS task_count, AVG(priority) AS avg_priority
FROM tasks
GROUP BY project
HAVING COUNT(*) > 1
ORDER BY task_count DESC;
```"""

PY_JSON_CSV_REPLY = '''```python
import csv


def to_csv(records, path):
    """Write a list of equal-keyed dicts to CSV. Empty input writes an empty file."""
    if not records:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
```'''

SHELL_REPLY = """```bash
#!/usr/bin/env bash
set -euo pipefail
for f in *.jpeg; do
  [ -e "$f" ] || continue
  target="${f%.jpeg}.jpg"
  if [ -e "$target" ]; then
    echo "skip: $target already exists"
    continue
  fi
  echo "rename: $f -> $target"
  mv "$f" "$target"
done
```"""

PYTEST_REPLY = '''```python
import pytest

from duration import parse_duration


def test_plain_seconds():
    assert parse_duration("45s") == 45


def test_minutes_and_seconds():
    assert parse_duration("2m") == 120


def test_hours_and_minutes():
    assert parse_duration("1h30m") == 5400


def test_invalid_raises():
    with pytest.raises(ValueError):
        parse_duration("banana")
```'''


def reply_for(prompt: str, system: str) -> str:
    """Deterministic canned answer chosen from the prompt shape."""
    p, s = prompt, system

    # Fault injection for testing the thinking-leak detector: when the task text
    # contains the sentinel below, answer in thinking style. The detector and the
    # suite's global tally must flag this; otherwise the check is decorative.
    if "LEAKTEST" in p:
        return "<think>\nLet me reason about this.\n</think>\nParis"

    # Test C - routing. NOTE: this must be tested BEFORE the handoff check below,
    # because routing case 9 mentions the word "handoff" in its task text and would
    # otherwise be answered with a handoff document instead of routing JSON.
    if '"route"' in s or "router" in s.lower():
        route = "local-fast"
        low = p
        if "OCR" in low or "扫描" in low:
            route = "local-ocr"
        elif "权限系统" in low or "架构" in low or "路线" in low or "安全事故" in low:
            route = "claude"
        elif "微服务" in low or "渲染引擎" in low:
            route = "codex"
        elif "竞态" in low or "内存泄漏" in low or "证明" in low:
            route = "deepseek"
        elif ("插画" in low or "图像" in low) and "生成" in low:
            route = "local-image"
        elif "界面截图" in low or "图表" in low:
            route = "local-vision"
        elif "是否执行" in low:
            route = "human"
        return json.dumps(
            {"route": route, "reason": "classified by the mock router", "confidence": 0.9},
            ensure_ascii=False,
        )

    # Test D - traceback
    if "Traceback" in p:
        return TRACEBACK_REPLY

    # Test F - handoff
    if "handoff" in p.lower():
        return HANDOFF_REPLY

    # Test B - extraction: echo the facts found in the sentence
    if "task_id" in s:
        tid = re.search(r"T-\d+", p)
        proj = re.search(r"属于\s*([A-Za-z0-9\-_]+)\s*项目", p)
        pri = re.search(r"优先级\s*(\d+)", p)
        worker = re.search(r"由\s*([A-Za-z0-9\-_]+)\s*处理", p)
        status = re.search(r"状态\s*([A-Za-z]+)", p)
        gpu = "不需要 GPU" not in p
        return json.dumps(
            {
                "task_id": tid.group(0) if tid else None,
                "project": proj.group(1) if proj else None,
                "priority": int(pri.group(1)) if pri else 0,
                "requires_gpu": gpu,
                "worker": worker.group(1) if worker else None,
                "status": status.group(1) if status else None,
            },
            ensure_ascii=False,
        )

    # Test A - Chinese summary
    if "5 个要点" in p or "要点" in p:
        return (
            "1. 旧架构用 Redis 队列加固定 worker，峰值排队延迟恶化到 6.8 秒。\n"
            "2. 新架构分为接入层、调度层、执行层三层，执行层引入 8 秒硬超时。\n"
            "3. 放弃自建向量检索改用 pgvector，检索延迟从 12 毫秒升到 35 毫秒。\n"
            "4. 只做完全相同查询的缓存，命中率约 23%，明确放弃语义缓存。\n"
            "5. 灰度环境错误率从 0.8% 降到 0.15%，调度耗时问题尚未解决。"
        )

    # Test E - code
    if "to_csv" in p:
        return PY_JSON_CSV_REPLY
    if "jpeg" in p or "rename" in p:
        return SHELL_REPLY
    if "GROUP BY" in p or "avg_priority" in p:
        return SQL_REPLY
    if "parse_duration" in p:
        return PYTEST_REPLY

    return "OK"


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence default logging
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json(
                {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mock"}]}
            )
        if self.path == "/props":
            return self._json({"default_generation_settings": {"n_ctx": 8192}, "model_path": "mock"})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json({"error": "not found"}, 404)

        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages") or []
        prompt = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")

        text = reply_for(prompt, system)
        prompt_n = max(1, len(prompt) // 3)
        out_n = max(1, len(text) // 3)
        timings = {
            "prompt_n": prompt_n,
            "prompt_ms": 40.0,
            "prompt_per_second": prompt_n / 0.04,
            "predicted_n": out_n,
            "predicted_ms": 500.0,
            "predicted_per_second": out_n / 0.5,
        }

        if not req.get("stream"):
            return self._json(
                {
                    "id": "mock", "object": "chat.completion", "model": MODEL_ID,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": prompt_n, "completion_tokens": out_n,
                              "total_tokens": prompt_n + out_n},
                    "timings": timings,
                }
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_chunk(payload: str):
            data = payload.encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode())
            self.wfile.write(data + b"\r\n")
            self.wfile.flush()

        # stream in a few pieces so TTFT is a real (non-zero) measurement
        step = max(1, len(text) // 4)
        for i in range(0, len(text), step):
            evt = {
                "id": "mock", "object": "chat.completion.chunk", "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": text[i:i + step]},
                             "finish_reason": None}],
            }
            write_chunk(f"data: {json.dumps(evt, ensure_ascii=False)}\n\n")
            time.sleep(0.02)

        final = {
            "id": "mock", "object": "chat.completion.chunk", "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_n, "completion_tokens": out_n,
                      "total_tokens": prompt_n + out_n},
            "timings": timings,
        }
        write_chunk(f"data: {json.dumps(final, ensure_ascii=False)}\n\n")
        write_chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock llama-server on http://{args.host}:{args.port} (model id: {MODEL_ID})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
