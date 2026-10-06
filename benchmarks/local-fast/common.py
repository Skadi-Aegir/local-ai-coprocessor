"""
common.py - shared HTTP client and scoring helpers for the local-fast benchmark suite.

Stdlib only: the worker must be testable without installing anything, and the
project rule is one isolated venv per Python component.

Talks to any OpenAI-compatible endpoint (llama-server for Phase 2).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------

@dataclass
class CallResult:
    """One inference call, with both the text and the server-reported timings."""

    text: str = ""
    ok: bool = False
    error: str = ""
    status: int = 0
    # wall-clock
    ttft_s: float | None = None          # time to first streamed content token
    total_s: float = 0.0                 # full request wall time
    # server-reported (llama.cpp `timings`)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def json_ok(self) -> bool:
        try:
            json.loads(self.text)
            return True
        except Exception:
            return False


class Client:
    def __init__(self, base_url: str = "http://127.0.0.1:8111", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> dict[str, Any]:
        return self._get("/v1/models")

    def props(self) -> dict[str, Any]:
        return self._get("/props")

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    # ------------------------------------------------------------ inference
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "local-fast",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        seed: int | None = None,
        enable_thinking: bool = False,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_schema: dict[str, Any] | None = None,
        stream: bool = True,
    ) -> CallResult:
        """Run one chat completion. Streaming is used so TTFT is a real measurement."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            # Qwen3.5 defaults to thinking mode; a fast worker must not
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        # only send sampling knobs the server understands, and only when non-default
        if top_k:
            body["top_k"] = top_k
        if min_p:
            body["min_p"] = min_p
        if repeat_penalty != 1.0:
            body["repeat_penalty"] = repeat_penalty
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        if grammar is not None:
            body["grammar"] = grammar
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema},
            }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        res = CallResult()
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                res.status = r.status
                if not stream:
                    payload = json.loads(r.read().decode("utf-8", "replace"))
                    res.raw = payload
                    res.text = payload["choices"][0]["message"]["content"] or ""
                    res.total_s = time.perf_counter() - t0
                    res.ok = True
                else:
                    chunks: list[str] = []
                    for raw_line in r:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload_s = line[5:].strip()
                        if payload_s == "[DONE]":
                            break
                        try:
                            evt = json.loads(payload_s)
                        except Exception:
                            continue
                        # final chunk carries usage/timings
                        if evt.get("timings") or evt.get("usage"):
                            res.raw = evt
                        for ch in evt.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                if res.ttft_s is None:
                                    res.ttft_s = time.perf_counter() - t0
                                chunks.append(piece)
                    res.text = "".join(chunks)
                    res.total_s = time.perf_counter() - t0
                    res.ok = True

            tim = (res.raw or {}).get("timings") or {}
            res.prompt_tokens = int(tim.get("prompt_n") or (res.raw.get("usage") or {}).get("prompt_tokens") or 0)
            res.completion_tokens = int(tim.get("predicted_n") or (res.raw.get("usage") or {}).get("completion_tokens") or 0)
            res.prompt_tps = float(tim.get("prompt_per_second") or 0.0)
            res.gen_tps = float(tim.get("predicted_per_second") or 0.0)
        except urllib.error.HTTPError as e:
            res.error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            res.status = e.code
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        return res


# --------------------------------------------------------------------------
# JSON extraction / schema checking
# --------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Best-effort JSON extraction: tolerate code fences and surrounding prose."""
    if text is None:
        return None
    s = text.strip()
    # strip ```json ... ``` fences
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # fall back to the outermost brace span
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


def check_schema(obj: Any, expected: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Compare a parsed object against an expected dict.
    Returns (all_correct, list_of_problems).

    Strict about types where the task defines them (bool vs int matters here:
    Python treats True == 1, which would silently pass a wrong answer).
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return False, ["not a JSON object"]

    for key, want in expected.items():
        if key not in obj:
            problems.append(f"missing field: {key}")
            continue
        got = obj[key]
        if isinstance(want, bool):
            if not isinstance(got, bool):
                problems.append(f"{key}: expected bool, got {type(got).__name__} ({got!r})")
            elif got != want:
                problems.append(f"{key}: expected {want}, got {got}")
        elif isinstance(want, int):
            if isinstance(got, bool) or not isinstance(got, int):
                problems.append(f"{key}: expected int, got {type(got).__name__} ({got!r})")
            elif got != want:
                problems.append(f"{key}: expected {want}, got {got}")
        elif isinstance(want, str):
            if not isinstance(got, str):
                problems.append(f"{key}: expected str, got {type(got).__name__} ({got!r})")
            elif got.strip().lower() != want.strip().lower():
                problems.append(f"{key}: expected {want!r}, got {got!r}")
        else:
            if got != want:
                problems.append(f"{key}: expected {want!r}, got {got!r}")

    extra = set(obj) - set(expected)
    if extra:
        problems.append(f"extra fields: {sorted(extra)}")
    return (len(problems) == 0), problems


# --------------------------------------------------------------------------
# tiny stats helper
# --------------------------------------------------------------------------

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


# --------------------------------------------------------------------------
# thinking-mode leakage
# --------------------------------------------------------------------------

THINK_MARKERS = ("<think>", "</think>", "<|think|>", "[Start thinking]", "Thinking Process:")


def thinking_leak(text: str) -> str | None:
    """
    Return the first reasoning marker found, or None.

    Qwen3.5 defaults to thinking mode and emits a <think> block before the answer.
    The worker must run with `chat_template_kwargs.enable_thinking = false`. If that
    ever fails to take effect, every mechanical task degrades: JSON extraction sees
    prose before the object, routing accuracy drops, and latency balloons.

    That failure would look like "the model is bad at JSON" rather than "the flag did
    not apply", so it is worth detecting explicitly and loudly instead of inferring it
    from crashed scores. Every suite run reports leakage counts for this reason.
    """
    if not text:
        return None
    for m in THINK_MARKERS:
        if m in text:
            return m
    return None
