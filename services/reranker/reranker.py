"""
reranker.py - Qwen3-Reranker-0.6B with the OFFICIAL scoring procedure.

The specification for Phase 5 is explicit: do not invent a scoring method that differs
from the official one. This module therefore reproduces the README's Transformers
implementation exactly:

    prefix = '<|im_start|>system\\nJudge whether the Document meets the requirements ' \\
             'based on the Query and the Instruct provided. Note that the answer can ' \\
             'only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n'
    body   = '<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}'
    suffix = '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'

    tokens = prefix_tokens + tokenize(body)[:max_length - len(prefix) - len(suffix)] + suffix_tokens
    logits = model(tokens).logits[:, -1, :]
    score  = softmax([logits[no_id], logits[yes_id]])[1]

Why the pieces matter:
  * the suffix ends with an EMPTY think block. Without it the model would be asked to
    produce a reasoning trace and the final logits would not be the yes/no decision.
  * truncation is applied to the BODY only, so the instruction and the decision
    scaffolding survive even when a document is too long. Truncating the whole sequence
    would cut the suffix and destroy the score.
  * padding_side is 'left' so that the final position of every row in a batch is the real
    last token rather than a pad token.
"""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- instructions
GENERAL_INSTRUCTION = (
    "Given the user query, determine whether the document contains information "
    "useful for answering the query."
)
CODE_INSTRUCTION = (
    "Given a programming question, determine whether the code or documentation is "
    "relevant to solving it."
)
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

SYSTEM_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str | None, query: str, doc: str) -> str:
    """The official body format. None means no instruction line is injected."""
    if instruction:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    return f"<Query>: {query}\n<Document>: {doc}"


@dataclass
class RerankerConfig:
    model_path: str = str(Path(__file__).resolve().parents[2] / r"models\reranker\Qwen3-Reranker-0.6B")
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_length: int = 2048
    batch_size: int = 20
    instruction: str | None = DEFAULT_INSTRUCTION
    # Ceiling on (rows x longest row) per forward pass. Attention activations grow with
    # the SQUARE of sequence length, so a fixed row count is unsafe: 3 maximal 2048-token
    # documents in one batch attempted a 768 MiB allocation and raised
    # torch.OutOfMemoryError. Batching by token volume instead keeps memory bounded no
    # matter how long the documents are. 20 x 2048 would be 40960, but real chunks are
    # far shorter, so this still packs a normal top-20 into one pass.
    max_batch_tokens: int = 8192
    # cache: bounded so a long session cannot grow without limit
    cache_size: int = 20000


@dataclass
class RerankResult:
    index: int
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None

    def to_dict(self) -> dict:
        d = {"index": self.index, "score": round(float(self.score), 6)}
        if self.dense_rank is not None:
            d["dense_rank"] = self.dense_rank
        if self.dense_score is not None:
            d["dense_score"] = round(float(self.dense_score), 6)
        return d


class ScoreCache:
    """
    Minimal in-process cache for query-document scores.

    Key = sha1(model_revision | instruction | query | document) so a change to ANY of the
    four invalidates the entry. That matters because the instruction is part of the
    prompt, and the model revision changes the weights - reusing a score across either
    would silently return a stale ranking.

    Deliberately not distributed, and bounded: eviction is insertion-order (the oldest
    entry goes first), which is adequate because Phase 5 re-ranks the same small corpus
    repeatedly and the working set fits comfortably.
    """

    def __init__(self, model_revision: str, capacity: int = 20000):
        self.revision = model_revision
        self.capacity = capacity
        self._d: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def key(self, instruction: str | None, query: str, doc: str) -> str:
        h = hashlib.sha1()
        for part in (self.revision, instruction or "", query, doc):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, k: str) -> float | None:
        with self._lock:
            v = self._d.get(k)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, k: str, v: float) -> None:
        with self._lock:
            if len(self._d) >= self.capacity:
                for old in list(self._d)[: max(1, self.capacity // 10)]:
                    self._d.pop(old, None)
            self._d[k] = v

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._d), "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else None,
        }

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0


class Reranker:
    def __init__(self, cfg: RerankerConfig | None = None):
        self.cfg = cfg or RerankerConfig()
        self._lock = threading.RLock()
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path,
                                                       padding_side="left")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[self.cfg.dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path, dtype=dtype)
        self.model.eval()
        dev = self.cfg.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.model.to(dev)
        self.load_s = round(time.time() - t0, 2)

        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.body_budget = self.cfg.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        self.revision = self._read_revision()
        self.cache = ScoreCache(self.revision, self.cfg.cache_size)

    # ---------------------------------------------------------------- info
    def _read_revision(self) -> str:
        """Best-effort model revision, used as part of the cache key."""
        p = Path(self.cfg.model_path) / "config.json"
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            return f"{cfg.get('model_type','?')}-{cfg.get('num_hidden_layers','?')}L-" \
                   f"{cfg.get('hidden_size','?')}H-{cfg.get('vocab_size','?')}V"
        except Exception:
            return "unknown"

    def vram_mb(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**20, 1)

    def info(self) -> dict:
        return {
            "model_path": self.cfg.model_path,
            "device": self.device,
            "dtype": self.cfg.dtype,
            "load_s": self.load_s,
            "max_length": self.cfg.max_length,
            "body_budget": self.body_budget,
            "prefix_tokens": len(self.prefix_tokens),
            "suffix_tokens": len(self.suffix_tokens),
            "yes_token_id": self.yes_id,
            "no_token_id": self.no_id,
            "revision_key": self.revision,
            "vram_mb": self.vram_mb(),
            "cache": self.cache.stats(),
        }

    def close(self) -> None:
        try:
            del self.model
        except AttributeError:
            pass
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __enter__(self) -> "Reranker":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # ---------------------------------------------------------------- tokenise
    def _encode_batch(self, pairs: list[tuple[str, str]],
                      instruction: str | None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build padded token ids for a batch.

        Truncation applies to the body only (README behaviour). Returns ids, mask and the
        number of pairs that were actually truncated, so the caller can report how often
        truncation bites instead of silently losing document text.
        """
        bodies = [format_pair(instruction, q, d) for q, d in pairs]
        enc = self.tokenizer(bodies, padding=False, truncation="longest_first",
                             return_attention_mask=False, max_length=self.body_budget)
        truncated = 0
        wrapped = []
        for ids in enc["input_ids"]:
            if len(ids) >= self.body_budget:
                truncated += 1
            wrapped.append(self.prefix_tokens + ids + self.suffix_tokens)
        # pad to the longest row in this batch
        maxlen = max(len(w) for w in wrapped)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids, attn = [], []
        for w in wrapped:                      # padding_side='left' -> pad on the left
            n = maxlen - len(w)
            input_ids.append([pad_id] * n + w)
            attn.append([0] * n + [1] * len(w))
        return (torch.tensor(input_ids, dtype=torch.long, device=self.device),
                torch.tensor(attn, dtype=torch.long, device=self.device),
                truncated)

    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[str, str]],
                     instruction: str | None) -> tuple[list[float], int]:
        """
        Score a batch and return (probabilities, number of truncated pairs).

        The official README calls `model(**inputs).logits[:, -1, :]`. That materialises a
        logits tensor of shape (batch, seq_len, 151669) even though only the LAST
        position is ever read. At a 2048-token sequence that single tensor is 1.32 GiB and
        it made scoring an over-long document fail outright with:
            torch.OutOfMemoryError: Tried to allocate 1.32 GiB
        even though the card had 8.87 GiB free - the spike, not the steady state, was the
        problem, and it is worse here because the embedding service shares this GPU.

        Computing the final position through the LM head explicitly is MATHEMATICALLY
        IDENTICAL (the head is a per-position linear map, so slicing before or after it
        gives the same vector) but allocates (batch, 1, 151669) instead of
        (batch, seq_len, 151669). test_reranker.py asserts the two agree, so faithfulness
        to the official method is verified rather than assumed.
        """
        ids, attn, truncated = self._encode_batch(pairs, instruction)
        out = self.model.model(input_ids=ids, attention_mask=attn)
        hidden = out.last_hidden_state[:, -1, :]          # (batch, hidden)
        logits = self.model.lm_head(hidden).float()       # (batch, vocab)
        pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
        probs = torch.softmax(pair, dim=1)[:, 1]
        return probs.tolist(), truncated

    # ---------------------------------------------------------------- public
    def score_pairs(self, pairs: list[tuple[str, str]], *,
                    instruction: str | None | object = ..., batch_size: int | None = None,
                    use_cache: bool = True) -> list[float]:
        """
        Score (query, document) pairs. Higher means more relevant, range 0..1.

        `instruction=None` means no instruction; omitting it uses the configured default.
        The Ellipsis sentinel keeps those two cases distinct, which the A/B test needs.
        """
        instr = self.cfg.instruction if instruction is ... else instruction
        bs = int(batch_size or self.cfg.batch_size)
        with self._lock:
            scores: list[float | None] = [None] * len(pairs)
            pending: list[tuple[int, str]] = []
            if use_cache:
                for i, (q, d) in enumerate(pairs):
                    k = self.cache.key(instr, q, d)
                    v = self.cache.get(k)
                    if v is None:
                        pending.append((i, k))
                    else:
                        scores[i] = v
            else:
                pending = [(i, "") for i in range(len(pairs))]

            truncated_total = 0
            if pending:
                # batch by token volume as well as row count (see _plan_batches)
                groups = self._plan_batches([pairs[i] for i, _ in pending], instr, bs)
                batches = 0
                for grp in groups:
                    chunk = [pending[j] for j in grp]
                    sub = [pairs[i] for i, _ in chunk]
                    vals, trunc = self._score_batch(sub, instr)
                    truncated_total += trunc
                    batches += 1
                    for (i, k), v in zip(chunk, vals):
                        scores[i] = float(v)
                        if use_cache:
                            self.cache.put(k, float(v))
                self.last_batches = batches
            else:
                self.last_batches = 0

            self.last_truncated = truncated_total
            return [float(s) for s in scores]

    def _plan_batches(self, pairs: list[tuple[str, str]], instruction: str | None,
                      batch_size: int) -> list[list[int]]:
        """
        Split indices into forward passes bounded by both row count and real token volume.

        The budget is (rows x PADDED length), not (rows x that row's own length). Padding
        makes every row in a batch as long as the longest one, so measuring each row
        individually underestimates the allocation and was how a 3-row batch of maximal
        documents still reached a 768 MiB attention allocation after this planner was
        first added.

        Cost grows with the square of length, so the budget is checked against the padded
        product, and a single row is always allowed through even if it alone exceeds the
        budget - otherwise an over-long document could never be scored at all.
        """
        lens = []
        for q, d in pairs:
            body = self.tokenizer.encode(format_pair(instruction, q, d),
                                         add_special_tokens=False,
                                         truncation=True, max_length=self.body_budget)
            lens.append(len(body) + len(self.prefix_tokens) + len(self.suffix_tokens))
        self.last_max_tokens = max(lens) if lens else 0

        budget = self.cfg.max_batch_tokens
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_longest = 0
        for i, n in enumerate(lens):
            new_longest = max(cur_longest, n)
            would = (len(cur) + 1) * new_longest
            if cur and (len(cur) >= batch_size or would > budget):
                batches.append(cur)
                cur, cur_longest = [], 0
                new_longest = n
            cur.append(i)
            cur_longest = new_longest
        if cur:
            batches.append(cur)
        return batches

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None,
               instruction: str | None | object = ...,
               batch_size: int | None = None,
               dense_ranks: list[int] | None = None,
               dense_scores: list[float] | None = None) -> list[RerankResult]:
        """
        Score documents against a query and return them ordered best-first.

        `index` is the position in the ORIGINAL `documents` list, which is what the caller
        needs to map back to its own metadata. The reranker never sees or manages an index
        of its own - that separation is deliberate.
        """
        if not documents:
            return []
        pairs = [(query, d) for d in documents]
        scores = self.score_pairs(pairs, instruction=instruction, batch_size=batch_size)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if top_n is not None:
            order = order[:top_n]
        out = []
        for i in order:
            out.append(RerankResult(
                index=i, score=scores[i],
                dense_rank=(dense_ranks[i] if dense_ranks else None),
                dense_score=(dense_scores[i] if dense_scores else None),
            ))
        return out

