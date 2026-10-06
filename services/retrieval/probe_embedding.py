"""
probe_embedding.py - first-contact verification for Qwen3-Embedding-0.6B.

Checks, before anything is built on top:
  * the model loads on CUDA
  * output dimension and dtype
  * whether outputs are already L2-normalised (the repo ships 2_Normalize)
  * whether Matryoshka truncation to 512 / 256 preserves sane similarities
  * that the query instruction measurably changes the query vector
  * VRAM footprint

Run: python probe_embedding.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL = str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

INSTRUCT_QUERY = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery:")

DOCS = [
    "local-fast runs Qwen3.5-4B at an 8192 token context and uses about 2981 MiB of VRAM.",
    "Concurrent inference of the 4B and 9B models reduces aggregate throughput by 47 percent.",
    "The vram_watch.ps1 script samples GPU memory every 500 milliseconds and writes a CSV.",
    "Chunking strategy determines whether a retrieved passage is semantically complete.",
]
QUERIES = [
    "What context size does the fast worker use?",
    "Why is running both models at once slower?",
]


def main() -> int:
    print(f"model path: {MODEL}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu free/total: {free/2**20:.0f} / {total/2**20:.0f} MiB")

    t0 = time.time()
    model = SentenceTransformer(MODEL, device="cuda", trust_remote_code=True)
    print(f"load time: {time.time()-t0:.1f}s")
    print(f"max_seq_length: {getattr(model, 'max_seq_length', None)}")
    print(f"prompts available: {list(getattr(model, 'prompts', {}).keys())}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu free after load: {free/2**20:.0f} MiB  (used≈{(total-free)/2**20:.0f} MiB)")

    # ---- dimension + normalisation
    t0 = time.time()
    d = model.encode(DOCS, batch_size=4, convert_to_numpy=True, normalize_embeddings=False)
    print(f"\nencode {len(DOCS)} docs: {time.time()-t0:.2f}s  dtype={d.dtype}  shape={d.shape}")
    norms = np.linalg.norm(d, axis=1)
    print(f"unnormalised norms: {norms.round(4).tolist()}")

    dn = model.encode(DOCS, batch_size=4, convert_to_numpy=True, normalize_embeddings=True)
    print(f"normalised norms  : {np.linalg.norm(dn, axis=1).round(6).tolist()}")

    # SentenceTransformer already normalises (2_Normalize in modules.json), so the
    # default output should already be unit length. Confirm rather than assume.
    default_out = model.encode(DOCS, batch_size=4, convert_to_numpy=True)
    print(f"default-arg norms : {np.linalg.norm(default_out, axis=1).round(6).tolist()}")

    # ---- Matryoshka truncation
    q_plain = model.encode([QUERIES[0]], convert_to_numpy=True, normalize_embeddings=True)[0]
    q_instr = model.encode([INSTRUCT_QUERY + " " + QUERIES[0]],
                           convert_to_numpy=True, normalize_embeddings=True)[0]
    print(f"\ninstruction changes query vector: cosine(q_plain,q_instr)="
          f"{float(np.dot(q_plain, q_instr)):.4f}  (1.0 would mean no effect)")

    print("\nMatryoshka truncation (docs at full 1024 vs truncated, re-normalised):")
    for dim in (1024, 512, 256, 128, 64):
        sub = dn[:, :dim]
        sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
        sims = sub @ (model.encode([QUERIES[0]], convert_to_numpy=True,
                                   normalize_embeddings=True)[0][:dim])
        # also normalise the query slice
        qs = q_plain[:dim] / np.linalg.norm(q_plain[:dim])
        sims = sub @ qs
        order = np.argsort(-sims)
        print(f"  dim {dim:5}: top doc index {order[0]}  sims={sims.round(3).tolist()}")

    # ---- timing
    texts = DOCS * 40  # 160 chunks
    for bs in (1, 8, 32, 64):
        t0 = time.time()
        model.encode(texts, batch_size=bs, convert_to_numpy=True)
        dt = time.time() - t0
        print(f"batch {bs:3}: {len(texts)/dt:8.1f} chunks/s  ({dt:.2f}s for {len(texts)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
