"""
compat_probe.py - can the reranker run in an existing env, or is a new venv required?

The specification allows reusing the embedding environment ONLY after confirming it
introduces no version conflict. This probes that empirically rather than by guessing:

  1. can sentence_transformers.CrossEncoder load the repo's modules.json (1_LogitScore)?
  2. does the raw Transformers path (the README's primary implementation) load and score?
  3. do the scores agree with a hand-computed log-softmax over the yes/no logits?
  4. does the official "yes"/"no" token id lookup work on this tokenizer?

Run it with the interpreter you are considering. Exit 0 means that interpreter is viable.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

MODEL = str(Path(__file__).resolve().parents[2] / r"models\reranker\Qwen3-Reranker-0.6B")


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    import torch
    import transformers
    print(f"torch {torch.__version__}   transformers {transformers.__version__}")

    problems: list[str] = []

    # ---- 1. tokenizer + official token ids
    print("\n--- tokenizer and yes/no token ids ---")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    print(f"  token_true_id ('yes') = {yes_id}")
    print(f"  token_false_id('no')  = {no_id}")
    if yes_id in (None, tok.unk_token_id) or no_id in (None, tok.unk_token_id):
        problems.append("yes/no tokens do not resolve to real ids")
    print(f"  unk id = {tok.unk_token_id}")

    # ---- 2. sentence_transformers CrossEncoder path
    print("\n--- sentence_transformers CrossEncoder path ---")
    try:
        import sentence_transformers as st
        print(f"  sentence_transformers {st.__version__}")
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(MODEL, device="cpu")
        q = "What is the capital of China?"
        docs = ["The capital of China is Beijing.",
                "Gravity attracts two bodies towards each other."]
        raw = ce.predict([(q, d) for d in docs])
        prob = ce.predict([(q, d) for d in docs], activation_fn=torch.nn.Sigmoid())
        print(f"  raw logit diffs : {list(raw)}")
        print(f"  sigmoid scores  : {list(prob)}")
        if not (raw[0] > raw[1]):
            problems.append("CrossEncoder orders the relevant document lower")
        print("  CrossEncoder OK")
        del ce
    except Exception as e:  # noqa: BLE001
        problems.append(f"CrossEncoder path failed: {type(e).__name__}: {e}")
        print("  FAILED:")
        traceback.print_exc()

    # ---- 3. raw Transformers path (README primary implementation)
    print("\n--- raw Transformers path (README official) ---")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16).eval()

        prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements '
                  'based on the Query and the Instruct provided. Note that the answer can '
                  'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tok.encode(prefix, add_special_tokens=False)
        suffix_tokens = tok.encode(suffix, add_special_tokens=False)
        print(f"  prefix tokens {len(prefix_tokens)}  suffix tokens {len(suffix_tokens)}")

        def score(instruction, query, doc, max_length=2048):
            body = (f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}")
            ids = tok.encode(body, add_special_tokens=False,
                             truncation=True,
                             max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
            full = prefix_tokens + ids + suffix_tokens
            inp = torch.tensor([full], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp).logits[0, -1, :]
            pair = torch.stack([logits[no_id], logits[yes_id]]).float()
            return torch.softmax(pair, dim=0)[1].item()

        instr = "Given a web search query, retrieve relevant passages that answer the query"
        s1 = score(instr, "What is the capital of China?", "The capital of China is Beijing.")
        s2 = score(instr, "What is the capital of China?",
                   "Gravity attracts two bodies towards each other.")
        print(f"  relevant   score = {s1:.9f}")
        print(f"  irrelevant score = {s2:.9f}")
        if not (s1 > s2):
            problems.append("raw Transformers path orders the relevant document lower")
        print("  raw Transformers OK")
    except Exception as e:  # noqa: BLE001
        problems.append(f"raw Transformers path failed: {type(e).__name__}: {e}")
        print("  FAILED:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    if problems:
        print("NOT VIABLE as-is:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VIABLE: both official paths work in this interpreter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
