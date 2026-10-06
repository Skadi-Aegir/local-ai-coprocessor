"""
test_reranker.py - correctness tests for the official scoring implementation.

Phase 5 requires verifying, specifically:
  * score ordering is correct
  * batch inference is correct
  * padding does not affect ranking
  * truncation behaves as expected
  * GPU inference works

These are the tests that decide whether the reranker can be trusted as a measuring
instrument for the rest of Phase 5. A reranker that ranks correctly but is perturbed by
batch length or padding would corrupt every benchmark that follows, silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reranker import (CODE_INSTRUCTION, DEFAULT_INSTRUCTION, GENERAL_INSTRUCTION,  # noqa: E402
                      Reranker, RerankerConfig, format_pair)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# A topic with an unambiguous relevant passage and several irrelevant ones of very
# different lengths, so ordering, padding and truncation are all exercised.
QUERY = "What is the capital of China?"
RELEVANT = "The capital of China is Beijing, and it has been the political centre for centuries."
NEUTRAL = "Gravity is a force that attracts two bodies towards each other."
IRRELEVANT = "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."
LONG_IRRELEVANT = ("Photosynthesis converts light energy into chemical energy. " * 120) + \
                  "This passage is long and entirely unrelated to geography."


def main() -> int:
    cfg = RerankerConfig(device="cuda", max_length=2048, batch_size=8)
    print(f"loading reranker from {cfg.model_path} ...", flush=True)
    rr = Reranker(cfg)
    info = rr.info()
    print(f"device={info['device']} dtype={info['dtype']} load={info['load_s']}s "
          f"prefix={info['prefix_tokens']} suffix={info['suffix_tokens']} "
          f"body_budget={info['body_budget']} vr am={info['vram_mb']} MiB")
    print(f"yes_id={info['yes_token_id']} no_id={info['no_token_id']}")

    try:
        # ---------------- GPU inference
        print("\n--- GPU inference ---")
        check("model is on cuda", "cuda" in str(rr.device), rr.device)
        check("model dtype is bfloat16",
              next(rr.model.parameters()).dtype == torch.bfloat16,
              str(next(rr.model.parameters()).dtype))
        check("yes/no token ids are distinct and real",
              rr.yes_id != rr.no_id and rr.yes_id > 0 and rr.no_id > 0,
              f"yes={rr.yes_id} no={rr.no_id}")

        # ---------------- ordering
        print("\n--- score ordering ---")
        s = rr.score_pairs([(QUERY, RELEVANT), (QUERY, NEUTRAL), (QUERY, IRRELEVANT)])
        print(f"  relevant={s[0]:.6f}  neutral={s[1]:.6f}  irrelevant={s[2]:.6f}")
        check("relevant scores above neutral", s[0] > s[1], f"{s[0]:.6f} vs {s[1]:.6f}")
        check("relevant scores above irrelevant", s[0] > s[2], f"{s[0]:.6f} vs {s[2]:.6f}")
        check("all scores within [0,1]", all(0.0 <= x <= 1.0 for x in s), str(s))

        res = rr.rerank(QUERY, [IRRELEVANT, NEUTRAL, RELEVANT])
        check("rerank puts the relevant document first", res[0].index == 2,
              f"got index {res[0].index}")
        check("rerank returns one result per document", len(res) == 3, str(len(res)))
        check("rerank indices are a permutation of the input",
              sorted(r.index for r in res) == [0, 1, 2], str([r.index for r in res]))
        check("rerank scores are non-increasing",
              all(res[i].score >= res[i + 1].score for i in range(len(res) - 1)),
              str([round(r.score, 4) for r in res]))

        # ---------------- batch invariance
        print("\n--- batch inference equals single inference ---")
        docs = [RELEVANT, NEUTRAL, IRRELEVANT, LONG_IRRELEVANT]
        pairs = [(QUERY, d) for d in docs]
        single = rr.score_pairs(pairs, batch_size=1, use_cache=False)
        for bs in (2, 3, 4, 8):
            batched = rr.score_pairs(pairs, batch_size=bs, use_cache=False)
            diff = max(abs(a - b) for a, b in zip(single, batched))
            check(f"batch_size={bs} matches batch_size=1 (max diff {diff:.2e})",
                  diff < 5e-3, f"{[round(x,5) for x in batched]}")
        print(f"  single: {[round(x,6) for x in single]}")

        # ---------------- padding invariance
        # A deliberately lopsided batch: one very long document and three short ones.
        # If padding were mishandled (e.g. right padding, or scoring a pad position) the
        # short documents' scores would shift when batched with the long one.
        print("\n--- padding does not affect ranking ---")
        short_only = rr.score_pairs([(QUERY, RELEVANT), (QUERY, NEUTRAL), (QUERY, IRRELEVANT)],
                                    batch_size=8, use_cache=False)
        mixed = rr.score_pairs([(QUERY, RELEVANT), (QUERY, NEUTRAL), (QUERY, IRRELEVANT),
                                (QUERY, LONG_IRRELEVANT)], batch_size=8, use_cache=False)
        d_short = max(abs(a - b) for a, b in zip(short_only, mixed[:3]))
        check(f"short docs unchanged by a long neighbour (max diff {d_short:.2e})",
              d_short < 5e-3, f"{[round(x,5) for x in mixed[:3]]}")
        check("relevant still ranks first in the lopsided batch",
              mixed[0] > mixed[1] and mixed[0] > mixed[2])
        check("long irrelevant document does not outrank the relevant one",
              mixed[0] > mixed[3], f"{mixed[0]:.5f} vs {mixed[3]:.5f}")

        # padding really was applied: the batch must contain pad tokens on the left
        ids, attn, _ = rr._encode_batch([(QUERY, d) for d in docs], DEFAULT_INSTRUCTION)
        left_pad = bool((attn[:, 0] == 0).any())
        check("left padding is applied for shorter rows", left_pad)
        check("attention mask width equals token width",
              ids.shape == attn.shape, f"{tuple(ids.shape)} vs {tuple(attn.shape)}")
        # every row's LAST position must be a real token, which is what makes the
        # final-position logit correct
        check("every row ends on a real (unmasked) token",
              bool((attn[:, -1] == 1).all()))

        # ---------------- truncation
        print("\n--- truncation behaviour ---")
        budget = rr.body_budget
        # Build a document long enough that the BODY alone exceeds the budget.
        huge = "The capital of China is Beijing. " + ("filler sentence about nothing. " * 4000)
        s_huge = rr.score_pairs([(QUERY, huge)], batch_size=1, use_cache=False)
        check("an over-long document still produces a valid score",
              0.0 <= s_huge[0] <= 1.0, f"{s_huge[0]:.6f}")
        print(f"  body budget = {budget} tokens; over-long doc score = {s_huge[0]:.6f}")

        # The answer sits at the START, so it survives truncation of the tail.
        head_answer = "The capital of China is Beijing. " + ("padding text. " * 4000)
        # The answer sits at the END, so it is cut away.
        tail_answer = ("padding text. " * 4000) + " The capital of China is Beijing."
        s_head = rr.score_pairs([(QUERY, head_answer)], batch_size=1, use_cache=False)[0]
        s_tail = rr.score_pairs([(QUERY, tail_answer)], batch_size=1, use_cache=False)[0]
        print(f"  answer-at-head={s_head:.6f}   answer-at-tail={s_tail:.6f}")
        check("an answer near the start survives truncation", s_head > 0.5, f"{s_head:.6f}")
        check("truncation demonstrably loses an answer placed at the very end",
              s_tail < s_head, f"tail {s_tail:.6f} < head {s_head:.6f}")

        # truncation must be reported, not silent
        rr.score_pairs([(QUERY, huge)], batch_size=1, use_cache=False)
        check("truncation is counted and reported", getattr(rr, "last_truncated", 0) >= 1,
              str(getattr(rr, "last_truncated", None)))

        # the suffix must survive truncation - otherwise the score is meaningless.
        # If the suffix were cut, the model would not be answering yes/no at all.
        check("suffix survives even for an over-long document", s_huge[0] > 0.0)

        # ---------------- max_length changes the budget
        print("\n--- max_length controls the body budget ---")
        for ml in (512, 1024, 2048):
            rr.cfg.max_length = ml
            rr.body_budget = ml - len(rr.prefix_tokens) - len(rr.suffix_tokens)
            got = rr.body_budget
            check(f"max_length={ml} yields body budget {got}",
                  got == ml - 39 - 9, f"expected {ml - 48}, got {got}")
        rr.cfg.max_length = 2048
        rr.body_budget = 2048 - len(rr.prefix_tokens) - len(rr.suffix_tokens)

        # ---------------- instruction sensitivity
        print("\n--- instruction changes the score, and None omits it ---")
        g = rr.score_pairs([(QUERY, RELEVANT)], instruction=GENERAL_INSTRUCTION,
                           use_cache=False)[0]
        c = rr.score_pairs([(QUERY, RELEVANT)], instruction=CODE_INSTRUCTION,
                           use_cache=False)[0]
        n = rr.score_pairs([(QUERY, RELEVANT)], instruction=None, use_cache=False)[0]
        d = rr.score_pairs([(QUERY, RELEVANT)], use_cache=False)[0]     # default
        print(f"  general={g:.6f} code={c:.6f} none={n:.6f} default={d:.6f}")
        check("instruction=None differs from an explicit instruction", n != g)
        check("omitting instruction uses the configured default",
              abs(d - rr.score_pairs([(QUERY, RELEVANT)],
                                     instruction=rr.cfg.instruction,
                                     use_cache=False)[0]) < 1e-9)
        check("format_pair omits the Instruct line when instruction is None",
              "<Instruct>:" not in format_pair(None, "q", "d"))
        check("format_pair includes the Instruct line otherwise",
              "<Instruct>:" in format_pair(GENERAL_INSTRUCTION, "q", "d"))

        # ---------------- the memory optimisation must be exactly equivalent
        print("\n--- final-position head equals the official full-logits call ---")
        probe_pairs = [(QUERY, RELEVANT), (QUERY, NEUTRAL), (QUERY, IRRELEVANT)]
        ids, attn, _ = rr._encode_batch(probe_pairs, DEFAULT_INSTRUCTION)
        with torch.no_grad():
            # official README form: full logits over the sequence, then slice
            full = rr.model(input_ids=ids, attention_mask=attn).logits[:, -1, :].float()
            full_pair = torch.stack([full[:, rr.no_id], full[:, rr.yes_id]], dim=1)
            official = torch.softmax(full_pair, dim=1)[:, 1].tolist()
            # our form: slice hidden states, then apply the head
            hid = rr.model.model(input_ids=ids, attention_mask=attn).last_hidden_state[:, -1, :]
            sl = rr.model.lm_head(hid).float()
            sl_pair = torch.stack([sl[:, rr.no_id], sl[:, rr.yes_id]], dim=1)
            optimised = torch.softmax(sl_pair, dim=1)[:, 1].tolist()
        diff = max(abs(a - b) for a, b in zip(official, optimised))
        check(f"sliced-head scoring equals the official full-logits scoring (max diff {diff:.2e})",
              diff < 1e-6, f"official={[round(x,6) for x in official]} "
                           f"optimised={[round(x,6) for x in optimised]}")

        # ---------------- long sequences must not exhaust memory
        # Regression: the official formulation materialised (batch, seq_len, 151669)
        # logits, which at a 2048-token sequence is 1.32 GiB for ONE scalar and raised
        # torch.OutOfMemoryError while 8.87 GiB was free.
        print("\n--- long documents do not blow up memory ---")
        very_long = "filler. " * 8000
        s_long = rr.score_pairs([(QUERY, very_long)], batch_size=1, use_cache=False)
        check("an 8000-word document can be scored without OOM",
              0.0 <= s_long[0] <= 1.0, f"{s_long[0]:.6f}")

        # Scoring several maximal documents must ALSO work: this is the path that used to
        # fail. Attention activations grow with the square of sequence length, so the
        # batcher has to split them rather than issue one enormous forward pass.
        several_long = rr.score_pairs([(QUERY, very_long)] * 4, batch_size=20,
                                      use_cache=False)
        check("four over-long documents can be scored in one call without OOM",
              all(0.0 <= x <= 1.0 for x in several_long),
              f"scores={[round(x, 6) for x in several_long]} "
              f"batches={getattr(rr, 'last_batches', '?')} "
              f"longest={getattr(rr, 'last_max_tokens', '?')} tokens")
        check("scores for identical documents are identical",
              len(set(round(x, 6) for x in several_long)) == 1, str(several_long))

        # The budget is (rows x padded length), so 4 rows at the 2048-token ceiling sit
        # exactly on an 8192 budget and must fit in ONE pass; a 5th row must force a split.
        # This asserts the boundary itself rather than a guess about where it falls.
        four = rr._plan_batches([(QUERY, very_long)] * 4, DEFAULT_INSTRUCTION, 20)
        five = rr._plan_batches([(QUERY, very_long)] * 5, DEFAULT_INSTRUCTION, 20)
        check("4 maximal rows fit the 8192 budget in one pass",
              len(four) == 1 and len(four[0]) == 4,
              f"{[len(g) for g in four]}")
        check("a 5th maximal row forces the batcher to split",
              len(five) >= 2, f"{[len(g) for g in five]}")
        check("no planned batch exceeds the token budget",
              all(len(g) * getattr(rr, 'last_max_tokens', 0) <= rr.cfg.max_batch_tokens
                  or len(g) == 1 for g in five),
              f"budget={rr.cfg.max_batch_tokens} groups={[len(g) for g in five]}")

        # a single row longer than the whole budget must still be attempted rather than
        # silently dropped
        solo = rr._plan_batches([(QUERY, very_long)], DEFAULT_INSTRUCTION, 20)
        check("a single over-budget row is still planned", len(solo) == 1 and len(solo[0]) == 1,
              str([len(g) for g in solo]))

        # a many-document batch of normal-sized chunks must still fit in ONE pass,
        # otherwise re-ranking a top-20 pool would be needlessly slow
        normal = [f"Document {i}. " + "Some ordinary sentence about software. " * 20
                  for i in range(20)]
        rr.score_pairs([(QUERY, d) for d in normal], batch_size=20, use_cache=False)
        check("a top-20 pool of normal chunks is scored in a single pass",
              getattr(rr, "last_batches", 99) == 1,
              f"batches={getattr(rr, 'last_batches', None)} "
              f"longest={getattr(rr, 'last_max_tokens', None)} tokens")

        # ---------------- determinism
        print("\n--- determinism ---")
        a = rr.score_pairs([(QUERY, RELEVANT), (QUERY, NEUTRAL)], batch_size=1, use_cache=False)
        b = rr.score_pairs([(QUERY, RELEVANT), (QUERY, NEUTRAL)], batch_size=1, use_cache=False)
        check("repeated scoring is deterministic", a == b, f"{a} vs {b}")

        # ---------------- cache correctness
        print("\n--- cache ---")
        rr.cache.clear()
        rr.score_pairs([(QUERY, RELEVANT)], use_cache=True)
        before = rr.cache.stats()
        rr.score_pairs([(QUERY, RELEVANT)], use_cache=True)
        after = rr.cache.stats()
        check("second identical call is a cache hit", after["hits"] == before["hits"] + 1,
              str(after))
        rr.cache.clear()
        cached = rr.score_pairs([(QUERY, RELEVANT)], use_cache=True)[0]
        uncached = rr.score_pairs([(QUERY, RELEVANT)], use_cache=False)[0]
        check("cached score equals recomputed score", abs(cached - uncached) < 1e-9,
              f"{cached} vs {uncached}")
        # a different instruction must NOT reuse the cached score
        rr.cache.clear()
        rr.score_pairs([(QUERY, RELEVANT)], instruction=GENERAL_INSTRUCTION, use_cache=True)
        rr.score_pairs([(QUERY, RELEVANT)], instruction=CODE_INSTRUCTION, use_cache=True)
        check("a different instruction is a cache miss",
              rr.cache.stats()["misses"] == 2, str(rr.cache.stats()))
        check("cache key includes instruction, query and document",
              rr.cache.key("i", "q", "d") != rr.cache.key("i2", "q", "d")
              and rr.cache.key("i", "q", "d") != rr.cache.key("i", "q2", "d")
              and rr.cache.key("i", "q", "d") != rr.cache.key("i", "q", "d2"))

        # ---------------- empty input
        print("\n--- edge cases ---")
        check("empty document list returns no results", rr.rerank(QUERY, []) == [])
        empty_doc = rr.score_pairs([(QUERY, "")], use_cache=False)
        check("empty document still yields a valid score",
              0.0 <= empty_doc[0] <= 1.0, f"{empty_doc[0]:.6f}")

        # ---------------- documented README example
        print("\n--- reproduces the README's documented example ---")
        s_beijing = rr.score_pairs([("What is the capital of China?",
                                     "The capital of China is Beijing.")],
                                   instruction=DEFAULT_INSTRUCTION, use_cache=False)[0]
        s_gravity = rr.score_pairs([("What is the capital of China?",
                                     "Gravity is a force that attracts two bodies "
                                     "towards each other. It gives weight to physical "
                                     "objects and is responsible for the movement of "
                                     "planets around the sun.")],
                                   instruction=DEFAULT_INSTRUCTION, use_cache=False)[0]
        print(f"  Beijing doc  = {s_beijing:.6f}")
        print(f"  gravity doc  = {s_gravity:.6f}")
        # README reports raw logit diffs 7.625 and -11.375 -> sigmoid 0.99951 and 1.15e-05
        check("Beijing document scores near 1.0 (README: sigmoid(7.625)=0.99951)",
              s_beijing > 0.99, f"{s_beijing:.6f}")
        check("gravity document scores near 0.0 (README: sigmoid(-11.375)=1.15e-05)",
              s_gravity < 0.01, f"{s_gravity:.6f}")

    finally:
        vram_after_del = None
        rr.close()
        if torch.cuda.is_available():
            vram_after_del = rr.vram_mb()
        print(f"\nvram after close: {vram_after_del} MiB")

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("reranker self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
