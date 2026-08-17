# Deep-read: `neurogolf-2026-4-27-lb-5126.ipynb` (Bug-Free Blend, LB 5126.37)

**Date read:** 2026-05-03
**Author chain:** layered on jonathanchan V7 → aliafzal9323 (286-task baseline) → artemnazemtsev part-3 → needless090 v5 → notebook author's custom 49 tasks

## Headline

The notebook is a **submission-packaging artifact** (loads bundled v84 ONNXs into `submission.zip` for Kaggle). The real content is the bundled dataset (NOT directly accessible from the notebook itself), but the markdown documents the techniques + composition + LB lineage in detail.

## Composition (v84)

| Layer | Tasks | Source |
|---|---:|---|
| Base ONNXs | 394 | jonathanchan V7 |
| Hand-coded cheap rewrites | 3 | t389, t282, t015 (this author) |
| `dim_value` scrub on v18 base | 47 | technique from `@poonszesen` |
| `dim_value` scrub on V7 base | 6 | additional |
| Unique to this notebook | 1 | t055 |
| **Total** | **395** | |

## Three "cheap rewrite" techniques

These are alternate ONNX implementations the author authored to **lower cost** on already-solved tasks (3 of them). Source files (`*_cheap.py`) bundled in the dataset; we don't have direct access but the markdown describes each:

1. **`t389_cheap.py` — color-swap via `ReduceSum + Greater + Mul-broadcast`.** Replaces a 5×5 dense Conv (cost 2.29M, ~10.35 pts) with a Sub+Greater+Mul pipeline at 184k cost → ~12.88 pts (+2.53 pts). **Note for us:** task389 is in our 200 external-only category — we currently don't solve organically. This rewrite is competitive with our potential color_perm/color_replace; worth investigating if we can replicate.
2. **`t282_cheap.py` — 3×3 stamp around each "5" cell, two parallel Convs with hand-set kernels.** +2.04 pts on task282. **Note for us:** task282 is SOLVED via our `bin-nbr-lut` at cost ~169k, ~12.96 pts. The stamp-rewrite likely lower cost; cross-check would be useful.
3. **`t015_cheap.py` — per-input-color stamp** (color-1 → ortho 7s, color-2 → diag 4s). +1.74 pts on task015. **Note for us:** task015 is in our truly-unsolved 150 set (no organic, no external in our blend). This pattern could be borrowable.

## `dim_value` / `dim_param` scrub technique

Not new in this notebook (originally from `@poonszesen`) but documented here: **clearing the H/W dims of the input/output tensors** (replacing concrete values with empty `dim_value`) **survives Kaggle's 1.19.2 grader for 53 tasks** and reduces the reported cost. This is one of the "exploit" families per `Research/A2_cost_metric_exploits.md` — out of scope for our ethical-play stance.

**12 confirmed-BAD tasks where scrub breaks scoring with `SubmissionStatus.ERROR`:**
`t004, t017, t034, t064, t066, t074, t119, t192, t231, t312, t329, t359`

These tasks have specific shape-inference dependencies that the scrub disrupts. **Useful for us as a "do not touch" list** if we ever experiment with dim_param clearing — though we shouldn't be doing that anyway per ethical-play.

## Two metric-update timeline (clarifies open bug)

The notebook header explicitly references the **April 21st update** ("the host's announced rescoring fixes"). Combined with the documented April 28th update, this **confirms two metric updates in 2026-04 — not one.**

This clears the open bug item in `03_PROJECT.md` "Two metric updates not one (2026-04-21 + 2026-04-28)". Specifically:
- April 21: introduced `Compress` ban + scoring fixes that disabled the "afr1ste 5395" lineage.
- April 28: documented in our existing notes; further refinements.

Specifics of April 21 still not fully documented in the notebook beyond the high-level "rescoring fixes" reference. May warrant a Kaggle discussion-forum search if we get access.

## Silent-fail handling

Notebook explicitly addresses **task363 silent-fail** (passes local visible-pair `verify_subset` but scores 0 on Kaggle's hidden test). Their fix: replaced with a custom 265/265-pair-verified ONNX based on a 5-AI rule consultation (GPT/Gemini/DeepSeek/GLM/Kimi). Rule: **shift-based correlation with sparse-line tie-break for 1×4 / 4×1 line seeds**.

**Implication for us:** task363 is in our truly-unsolved 150 set. The author's rule (shift-correlation with line-seed tiebreak) would be the implementation target if we wanted to add task363. Pre-condition: would need to verify against arc-gen pairs.

## What this implies for NG

1. **We already cleared the "Two metric updates" open bug** — close it.
2. **Worth checking** whether our `bin-nbr-lut` task282 ONNX is more or less expensive than the 3×3 stamp rewrite. If theirs is cheaper, we could lift it as a `[B→A]` Tier-1 candidate.
3. **task389 cheap-rewrite** (color-swap with ReduceSum + Greater + Mul) is a potential `[B→A]` since task389 is currently external-only (we don't solve it organically). Worth attempting if we can reverse-engineer the exact pipeline from the description.
4. **task015 cheap-rewrite** (per-input-color stamp) is a potential `[B→A]` for an UNSOLVED task — direct LB gain if implementable.

## GAPS

- Don't have direct access to `t015_cheap.py` / `t389_cheap.py` / `t282_cheap.py` source files. The markdown gives high-level descriptions but not enough for line-by-line replication. Owner can pull these files if Kaggle dataset access is available.
- Don't have the `scrub_list.json` content beyond the BAD list. Knowing the 53 SAFE tasks could be educational but we're not exploiting scrub anyway.

## Sources

- `Research/primary_sources/neurogolf-2026-4-27-lb-5126.ipynb` (this notebook)
- `Research/A2_cost_metric_exploits.md` (background on `dim_param` scrub family)
- `03_PROJECT.md` Open bugs (the "Two metric updates" item this clears)
