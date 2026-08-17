# task133 LIVE robustness finding — flat-Gather col index is unclamped (OOB crash on ~0.1% arc-gen)

**Recorded:** 2026-07-12 by `bo_ng_cld_worker3` (Claude lane). Affects the LIVE ship AND every byte-equal descendant.

## The bug
Live 16,288 (owner Boladi, sha `a622a54e…`) node `g4 = Gather(gflat, flat_i, axis=0)` (the template-color probe at 4 sample points/channel) uses a flat index
```
flatf = r4*30 + c4b        # r4 = Clip(r4b, 0, 29)   <- clipped
                           # c4b = leftf + sxlH + h2  <- NOT clipped
```
The row `r4` is clipped to `[0,29]` but the col term `c4b` is **not**. When `c4b` reaches 30, `flatf` reaches `29*30+30 = 900`, which is OOB for the 900-element `gflat` (`[0..899]`). ORT then raises `indices element out of data bounds, idx=900` and the whole inference throws.

## Rate / where it bites
- Deterministic gate corpus (`tools.ng_harness.corpus`, seed=SEED_OFFSET+raw_id): first crash at **corpus index 538**; ~3 in 3,000 (~0.1%).
- Random generator sweep: ~2 in 6,000.
- **Official corpus is clean**: all 5 train/test + 262 arc-gen officials run fine (that is why the ship gated OK on officials and why `profile_real_cost --arcgen 266` runs 267/267).
- Consequence: `ng_gate ab --n 2000` / `truth --n 2000` **abort** on this graph family (they stream past corpus-538). Max clean is `--n 538`.

## Why it matters to the team
- The live graph almost certainly **loses ~0.1% of the hidden arc-gen test set to crashes** (a throwing graph scores 0 on that pair). Any byte-equal descendant inherits this exactly (no regression, but no free fix either).
- Any future task133 gate MUST use `--n <=538` or a crash-tolerant comparator; do not read an aborted `--n 2000` as a candidate failure.

## Candidate fix (NOT taken this session — separate from the cost win)
Add `c4b = Clip(c4b, 0, 29)` (or clip `flatf` to `[0,899]`) before the Gather. Cost: +~80 B (one extra `[1,10,1,4]` F16/INT tensor) — eats into savings and is **NOT byte-equal** to live (diverges on the OOB inputs: live crashes, fixed runs). Correctness of the clamped probe on those inputs is **uncertain** (off-grid col 30 clamped to 29 may misreport template color), so it is a `RISK_EXPERIMENT` needing its own truth-gate on the OOB slice, not a clean win. Left for a future correctness-focused card; the cost win (`drop_s3_topk3` 15,208) is byte-equal and independent of this.

## Hypothesis to test if anyone revisits
On the 3 known OOB corpus indices (start 538), build the clipped variant and check its output vs arc-gen ground truth. If it matches GT → the fix recovers ~0.1% hidden (correctness upside worth the +80 B); if not → it only trades a crash for a wrong answer (skip).
