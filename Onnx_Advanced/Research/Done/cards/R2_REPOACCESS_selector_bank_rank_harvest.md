# Research Card R2 (REPO-ACCESS) — Execute exact selector-bank rank reduction on the live einsum-final graphs

**Type**: repo-access research + execution. **Deliverable**: `Logs/research/R2_selector_bank_rank.md` (findings + per-bank verdicts) and, for any bank where a smaller exact decomposition is found, a staged rebuilt candidate in `staged_submits/` (gated, for captain to ship). This is the EXECUTION of the L1/M2 method (`../Done/l1_exact_cp_decomposition.md`) on our ACTUAL banks — the thing internet-only cards can't do.

## Why this is high-value

The einsum-final class (#68/#87/#89) made selector-bank element count the params bottleneck. Several shipped graphs carry banks at a known construction rank that may not be the true minimum — the registered residual is **task312 (+0.51, T=19 vs theoretical ≥10)** plus siblings (304, 357, and any R1 surfaces). L1 proved the method (flattening-rank kill-test → bounded-integer/dyadic CP via MIQCP). M2 (`Todo/M2_*`, internet card) is producing the solver recipe. R2 RUNS it on our real tensors.

## Read first
`../Done/l1_exact_cp_decomposition.md` (the method), `techniques.md` #87 (flattening-rank kill-test, FREE-OUTPUT SEPARABILITY TRAP) + #68 discriminator + #89 (einsum operand order = contraction path), `WORKER_PROTOCOL.md`.

## Method
1. **Inventory the banks**: from the shipped einsum-final tasks (312, 304, 357 — grep `live_costs.json` notes for "einsum"/"selector"/"contraction"; add any R1 flags), extract each selector-bank initializer tensor from the task's onnx (staged archive `staged_submits/_shipped_archive_*/` or `submissions/submission.zip`). Record shape + current element count.
2. **Step 0 kill-test (cheap, do first)**: compute all matricization/flattening ranks of each bank — these are unconditional CP lower bounds. If a flattening rank == the current construction rank, that bank is PROVEN minimal → record DEAD, move on. (This alone may close 312's residual either way.)
3. **Bounded-integer CP search** (only if flattening rank < current): per L1/M2, run a bounded-alphabet feasibility search (S={0,±1} then {0,±1,±2}) with first-nonzero=+1 normalization + lexicographic support ordering, dictionary-restricted (basis/ones/class-indicator/cyclic-shift/diagonal) before free factors. Use whichever solver M2 recommends (or OR-Tools CP-SAT / Z3 if M2 not yet back). Enforce exactness (integer/rational, not rounded-float) and the f32≤2^24 contraction bound.
4. **Rebuild + gate** any bank that shrinks: rebuild the task's onnx with the smaller bank, then ALL 5 gates (visible 100% → truth --full → ab --full byte-equal vs LIVE → score_v2 → gate-5 real-input profile). Stage to `staged_submits/taskNNN_rankcut_cost<C>/` + manifest. Captain ships.

## Output
Per bank: `task, bank_shape, current_rank/elements, flattening_ranks, min_exact_rank_found, alphabet, new_elements, point_gain, status(DEAD-proven-minimal / SHRUNK-staged / OPEN-timed-out)`.

## Discipline
- A flattening rank == construction rank is a PROOF the bank is minimal — that closes the residual honestly (record it; it is a valid "no headroom" result, not a failure).
- Exact only: the rebuilt bank must reproduce the einsum output byte-equal vs LIVE on the full corpus — that is the ship gate; a rank cut that breaks byte-equality is wrong.
- Time-box per L1 (~2–3 h/bank at S={0,±1}/{0,±1,±2}); record OPEN if it doesn't converge.
- End with a one-paragraph verdict on task312 specifically: is the +0.51 real (shrinkable) or proven-minimal?
