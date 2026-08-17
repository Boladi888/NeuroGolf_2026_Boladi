# Research Card R4 (REPO-ACCESS) — Harvest task312 selector-bank ranks 17→13 with OR-Tools CP-SAT

**Type**: repo-access execution (the narrowest, most concrete card on the board). **Deliverable**: `Logs/research/R4_task312_deep_rank.md` + a staged rebuilt candidate in `staged_submits/task312_rankNN_cost<C>/` for any exact sub-18 rank found. This is the direct continuation of **R2** (`../Done/R2_REPOACCESS_*`, which shrank the bank to rank-18 and shipped 600→570) using the **M2 recipe** (`../Done/m2_exact_rank_tooling_recipe.md`). R2 stalled ONLY because the exact solver wasn't installed locally — M2 resolved that. This card just runs it.

## Why this is worth it
task312's selector-bank rank residual is the single biggest known remaining edge (+~0.46 at theoretical rank-10; +0.2–0.3 realistic at ranks 16–13). The bank is currently an exact rank-18 CP (shipped, cost 570). Every rank you remove exactly is a direct, byte-equal team gain (boladi is team-best on 312).

## Setup (the one thing R2 couldn't do)
`pip install ortools` (OR-Tools CP-SAT — free, MIT, exact-integer; M2's verdict-winning solver). That's the whole blocker R2 hit.

## Read first
`../Done/m2_exact_rank_tooling_recipe.md` (the runnable CP-SAT script + the recipe: S={0,±1} first, **table-encoded products** not raw bilinear, first-nonzero-positive normalization on A&B, support-code ordering of rank-1 terms, explicit `|prefix| <= 2^24` guards), `../Done/R2_REPOACCESS_*` (how the rank-18 bank was found + extracted), `techniques.md` #87 (flattening-rank kill-test + the FREE-OUTPUT SEPARABILITY TRAP) + #68 discriminator, `WORKER_PROTOCOL.md`.

## Method
1. **Extract the bank**: pull the rank-18 selector-bank tensor from the shipped task312 onnx (`submissions/submission.zip::task312.onnx` or the archived `staged_submits/_shipped_archive_*/task312_rankcut_cost570/`). Confirm its shape + current element count.
2. **Step-0 kill-test FIRST** (cheap): recompute the three matricization/flattening ranks. If any == 18, the bank is PROVEN minimal at 18 → record DEAD, stop (a valid "no headroom" close).
3. **CP-SAT descent**: per M2's script, search T=17, then 16, 15, 14, 13 at S={0,±1}; widen to {0,±1,±2} and the structured dictionary (basis/ones/class-indicator/cyclic-shift/diagonal) before free dense factors. Enforce exactness (integer vars) + the `|prefix|<=2^24` contraction bound. Time-box per M2's budget; record OPEN if a rank doesn't converge (do not claim minimality you didn't prove).
4. **Rebuild + gate** any exact shrink: rebuild task312.onnx with the smaller bank, run ALL 5 gates (visible 100% → truth --full → ab --full byte-equal vs LIVE → score_v2 → gate-5 real-input profile). Stage `staged_submits/task312_rank<NN>_cost<C>/` + manifest. Captain ships.

## Output
`Logs/research/R4_task312_deep_rank.md`: per rank tried — `T, alphabet, dictionary(Y/N), result(FEASIBLE+staged / INFEASIBLE-in-alphabet / OPEN-timeout), new_elements, point_gain`. Plus the final verdict: the lowest exact rank found, whether it's a proven floor (flattening-rank match or exhausted alphabet), and the captured point-gain.

## Discipline
- Exact only: the rebuilt bank must reproduce the einsum output BYTE-EQUAL vs LIVE on the full corpus — that's the ship gate.
- Honest certificates: "infeasible at S={0,±1}" ≠ "rank-17 impossible over ℝ" — label results by alphabet (M2 covers this).
- If the flattening-rank kill-test shows 18 is minimal, that's a clean, valuable close (the +0.46 was theoretical; record it dead and move on).
- Captain ships from the staged dir; never submit.
