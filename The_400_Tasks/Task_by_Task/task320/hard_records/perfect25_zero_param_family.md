# task320 — perfect-25 zero-param family DEAD (ceiling-audit rumor disproven)

**Live bar:** 88 bytes, SHA256 `3b60b3cd73e4985df112d876a12b57ad7dbc9cd62e7ed87b71d9e2b22ee1c016`,
source `python tools/ng_current_bar.py --task 320 --json` → manifest (owner Vamshi, 2026-07-02).
(Note: local `task_state.csv`/`live_costs.json` show 706 / 10,675 — STALE. Authoritative = 88.)

**Rumor audited:** prompt `Prompt_...5646_Task320_ZeroopsweepCeilingAudit.md` posits task320 is a
"perfect-25" task reachable by a no-payload direct-output op priced ≤1 (Identity / any Transpose perm /
whole-grid flip / rotate). Prize claim: 88 → floor_est ~1.

**Attempted route:** analytic one-shot sweep of the entire zero-param shape-only family across the
FULL corpus (4 visible train/test + 262 arc-gen = 266 pairs) BEFORE any ONNX build, per the prompt's
fast-probe kill rule.

**Failed gate:** exact-match on the corpus. Every family member is 0/266 exact:

| transform | exact | shape-match |
|---|---|---|
| identity | 0/266 | 266/266 |
| flipud | 0/266 | 266/266 |
| fliplr | 0/266 | 266/266 |
| rot180 | 0/266 | 266/266 |
| transpose | 0/266 | 44/266 (square grids only) |
| rot90 | 0/266 | 44/266 |
| rot270 | 0/266 | 44/266 |
| transpose_anti | 0/266 | 44/266 |

A visible/corpus miss = certain hidden 0 (traps §VISIBLE-FAIL). No ONNX was built and no
truth/ab/battery/profile gate was run — correctly, since there is no visible-correct cost≤1 candidate
(kill rule: do not gate a visible-zero).

**Why it's dead — forced-component floor argument (cost ≤1 is impossible):**
1. `input.shape == output.shape` in 266/266 AND the nonzero mask is identical in 266/266 → the transform
   is a **pure recolor**, never a shape permutation. The only no-payload cost-1 ops (Identity / Transpose)
   reproduce a fixed shape-permutation of the input; a pure recolor is not one → structurally excluded
   (confirmed 0/266).
2. The recolor is **not** a single global per-cell color bijection (tested: inconsistent across pairs) →
   it cannot be a fixed color-map op either; the output color of a cell depends on its position.
3. The recolored region is **data-dependent and positional**: only non-background cells change
   (0/2933 changed cells were background) and changes concentrate in the bottom portion of each grid
   (changed-row fraction ≈0.63→1.0) — i.e. the "staircase bottom-half recolor" whose boundary is set by
   per-column counts (this is exactly what the live 88-byte graph computes:
   Slice→ReduceSum(cols)→Conv cumsum→Greater threshold→Where recolor→Mul delta→Add→Pad).
4. Producing a data-dependent region mask forces ≥1 materialized intermediate AND a color/delta payload
   tensor — both charged. Therefore any correct graph charges >1 byte. **cost ≤1 is forced-impossible.**

**Verdict:** WALL on the perfect-25 hypothesis. task320 is a genuine algorithmic recolor, not a
zero-param shape task. The captain's ceiling table should drop the ~1-byte floor_est for task320; the
real forced floor is a data-dependent-recolor graph, and the live 88-byte v4 graph already realizes it.

**One remaining hypothesis (a byte-SHAVE below 88, NOT a perfect-25 route — out of scope for this
rumor-hunt slot, and separately walled):** shrink the live basis/delta payload of the 88-byte graph.
The prior sparse-initializer probe already died (see `sparse_initializer_probe.md`: scorer returns
`None` on sparse inits). Any future shave must be a dense param-factorization of the recolor payload that
adds no charged intermediate, and — because task320 is flagged **hidden-bad** (`LIVE_ALGORITHM.md`) — must
byte-match the live algorithm on hidden inputs, not merely pass visible+arc-gen.
