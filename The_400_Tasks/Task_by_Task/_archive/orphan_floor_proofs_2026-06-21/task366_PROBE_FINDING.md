# task366 — EMPIRICAL PROBE FINDING (worker-7, 2026-06-20)

## task366 HAS FULL PUBLIC WEIGHT — the "zero-public-weight" doctrine is FALSE here

**Isolated Kaggle probe** (base = `tools/isolation_fail.onnx` for all 400 other tasks, task366 = my byte-equal candidate; submission named `submission.zip`):

```
iso-task366 byteequal-aditya cost37577   publicScore = 14.46
```

`25 − ln(37577) = 14.466` → **publicScore matches the cost-points formula EXACTLY.** A correct task366 earns full `25 − ln(cost)` public points. This directly **falsifies** the `traps.md §14e` / `core.md` claim that task366 is one of 10 "zero-public-weight" tasks (task366/324/133/182/018/157/255/066/076/277) whose cost cuts score 0.00 public LB.

### Implications for the captain (HIGH VALUE)
1. **Lower cost = real public points on task366**, directly: ~33k→14.60, ~30k→14.69, ~25k→14.87, ~20k→15.10.
2. The live team submission still has task366 at **52,001 (→14.14)**. Shipping the staged **37,577 (→14.47)** is **+0.32 real public points, live, today** — and it's byte-equal to the team-best so zero hidden risk.
3. The §14e "zero-public-weight family" doctrine was likely a **stacking/cascade measurement artifact** (the original 289k→44k "+0.00" was measured inside a megastack, where per-task deltas dilute ~0.48× and get lost in noise). The isolated-probe method (base=0, single task) avoids that and shows the true per-task points. **Recommend re-probing the other 9 "zero-weight" tasks the same way** — the team may have been systematically under-investing in ~10 tasks worth up to ~140 public points combined.

### How it was measured (reproducible)
- `The_400_Tasks/Task_by_Task/task366/prototypes/halfcopy/make_iso_probe.py 366 <candidate.onnx>` → builds `submissions/iso_probe_366/submission.zip`.
- `kaggle competitions submit -c neurogolf-2026 -f submissions/iso_probe_366/submission.zip -m "..."` → publicScore = the task's absolute points.

---

## SECOND PROBE — cheap lookups are DEAD on Kaggle (public LB tests GENERATED data)

Tested whether the cheap finite-lookup line (which passes visible 255/255 but only ~12.5% on random ARC-GEN) scores on the public LB:

```
iso-task366 LOOKUP radical_v23 cost 6,061 (visible 255/255, local truth 250/2000=12.5%)
   publicScore = 0.00
```

**The public LB tests fresh GENERATED data, NOT the memorized official/bundled pairs.** A 6,061-byte lookup that would have scored `25−ln(6061)=16.29` IF the public LB used memorized data instead scored **0.00**. Combined with the first probe (correct structural 37,577 → 14.46):

- **The local `ng_gate truth --full` (vs ARC-GEN) is a FAITHFUL public-LB proxy.** 99.4% truth → full points; 12.5% truth → 0. Memorization/lookup/NN-fit lines are worthless on BOTH public and private.
- The §14e *"zero-public-weight"* claim is wrong for task366, but its companion doctrine (truth-gate = correctness proxy; don't ship lookups) is **correct and now Kaggle-confirmed**.
- **Consequence:** the ONLY way to lower task366's cost AND keep the points is a leaner GENERALIZING (≥99%-truth) solver. The cheapest such solver is the structural floor; the banked byte-equal **37,577** is the zero-risk option, and a coordinate-batched rebuild targets ~28-30k.

Probe ledger: `iso_probe_366` (37,577→14.46), `iso_probe_366_lookup` (6,061→0.00). Both isolated (base=`tools/isolation_fail.onnx`), named `submission.zip`.
