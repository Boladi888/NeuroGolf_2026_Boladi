# Transferable insight — gated mutually-exclusive union: SELECT-then-threshold beats per-branch-threshold-then-OR

**Source**: task042 batch 54 (`bo_ng_cld_worker3`, 2026-07-08). Live 2,522 (Boladi) → **2,322** staged (−200), byte-equal to LIVE (A/B 200266/200266). Beat a standing WALL that two prior workers (W4/W8) had signed off as floor-minimal.

**Task shape**: N mutually-exclusive detector branches selected by a data value (here a foreground **count** → distance n∈{1,2,3}). Each branch = a wide (u8) signal `qmax_nk[1,1,10,10]` thresholded to a BOOL "fired" plane, then all BOOLs OR'd into one paint mask. The classic realization:
```
fired_k = Greater(qmax_nk, thr_k)      # N BOOL planes
paint    = fired_1 Or fired_2 Or …      # (N-1) more BOOL planes  →  (2N-1) planes total
```

**The lever**: because the branches are **mutually exclusive** (a data value picks exactly one), you can SELECT the active branch's *pre-threshold* signal first, then threshold ONCE:
```
sel   = Where(g1, qmax_1, Where(g2, qmax_2, … qmax_N))   # (N-1) wide planes
thr   = Where(any_gate, thr_val, HIGH)                    # scalar
paint = Greater(sel, thr)                                 # 1 BOOL plane  →  N planes total
```
Cost drop = **(N-1) × plane_bytes** of memory (N=3 → 5 planes→3 planes = −200 B here). `Where(bool, u8, u8)` is ORT-supported; nest for N>2.

**Why it's byte-equal (the correctness argument)**: gates are disjoint & exhaustive, so `OR-of-independently-gated ≡ select-active-then-threshold`. Off-branch signals are *dropped by the select* instead of *suppressed by a sentinel threshold*. The no-branch (invalid data) case is held at "no fire" by selecting `HIGH` (255) into the threshold via an `any_gate` OR.

**The trap that hid this from prior workers**: the WALL note rejected the obvious `Max(u8)+single Greater` because "the branch thresholds differ (taps 2/8/18) — needs a shared threshold." Two things were missed:
1. `Max` leaks an inactive branch's high signal; **`Where`-select does not** — it reads only the active branch, so a per-branch (or here uniform) threshold is valid.
2. The live graph did **not** threshold at the full per-branch tap count; it threshold at a **uniform `>1`** folded with the gate (`Where(gate, 1, 255)`). So the "thresholds differ" objection was a misread of the actual graph. **Always re-derive the live threshold semantics from the ONNX before trusting a floor note that says a combine is minimal.**

**Applies to**: any task with a runtime-selected 1-of-N branch union where the per-branch signal is *wider than BOOL* (u8/f16/f32 conv or arithmetic planes) — argmax-gated renderers, count-gated / size-gated / color-gated detector stacks. Net win = (N−1)×(sizeof branch plane). No gain if branches are already BOOL and N=2 (select=1 plane = Or=1 plane).

**Gate discipline that caught it cleanly**: visible 266/266 → profile static=profile=2322 (<2522) → capped truth/ab 20k → full truth 200k / ab 200266 byte-equal. Fully static graph ⇒ profile==static (no credit-probe risk).
