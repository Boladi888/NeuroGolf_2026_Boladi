# Transferable insight — a MONOTONIC priority/weight LOOKUP TABLE feeding `Where(mask, table, 0) → ArgMax` folds for free into `ArgMax(mask, select_last_index=1|0)`: delete the whole table (−N params). NOT a duplicate-init prune — a positional-encoding fold.

**Source**: task209 (`bo_ng_cld_worker1`, Claude lane, 2026-07-11). LIVE/team-best **4917**
(v15, sha `9662e34a…`, byte-equal to `submission.zip`). New staged rung **v16 = 4902 (−15)**,
sha `ef2d7e65…`, status **PROBE_READY**. First application of this fold on the team.

## The pattern it kills
A hypothesis/candidate selector that picks "the valid cell with the highest priority" is often built as:
```
sel  = Where(mask_bool[R,C], priU[R,C], 0)   # priU = a stored priority WEIGHT table
flat = Reshape(sel, [R*C])
bidx = ArgMax(flat)                          # first-max = the max-priority valid cell
```
where `priU` is an initializer (R*C params). task209 had `priU[3,5] = 1 + 8r + c` = **15 params**.

**Key observation:** if the priority table is STRICTLY MONOTONIC in the row-major flat index
(here `priU=1+8r+c` and flat=`5r+c`; both are monotone in (r then c) because `c ≤ C-1 < ` the row
coefficient), then `argmax(priU over the valid cells)` is *identically* **the last valid cell in
flat order**. So the whole table + Where collapses to:
```
bidx = ArgMax(Cast(mask,u8) reshaped to [R*C], select_last_index=1)
```
- Deletes the priority initializer entirely → **−(R*C) params** (task209: −15 → 4902).
- `select_last_index=1` ⇒ max flat index; use default (`=0`, first index) if priority is
  DESCENDING in flat order. Pick the attribute to match the existing tie-break direction.

## Two gotchas (both hit on task209)
1. **ArgMax has NO bool kernel** (ORT 1.24.4: `Type 'tensor(bool)' … invalid`). You must `Cast`
   the mask to a numeric type (u8/i8) first. That Cast is an [R,C] tensor the same size as the old
   `Where` output, so the **memory is unchanged — the win is params-only** (−N). Do not expect a
   memory saving too.
2. **The all-zero-mask fallback CHANGES.** `Where+ArgMax(first-max)` on an all-zero mask returns
   flat index **0**; `ArgMax(mask, select_last_index=1)` returns the **last** index (R*C−1). So the
   candidate is **NOT byte-equal** to the old graph on any input where no cell is valid. On task209
   this fired on 4/2000 arc-gen pairs (all-zero `cons`). Forcing the fallback back to 0 needs a
   sentinel-True at index 0, which costs ≥ the params saved (Or + sentinel init) — so byte-equality
   and the param win are mutually exclusive here.

## Why it can still ship (the correctness-neutral proof)
Classify the divergences with `corpus.get_tensors(task, n)` + run BOTH models vs ground-truth Y:
on task209 **all 4 divergences were BOTH-WRONG** (v15 wrong AND v16 wrong) — 0 regressions,
0 improvements, truth **1982/2000 == 1982/2000**. So the new graph passes exactly the tasks the
old one passes and fails exactly the ones it fails ⇒ correctness-neutral, hidden pass/fail set
unchanged on the tested window. Residual hidden risk is confined to all-zero-mask cases whose TRUE
answer is exactly the old fallback cell (flat 0) — small, characterized ⇒ **PROBE_READY**, not
LOCAL_GOLD (byte-equal is the LOCAL_GOLD gate and is absent).

## Rule of thumb
When auditing an over-parameterized selector, don't only look for duplicate/unused inits — look for
**monotonic-weight LOOKUP TABLES that only feed an ArgMax**. If the table is monotone in flat order,
`select_last_index` (or default first-index) reproduces the argmax and the table is free to delete.
Then: Cast the bool mask to u8 (ArgMax rejects bool), expect a **params-only** win, and gate the
**fallback divergence** with a both-wrong classifier before calling it shippable (PROBE_READY unless
you can also pin the all-zero fallback for byte-equality).
