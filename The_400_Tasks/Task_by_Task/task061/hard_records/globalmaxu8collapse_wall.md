# task061 GlobalMaxU8Collapse — WALL at 1,668 (hard record)

**Live bar:** 1,668 (manifest, owner Aditya, `2026-06-19T12:21:50Z`).
**Live sha256:** `ca895cf016d83d9fdbcadf4e03a66caa491ec02314138a789a56ccd0a62addcd`
(Team_Best_Onnx/task061.onnx).
**Verdict (batch 5405, 2026-07-08):** WALL. No static opset-16 graph honouring the
`[1,10,30,30]` ABI beats 1,668. Prompt `floor_est ~500` is an artifact — it ignores
the ABI-forced 900 B canvas and the two mul-then-mod-forced 324 B planes.

## Baseline reproduced this batch (guarded plain-Mul prototype v9)
- `ng_gate visible --task 61`: **267/267 (100.00%) PASS**
- `_score_v2`: `(0, 1634, 34)` → **static 1,668**
- `profile_real_cost --arcgen 266`: **static=1668 profile=1668** (mem 1634 + params 34; 267 real inputs)

## Forced-component floor proof (why 1,668 is the floor)
| component | bytes | why forced | best alt tried → cost |
|---|---:|---|---|
| `idx30` 30×30 U8 canvas | 900 | ABI output is `[1,10,30,30]`; Equal-LAST charges one 30×30 U8 plane (min for a 30×30 intermediate; U8/BOOL are 1 B/elem, opset-16 has no sub-byte dtype) | Pad-LAST **v18=4012** (10×18×18 BOOL core = 3,240 B); OneHot-tail **v15=15,347** |
| `prod` 18×18 U8 | 324 | `(a·b)%n` needs `a·b` materialised; no fused mul-mod op in opset-16 | INT16 raw product = 648/plane (2×) |
| `mod0` 18×18 U8 | 324 | the `%n` residue map fed to Equal; periodicity can shrink the *block* but re-tiling to 18/30 costs more | block-tile **v16=1726**, tile-to-30 **v17=1763** (Gather INT index + [30,ncol] surface > 648 saved) |
| detector | 50 | `ReduceMax[1,10]F32`(40)+`ArgMax`i64(8)+casts(2) — per-channel presence is minimal | — |
| row_mod+col_mod | 36 | outer-product vectors; transpose-elim is net-zero (+18 params for −18 mem) | — |
| params | 34 | row(18)+channel_ids(10)+pads(4)+sentinel(1)+guard(1) | — |
**Sum = 1,668 = v9.**

## Routes killed (do NOT re-explore)
1. **Pad-last / Equal-at-18** (`v18`, built 2026-07-08): score **4012**. The `[1,10,18,18]` BOOL
   core is 3,240 B (matches the #81 crossover: 10·h·w<900 ⇒ h·w<90; 18²=324 is 3.6× over). DEAD.
2. **Tile a 9×9 block straight into the 30×30 canvas** (`v17`, built 2026-07-08): score **1763**.
   Deletes BOTH 18×18 planes but the `tile_rows`[30,10]=300 + INT32 `idx30_vec`[30]=120 +
   `row_idx`[18]=72 + block(262) exceed the 648 saved. Gather-tile family (v16 1726 / v17 1763) is DEAD.
3. **OneHot tail** (`v15`): 15,347 — INT32 indices + F16 one-hot dwarf the U8 canvas. DEAD.
4. **Separable free-output bank / signed polynomial**: rank lower-bound 28; monomial-basis
   separator LP infeasible before d=5 (dense params already > 1,668). DEAD.
5. **Dynamic-shape collapse of the tail** (`v11`/`v12`): scorer-fiction; ORT reuses the annotated
   small buffer and crashes on real visible inputs. DEAD (do not revive).
6. **No-pad `[1,10,18,18]` output** (`v8`/`v10`, 763–942): raw-shape ABI gate 0/267. DEAD.

## One remaining hypothesis (only genuine sub-floor lever)
A legal (non-crashing) way to emit `[1,10,30,30]` while charging **less than one 30×30 U8 plane** —
i.e. an op that produces the 30×30 output from an ≤18×18 input WITHOUT materialising a 30×30
intermediate. None exists in opset-16 (Pad/Equal/Expand/Resize/Scatter all charge a 30×30
intermediate or a bigger BOOL plane). Reopen only if a higher opset with a sub-byte or lazy-pad
output op becomes allowed. Absent that, 1,668 is the floor.

## ⚠ Tooling caveat (transferable — flag to runner/captain)
The **LIVE team-best** (`ca895cf0…`, obfuscated `QLinearMatMul`, **NO `Max(n,4)` guard**) **natively
crashes `_score_v2.score_network_v2` and `profile_real_cost.py`** under ORT **1.24.4**
(`0xC0000094` INTEGER_DIVIDE_BY_ZERO): the scorer's `ch0=1` dummy → `ReduceMax`→`ArgMax`=0 →
`Mod(positions, 0)`. Real inputs run fine (n≥4); only the guard-less dummy path divides by zero.
Any re-scoring/re-profiling of the shipped task061 in this env will hard-crash the worker process
(escapes Python `try/except` — it is a native SEH exception). The guarded plain-`Mul` prototype v9
scores/profiles cleanly at 1,668 and is the correct thing to gate against. This likely bites other
tasks whose obfuscated ships dropped the divisor guard.
