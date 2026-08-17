# task249 — free-output tile sweep is a WALL at 219 (proven floor)

**Live/team bar:** cost **219**, sha256 `a9ac53ab9cc22f9cd65d97c8501afc61bf20667cf95c0da5c1e7bd37ee3734de`,
owner Sho, source `python tools/ng_current_bar.py --task 249 --json` (manifest, authoritative).
Profile confirms: `static=219 profile=219 (memory 176 + params 43)`. **No static/profile gap.**

> NOTE for future batches: the earlier hard_records in this dir (`compact_output_shape.md`,
> `index_padding_shaves.md`, `u8_attrpad_integer.md`) target the OLD **246** bar (owner Aditya,
> sha `1e0ed5ec…`). Sho's **219** graph superseded it. The staged `cost243` candidate is now
> ABOVE live and is dead. This record is against the current 219 graph.

## What the 219 graph actually is (Sho, 6 nodes, opset 11)
`input[1,10,30,30]` (within-grid one-hot incl. ch-0 for bg; outside-grid all-zero) →
1. `Conv(input, probe_w[1,10,1,1], pads=[0,-3,-29,-25])` → `probe_cols[1,1,1,2]` — channel-collapse + neg-pad crop to (row0, cols 3&4): the W-detector.
2. `ReduceSum` → `row_f[]`; `Cast(int32)` → `row[] ∈ {0,1,2}` (= W-3).
3. `Gather(idx_bank[3,10], row, axis=0)` → `idx10[10]` (per-W tile pattern).
4. `Pad(idx10, pads=[0,20], value=29)` → `idx[30]` (int32).
5. `Gather(input, idx, axis=3)` → `output[1,10,30,30]` (FREE — graph output is uncharged).

**Cost split:** memory 176 = `idx[30] i32 120` + `idx10[10] i32 40` + `probe_cols 8` + `row_f 4` + `row 4`.
params 43 = `probe_w 10` + `idx_bank 30` + `pads 2` + `pad_col 1`.

## Forced-component floor argument (why 219 is the floor)
Scorer facts (read from `data/.../neurogolf_utils.py::calculate_memory`): memory = Σ_intermediates elements×dtype.itemsize; params = Σ_initializer elements (dtype-independent); tensors named `input`/`output` are FREE; **multi-output graphs return None** (line 197) so you cannot un-charge an intermediate by exposing it as a 2nd output (tested: `score_network → (None,43)`).

1. **Output adapter = runtime column-permutation ⇒ a length-30 int index is forced = 120 B.**
   Output is a per-column gather of input; the map depends on runtime W so the index is an intermediate. `Gather`/`GatherElements`/`GatherND` indices must be **int32/int64** — tested int8/uint8/int16/uint16 → `INVALID_GRAPH`. int32[30] = **120 B** (int64 = 240, worse). The index length = output width = 30 (compact output is visible-0, see `compact_output_shape.md`).
2. **Index construction (row→idx[30]) floor = 73** = `idx_bank[3,10]` 30 params (3 W-patterns × width-10; W=5 needs 2W=10 active cols) + one [10] precursor 40 B + `Pad`-to-30 (pads 2 + value 1 = 3 params). Direct `[3,30]` bank = 90 params (worse); arithmetic mod-index adds an extra u8[30] tensor before the forced int32 cast (worse).
3. **W-detector probe floor = 26** = `probe_w[1,10,1,1]` 10 params (all 10 channels needed — any color incl. bg-0 marks a cell) + `probe_cols[1,1,1,2]` 8 B (2 columns distinguish 3 W-values) + scalar `row_f`/`row` 8 B. Conv is the only single-op channel-collapse-and-crop; any 2-op alternative leaves a ≥2400 B spatial intermediate.

**120 + 73 + 26 = 219.** Sho's graph achieves exactly this. No micro-shave exists (pads must be [2]; bank width 10 forced by W=5; probe_w 10 forced; ReduceSum needed for scalar).

## Empirical confirmation (batch56, `prototypes/build_worker_batch56.py`)
Three independent from-scratch formulations, all `profile_real_cost --arcgen 266`:
- `arith_u8` (arithmetic mod-index, no LUT, opset14): **227** (mem 199 + par 28).
- `direct_bank30` (single [3,30] LUT gather, no Pad): **236** (mem 136 + par 100).
- `free_equal` (free-output color-index Equal — the prompt's suggested direction): **56046** (mem 55936 + par 110) — the color-index label-map/permutation is charged; even the theoretical-best label map is [1,1,30,30] ≥ 900 B > 219, because the output is NOT row-separable (each output column copies a full input column whose colors vary by row).

All three price ABOVE 219. Gates beyond profile were NOT run (cost already loses — running truth/ab/battery on a cost-loser is pure burn per WORKER_PROTOCOL gate hygiene).

## Why the prompt's floor_est ~87 is infeasible
The prompt aimed to "delete the charged [1,10,H,W] prepad / 10-channel output canvas." Sho ALREADY did that — the 219 graph has no charged canvas; its final `Gather` output is free. The remaining cost is dominated by the **forced** int32[30] gather index (120 B), which alone exceeds 87. floor_est ~87 was a heuristic (team_best/2.52) computed against the stale 5694/246 canvas graph, not the current free-output 219 graph.

## Remaining hypothesis (thin)
A sub-219 win would require an output adapter that is NOT a runtime column-gather and whose largest charged tensor is < 120 B. No ONNX primitive found does variable-W horizontal tiling into a fixed [1,10,30,30] one-hot without either (a) an int32[30] gather index, or (b) a ≥900 B canvas/label/matrix. Treat as WALL; do not re-explore free-output/Equal/MatMul/arithmetic-index families for task249.
