# DEAD ROUTE — task243 single-Einsum 15-combined-round flood (5689, sub-6985 exact attempt)

- **Batch/worker:** bo_ng_cld_worker3 (Claude lane), 2026-07-12
- **Live bar (ng_current_bar):** manifest **6985** (Vamshi), onnx sha `8c3e6ba54a75611d8ff629caa97ba1db40c576647b6465b1d3c41c7b45b09745`.
- **Route:** collapse Vamshi's 2-Einsum combined flood (6985 = 15+13 = 28 rounds) into ONE Einsum
  (15 rounds), dropping the inter-Einsum `state_0` tensor (F32 [1,1,18,18] = 1296) → cost **5689**.
  Builder: `prototypes/build_task243_single_einsum_L15.py` (edits `teambest_6985.onnx`, removes the
  `flood_rank2_L13` node, rewires Cast to `state_0`).

## Verdict: DEAD (visible FAIL) — the 2nd Einsum / state_0 (1296 B) is CORRECTNESS-forced, not a choice.
- `score_v2` = mem 3816 + par 1873 = **5689** (−1296 vs 6985; the exact saving = dropping `state_0`).
- `ng_gate visible --task 243` → **255/265 (96.23%) FAIL**. 10 pairs need combined-reach > 15.
  Per traps §VISIBLE-FAIL a visible miss = certain hidden 0 → dead; no truth/ab run (hygiene).

## Why forced (this closes the "1 state instead of 2" lever below 6985)
- The corpus combined-diameter is ~28 (don't-do: dropping any round of the 36-round schedule fails the
  265-pair corpus; LIVE runs 28 combined). One Einsum caps at **15 combined rounds**: the equation
  `flood_direct_L15` uses **exactly 52 letters** (4 base `nqas` + 15×3 [rank+row+col] + 3 tail). 28 rounds
  = 84 letters ≫ 52 → 28 combined rounds CANNOT fit one Einsum → **2 Einsums forced → 2 F32 state tensors**.
- So 6985's memory floor (ch0 gate 1296 + state_0 1296 + state_1 1296 + fill_b 324 + fill_b30 900 = 5112)
  has **no removable state**. The only sub-6985 *exact* lever (fold to 1 Einsum) is gated 96.23% FAIL.

## The sub-6985 win that DID survive (for the next batch): rank1_alt 6014
The cheaper-exact route is NOT "fewer states" but "cheaper bands": single-axis rounds need only ONE
`M[18,18]=324` band vs combined's rank-2 `A+C=1296`. 64 single-axis rounds (32V+32H, 2 Einsums) →
**6014** (−971), memory identical 5112, params 902. Provably ⊇ LIVE (path-reach ≥32 > 28), 0 regressions
on adversarial combs, truth n=4000 = 100%. STAGED as PROBE_READY:
`staged_candidates/task243_rank1alt_safe64_cost6014/`. See its manifest for the full ⊇-LIVE proof.

## Remaining hypothesis below 6014 (none viable found)
- Below 6014 requires either (a) 1 Einsum with ≥56 single-axis rounds (letter budget caps ~45 → the
  hidden-failed max45 4718, under-reaches) or (b) narrowing a forced F32 carrier (F16 NaNs on 100% of
  grids — `hard_records/carrier_narrow_equaltail_900.md` PASS 2) or dropping the gate (visible 0/265).
  All dead. 6014 is the cheapest SAFE (⊇ LIVE) flood; 6985 the cheapest byte-safe.
