# task096 history (archived 2026-06-22; pointer in task096.md ## COMPRESSED HISTORY)

## Superseded attempts (older → newer)

| Date | Version | Cost | Delta | Notes |
|---|---|---|---|---|
| 2026-05-19 | f16decode (BIG_GATHER_a) | 25364 | -4004 | Cast UINT8→F16 at decode entry |
| 2026-05-24 | structural_layers (not staged) | 9545 | -15819 vs f16decode | Structural rule graph; synth-LIVE 0/200 (LIVE is known-bad lookup) |
| 2026-05-24 | SHIPPED structural_layers | 9545 | - | Kaggle → 6100.67 (+15.83 LB) |
| 2026-06-08 | v2 unified-shell rewrite | 7173 | -2372 | #108 per-cell predicate; dissolved 23 mask planes; 7350→7213→7173 micro-rungs |
| 2026-06-12 | v3 bitrun | 6443 | -730 | Power-of-two row mask + BitShift/BitwiseAnd for run3/4/5; dissolved QLinearConv row detectors |
| 2026-06-19 | v6 freeout_einsum | 6119 | -324 | Free-output 13-term Einsum tail (#109/#119); dissolved 900B Equal-tail AND ~1450B drawing planes |
| 2026-06-21 | v7 selected_marginals | 6024 | -95 | Selected-color (#42/#63) marginals; F32[1,10,30]→F32[1,6,30]; +sel_oh_f 240B |

## Key archived attempt details

### v3 bitrun (6443, 2026-06-12)
builder: `prototypes/build_task096_v3_bitrun.py`. SHA `d8cd04f9ebe254f225a41c4da3c777c46793b856b0dc240dd626060484bc7b24`.
Win: replace grouped QLinearConv row-run detectors + row_has_u8/slice intermediates with 19-row power-of-two row mask
(Einsum(row_has_f, pow2_rows_f) → UINT32, then BitShift/BitwiseAnd for run3/4/5). Rows ≥19 zero-weighted.
Additional small cuts: remove generator-impossible clamps, reuse count_eq1 for centre singleton, drop
rank-invariant extent/arm constants, convert opset-conversion axis Constant nodes into initializers, dedupe axes.
Rejected: row-only/col-only/flattened extents fail clipped cases; single 5-row window count overestimates split
row runs (324 stored color instances); v2 safe-opt is a no-op; Pad-BOOL/pre-Equal one-hot tail worse.

### v6 freeout_einsum (6119, 2026-06-19)
builder: `prototypes/build_task096_v6_freeout_bitrun.py`. SHA `212841526dfadbf87e79065315640a7dbb65d4d4be27dd822167bd26e5c77360`.
Win: free-output 13-term separable Einsum (#109/#119): `output = Einsum('tk,td,te,dr,ec,b->bkrc', KT, RF, CF, E, E, ones[1])`.
13 terms = 1 bg + 2/shell (strict-band split killed 6 corner-correction terms). Score in-kernel, scorer-invisible.
Dissolved 900B Equal-tail canvas AND ~1450B drawing/s_idx planes. Gates: visible 266/266; ab 200266/200266; truth 98.826%; static==profile==6119.
Floor then: 2400B F32 row+col marginals. Rejected: F32→F16 cast keeps F32 plane; input-Slice-to-19 → 22.8KB; moment extent (#115) mismatches 325/2000.

### v7 selected_marginals (6024, 2026-06-21)
builder: `prototypes/build_task096_v7_selected_marginals.py`. SHA `c0b8b2bead4851fb5b8a7a54f49cb1e9747583fc894faa977864be6c7e927819`.
Win: selected-color free-input marginals (#42/#63): TopK(nonbg_f, 6) → sel_idx → OneHot → sel_oh_f F32[1,6,10]=240B;
Einsum('bchw,bsc->bsh', input, sel) → row_count_f F32[1,6,30]=720B (×2 for col). Replaced F32[1,10,30]=2400B pair.
Gates: visible 266/266; ab 200266/200266; truth 197652/200000 (98.826%); static==profile==6024.
Rejected on the way: direct count marginals fed to ArgMax failed 142/266 (densest rows ≠ first/last); thresholding to U8 before bbox fixed it.

### v2 unified-shell rewrite (7173, 2026-06-08)
builder: `prototypes/build_task096_v2.py`. SHA `not-recorded` (gates confirmed at 7213 variant).
Win: replaced 23 [1,1,11,11] mask planes + ring/centre logic (~2783B) with unified per-cell rule:
border coords a=min(r,S-1-r), b=min(c,S-1-c), shell s=min(a,b), paint iff max(a,b)<=s+arm[s]-1.
Two Gathers (color_table[s], reach_table[s] via int32 s_idx) + [6]-vector ScatterElements for invisible-bg centre.
Ranking in F16 (extent*16+arm*16.125-count*0.25) with is_layer=extent≥4 penalty for ring/centre separation.
Micro-rungs: 7350→7213 (run-slice to ≤19 active rows), 7213→7173 (drop rank-invariant -32 constant), both byte-identical.
Floor then: row_has_f+col_has_f F32[1,10,30]=2400B + out_idx30 U8[1,1,30,30]=900B.
