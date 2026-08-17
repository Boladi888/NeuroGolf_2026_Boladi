# task289 — archived rung history

Moved out of `task289.md` 2026-07-09 (batch 5608) to keep manual content < 15 KB.
Current state / bar / WALL live in `task289.md`; superseded local Einsum lineage (all beaten by
the team's live Resize-family **324**) is preserved below. One-line-per-rung summary stays in the
`task289.md` Attempts table.

---

### expand_reuse (2026-06-24) — local Einsum best 529 (SUPERSEDED by live 324)

Reviewer-r289 spotted that `expand[0,p,x]` is already the `src[30,3]` selector in transposed form: for k=1, it maps source rows/cols 0..2 to p/q and zeroes the rest. Selecting that slice with `e0=[1,0,0,0,0]` removes the separate `src` initializer.

```
presence = ReduceMax(input, axes=[2,3])        # [1,10,1,1]
s        = ReduceSum(presence)                 # [1] == k + 1
ksel     = Cast(Equal(s, [2,3,4,5,6]))          # [5]
output   = Einsum("bnxy,j,jpx,l,lqy,kpr,kqc,k->bnrc",
                  input, e0, expand, e0, expand, expand, expand, ksel)
```

Charged surface: memory 69 = `presence[1,10,1,1]F32` 40 + `s[1]F32` 4 + `ksel_bool[5]BOOL` 5 + `ksel[5]F32` 20. Params 460 = `expand[5,3,30]` 450 + `s_values[5]` 5 + `e0[5]` 5. Static cost 529; real-input profile with `--arcgen 266` is 529.

Gates: visible 268/268; truth 200000/200000; AB 200268/200268 byte-equal; profile static=529 over 268 real inputs. SHA256 `674d3a54eec38f41b1c0edc3f1a273cc5cc1c51f37b7855118cd5229360f2b88`. Staged: `staged_submits/task289_expand_reuse_cost529/`.

### kbasis5 (2026-06-21) — previous best 614

The 634 prefix/backdiff wall was scoped to building a runtime `[30,3]` band selector. A cheaper formulation stores the five possible expansion bases as a constant bank:

```
presence = ReduceMax(input, axes=[2,3])        # [1,10,1,1]
s        = ReduceSum(presence)                 # [1] == k + 1
ksel     = Cast(Equal(s, [2,3,4,5,6]))          # [5]
output   = Einsum("bnxy,xp,yq,kpr,kqc,k->bnrc",
                  input, src, src, expand, expand, ksel)
```

`src[30,3]` selects input rows/cols 0..2 from the padded 30x30 canvas. `expand[5,3,30]` contains one row/col placement basis for each k in 1..5; rows outside the active `3k` band are zero. This trades params for runtime memory: the candidate pays `expand` 450 params and `src` 90 params, but only `ksel_bool[5]` + `ksel[5]` at runtime instead of `prefix_bool[30,3]` + `prefix[30,3]`.

Charged surface: memory 69 = `presence[1,10,1,1]F32` 40 + `s[1]F32` 4 + `ksel_bool[5]BOOL` 5 + `ksel[5]F32` 20. Params 545 = `expand[5,3,30]` 450 + `src[30,3]` 90 + `s_values[5]` 5. Static cost 614; real-input profile with `--arcgen 266` is 614.

Gates: visible 268/268; truth 200000/200000; AB 200268/200268 byte-equal; profile static=614 over 268 real inputs. SHA256 `7310c5737ae51172d5db512107dad944729589b5a1f802679f382679cdfa234e`. Staged: `staged_submits/task289_kbasis5_cost614/`.

### einsum_sum_minus_one (2026-06-12) — previous best 634

Beats the 679 team-best (Boladi) by 45 with the SAME single fused free-output Einsum; only the `k` detector changed. The 3x3 source is padded into the 30x30 one-hot canvas, so the background channel 0 is present in every input (checked on all 268 official examples — 0 had ch0 absent). Hence:

```
presence = ReduceMax(input, axes=[2,3])   # [1,10,1,1]
s        = ReduceSum(presence)            # [1]  == k + 1   (ch0 always counted)
sm1      = Sub(s, one)                    # [1]  == k
k_bounds = Mul(sm1, bounds)               # [3]  == (p+1)*k, bounds=[1,2,3]
prefix   = Cast(Less(row_idx, k_bounds))  # [30,3] F32
output   = Einsum("bnxy,yq,xp,rp,cq->bnrc", input, backdiff, backdiff, prefix, prefix)
```

This drops `ch_nonzero` (10 params), `presence_nz[1,10,1,1]` (40 B mem), and the standalone `k[1]` tensor, adding only `one` (1 param) and `sm1[1]` (4 B). Net -36 mem -9 params.

Charged surface (memory 510): prefix[30,3]F32=360, prefix_bool[30,3]BOOL=90, presence=40, s=4, sm1=4, k_bounds=12. Params (124): backdiff[30,3]=90, row_idx[30,1]=30, bounds=3, one=1.

Gates: visible 268/268; truth 200000/200000; AB 200268/200268 byte-equal (div=0); static=634; profile (arcgen 266, 268 inputs)=634 (gap 0.0%). sha256 970d3b1736c628c73b4c6b22ab053f295350131b14749dbb4a8ac11b05ad2276. Staged: `staged_submits/task289_einsummicro_cost634/`.

### einsum_backdiff (2026-06-10)

Full rewrite from the ARC-GEN rule. Instead of building a 15x15 upscaled canvas with two MatMuls and then Pad, compute `k` with `ReduceMax` colour presence, build one prefix mask `[30,3]`, and emit the fixed 30x30 output directly as the graph output with:

`Einsum("bnxy,yq,xp,rp,cq->bnrc", input, backdiff, backdiff, prefix, prefix)`

The shared `backdiff[30,3]` has nonzero rows only for source rows/cols 0..2, so the final Einsum reads the source 3x3 without a visible Slice. Score: memory 546 + params 133 = cost 679. Gate 5 real-input profile over all 268 official visible+arc-gen inputs was exactly 679.

### equalfold (2026-05-19)

Two optimizations on the 27-node task289 model:

1. **Equal-fold**: Replace `Sub+Abs+Less(0.5)` with `Equal` for both row and col matching. Since `r_in = Floor(row_idx/k)` is integer-valued, `|r_in - r_src| < 0.5` ↔ `r_in == r_src`. Removes nodes 14-21 (2 Sub + 2 Abs + 2 Less + 2 Cast) and replaces with 2 Equal + 2 Cast. Saves d_rr,adrr,d_cc,adcc intermediates (4×180B = 720B).

2. **valid_mask fold**: Instead of computing `valid_mask = Outer(r_valid, c_valid)` [1,1,15,15]=900B and then `output_inner = upscaled * valid_mask` [1,10,15,15]=9000B, fold validity directly into the match masks before MatMul: `m_row_v = Mul(r_valid, m_row)` → zeros invalid rows, `m_col_v = Mul(c_valid, m_col)` → zeros invalid cols. MatMul chain uses masked versions directly. Saves 9900B, adds 360B (m_row_v + m_col_v). Net -9540B.

## Cost analysis (local lineage)

- Baseline: 31793 (memory=31714, params=78)
- equalfold candidate: 21532 (memory=21454, params=78) — output_inner[1,10,15,15]=9000B eliminated via valid_mask fold
- LIVE after queued equalfold refresh: 21139 (memory=21064, params=75)
- einsum_backdiff: 679 (memory=546, params=133, profile_cost=679) — deletes input_inner/step1/upscaled 15x15 canvases
- einsum_sum_minus_one: 634 (memory=510, params=124) — k = sum(presence) - 1 removes ch_nonzero, presence_nz, standalone k. Net -45.
- kbasis5: 614 (memory=69, params=545) — constant expand[5,3,30] bank replaces runtime prefix. Net -20.
- expand_reuse: 529 (memory=69, params=460) — deletes src[30,3] via the bank's k=1 slice. Net -85.
- **2026-07-09: team LIVE = 324 (Sho, Resize family) — beats the whole Einsum lineage above.**

## Current cost surface note (retired)

The 634 proof was retracted as a global floor (it only proved the runtime prefix/backdiff formulation);
kbasis5 moved the band structure to a constant k-basis bank; expand_reuse reused its k=1 slice as the
source selector (memory 69, params 460 = expand 450 + s_values 5 + e0 5). No sub-529 EINSUM floor was
claimed. As of batch 5608 the whole Einsum lineage is dominated by the Resize-family 324 — see the WALL
record. Einsum floor ≈519 (bank 450 or computed [3,30] F32 360B); low-rank/CP dead (`>0` grader).

### 2026-06-24 sub-614 factorization check

`cp_rank11` found an exact 11-dimensional rank-one slice basis for the expansion tensor, reducing static cost from 614 to 582 in scorer terms: memory stays 69 and params become 513 (`cp_c[30,11]` 330 + `src[30,3]` 90 + `cp_a[5,11]` 55 + `cp_b[3,11]` 33 + `s_values[5]` 5). It is not shippable: ORT/F32 cancellation leaves false off-band outputs positive by roughly 1e-6, and `ng_gate visible --task 289 --cand ...task289_cp_rank11.onnx` returns 0/268. Adding a bias would require a charged output-sized intermediate or a dense coefficient matrix that prices above 614. A direct low-rank pair selector is also out: flattening the exact pair selector by `(k,p,q)` vs `(r,c)` has rank 41, so no <=11-term direct CP can stay in budget.

### 2026-06-21 prefix/backdiff wall check

Visual + generator reconfirmed `grid_enhance(size=3, enhance=len(colors))`: k is distinct nonzero colors, each 3x3 source cell becomes k x k, and active output is 3k x 3k. Falsifications: copied constant blocks, not interpolation; TR1 has four nonzero pixels but k=2 -> 6x6 while TR4/test have five pixels but k=3/4 -> 9x9/12x12.

The only named lever was selector sharing. It fails because `prefix[:3]` is runtime cumulative bands (`k=1`: `[[1,1,1],[0,1,1],[0,0,1]]`) while `backdiff[:3]` is fixed signed finite difference (`[[1,0,0],[-1,1,0],[0,-1,1]]`); for k>=2 prefix source rows collapse further toward all-ones. Computing backdiff from prefix adds a charged [30,3] surface. No cheaper prefix producer was found: ONNX has no fused Less-to-F32, and arithmetic prefix first materializes an F32 [30,3]. Source-slice Path-B saves 81 params but adds at least `input_3x3`=360 B, so it rises to >=913; explicit canvas/Resize/ConvTranspose/sparse-write families also price above 634.

Re-measured 2026-06-21 before `kbasis5`: score_v2 (0, 510, 124), profile-real cost 634 over 268 inputs, visible 268/268, truth 200000/200000, AB 200268/200268 byte-equal.

### 2026-05-20 — SG_ALL worker (no candidate)

- Tried: f16 conversion (robust + direct + opset bump); drop-cast-before-Pad scan; bool-throughout rebuild scan
- Result: NO CANDIDATE
- Reason: Cast(BOOL->F32) feeds MatMul which requires F32. Cannot drop the cast. Algorithm-specific structure.
