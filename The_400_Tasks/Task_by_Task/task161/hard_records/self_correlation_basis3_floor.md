# task161 hard record — self-correlation `basis3` is at its byte floor (428)

- **Live bar at investigation**: effective best rung **428** (local staged
  `task161_basis3_cost428`, sha256 `b2498519…eeecf`; team-best in manifest.csv /
  workbook = **459** Aditya sha12 `cb4eade24614`). Target was strict-below 428.
- **Batch/worker**: batch 46, `bo_ng_cld_worker1` (Claude lane), 2026-07-03.
- **Attempted route**: the prompt's `Basis3SelfCorr` direction — "share or shrink
  the repeated row/column self-correlation selectors in the basis3 sparse Einsum
  path, delete one selector or factor without materializing F32 edges."
- **Verdict**: **WALL at 428.** No factor can be deleted or shrunk without either
  (a) breaking byte-equality to LIVE (hidden-bad task) or (b) *increasing* cost.

## Cost decomposition (measured, static == profile == 428)

`cost = memory 350 + params 78` (params = initializer **element count**, dtype-
and sparsity-independent; memory = intermediate-tensor **bytes**, input/output
free; source: `neurogolf_utils.calculate_memory`/`calculate_params`).

| memory tensor | shape/dtype | bytes | forced because |
|---|---|---:|---|
| `basis3` | [3,10] F32 | 120 | render is rank-3 (below) |
| `color_counts` | [1,10] F32 | 40 | count==4 gate (ReduceSum) |
| `row_cover_counts` | [1,10] F32 | 40 | row boundary-pair Einsum |
| `col_cover_counts` | [1,10] F32 | 40 | col boundary-pair Einsum (distinct contraction) |
| `cover_counts` | [1,10] F32 | 40 | row+col Add, needed before Equal==4 |
| `target` | [1,10] F32 | 40 | F32 coeff for render Einsum (BOOL can't enter F32 Einsum) |
| `count4_bool` `cover4_bool` `target_bool` | [1,10] BOOL | 30 | two Equals + And |
| **total** | | **350** | |

| params init | shape | params | forced because |
|---|---|---:|---|
| `edge_pick` | [30] | 30 | one-hot selects boundary idx 0 from the size-30 spatial axis *inside* the Einsum — memory-free; a `Slice` alternative materializes a [1,10,30,1] F32 = 1200 B |
| `ones` | [1,10] | 10 | basis row: color-presence sum |
| `e0` | [1,10] | 10 | basis row: background output channel 0 |
| `rowmap3`+`colmap3`+`valmap3` | 3×[3,3] | 27 | rank-3 selection maps (below) |
| `four` | scalar | 1 | gate constant |
| **total** | | **78** | |

## Why every component is forced

1. **Both gate conditions are empirically necessary** (measured over all 266
   officials train+test+arc-gen): 62 colors have `cover==4` but `count!=4` (so
   `count4` is needed to reject them) and 4 colors have `count==4` but `cover!=4`
   (so `cover4` is needed). Neither Equal can be dropped. This matches the LIVE
   graph, which explicitly computes `Equal(color_count, four)` (node 7).

2. **`row_cover` and `col_cover` are distinct contractions** (`bkhw,bkhv,v->bk`
   same-row-vs-col0 vs `bkhw,bkuw,u->bk` same-col-vs-row0). Einsum is a single
   product-sum and cannot emit their *sum*; the `Add` (40 B) is required before
   `Equal==4`. Building the combined partner tensor
   `B[k,h,w]=input[k,h,0]+input[k,0,w]` to fuse them costs a [1,10,30,30] = 36 kB
   intermediate — catastrophic.

3. **The render is provably rank-3.** Expanding the output Einsum:
   `out[o] = e0[o]·(lP·tP) + (target-e0)[o]·(lT·tP + lP·tT)` — three independent
   rank-1 (in h,w) terms: `e0·lP·tP`, `(target-e0)·lT·tP`, `(target-e0)·lP·tT`.
   Terms 2 and 3 share the color coeff but have distinct h/w factors, so they
   cannot merge. Three terms ⇒ 3 basis rows {ones, target, e0}. Sharing all
   three in ONE `[3,10]` Concat (120 B) is strictly cheaper than any split:
   row/col-basis `[2,10]` + val-basis `[2,10]` = 160 B (this IS `task161_coeff8`,
   measured **449** = +21). Each of the 3 rows is used (ones→presence,
   target→color, e0→background), so none can be dropped.

4. **Selectors are already maximally shared.** `edge_pick` is a single [30] init
   reused 4× (row_cover, col_cover, and twice in the render). `basis3` is a
   single tensor reused 3× in the render. There is no second selector to fold.

5. **Map factor is at floor (built + measured this session).**
   `task161_cldw1_wmerge.onnx` folds the three [3,3] maps + the `t`-contraction
   into one merged coefficient `W[p,q,r]=Σ_t rowmap[t,p]colmap[t,q]valmap[t,r]`
   (equation `bkhx,x,pk,bmyw,y,qm,ro,pqr->bohw`). W is dense [3,3,3] = **27
   params** (only 5 nonzero, but the scorer charges all 27 — element count,
   not sparsity). Maps are initializers (0 memory), so memory is unchanged.
   Result: **exact tie 428**, visible 266/266. Confirms the maps cannot be
   shrunk. (sha256 `334a2240…6945f`.)

6. **Sparse-tensor operands are rejected by the scorer path** (prior worker
   probes): `Concat`/`Einsum` raise ShapeInferenceError "unsupported type
   sparse_tensor(float)" — so sparsifying `basis3`/W to dodge the dense param
   charge is not available.

## One remaining hypothesis (low confidence, NOT pursued — hidden-bad risk)
Dropping the `count4` gate entirely would save ~60 B (color_counts 40 + one bool
10 + And 10) IF `cover==4` alone reproduced LIVE. It does **not**: 62 colors
would be misclassified (item 1), and the LIVE graph computes count4 explicitly,
so this diverges from the LIVE algorithm and is a near-certain hidden-0. Do not
pursue.

## Do-not-re-explore
- Splitting basis3 into two [2,10] concats (coeff8 = 449). ✗
- GatherND / edge-read with four F32 edge tensors (F32 edges charge). ✗
- Slice-before-cast crop (129,909 — raw F32 slice). ✗
- boolfirst separable (37,528), label-plane U8 (9,516), organic (hidden-bad). ✗
- Sparse `basis3`/coeff operands (scorer rejects sparse). ✗
- Merged [3,3,3] / [2,2,2] coefficient (tie 428 / needs split basis = 449). ✗
