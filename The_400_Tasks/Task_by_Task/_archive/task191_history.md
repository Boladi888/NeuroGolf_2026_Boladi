# task191 — archived history (F16-Conv lineage v1–v7)

> Compressed out of `task191.md` 2026-06-30 by bo_ng_cld_worker2 (WORKER_PROTOCOL
> staging rule: manual content must stay < 20 KB). This is the **worker-7 F16-Conv
> lineage** that floored at 24,807 (opset-20) / 26,917 (safe opset-13). It was
> **superseded by Aditya's QLinearConv-INT8 graph at cost 11,169** (team manifest,
> 2026-06-10), which the local docs below never recorded. The v1–v7 floor proofs
> claiming "F16 Conv forced / ~24–27k floor" are FALSIFIED by Aditya's INT8 rewrite —
> keep them only as a record of the F16 path, NOT as a live floor. Current floor
> analysis lives in `task191.md`.

## Lineage summary (F16-Conv, worker-7, 2026-05-30 → 2026-06-01)

| version | cost | Δ | gate | note |
|---|---:|---:|---|---|
| LIVE (pre) | 85,921 | — | — | distilled MLP, 46-MatMul stamp bank, F16 opset10 |
| v1 BOOL-restructure (Path A byte-equal) | 63,060 | −22,861 | 267/267 byte-eq, 1200/1200 GT | F16 compute core kept; BOOL I/O + tail; Kaggle +0.31 |
| v2 Path-B full rewrite (decoded rule) | 36,469 | −26,591 | truth 5000/5000 | from generator rule; 8 D4 Conv match + ConvTranspose paint; +0.55 pred |
| v3 + 2-Conv group-reduce paint | 32,841 | −30,219 | truth 20000/20000 | 8→2 paint channels; **Kaggle 6446.86 (+0.65 LB)** |
| v4 + single-ch U8 canvas output | 29,096 | −3,745 vs v3 | truth 20000/20000 | 3 BOOL planes + 7 ZEROCH → 1 U8 canvas + Equal-expand |
| v5 + positional-hash match core | 26,917 | −5,924 vs v3 | truth 20000/20000 | 2 windowmasked code Convs + 8 Equal + 6 Or membership |
| v6 + opset-20 broadcast-Equal/BOOL-ReduceMax | 24,807 | −2,110 vs v5 | truth 20000/20000 | opset-20 only; AGGRESSIVE; this is the shipped LIVE-local src name |
| v7 = v5 + INT32 index plumbing (opset-13 SAFE) | 26,545 | −372 vs v5 | truth 20000/20000 | grader-safe; byte-identical to v5 |

Builders: `prototypes/build_task191_v{1..7}.py`; candidates `prototypes/task191_v{1..7}.onnx`;
numpy mirror `prototypes/faithful_v2.py`.

## v5/v6 cost map (F16-Conv lineage floor, NOW SUPERSEDED)

The F16 path's dominant tensor was the **8 D4 match Convs in F16** (`mA`+`mB` `[1,4,23,23]`
F16 ×2 = 8,464) plus the F16 paint path. The lineage's floor proof argued "F16 dtype
FORCED by ORT (Conv/ConvTranspose reject INT/BOOL/UINT8)". **That objection is true for
plain `Conv`, but Aditya's graph uses `QLinearConv` (INT8), which DOES correlate in INT8** —
halving the match map (`[1,8,23,23]` U8 = 4,232 vs the F16 8,464) and avoiding the F16 paint
path entirely (INT8 stamp QLinearConv). That single substitution is the 24,807 → 11,169 win.

v5 per-tensor map (memory 26,093 + params 824 = 26,917):

| tensor | shape / dtype | bytes | role |
|---|---|---:|---|
| `yel25f` | [1,1,25,25] F32 | 2,500 | yellow read (F32 input) |
| `blue23` | [1,1,23,23] F32 | 2,116 | blue read (bbox + recolor) |
| `ps` | [1,2,23,23] F16 | 2,116 | 2-ch paint seed (ConvTranspose input) |
| `yel_f16` | [1,1,25,25] F16 | 1,250 | F16 Conv feed |
| `eA0..eA3,eB0..eB3` | [1,1,23,23] BOOL ×8 | 4,232 | 4-way membership per group |
| `oA01,oA23,redA,oB01,oB23,redB` | [1,1,23,23] BOOL ×6 | 3,174 | OR-fold to 2 group seeds |
| `code_A,code_B` | [1,1,23,23] F16 ×2 | 2,116 | 2 window-shape hash maps |
| `paint23` | [1,1,23,23] F16 | 1,058 | ConvTranspose paint output |
| `ps_bool` | [1,2,23,23] BOOL | 1,058 | seed Concat before F16 cast |
| `canvas30` | [1,1,30,30] U8 | 900 | output color-index canvas |
| recolor/bbox/patch/scalars | various | ~3,650 | logic |
| params | permA/B 864 + fill 300 + indices | 824 | |

## Falsified floor proofs (kept as record — DO NOT cite as live)

- "**<1k structurally impossible / ~26.9k realistic floor**" (v5) and "**<20k infeasible, 24,807
  aggressive floor**" (v6): both reasoned from a **plain-`Conv` F16 match map**. Aditya's
  `QLinearConv` INT8 path reached 11,169, so these floors are obsolete. The <1k claim itself
  (F32 reads alone > 1k) is still true, but the realistic floor is ~8.8k, not ~24–27k.
- "**8,464 match-Conv core irreducible, F16 forced**": the *8-orientation* requirement is real
  and survives (see `task191.md` forced-component proof), but the *F16* dtype was NOT forced —
  INT8 QLinearConv halves it to a `[1,8,23,23]` U8 = 4,232 map.

## Don't-do (F16 lineage, still valid as graph-surgery notes)

- **Einsum-collapse the 46-MatMul bank** (the original Path-A prompt's headline idea): DON'T.
  The bank intermediates are `[5,5]/[5,30]` (50–300 B); the whole bank is ~9k, not the cost
  center; fusable pairs save ~1.3k; and `Einsum` ≠ chained `MatMul` in F16 (max abs diff 0.042)
  → hidden-set divergence risk under downstream ArgMax/threshold.
- **Crop below 23×23 / assume a sub-23 grid**: generator `size=23` is hardcoded; the matcher's
  Conv needs the 25×25 yellow feed (23 active + 2 zero pad) to emit 23×23.
- **input cast F16 then slice-cheaply**: F16 input cast is 18,000; BOOL 9,000; slicing F32
  per-channel then casting is worse (F32 slices are 2× bytes).
- **Union-paint vs generator's last-orientation-wins**: verified equivalent (0/20000 anchors
  match >1 orientation with differing rects).
