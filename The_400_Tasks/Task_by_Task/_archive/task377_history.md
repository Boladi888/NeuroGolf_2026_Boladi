# task377 - archived superseded surgical-floor history

(Pre-rewrite ladder: ringdyn 77,530 -> blunt-f16 63,682 -> surgical v1-v4 ~44,882. All superseded by the from-scratch rewrite at 9,887 and the output-fold floor at 9,166. Kept for provenance.)

### ringdyn (2026-05-19)

Graph surgery on nodes 190-203: removed 5 × [5,1,1,9,9] ring-mask tables (v11-v15, 2025 param elements) and replaced with 23 dynamic nodes computing `ring_index(r,c) = min(r,c,2L-r,2L-c)` using `Min(range9_row, Sub(2L,range9_row))` etc., then `Gather(frame_colors_stacked, ring_index)` to get the color per cell, then `in_box = Less(r,2L+0.5) * Less(c,2L+0.5)`, then one-hot encode and Pad. New initializers: range9_row_f [9,1], range9_col_f [1,9], scalars (0.0, 4.0, 1.0, 10.0) = 23 elements. Net params saved: 2025-23=2002. Memory also dropped (-694 bytes). Total cost -2709 vs baseline.

### v1 / v2 — the blunt-f16 lever (2026-06-01)

**Recon correction.** The 2026-05-31 prompt (`Prompt_..._Task377_INT64ArgMaxGather.md`)
targets a "72,000 B [1,10,30,30] INT64" intermediate and proposes INT64→INT32
narrowing + crop down to a "~10,000-20,000 floor". **That recon is stale.** Since it
was written, LIVE was fp16-converted (2026-05-20 OB_HUGE, −13,870) and the color
index had already collapsed to `v17` [1,1,30,30] INT64 (7,200 B). The current LIVE
(63,682) has **no** 72,000 B tensor, **no** crop opportunity (input is 30×30), and the
only big INT64 is `ArgMax`'s `v17` (must be INT64 — can't narrow to INT32; INT32 only
helps *Gather indices*, and `v17` isn't one). `recon_task377.py` dumps the real
per-tensor breakdown.

**Real lever = the blunt f16 cast.** The whole-graph fp16 pass inserted
`node0: Cast(input)→_input_f16_cast` **f16 [1,10,30,30] = 18,000 B** — the single
biggest intermediate — just to feed the two channel-reductions:
`node1 ReduceMax→v16` and `node2 ArgMax→v17`. recon confirms `_input_f16_cast` has
**no other consumers**. But `ReduceMax`/`ArgMax` over a **one-hot** input are
**dtype-invariant** (per-cell values are exactly {0,1}, exact in f16): so
`ReduceMax(Cast_f16(x)) == Cast_f16(ReduceMax(x))` and `ArgMax(Cast_f16(x)) == ArgMax(x)`,
byte-for-byte. **v1** moves both reductions onto the F32 `input` and drops the canvas
cast: −18,000, +3,600 (the small F32 `ReduceMax` result, then cast to f16) = **−14,400**
→ 49,282. Pure op-reorder within the existing vocabulary (ReduceMax/ArgMax/Cast all
already present) ⇒ grader-faithful; **byte-identical on 200,266/200,266 corpus inputs**.

**v2** additionally replaces `ArgMax→v17 (INT64 7,200) → Cast→v18 (f16)` with a 1×1
**color-ramp `Conv`** (`weight=[0,1,…,9]`, shape [1,10,1,1]): `Σ_c c·onehot[c]` == the
ArgMax color index, exactly, for every one-hot input (padding/all-zero → 0 == ArgMax of
zeros). −7,200 +3,600(f32 Conv out) +10(weight params) = **−3,590** → 45,692, still
byte-equal (2266/2266 + 200k bg). **Caveat:** `Conv` is a *novel op* for this graph →
traps.md §NOVEL-OP flags Or/And/Where/Conv introductions as grader-unreliable. The
documented failures were *approximations* that diverged on hidden; this Conv-ramp is
provably exact on the whole one-hot input space, so the mechanism shouldn't apply — but
the rule is a hard constraint and the marginal gain is only ~+0.076 LB. **v1 is the safe
default; v2 only if the captain accepts the novel-op risk.**

**Structural floor reached.** After v2/v4, the residual ~44.7k is ~15× [1,1,30,30] f16
(27,000, the 5-iteration seed-finding — confirmed minimal by the from-scratch audit
below AND by `redundancy_scan.py`: zero CSE / dead-cast / dead-node leftovers), 2×
[1,1,30,30] F32 reduce-transients (7,200, unavoidable: ReduceMax/Conv output F32 because
`input` is F32), 5× bool Equal masks (4,500), the [1,10,9,9] output one-hot (~810 uint8),
ring machinery (~1,000). The prompt's "10-20k floor" is unreachable for this structure —
it assumed the stale 72k-INT64 + crop, neither of which exists.

**Proven byte-equal ladder (all vs the live `submission.zip` parent, captain ships one):**

| Cand | Cost | Δ LB | Levers | Risk | Proof |
|---|---:|---:|---|---|---|
| LIVE | 63,682 | — | — | — | shipped |
| `task377_v1.onnx` | 49,282 | +0.256 | kill f16 canvas | **safe** (op-reorder) | byte-equal **200,266/200,266** |
| `task377_v3_boolout.onnx` | 48,472 | +0.273 | v1 + uint8 out | **safe** (dtype-narrow) | byte-equal 2,266/2,266 |
| `task377_v2_conv.onnx` | 45,692 | +0.332 | v1 + ramp Conv | ⚠ novel `Conv` | byte-equal **200,266/200,266** |
| `task377_v4_conv_u8.onnx` | 44,882 | +0.350 | v1 + Conv + uint8 | ⚠ novel `Conv` | byte-equal 2,266/2,266 |

**Recommendation:** ship **v3 (48,472)** as the safe default (strictly dominates v1; both
are pure op-reorder/dtype-narrow within the parent's op vocabulary). Consider **v4
(44,882)** only if the captain accepts the `Conv` novel-op grader-risk for the extra
~+0.077 LB — the Conv-ramp is provably exact on the whole one-hot input space (so the
§NOVEL-OP divergence mechanism doesn't apply), but the rule is a documented hard
constraint. `task377_v1.onnx` is the prompt's named deliverable / ultra-conservative
fallback. **Not submitted — captain ships.**

