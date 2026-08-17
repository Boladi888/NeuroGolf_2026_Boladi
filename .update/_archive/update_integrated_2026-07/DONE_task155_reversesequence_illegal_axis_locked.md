# Transferable insight — `ReverseSequence` is a dead reversal op: scorer-illegal + ORT axis-locked to {0,1}

**Source**: task155 batch 53 (`bo_ng_cld_worker1`, 2026-07-07). Perfect-25/cost≤1 ceiling audit of vflip; LIVE 158 (Aditya) re-confirmed at the forced floor, no improvement (WALL).

**Task family**: axis-reversal transforms — vflip/hflip/rotate on a one-hot `[1,10,30,30]` top-anchored canvas (task155/087/140/150/179/241 and any "reverse the rows/cols of the active N×N grid").

**The tempting idea**: a top-anchored variable-N flip `out[r]=in[N-1-r] (r<N), else pad` is *exactly* the semantics of ONNX `ReverseSequence(input, sequence_lens=[N], time_axis=<row-axis>)` — it reverses `input[0:seq_len]` along an axis and leaves the tail untouched, using only a **scalar** length. That would drop the whole `[30]` INT32 Gather index (120 B, the dominant term of the shipped 158 floor) and replace it with an ~8 B `[1]` seq-len. Looks like a big win.

**Why it's dead (two independent blockers, both measured)**:
1. **`score_v2` op allowlist rejects it** — building any graph with a `ReverseSequence` node makes `score_network_v2` return `None` (`"Op type ReverseSequence is not permitted"`). Unscoreable ⇒ ineligible, regardless of correctness. (Same class of hard block as sub-INT32 Gather indices → None.)
2. **ORT restricts `time_axis` and `batch_axis` to {0,1}** — `reverse_sequence.h:21 "Invalid time_axis of 2. Must be 0 or 1"`. The canvas row axis is 2 (`[1,10,30,30]`), so ReverseSequence can only touch the batch(0)/channel(1) axes directly. Moving the row axis into {0,1} needs a Reshape/Transpose that **materializes a `[10,30,30]` F32 = 36,000 B intermediate** — 228× worse than the 120 B Gather index it was meant to replace.

**General rule**: don't reach for `ReverseSequence` (or Transpose-then-reverse) to cheapen an axis flip. On a `[1,10,H,W]` canvas the reversal axis is ≥2, and (a) the scorer won't score the op and (b) ORT won't reverse axis ≥2 without a full-tensor transpose. The forced floor for a **correct variable-N** flip stays the single `Gather(axis=2)` with a data-dependent `idx[H]` INT32 index (a static/constant index is byte-wrong because the anchor `π(0)=N-1` varies with N — one `Identity`/full-`Slice(steps=-1)` visible gate = 0/266 confirms the fixed-permutation family is dead).

**Applies to**: any flip/reverse task on the 4-D one-hot canvas; and generally as a "not on the ORT/scorer allowlist" entry alongside the existing sub-INT32-Gather-index and U8-OneHot/Einsum NOT_IMPLEMENTED facts.
