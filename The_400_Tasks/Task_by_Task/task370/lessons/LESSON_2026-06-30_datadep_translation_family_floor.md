# Lesson — data-dependent diagonal/ray TRANSLATION has a 3-family floor; dynamic-conv wins by construction

**Where:** task370 (DiagonalZeroMaskRewrite), session bo_ng_cld_worker1 2026-06-30.
Worked example graph: `K-NeuroGolf-OnnxlyFans/.../Team_Best_Onnx/task370.onnx` (cost 8868, byte-equal LIVE).

**What I discovered (transferable):**
When the transform is "translate a binary mask by `k·(dr,dc)` for `k=1..K`, with `(dr,dc)` decided at RUNTIME from the input", there are exactly three implementable ONNX families, and two are eliminated by their *forced dominant tensor* — no tuning can save them:

| family | forced dominant tensor | bytes (task370 K=9, 20×20=400 cells) |
|---|---|---:|
| B — Gather-shift (build `[K,cells]` runtime shift-index, Gather flat mask, ReduceMax) | `[K,400]` index | INT32 **14,400** / INT64 28,800 |
| C — coordinate-formula free-output (per-cell residue along data-dependent diagonal) | INT32 coord grids `R[20,20]`+`C[20,20]` | **3,200** (+ residue/render on top) |
| A — dynamic-conv (ScatterND a `[1,1,(2D+1),(2D+1)]` ray kernel, QLinearConv the mask) | U8 conv kernel `[1,1,29,29]` | **841** |

Family B's single index tensor (14,400 B) **alone exceeds the entire 8868-B Family-A graph** (17.1× the conv kernel). Family C's coord grids (3,200 B) beat the kernel ~4× before any rendering. So Family A (dynamic-conv) is provably the minimal family for data-dependent ray/translation spreads.

**Why it matters:** This is the *quantitative* "why" behind the existing qualitative verdicts library #132 ("diagonal/ray spreads are Conv/render floors, not free-output separable") and #110 ("data-dependent source values → bit-pack + Conv, not Gather"). Gives a fast triage rule: as soon as the step vector is runtime-decoded, price `K·cells·{4,8}` for the Gather index and `2·cells·4` for the coord grid and compare to the `(2D+1)²` conv kernel — the conv kernel is almost always smaller because `K·cells ≫ (2·max_step+1)²` once cells ≳ a few hundred.

**Floor corollary for task370:** combined with the per-tensor floor (F32 slice 1600 §INPUT-DTYPE-LOCK, paint30 900 = output-forced 30×30, kernel 841 = 29 minimal-passing, 7×400 pipeline minimal), 8868 is a hard floor under the byte-equal-to-LIVE gold gate. Only off-gate (dual-truth more-correct-than-LIVE) headroom remains, out of scope for a cost pass on a Solved/shipped task.

**What's missing from the library:** #132/#110 state the verdict qualitatively but not the `K·cells` vs `(2D+1)²` triage arithmetic. This adds the numeric decision rule.

## Sorter note: techniques.md (ray/translation family, near #132/#110) — extend with the "K·cells index vs (2D+1)² conv-kernel" triage arithmetic; task370 worked example, dynamic-conv minimal-family proof.
