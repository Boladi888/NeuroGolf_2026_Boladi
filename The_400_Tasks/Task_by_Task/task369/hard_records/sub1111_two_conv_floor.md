# task369 — WALL at 1111 (two-conv integer stack is the floor)

- **Live bar:** cost **1111** (memory 800 + params 311), creator Vamshi.
  - sha256 `3583d772e665e5eb9bfad145de708f8e7f916af80642d92ef96da6e1cc3d8d85`
  - Sources agree: manifest.csv `task369,Vamshi,3583…,1111,17.9870`; workbook row (points 17.987, updated 2026-07-02T02:35:15Z); profile `static=1111 profile=1111 (memory 800 + params 311; 265 real inputs)`.
  - Ignore stale main-repo values (`show_task.py`=1599, `live_costs.json`=5339, lib_query recall=7729). Library DB is STALE.

## Team-best graph (batch 42, the thing to beat)
`Slice`(ch0→F32[1,1,10,10]) → `Cast`(U8[1,1,10,10]) → `QLinearConv`(3×3, w1[3,1,3,3], 3 U8 feature planes) → `ConvInteger`(3×3, w2[10,3,3,3], direct INT32[1,10,30,30] graph output).
- memory = 400 (x_f F32) + 100 (x_u8) + 300 (h, 3 U8 feats) = **800**
- params (score_v2 = initializer element count) = w2 270 + w1 27 + b1 3 + slice_starts/ends 8 + {u8_zero,scale,i8_zero} 3 = **311**

## Rule (why 2 nonlinear spatial layers are forced)
Colors {0,5} only; 5 preserved. Zero-mask `z`, degree `deg = z·conv4(z)`.
- `deg==0` → 3; `deg>=2` → 1; `deg==1` → **1 if the lone zero-neighbor itself has deg>=2, else 2**.
The `deg==1` branch is a genuine **2-hop** query (cell → lone neighbor → that neighbor's degree), and the "neighbor has deg>=2" test is a **threshold applied per-neighbor before aggregating**. That requires two sequential nonlinear spatial layers; it cannot be folded into one clamped conv.

## Falsified sub-1111 routes (first-hand this batch)
| route | result | why dead |
|---|---|---|
| Single 5×5 conv (linear-argmax upper bound) | **88.5%** (color2 0/1720, color3 7/1336) | one linear layer can't make the non-monotonic deg-decision; single QLinearConv (one clamp, monotonic) is weaker still. "raw 5×5 unsafe" confirmed. |
| 5×5 conv1 + 1×1 head (collapse w2 to [10,K,1,1]) | 100% needs **K=5**; K=4 only 99.95% | cost = mem 400+100+**500** + params 125+50+5+8+3 = **1191**. The 5th U8 feature plane (+200 mem) outweighs the w2 saving (−220 params). |
| 2-feature 3×3+3×3 (`k2_bucket`, drops h to 200) | **visible 0/265 FAIL** | LP/CP-SAT (worker2): colors 1&2 not linearly separable in any 2-channel code. Visible-zero ⇒ certain hidden 0. |
| Grouped final head, group=2 (`grouped4_probe`) | visible 100% but cost **1131** | group=2 needs input-channels÷2 ⇒ 4th feature plane (+100 mem); nets +20. gcd(feat=3, out=10)=1 ⇒ no grouping without a 4th feature. |

Probes: `prototypes/` MLP/linear-separability run over the full 265-pair state corpus (train+test+arc-gen). MLP hidden-unit sweep: 3f→98.5%, 4f→99.95%, **5f→100%** (this is the 5×5+1×1 result above).

## Forced-component floor argument (why 1111 is the floor)
- **memory 800 forced:** entry must reduce F32[1,10,30,30]→10×10 active ch0; `Slice` preserves F32 ⇒ x_f = 400 (casting the full input *before* slicing = 9000, far worse). Integer conv needs a U8 quantize ⇒ x_u8 = 100. Feature planes: K=2 falsified (visible-zero) ⇒ K≥3 U8 ⇒ h = 300. QLinearConv (U8 out) is chosen over ConvInteger (INT32 out, h would be 1200). No cheaper entry/feature representation exists.
- **params 311 forced:** final head must emit [1,10,30,30] *as the graph output* (any Pad/Concat to widen channels makes the pre-op [1,≤6,30,30] INT32 a charged 14–22 KB intermediate). 2-hop ⇒ head is 3×3 (1×1 head needs K=5 ⇒ +200 mem, net +80). K=3 feats ⇒ w2 = 10×3×3×3 = 270; the 6 always-zero output channels (162 of the 270) are still cheaper as zero-weights than a zeros initializer (4×900). Overhead 41 (w1 27 + b1 3 + slice 8 + scalars 3) is minimal; b1/scalar shaves are the forbidden zero-point route (traps dead-end).

**Verdict: WALL. 1111 is the structural floor for this F32-input / 2-hop-rule task. No staging candidate.** Both prompt-named levers (grouped final-output layout; remove F32 slice / U8 feature stack) are falsified above. One remaining hypothesis if ever revisited: a legal op that emits a sub-byte (BOOL) conv *feature* plane would drop h below 300 — no such ONNX integer-conv output dtype exists today.
