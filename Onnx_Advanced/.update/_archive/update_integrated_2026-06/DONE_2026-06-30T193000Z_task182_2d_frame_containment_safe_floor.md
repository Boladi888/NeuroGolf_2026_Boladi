# task182 FrameTemplateRewrite — 2D frame-containment safe floor (WALL at 7649)

Worker `bo_ng_cld_worker1`, 2026-06-30. Task182 = recolor every blue sprite whose
SHAPE matches a colored "key" sprite that sits inside a complete 7×7 gray frame
(generator `task_776ffc46.py`). Team-best = Aditya's `task182_exact_render`,
cost **7649** (mem 6502 + params 1147), truth 199990/200000 (CLASS-CHECK; byte-equal
is the gold gate), visible 267/267. Verdict after deep analysis + 2 from-scratch
builds + a kernel sweep: **WALL — 7649 is the safe floor.** Three transferable
insights below.

## Insight 1 — 1D profile hashes are UNSAFE for "contained-object" disambiguation; 2D spatial conv is forced

When a rule must pick the object *inside a frame/container* and ignore look-alike
decoys elsewhere, a 1D column/row-profile hash (`Conv(input, kernel[K,10,30,1])` →
tiny `[1,K,1,30]`) **cannot** encode the 2D containment relationship and will
mis-select among decoys. task182's prior v49–v53 line dropped frame detection for
exactly this cheap profile trick, hit 264/267 visible, and **scored 0 on hidden**
(the 3 multi-color decoy pairs ARE in hidden). The safe construction folds frame
+ shape into ONE 2D conv: an input channel for the frame perimeter carries a huge
baseline (here ch5 = 24 perimeter cells × 1000 = +24000) so the per-window score
lands in the matching range ONLY when a full frame is present; the colored-shape
channels (ch2≡ch3, color-agnostic) add the small shape-delta. A second conv hashes
the candidate (blue) channel with the SAME shape kernel + the same +baseline bias,
so `Equal(white_score, target_score)` matches forms across colors. Net: the 2D box
hash is a *safety premium*, not waste — don't "optimize" it away with profiles.

## Insight 2 — one-hot 10-channel F32 input is the cost-floor root cause for shape-match-render tasks

The credited cost (`neurogolf_utils.calculate_memory`) = Σ(named-intermediate bytes,
max(static,runtime)) + Σ(init element COUNTS); `input`/`output` tensors are FREE.
Consequence: convolve on the FREE `input` directly — never materialize a
multi-channel 30×30 plane. Every "exploit the dead/duplicate channels" idea dies
on this: a box-hash weight `[1,10,k,k]` is nonzero on only 3 of 10 channels, but
slicing to them (non-contiguous → ≥2 slices) costs ≥10,800B of 30×30 F32 planes ≫
the ~567 params saved; a 1×1 channel-projection costs a `[1,2,30,30]`=7200B plane.
Measured: a merged-shape-channel reformulation's three forced intermediates alone =
11,840B vs the direct-on-input design's 1936+1444+900. Params count ELEMENTS not
bytes, so dtype tricks never reduce params — only fewer weights do, and fewer
weights require the forbidden slice. Corollary: hash values that must be
collision-free across N shapes run ~10⁴ in magnitude → F16 (≤2048) and UINT8 (≤255)
both overflow, so the score maps are F32-locked; QLinearConv can't help because it
needs a 9000B quantized-input plane.

## Insight 3 — sliding-hash kernel size has an interior cost optimum

For a `k×k`, C-channel hash on a 30×30 free input that is then reduced to a scalar,
cost = `C·k² + dtype·(31−k)²` (params + output plane). Bigger kernel ⇒ fewer output
positions but more params. For C=10, F32 (dtype=4): k=8→2756, **k=9→2746**, k=10→2764
— k=9 is the integer optimum. Don't assume "smaller kernel = cheaper"; check the
derivative `2·C·k − 2·dtype·(31−k)`.

## Sorter note: traps.md — add as new section "profile_hash_unsafe_for_containment" (Insight 1; pairs with task182 + the §FILTERED-CLEAN-vs-LOOKUP class-check note; cross-link the v49–v53 hidden-0 result)
## Sorter note: techniques.md — extend the "conv-on-free-input vs channel-slice" technique with Insight 2's cost-floor rule (one-hot input ⇒ never materialize a multi-channel 30×30 plane; merged-channel measured 11,840B floor) and Insight 3's kernel-size optimum `C·k²+dtype·(31−k)²`.
## Sorter note: by_symptom.md — route "framed/contained object disambiguation, decoy colored shapes, multi-frame" → the new traps section + 2D-box-hash technique.
