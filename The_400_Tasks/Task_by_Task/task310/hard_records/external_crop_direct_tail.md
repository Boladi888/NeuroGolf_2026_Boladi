# task310 — ExternalCropDirectTail: WALL (sub-148 infeasible under the output-shape contract)

**Batch 56 / bo_ng_cld_worker2 (Claude lane), 2026-07-09.**

## Live bar used
- **Authoritative live: cost 1485**, sha256 `494e4b3a3bc67deeadf82dea2222ec3dd8dca2cce4cf3285ff8d9a27075b17b5`, owner Sho, source = team manifest.csv (via `ng_current_bar.py --task 310`).
- **Pending ceiling (Sho): cost 1162**, sha `a1940b30…` (`Staged_Sho/task310.json`, "wave36 1485->1162"). Any real win must be **< 1162**.
- Prompt floor_est **~148** (external-LB "20+ pts"; confidence MED). 148→20.003 pts, 1162→17.942 pts, 1485→17.697 pts.

## The task (decoded from ARC-GEN, 265 pairs)
Find the rare **marker color** (least-count *present* color) which frames a rectangular box; **crop that bbox** and emit it at the output top-left. Output sizes 5×5..8×8 (always square); inputs 20×20..30×30. LIVE front-end (marker=min-count → row/col marker-count Einsum → ArgMax first/last = r0,r1,c0,c1) is 266/266 and was reused verbatim in all probes below.

## Why the prompt's premise is STALE
Prompt says "replace the ArgMax/Gather crop pipeline." **There is no Gather pipeline anymore.** LIVE (1485) is already a **mixed-radix (5,2,3) positional-adder free-output Einsum** (`task310_radix23`, 37 nodes, 0 Gather): it implements the data-dependent shift `r = Y + r0` as one Einsum whose only large tensor is `output` (free). memory 880 + params 605. Params = the adder network: Phi[30,5]=150, Ahigh_fA[5,5,5]=125, Plo3[30,3]=90, Plo2[30,2]=60, A3=54, Ahigh_fB=50, A2=32, iota/vals=44. The Gather→Einsum structural rewrite (81924→…→1485) is **already done** by Sho.

## The forced floor (why 148 is unreachable)
**Scoring facts** (`neurogolf_utils.py`): cost = `memory`(Σ bytes of every *intermediate*; `input`/`output` skipped, line 234) + `params`(Σ initializer *element counts*, dtype-independent). **Every intermediate must have a statically-inferable shape or the score is `None`** (lines 229-233). **The grader compares `np.array_equal(user_output, expected[1,10,30,30])`** (line 546) — output must be exactly [1,10,30,30], crop at top-left, all-channel-zero elsewhere.

**Three from-scratch formulations built & priced (all reuse the 266/266 front-end):**

| # | Formulation | profile cost | mem+params | visible | verdict |
|---|---|---|---|---|---|
| 1 | **bare dynamic-Slice as output** (`build_task310_direct_slice/`, sha E268AD05…) | **469** | 464+5 | **0/266** | content 266/266 correct, but output [1,10,H,W] ⇒ `np.array_equal` vs [1,10,30,30] = False. **GRADER-DEAD.** |
| 2 | **Slice→Pad to [1,10,30,30]** (`build_task310_slice_pad/`, sha C0F57942…) | **3133** | 3120+13 | **266/266** | correct but the [1,10,8,8] crop intermediate (2560 B) dominates. static=36573 is GRADER-GAP fiction (dummy=full grid). LOSES. |
| 3 | materialized selection-matrix Einsum (output [1,10,30,30] ⇒ RowSel,ColSel [30,30] F32 = 7200 B) — analytic | ~7200 | — | — | LOSES (worse than radix; radix exists precisely to keep the [30,30] selection *virtual*). |

**The argument:** producing the forced static [1,10,30,30] top-left crop needs a crop-place operator. Its three realizable forms floor at: Slice/Gather crop intermediate ≥ 2560 B (#2), materialized [30,30] selection ≥ 7200 B (#3), or the **virtual-selection free-output radix Einsum** — memory ~880 + params ~600 (#LIVE). The minimum is the radix at **~1162–1485**. The only mechanism that reaches the 148 regime is the bare-Slice (469), and it is **grader-invalid**. 148 would additionally require params < 50 while the positional-adder param floor is ~300 (positional bases already at the Σdᵢ=10 minimum → 30×10) + ~200 (carry adders) ≈ 500, plus the forced [30]-F32 rowcnt/colcnt front-end vectors (≥240 B). **148 does not survive the output-shape contract.**

## Exact failed gate
- Probe 1 (bare-Slice): `ng_gate visible` = **0/266 FAIL** (ORT `VerifyOutputSizes {1,10,8,8}≠{1,10,30,30}`); profile 469. Mirrors official `np.array_equal` shape rejection.
- Probe 2 (Slice+Pad): visible 266/266 PASS, profile **3133 >> 148 > bar**. No `--full`/truth run (cost already loses — gate-ladder stop).

## One remaining hypothesis (unexplored)
Replace the digit-decomposed positional **adder** (the ~500-param radix core) with a **rank-1 F32 exp-ramp positional code** (cost_floor_reference task138: "exp-ramp positional codes replace ArgMax planes") so the shift becomes a scalar multiply instead of a carry network. **Risk:** exp-ramp gives *soft/approximate* selection; a pixel-exact integer crop needs exact one-hot recovery, which likely re-introduces a plane. Unverified — the single mechanism that could break the ~1150 radix floor; test before calling the floor absolute. Absent that, the frontier is **Sho's ongoing radix refinement (1162, actively being mined)**, not a new mechanism.

## Do-not
- Do NOT rebuild the bare-Slice / dynamic-shape-output route (any variant): grader requires exact [1,10,30,30] element-equality (`neurogolf_utils.py:546`). The cheap 469 is unshippable, full stop.
- Do NOT grind radix micro-variants to shave 1485→~1385: still > Sho's 1162 (forbidden rung-minus-ε shave for a STRUCTURAL slot).
- No candidate staged (nothing < 1162, let alone < 148).

---

## Batch 57 / bo_ng_cld_worker3 (Claude lane), 2026-07-11 — RE-CONFIRMED + exp-ramp loose end CLOSED

**Bar is now FIRM at 1162.** Sho's pending ceiling from batch 56 was promoted: `ng_current_bar --task 310` -> live_cost **1162**, sha `a1940b30...`, owner Sho, source=manifest, `stale:false` (workbook + manifest both agree at 1162; the 1485 the prior batch used as "live" is retired). No active `Staged_Boladi`/`Staged_Sho/task310.json` (only `_hidden_failed` archives) -> fully promoted, no pending rung. Any win must be **< 1162**.

**Charged component named in the Direction is CONFIRMED GONE.** Re-inspected the live 1162 graph (`Team_Best_Onnx/task310.onnx`, sha `a1940b30...`, `t310_bin_transducer`, 27 nodes): a **pure-binary radix** free-output Einsum -- E0..E4 = the 5 binary bit-planes of positions 0..29 (`[30,2]` each = 300), `Amid[2,2,2,2,2]`=32 the carry tensor; **zero F32 crop plane, zero Gather index plane** (the 10 Gathers are tiny `E[30,2]`-bit lookups, `[1,2]` each). Profile = static = **1162 = memory 796 + params 366**. Per the 2026-07-10 workflow-update rule ("if the charged component is already gone, return DROP with the refreshed floor note") => **DROP**.

**Reprices reconfirmed against the firm env (2026-07-11):**
- `direct_slice` (bare dynamic-Slice output): profile **469** (mem 464 + params 5); `ng_gate visible` = **0/266 FAIL** -- ORT `VerifyOutputSizes {1,10,5,5}/{6,6}/{7,7} != {1,10,30,30}`. Grader-dead (shape), unchanged.
- `slice_pad` (Slice->Pad to static [1,10,30,30]): profile **3133** (mem 3120 + params 13); `ng_gate visible` = **266/266 PASS**. Correct but loses (crop intermediate dominates); static 36573 is GRADER-GAP fiction.

**Exp-ramp hypothesis (the prior batch's one open loose end) -- REFUTED numerically** (`task310/prototypes/exp_ramp_refute.py`). The per-axis crop-place is a shift matrix `S[Y,R]=1 iff R==Y+r0`. Measured `rank(S)` = **25-30** on its support (partial permutation, all singular values 1.0). A rank-1 exp-ramp `a^Y*b^R*c^{r0}` is rank <= 1, so it **cannot** represent S: best rank-1 (SVD) reconstruction Frobenius err **4.90**, crop max pixel err **9.0** (wrong color indices => visible FAIL). Making it exact needs either the binary 5-factor log-decomposition (= Sho's radix, the MINIMAL exact factorization) or an explicit `Equal` exact-match plane (`[30,30]`, >=3600 B). **The exp-ramp does not break the floor.**

**Why 148 is unreachable -- tightened.** Even a *hypothetical zero-param* shift cannot reach 148: the forced front-end memory alone = counts`[1,10]`F32 (40) + rowcnt`[1,30]`F32 (120) + colcnt`[1,30]`F32 (120) = **280 B > 148**, before any mask/shift/params. rowcnt/colcnt are Einsum sum-product outputs (F32; no U8 kernel) and are the minimal signal ArgMax needs to locate first/last marker row & col. 148 is a cross-representation artifact of the external LB (raw-grid), not achievable under the NeuroGolf `[1,10,30,30]` one-hot output-shape contract. The exact floor for THIS representation is Sho's radix at **1162**.

**Verdict: DROP / WALL (no candidate < 1162).** One micro-shave exists but is FORBIDDEN for this STRUCTURAL slot and does not approach 148: swap `mr=(iota<=r1)` for `mr=(rowcnt>0)` (the frame spans every bbox row) making `iota30` unused -> -30 params -> 1132. Rung-minus-epsilon; not staged.

### Batch 57 PASS 2 (forced second pass) — occupancy variant BUILT+GATED (regression), bbox stats, shift-floor law

**bbox distribution (all 266 pairs, marker=min-count present color, crop=bbox => output byte-exact 266/266):**
`r0,c0 in [0,23]` (max 23 -> **5 shift bits genuinely required**; a 4-bit radix would FAIL), `r1,c1 in [4,28]`, `H,W in [5,8]`, **all outputs square (H==W)**, input sizes 20..30. The small crop (`<=8`) and square invariant do NOT shrink the `[1,10,30,30]`-output-forced 30-length axis intermediates (rowcnt/colcnt/mask are all `[30]`).

**Probe #5 — occupancy-mask radix variant** (`prototypes/build_task310_occ_mask.py`, genuinely different front-end): replace input-side `mr=[R<=r1]` (needs ArgMax r1,c1 + iota30) with hidden-safe **occupancy `mr=(rowcnt>0)`** reusing rowcnt/colcnt; drop 2 ArgMax + iota30. Result: `ng_gate visible` **266/266 PASS**, `profile_real_cost` **1177 = mem 840 + params 337** -> **LOSES to 1162 by +15**. `ng_graph_cost_diff`: −30 params (iota30) is out-weighed by +60 memory (two materialized `[1,30]` BOOL occupancy tensors mrb2/mcb2) => net **+15 regression**. Measured proof that Sho's input-side `iota<=r1` mask is *already* the byte-minimal mask topology. Gate ladder honored (profile does not beat live => no `--full` battery; that would be pure burn).

**Library floor-law grounding (cost_floor_reference.md):**
- **§ "Shift/translation floor = gathered-axis length" (task053):** a data-dependent shift's cheapest realization is O(shifted-axis-length)=**30**; multi-op 2-axis shift assemblies measured **3603 (Gather3+Pad) / 7205 (Slice+Concat)**; only a free-output Einsum fusion stays sub-3600. The live radix IS this floor realized — E factors `[30,2]`, masks `[30]`, rowcnt/colcnt `[30]` are all O(30). My 2-axis data-dependent crop-place cannot beat it.
- **Line 18 (dtype-narrow never nets out):** a U8 color-index crop needs the `[1,30,30]` channel-collapse as a >=plane-bytes F32/INT cast source (>=3600) OR a *static* crop location (Conv); mine is data-dependent -> dead.
- Post-Einsum mask multiply = `[1,10,30,30]` intermediate 36000 -> dead.

**Conclusion unchanged and tightened: WALL.** Every genuinely-different mechanism (Gather-window ~20112, Slice+Pad 3133, materialized `[30,30]` selection ~7200, U8-crop >=3600, occupancy-mask 1177, post-mul 36000, exp-ramp non-exact) is >= the radix or invalid. The radix (mem ~796 + params ~366 = 1162) is the axis-length floor for an exact `[1,10,30,30]` crop-place; 148 is a raw-grid external-LB artifact.
