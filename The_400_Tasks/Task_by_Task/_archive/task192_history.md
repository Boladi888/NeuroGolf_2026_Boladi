# task192 — superseded attempt history (archived 2026-06-21 by W11 per WORKER_PROTOCOL md-size rule)

Current best, Rule, Cost-analysis floor, the three most-recent wall proofs, and the full Don't-do log
remain in `The_400_Tasks/Task_by_Task/task192.md`. This file holds the superseded v1..v20 attempt rows
(the ladder down from 56,531 -> 3,031) and the older spread-knowledge scan, kept for provenance only.

## Superseded attempts (v1 .. v20 ladder)

| Date | Approach | Result | Cost | Notes |
|---|---|---|---|---|
| 2026-05-19 | crop-to-active-grid: tried 14×18, 18×20, 19×20, 20×19 etc. | FAIL | — | Live data needs exactly 20×20 (19×anything fails; 20×19 fails 184/265) |
| 2026-06-01 | v1 single-channel color-index canvas, full-canvas (`build_task192_v1.py`) | PASS | 20,873 | Conv on the *free* F32 input (no 18k pre-cast, no `[1,10,20,20]` tensors) → solid-rect mask + dominant color → `[1,1,30,30]` index canvas → `Equal(canvas,[0..9])`. visible 265/265, ab 2265/2265. |
| 2026-06-01 | v2 = v1 + crop single-channel intermediates to 20×20 + Pad canvas (`build_task192_v2.py`) | PASS | 18,389 | Shrinks every `[1,1,30,30]` except the two F32 input-reads to `[1,1,20,20]`. visible 265/265, ab 2265/2265, truth 4000/4000. |
| 2026-06-01 | v3 = v2 + fold the TWO F32 input-reads into ONE weighted conv `[0.25,1..1]` (`build_task192_v3.py`) | PASS* | **13,597** | empty→0/bg→0.25/colored→1; `active=>0.1`, `colored=>0.5` byte-identical to LIVE's two masks. Former cheapest. visible 265/265, ab 2265/2265, truth 6000/6000. **\*BUT diverges from LIVE on dominant-color ties** (522/5000 random multicolor, all tie-grids; single-channel canvas can't multi-hot). Real data has 0 ties — aggressive option, +1.425 LB. |
| 2026-06-01 | **v4 = v3 single-read + multi-hot `Where(BOOL,U8,U8)` painting (opset 13)** (`build_task192_v4.py`) | **PASS (TIE-SAFE)** | **18,166** (was 56,531) | Tie-safe: reproduces LIVE's multi-hot-on-tie via 2× `[1,10,20,20]` U8 (8,000 B vs LIVE's 24,000 B F16). Gate: visible 265/265, ab byte-equal 2265/2265, truth 8000/8000, multicolor 5000/5000 incl ties. Byte-equal on ALL inputs. |
| 2026-06-09 | **v5 = v3 + #98 dilated crop-collapse + U8 morphology + U8 canvas** (`build_task192_v5.py`) | **PASS (STAGED)** | **7,577** | Replaces charged `[1,1,30,30]` `combined` with direct `[1,1,20,20]` dilated Conv carrier (2×2 kernel, dilation 10), U8 MaxPool erode/dilate, U8 Equal-tail canvas. Gates: visible 265/265, truth/ab `--full` byte-equal. Staged at `staged_submits/task192_dilated_u8_canvas_cost7577/`. |
| 2026-06-09 | v6 = v5 color-inverse fuse (`build_task192_v6.py`) | PASS | **7,177** | Exact: `Greater(comb,0.5)->Not` becomes `Less(comb,0.5)`, dropping one 20×20 BOOL surface. Superseded by v8. |
| 2026-06-09 | v7/v8 q2-max rewrite (`build_task192_v8_opt.py`) | **PASS (STAGED)** | **5,395** | 1×1 negative-pad Conv crop carrier, direct `Cast(comb->BOOL/UINT8)`, `QLinearConv(2×2)` colored counts, `MaxPool` counts before threshold, `Einsum("bchw,c->bc")` non-bg color counts. Gates: visible 265/265, truth/ab `--full` byte-equal. Staged at `staged_submits/task192_q2max_negpad_cast_einsum_cost5395/`. |
| 2026-06-09 | sub-5k probes: no-pad20 and one-pass q4/linear stencils | FAIL | 4,487 / 5,000 | `task192_v9_nopad20.onnx` exact vs cropped LIVE, scores 4,487, but visible rejects 20×20 output shape (`0/265`) — 30×30 output mandatory. Optimized q4-support 5,000 fails visible (`157/265`). No separating 3×3 stencil; no viable sub-5k one-pass detector. |
| 2026-06-09 | v10/v11 free-output Conv probes | **DEAD END** | 910 | One F32 `Conv(input)->output`. Visible 265/265, but best soft 3×3 full truth only `137779/200000` (68.889%). Zero hidden credit; do not submit/continue this family. |
| 2026-06-09 | v12 q2-paint rewrite (`build_task192_v12_q2paint.py`) | **PASS (STAGED)** | **4,995** | `QLinearConv` int32 bias `-3` -> binary `q2bin`; `Mul(q2bin, main_idx)` writes dominant color at full-block top-lefts; `MaxPool` dilates; one `Where(active, paint20, 10)`. Gates byte-equal `--full`. Staged at `staged_submits/task192_q2paint_exact_cost4995/`. |
| 2026-06-09 | v13/v14 #137 U8-carrier probes | FAIL / WORSE | invalid / 39,395 | Direct `QLinearConv(raw F32 input)` fails shape inference. Reserved-name bridge (`raw -> Cast("input") U8 -> QLinearConv`) visible-clean but scorer charges the 9,000-byte cast, profiles `memory=39349`; do not pursue input-name accounting tricks. |
| 2026-06-09 | v15→v20 exact qcarrier/free-output tail (`build_task192_v20_prefused_q2direct.py`) | PASS (superseded) | 3,031 | v15 removed `Mul(q2bin, main_idx)` w/ dynamic `QLinearConv` scale (4,598); v17 moved final tail into graph-output `QLinearConv` (4,566); v18 used 0/1/4 carrier to delete active masks (3,766); v19 fused rect dilation into final 2×2 output QLC (3,426); v20 prefuses `[carrier_copy, q2bin]` in one 2-output QLC. Gates byte-equal `--full`. Staged at `staged_submits/task192_prefused_q2direct_exact_cost3031/`. |

## Older spread-knowledge scan (20260518)

Canonical cost at scan time: 76199. Pattern sweep: `slice_axis_collapse` saved 4 bytes (byte-exact);
`crop_to_size_advisory` rough est ~20000 (already realized by the v1..v21 ladder); `fp16_selective`
failed verify. `polish_candidate.py` applies `slice_axis_collapse` (-4 B). f16_robust prior worker
delta -1,664 (verify 265/265) — subsumed by the U8 ladder. These are all superseded by v21 @ 3030.

---

## 2026-07-15 cleanup compression source for task192.md

Full pre-compression task file archived from `Task_by_Task/task192.md` because repo_lint manual content exceeded 20 KB.

```markdown
# task192

<!-- AUTO:BEGIN -->
<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->

**STATUS:** LIVE cost **13,597** (live_costs.json, refreshed 2026-06-05).
(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)

**Linked research / prototypes** (auto-detected by filename):
- _none found_
<!-- AUTO:END -->

<!-- HIST:BEGIN -->
<!-- mined by tools/mine_task_history.py; rerun to refresh -->

**Git history** (commits touching files or subjects mentioning this task):
- `648e1306` — docs+prompts: Onnx_Advanced discoverability restructure + drive-to-floor prompts
- `a81a00b0` — agent_helpers/ (9 scripts) + auto-graph-diag in all 401 task .md files
- `1911db7d` — OB_HUGE worker: final wrap-up notes for 8 unstaged tasks
- `9c10e3fe` — OB_HUGE worker: add task137/008/014/088/157/195/290 candidates and notes
- `158422cb` — HUGE_CONV_c: 2 staged candidates, 94545 bytes saved
- `f0d0a66a` — HUGE_RSUM_b: 3 staged candidates, -229449 bytes total, +3.285 pts expected
- `751fd1d6` — session 2026-05-18: accumulated task notes across worker runs
- `3c61b5d5` — Junction-relocate BatchPilotArchive/ out of tree to .notongit/
- `54ec3c87` — Mine session logs + recipe inventory + LIVE_ALGORITHM for all recipe-tasks
- `f2023f06` — Update docs + re-mine HIST (401 tasks) + fix re.sub backslash escape bug
- `e69612d9` — Mine git/session/lessons/notes history into per-task .md HIST blocks
- `6668c221` — Per-task .md folder (Task_by_Task/) + migrate task_cards/ + update docs
- `b2e3cd93` — File away scoring artifacts
- `315ed3f3` — Archive pilot artifacts
- `316414b0` — Track generated NeuroGolf artifacts and env path overrides
- _...and 3 more_

<!-- HIST:END -->


## Rule

**Symmetry/rectangle completion.** Find the solid colored rectangle(s) and the dominant non-bg
color; output = paint the rectangle region with the dominant color, paint every other *active*
(non-empty) cell as background (ch0), leave padding empty. Decoded from LIVE 2026-06-01 (initializer
dump below).

## Algorithm sketch

```
# all on the 20x20 active region (LIVE crops; full-canvas processing is byte-equal since grids <= 20x20)
col   = sum(input[:, 1:10], axis=channels)          # colored-cell (ch1..9) indicator, 0/1
s3    = conv2d(col, ones[2,2], pad=bottom-right)     # count colored cells in each 2x2 block
gt9   = s3 >= 4 (thr 3.5)                            # top-left of a FULLY colored 2x2 block
rect  = conv2d(gt9, ones[2,2], pad=top-left) >= 0.5  # dilate back -> solid-rectangle mask
counts= sum(input[:, 1:10], axis=spatial)            # per-color pixel count (ch0 zeroed)
main  = (counts >= max(counts) - 0.5)                # dominant non-bg color one-hot (no ties in data)
active= sum(input, axis=channels) > 0                # inside-grid indicator (incl bg ch0)
out[rect]            = main_color                     # dominant color in the rectangle
out[active & ~rect]  = ch0 (background)              # everything else inside the grid -> bg
out[~active]         = empty                          # padding -> all-zero one-hot
```

Initializers (LIVE): `Wc=[0,1,1,1,1,1,1,1,1,1]`, `W2=ones(2,2)`, `Wall=ones(10)`, `CH0=[1,0..0]`,
`thr9=3.5`, `half=0.5`; crop `[:, :20, :20]`; pad back `+10/+10` rows/cols.

## Attempts

_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._

## COMPRESSED HISTORY
Superseded v1..v20 attempt rows (ladder 56,531 -> 3,031) + the older 20260518 spread scan moved to
`The_400_Tasks/Task_by_Task/_archive/task192_history.md` (2026-06-21, W11). Below: current best (v21)
and the three most-recent wall proofs only.

| Date | Approach | Result | Cost | Notes |
|---|---|---|---|---|
| 2026-06-12 | **v21 INT64-Equal skip-cast (`build_task192_v21_int64equal.py`)** | **PASS (STAGED)** | **3,030** | Skip `Cast(main_idx_i64 INT64→U8)→main_idx_u8 (1B)`. Use `Equal(INT64,INT64[9])` directly (INT64 Equal valid at opset 14). `idx8_colors` changed U8→INT64 (same 9 params). Net: -1B memory, same 123 params. Gates: visible 265/265, truth `--full` 200000/200000, ab `--full` 200265/200265 byte-equal, score/profile `2907+123=3030`. Staged at `staged_submits/task192_carrierdel_cost3030/`. |
| 2026-06-19 | neighbor-count + Where-passthrough analysis (Worker-H) | DEAD END — cannot match LIVE | N/A | Full analysis: (1) `same-color-neighbor ≥ 1` as clustered flag fails for IN-BOX DISTRACTORS — distractor pixels inside the rectangle have `color ≠ boxcolor` neighbors; same-color count = 0 → outputs bg, but LIVE outputs boxcolor → diverges on every real input (in-box distractors occur ~100% of the time given 5% pixel density over ~40–60% box coverage). (2) `any-color-neighbor ≥ 1` as clustered flag fails for EDGE-ADJACENT DISTRACTORS — a distractor adjacent to the box edge has non-bg neighbors → incorrectly included in rect → kept as distractor color, not bg. (3) Even if semantically correct, F32 input passthrough requires `Cast(input, U8)` = 9000B charged intermediate (graph output must be U8, not F32). Total cost of passthrough approach ≥ 9000B memory; far worse than v21 (2907B). (4) All other approaches to reduce v21 (ReduceSum vs Einsum, F16 intermediates, Pad vs Concat, Range vs initializer, 1-QLC merge, smaller kernels) were priced and all come out ≥ 3030. The carrier_f F32 [1,1,20,20] = 1600B is structurally unavoidable for any Conv-based morphology approach. CONFIRMED FLOOR: v21 at 3030 is the task floor. |
| 2026-06-20 | Worker task192 free-output/Einsum rank audit | DEAD END below v21; v21 re-gated | 3,030 | Prompt lever tested: target non-bg output mask is not separable. Official 265 pairs rank histogram `{2:4, 3:102, 4:86, 5:73}`; 2000 fresh ARC-GEN samples from `external/ARC-GEN/tasks/task_7e0986d6.py` rank histogram `{2:34, 3:763, 4:665, 5:538}`, rank1=0. Source has `num_boxes = randint(3,5)`, so #109 would need paired dynamic row/col factors for up to 5 boxes, not one row-band x col-band. No sub-3030 candidate found. Re-gated `staged_submits/task192_carrierdel_cost3030/task192.onnx`: visible 265/265, truth 200000/200000, AB 200265/200265 byte-equal, static/profile 3030. |
| 2026-06-21 | W11: free-input marginal-AND 2x2-solidity detector priced (PRIMARY lever) | DEAD END — detector alone > whole v21; cross-formulation floor | 3,030 | The prompt's PRIMARY route (delete the Conv carrier via `AND of 4 shifted U8 marginals`) was priced and FAILS. Root cause: ONNX Einsum/Conv/Reduce over the free F32 input ALL output F32 (Einsum CPU dtypes = float/int32/int64 only — no U8/BOOL; Conv out=input dtype=F32); only a comparison (`Greater`) emits BOOL, and it needs the already-charged F32 marginal as input. So the `colored` spatial map is FORCED to F32 `[1,1,20,20]`=1600B — identical to v21's `carrier_f`. The marginal route does NOT delete carrier_f; it renames it AND adds bool-conversion + 4 shifted-slice BOOL planes + dilate on top. Measured cheapest cropped detector portion alone (Conv colored_f 1600 + Greater bool 400 + 4-way AND + Cast + MaxPool dilate) = **5121B** (under-estimate; proper re-padded shifts add more) — already EXCEEDS the entire v21 model (3030) before the dominant-color path (107B) or output assembly. v21's QLinearConv morphology (carrier_u8 400 + carrier_q2 800 = 1200B for full erode+dilate) is strictly cheaper than the bool-shift-AND chain. **3030 is a cross-formulation floor: the forced 1600B F32 spatial marginal is paid by ANY detector (Conv morphology OR marginal-AND OR separable Einsum).** v21 re-gated this session: visible 265/265, AB 200265/200265 byte-equal, static=profile=3030 (arcgen 266). |
| 2026-06-26 | sub-1000 zero-memory final-op audit | DEAD END — no shippable sub-1000 rewrite found | 3,030 | Tested the only honest sub-1000 surfaces: final-output Conv families and tiny-vector dynamic-bias variants. Existing `task192_freeconv3x3_soft_risky_cost910` remains visible-clean but generator-wrong (`truth --n 2000`: 1466/2000; manifest full truth 137779/200000). Competitor cost-0/1 task192 artifacts are Identity/`Greater(input,0)` stubs and fail visible 0/265. A grouped 9x9 final Conv prices at 820, but color-channel LP is infeasible on 300 fresh ARC-GEN examples; adding a 10-count dynamic bias matrix keeps the price near 990 and makes channels 1-9 separable, but channel 0 is infeasible because background requires local all-color access, not just the ch0 plane. Dense 3x3 final Conv variants price at 910; fresh ARC-GEN patch-conflict scan found an exact 3x3 conflict by example 800, and dilated 3x3 variants hit ch0 conflicts/infeasibility. No candidate staged. |
| 2026-06-26 | retry: dense final Conv + full-count dynamic bias | DEAD END — visible-clean 2680 probe is generator-wrong | 3,030 | Tested stronger final-output Conv family: `counts = Einsum(input)->[10]`, `bias = Einsum(counts,A[10,10])->[10]`, final dense Conv to graph output. Area-8 kernels price ~980 but hit ch0 conflicts/infeasibility even with full counts. Area-9 3x3 prices 1080; sample LP fit existed, but official+1000 constraints make ch0 infeasible. Wider 5x5 prices 2680 and has no feature conflicts on 5000 fresh ARC-GEN examples, but the official+300 LP probe only gates visible 265/265 and fails truth 706/2000 (35.3%). This family is not stageable without a much stronger proof/training loop, and it does not meet the requested <1000 target. |
| 2026-06-26 | risk probe: projected SVM 5x5 count-bias final Conv | RISK-STAGED — visible-clean, below v21, not exact | 2,730 | A max-margin 5x5 final-output Conv with full-count dynamic bias generalized better than LP: after minimal L1 projection onto official constraints, staged at `task192_dense5x5_countbias_risk_cost2730/`. Cost/profile 2730 = memory 120 + params 2610; visible 265/265; truth 1727/2000 and 16850/20000 (84.25%); AB n=2000: 1992/2265, 273 divergences. This is an explicit risk probe only (`do_not_submit_without_captain_probe=true`), not an exact replacement for v21. |

## Cost analysis

**Measured 2026-06-01 (authoritative, on shipped `submission.zip::task192.onnx`).** Cost 56,531
= memory 56,474 + params 57. The shipped model is **already F16 + cropped to 20×20** — there is
**NO `[1,10,30,30]` INT64 tensor** (the Task192 prompt's "kill the 72,000 INT64" premise was a
mis-pattern-match against `cost_floor_reference.md`; 72,000 > the entire 56,474 memory, so it was
impossible). The auto-graph-diag block below (74,531) is a STALE f16 prototype snapshot, not the
shipped model — f16 *raised* cost there.

LIVE memory breakdown (per-tensor): `input_cast_to_crop_input_inner [1,10,30,30] F16 = 18,000`
(31.9% — the full-canvas Cast done BEFORE the crop Slice) + four `[1,10,20,20]` F16 = 32,000
(`input_inner`, `paint`, `bg_paint`, `output_inner`) + ~6,400 (`[1,1,20,20]` masks).

**Levers:** (1) the three `[1,10,20,20]` one-hot tensors (paint/bg_paint/output_inner, 24,000 B)
collapse to ONE `[1,1,30,30]` single-channel index canvas via the `Equal(canvas,[0..9])` floor
trick — output is BOOL and not counted; (2) `input` `[1,10,30,30]` F32 is **not counted**, and the
1×1 convs commute with the crop with zero-padded boundary == LIVE, so Conv directly on `input`
eliminates BOTH the 18,000 cast and the `[1,10,20,20]` working tensors. v1 applies both → 20,873.

**Floor analysis (2026-06-09 update).** The earlier "one mandatory F32 input-read (~5.2k incl crop)"
floor was too pessimistic because the 30×30 carrier was not mandatory. #98 applies directly: a 2×2
Conv with dilation 10 has effective size 11, so it reads the free `[1,10,30,30]` input and emits the
top-left 20×20 weighted carrier in one node. Only the top-left tap is nonzero (`[0.25,1..1]`), so the
semantics are identical to v3's `Conv -> Slice`, but the charged 3,600-byte `combined` tensor is gone.

Current best (v21 INT64-Equal, 3,030) is byte-equal to LIVE on the full ARC-GEN corpus for this
generator: visible 265/265, truth `--full` 200000/200000, ab `--full` 200265/200265, gate profile==static.
The remaining charged surfaces are `carrier_f` F32 1,600, `carrier_u8` 400, `carrier_q2` U8[1,2,20,20]
800, `main_idx_i64` INT64[1] 8, and 99 bytes for the rest of the color-scale recovery. The F32 carrier
(1600B) is structurally unavoidable: QLC requires U8 input, and any full Cast of the free F32 input
costs 9000B. The two-QLC architecture (q2 erosion + output dilation/routing) is algorithmically
necessary; no single-pass QLC can compute OR-of-four-2x2-blocks (non-linear). The 1B savings in v21
(skip Cast INT64→U8 in the color-scale path) is the only recovery found under the carrier deletion
direction. The exact sub-2k gap would require deleting the F32 carrier entirely, which is not feasible.

**Free-output/Einsum audit (2026-06-20).** The #109/#119 single/separable output premise is false for
this generator. The output is the union of 3-5 non-overlapping rectangles, and the non-bg output mask
has rank 2-5 on both officials and fresh ARC-GEN samples (0/2000 rank-1 fresh masks). A cheap
row-band x col-band final Einsum would overpaint cross-products between boxes. An exact #109 rewrite
would first need dynamic pairing/labeling of up to five row and column intervals; that is the same
information the current `carrier_q2` + final QLinearConv keeps in a 800B local morphology plane. This
is a formulation-relative wall for the requested free-output separable route, not a universal proof
that no possible future formulation can beat 3030.

**Free-input marginal-AND detector wall (2026-06-21, W11).** The PRIMARY prompted escape — delete the
Conv carrier by computing the rect mask as `AND of 4 shifted U8 marginals` read from the free F32 input
— was priced and is strictly worse than v21. The 1600B `carrier_f` (F32 `[1,1,20,20]` spatial marginal)
is NOT a Conv artifact; it is FORCED by the ONNX dtype rules for reading the free F32 input: Einsum CPU
supports float/int32/int64 only (no U8/BOOL), Conv output dtype = input dtype = F32, and ReduceMax/Sum
over the F32 input also emit F32. The ONLY op that produces BOOL directly is a comparison (`Greater`),
which requires the already-charged F32 marginal as its input. Therefore EVERY detector formulation pays
the same 1600B F32 spatial map once. The marginal-AND route then ADDS a BOOL `colored` (400B), four
shifted-slice BOOL planes, the AND chain, and a MaxPool dilate on TOP of that 1600B — measured cheapest
cropped detector = 5121B, which already exceeds the entire v21 model (3030). v21's two-QLinearConv
morphology (carrier_u8 400 + carrier_q2 800 = 1200B for erode+dilate) is cheaper than the bool-shift-AND
chain. **Conclusion: 3030 is a CROSS-FORMULATION floor across all three audited routes (Conv morphology,
separable free-output Einsum, free-input marginal-AND). The forced 1600B F32 spatial marginal is the
hard floor component; no charged-tensor route deletes it.** No candidate staged (a tie/loss adds 0 pts).

**Sub-1000 audit (2026-06-26).** The only cost model that can land below 1000 is a zero-memory final
op: no charged spatial intermediate, and params below 1000. The known final Conv rung at 910 is not
generator-clean. New checks did not find a repair: grouped 9x9 final Conv (820) cannot solve channel 0;
the stronger grouped 9x9 + dynamic 10-count bias matrix (~990) separates color channels 1-9 but leaves
background infeasible; dense 3x3 final Conv has real ARC-GEN local-patch conflicts, and dilated 3x3
variants conflict or fail at channel 0. This does not supersede v21; it just closes the requested
sub-1000 zero-memory rewrite pass.

**Dense final Conv + full-count bias retry (2026-06-26).** A near-miss family computes a [10] full-color count vector, maps it to a dynamic Conv bias, then makes the dense Conv the graph output. Area-8 kernels fit under 1000 (~980) but fail ch0. Area-9 3x3 costs 1080 but is infeasible once officials+1000 fresh samples are hard constraints. The arbitrary LP 5x5 version costs 2680 and gates visible but fails truth 706/2000. A max-margin 5x5 version projected onto visible constraints is stronger and staged as a risk probe at 2730, but still fails generator truth at 16850/20000; it is not an exact ship candidate.


**Under-2000 dead end (2026-06-09).** Single free-output Conv variants (910/1510/1610) are visible-clean
but generator-wrong; best soft 3×3 full truth is 68.889%, so expected hidden credit is zero. Do not
submit or continue this family.

Correctness contracts:
- **Generator-real distribution (v5, 7,577; staged)**: same single-channel index canvas contract as v3.
  ARC-GEN 7e0986d6 uses one box color plus isolated distractor pixels; dominant-color ties are absent
  on the real corpus, and full truth/A-B gates pass.
- **All-input tie-safe contract (v4, 18,166)**: keep the multi-hot `Where(BOOL,U8,U8)` painting if an
  off-generator dominant-color tie must reproduce LIVE exactly.

## Don't-do log

- **Crop-to-active-grid**: Training max grid is 13×17, but live data has 265 examples needing exactly 20 rows × 20 cols. Probing showed: 20×20 passes (265/265), 19×anything fails (~183/265), 20×19 fails (184/265). The model already crops to the minimum viable size.
- **F16 input/output trick**: Model has no existing F16 chain. A Cast-before-Slice approach creates a large [1,10,30,30] F16 intermediate (18k bytes) — more than it saves.
- **fp16_selective**: failed verify per spread scan.
- **Bool dilation replacement**: replacing Conv-based 2×2 dilation with boolean Pad+Slice shifts adds more memory than it saves.
- **DON'T trust the "kill the 72,000 INT64" prompt premise** (2026-06-01): there is no INT64 tensor; the shipped model is F16+cropped at 56,474 memory (72,000 > total, impossible). Always run `largest_intermediate.py` on the *shipped* `submission.zip::task192.onnx` before believing any recon — the auto-graph-diag block in this file is a stale f16 prototype.
- **DON'T crop the 10-channel input to chase the floor**: a `[1,10,20,20]` working tensor costs 8k (F16) / 16k (F32). The `input` `[1,10,30,30]` F32 is *free* (not counted) — Conv on it directly and keep only single-channel `[1,1,30,30]` intermediates. (v1 floor ≈ 20.8k = two mandatory F32 input-reads + rect chain.)
- **DON'T reintroduce a full `[1,1,30,30]` combined carrier before cropping**: v5 proves the #98 dilated crop-collapse route for this task. Use a 2×2 Conv with dilation 10 and only the top-left tap nonzero to emit the 20×20 carrier directly.
- **Prefer the v8 negative-pad 1×1 carrier over the v5 dilated 2×2 carrier**: `Conv(kernel=1×1, pads=[0,0,-10,-10])` emits the same 20×20 weighted carrier with 10 params instead of 40, and enables direct `Cast(comb, BOOL/UINT8)` for active/colored masks.
- **DON'T drop the final 30×30 tail even though the cropped tensor is exact**: `task192_v9_nopad20.onnx` scores 4,487 and is byte-equal to cropped LIVE, but `ng_gate visible` rejects the output shape at `0/265`. The fixed 30×30 output contract is mandatory here.
- **DON'T stage the one-pass q4-support detector**: optimized q4-support scores 5,000 and fails public (`157/265`). The failure is algorithmic; public LP checks showed no separating 3×3 linear stencil over colored occupancy or bg-aware carrier values.
- **DON'T submit or continue the 910 free-output Conv family**: it is visible-clean and below 1600/2000, but full ARC-GEN truth is only 68.889%. The local-patch family has real conflicts and should be treated as a zero-credit dead end, not a learning probe.
- **DON'T expect a one-pass linear rect detector to replace q2 morphology**: public/generator LP checks show the OR-of-four-full-2×2-block predicate is not linearly separable as a single local threshold. v12's exact lever is different: binarize q2, then propagate the color before the final active/sentinel Where.
- **DON'T apply #137 by casting/quantizing the full input**: QLinearConv cannot consume the fixed F32 NeuroGolf input directly. A full `Cast(input, UINT8)` bridge costs 9,000 B and a reserved-name bridge still scores worse (`task192_v14_named_input_qcarrier.onnx`: 39,395). The U8 carrier would only be a win if the quantized input were genuinely free, which this task's scorer/gate path does not allow.
- **DON'T F16-cast the input-reads or `s3`**: input is F32, so a Cast to F16 adds a tensor equal to what it saves (no net win). F16 only helps the post-threshold chain (fullmask/dil_raw/inner/canvas).
- **DON'T pursue same-color OR any-color neighbor-count passthrough (2026-06-19 confirmed dead):** Same-color-neighbor ≥ 1 diverges on IN-BOX DISTRACTORS (distractor inside the box has color ≠ boxcolor neighbors → same-color count = 0 → outputs bg, but LIVE outputs boxcolor). Any-color-neighbor ≥ 1 diverges on EDGE-ADJACENT DISTRACTORS (distractor next to box edge has non-bg neighbors → incorrectly classified as in-rect → kept as distractor color not bg). Even if the semantic were correct, the passthrough requires `Cast(F32 input, U8)` = 9000B — worse than v21. The F32 input passthrough trick from task193 does NOT transfer to task192 because task193 preserves per-cell color identity (Where keeps whatever color the cell had), while task192 requires overriding in-box distractors with boxcolor.
- **DON'T chase the free-input marginal-AND 2x2-solidity detector to delete the carrier (2026-06-21, W11 wall):** the `AND of 4 shifted U8 marginals` route does NOT delete the 1600B F32 `carrier_f`. ONNX gives no op that emits a U8/BOOL spatial map directly from the free F32 input (Einsum=float/int32/int64 only; Conv out=input dtype=F32; ReduceMax/Sum out=F32). The `colored` spatial marginal is FORCED to F32 1600B for any detector; the marginal-AND chain then piles a 400B BOOL `colored` + 4 shifted BOOL planes + dilate on top (measured cheapest cropped detector = 5121B > the whole v21 = 3030). v21's QLinearConv erode+dilate (1200B) is cheaper than the bool-shift-AND chain. 3030 is a cross-formulation floor (Conv morphology / separable Einsum / marginal-AND all pay the same forced 1600B F32 marginal).
- **DON'T continue sub-1000 zero-memory final Conv variants unless a new non-linear final op is found:** single final Conv is the only honest <1000 surface, but exact ARC-GEN behavior is not linearly separable. Dense 3x3 has real local-patch conflicts; grouped 9x9 lacks local all-color access for ch0; grouped 9x9 plus a dynamic 10-count bias matrix still leaves ch0 infeasible. Dense area-8 plus full-count bias still fails ch0 at ~980, and area-9 rises above the target before also failing official+1000 constraints. Final-output deep Einsum would need local shift selectors/term banks whose params exceed the <1000 target before correctness is addressed.
- **The single-channel index canvas (`Equal(canvas,[0..9])`) CANNOT reproduce LIVE's multi-hot-on-tie** (v3/v5). LIVE's dominant-color one-hot is `Greater(counts, max−0.5)`, which is multi-hot on a count tie → paints 2+ channels in the rectangle. A single canvas value paints one. This is invisible on real data (0/265 ties, full ARC-GEN truth/A-B clean for v5) but shows as 522/5000 divergence on random multicolor (ALL tie-grids). If you ship a single-channel-canvas candidate for a "dominant color" task, EITHER prove 0 ties in the distribution OR use the multi-hot `Where(BOOL,U8,U8)` painting (v4, +4.5k, tie-safe fallback).

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

Current canonical cost: **76199** (memory=76138, params=61, points≈13.758896).

Pattern sweep verdicts (each pattern measured INDEPENDENTLY vs the original — patterns overlap, so the sum overstates the real cumulative savings):

| Pattern | Verdict | Δ cost (vs original) | Notes |
|---|---|---:|---|
| `clip_mul_to_min` | — | — | no Clip[0,1] + Mul(mask) pairs detected |
| `const_to_init` | — | — | no Constant nodes with a 'value' attribute to convert |
| `crop_to_size_advisory` | ? | -20000 | max active grid is 20×20 but graph has 1 [...,30,30] intermediate(s). Crop-pr... |
| `fp16_selective` | ✗ verify | — | fp16-converted; driver must verify cost actually drops and output is within t... |
| `onnxsim` | — | — | onnxsim made no changes |
| `safe_opt` | — | — | no changes from safe-opt + extended passes |
| `scalar_broadcast_ones` | — | — | no uniform-valued float initializers consumed only by broadcasting ops |
| `slice_axis_collapse` | ✓ SAVES | -4 | collapsed axis-0 no-op from 1 Slice node(s) |

**Free LB available (sequential pipeline truth)**: running `tools/polish_candidate.py` on the current ONNX applies `slice_axis_collapse` and saves **4 bytes** (byte-exact verified on 3 inputs). Note: the per-pattern Δ above was measured against the ORIGINAL, so the per-pattern numbers do not sum to this — patterns subsume each other when applied sequentially.

**Advisory** (crop_to_size_advisory): max active grid is 20×20 but graph has 1 [...,30,30] intermediate(s). Crop-process-pad could save ~20000 bytes (rough est). — estimated savings if applied manually: ~20000 bytes.

_Re-run: `.venv/Scripts/python.exe tools/spread_scan.py` (updates JSON in `Logs/spread_scan_results/`); then `.venv/Scripts/python.exe tools/spread_annotate_notes.py`.
<!-- SPREAD:END -->

-  [OB_HUGE worker]: pattern=f16_conversion delta=-1664 verify=265/265 cand=task192_f16_t_d890b2e4_on_5835_30

### 2026-05-20 � PR_HUGE worker (this session)
- Technique: **f16_robust** (delta=-1,664 bytes)
- Verify: 265/265
- Adversarial: 100%
- Notes: Robust f16 converter.
- Staged at: `.hermes/board/staged_submits/task192_f16_*_on_5835_30/`

<!-- BEGIN AUTO-GRAPH-DIAG -->

## Graph Diagnostic (auto-generated)

**Cost**: 13,597 | **Memory**: 13,540 | **Params**: 57 | **Points**: 15.4824
**Nodes**: 22 | **Initializers**: 16 | **Opset**: 11 | **Graph output dtype**: BOOL
**Max active grid (from `task192.json`)**: 20×20

### Quick-look flags

- has_Conv: ✓
- has_MatMul: ✗
- has_ReduceSum: ✓
- has_ReduceMax: ✓
- has_Where: ✓
- has_Gather: ✗
- has_OneHot: ✗
- first_node: `Conv`
- last_node: `Equal`
- ends_with_Cast(F16→F32)→output: ✗
- first_node_is_Slice: ✗

### Op-type histogram

| op | count |
|---|---:|
| `Greater` | 5 |
| `Conv` | 3 |
| `Cast` | 3 |
| `ReduceSum` | 2 |
| `Mul` | 2 |
| `Where` | 2 |
| `Slice` | 1 |
| `ReduceMax` | 1 |
| `Sub` | 1 |
| `Pad` | 1 |
| `Equal` | 1 |

### Initializers (top 10 by params)

| name | dtype | shape | params |
|---|---|---|---:|
| `Wcomb` | F32 | [1, 10, 1, 1] | 10 |
| `Wc` | F32 | [1, 10, 1, 1] | 10 |
| `idx` | F16 | [1, 10, 1, 1] | 10 |
| `pad_canvas` | INT64 | [8] | 8 |
| `W2f16` | F16 | [1, 1, 2, 2] | 4 |
| `c_starts` | INT64 | [2] | 2 |
| `c_ends` | INT64 | [2] | 2 |
| `c_axes` | INT64 | [2] | 2 |
| `c_steps` | INT64 | [2] | 2 |
| `t_active` | F32 | [] | 1 |
| ... (6 more) | | | |

*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*

*Node list (22 nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py task192` for the full sequence.*

<!-- END AUTO-GRAPH-DIAG -->
```
