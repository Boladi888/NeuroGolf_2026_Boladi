# task156 FreeOutputPadTail — FALSE FLOOR / DROP

Date: 2026-07-11
Worker: bo_ng_cld_worker2 (Claude lane)

## Live/team bar (refreshed 2026-07-11)

- Authoritative bar: **cost 977 / profile_cost 977** (fresh profile, 265 real inputs: `static=977 profile=977 (memory 725 + params 252)`).
- Source: `python tools/ng_current_bar.py --task 156 --json` → authoritative_source=`manifest`; workbook agrees (977); `Staged_Boladi/_shipped/task156.json` shipped 2026-07-10T17:54Z at 977.
- Owner/sha: **Boladi**, `de09544e65761459dcbef871afca238d281618a26c6c349c224e48bad643586b`.
- The prompt's `980` (and the older `2de15be6…` Vamshi sha) is STALE: the affine-recode 980→977 already shipped after the prompt was authored (2026-07-09).

## The prompt's premise is already dissolved (why this is a DROP, not a WALL)

Prompt = false-floor attack: "dissolve the FLOAT16 1x10x10x10 Add/Cast prepad canvas / 2000-byte prepad tensor into a Pad/Where/Equal free-output tail" → target ~420.

The live 977 graph (`classcode_u8_qlinear3x3_final`, 7 nodes) has **already applied the FreeOutputPadTail optimization**. Direct inspection (`inspect_graph.py` + dtype/op scan of `Team_Best_Onnx/task156.onnx`):

- nodes = `[Slice, Cast, Einsum, Greater, Where, Where, QLinearConv]`
- **has Pad node: FALSE. has Add node: FALSE.**
- **FLOAT16 initializers: 0. FLOAT16 intermediates: 0.**
- The 30×30 output canvas is produced by `QLinearConv` with `pads=[1,1,21,21]` — the asymmetric pad absorbs the canvas into the conv at ZERO tensor cost; padded region = conv zero-point → background channel.

So the charged component the prompt wants me to delete **does not exist** in the live graph. Per the 2026-07-10 workflow rule ("if it is already gone … return DROP with the refreshed floor note"), this is a DROP.

## The ~420 free-output target is a phantom (structural infeasibility)

The rule (decoded): two stacked yellow(4) rectangles; output = yellow borders preserved, SMALLER rectangle interior→blue(1), LARGER→red(2), background 0.

- Border vs interior needs a **3×3 neighborhood** decision → forces a Conv/pool; a pure elementwise Equal/Where tail cannot compute "adjacent-to-non-yellow." (forced-component argument)
- The interiors are **2-D regions at data-dependent positions** with a **runtime small/large selector** (Einsum orientation → Greater → Where). Library verdicts #109 (2-D region complement forces canvas + Gather, not separable-Equal), #10 (runtime-decoded selectors kill free-output regardless of rank), #132 (variable placement = Conv/render floor) all independently rule out a free-output separable rendering here.

The current 2-channel-feature + QLinearConv IS the render floor for this rule.

## Fast probe already executed & killed (both prompt kill-conditions met)

Prompt fast probe = "one-channel predicate/index + Equal/Where tail; kill if it prices >420 OR falsifies on fresh ARC-GEN." Prior batch (2026-07-03) built exactly this:

- `prototypes/task156_onechan_qlinear.onnx` (see `hard_records/onechan_qlinear.md`): static **758 (>420 → kill-1)** AND `[visible] 0/265 (0.00%) FAIL (→ kill-2)`. One channel collapses padding with active background.

Plus 3 more independent sub-980 micro-recodes already dead (`hard_records/micro_recode_failures.md`): scalar-background `visible 0/265`; INT8 features `ORT NotImplemented Where`; UINT8-weights-zp1 `visible 168/265`.

## Every charged byte of the 977 floor is forced

memory 725 = c4_f F32 400 (F32-input slice; needed by Cast→c4 AND Einsum→orientation; a cheaper dtype needs a costlier whole-input Cast) + features U8 200 (2-channel; 1-channel is visible-dead above) + c4 BOOL 100 (class selector) + yellow_codes 20 + orientation 4 + top_small 1.
params 252 = weights 180 (10-color output from ONE conv; colors 3/5/6/7/8/9 are all-zero filters but shrinking to 4 out-channels then padding channel-dim 4→10 needs a [1,6,30,30]=5400 B zero tensor ≫ 108 params saved) + bias 10 (must be full length-10, traps §8h: sub-declared Conv bias is visible-unstable) + yellow_codes_top/bottom 40 + row_weight 10 + scalars ~5.

Note on the tempting yellow_codes redundancy: `yellow_codes_bottom == row-reverse(yellow_codes_top)`. But deriving it via a reverse-Slice materializes a NEW [1,2,10,1] intermediate (+20 B memory) to save 20 params minus ~4 index params ⇒ **net +4 (977→981, WORSE)**. This is why the affine worker left both.

## ng_safety (live best)

`verdict: WARN`, 0 failures, 1 warning `WARN_X86_QUANTIZED_OP_FAMILY` (QLinearConv). Inherent to the render floor; grader-confirmed (graph is live on the leaderboard). A free-output alternative would avoid it — but free-output is structurally infeasible (above).

## Remaining hypothesis (only one, and it is not the prompt's route)

A sub-977 win, if it exists, is NOT a free-output/prepad-dissolve route (that component is gone and the free-output route is infeasible). It would require a provably-correct non-Where 2-channel feature constructor OR a source-derived weight/bias recode that deletes a charged scalar/vector while passing visible+truth — i.e. a `rung−epsilon` micro-shave, explicitly out of scope for this false-floor prompt and already picked over by the affine-recode batch (which extracted the last easy 3 B, 980→977). No structural 2.33× floor route exists.

## PASS-2 (2026-07-11): free-output route BUILT, gated, and priced — empirical WALL

Forced second pass built the prompt's literal suggested route ("dissolve into Pad/Where/Equal-as-output or color-index") from scratch and priced it:

- Builder: `prototypes/build_task156_freeoutput_colorindex.py` → `prototypes/task156_freeoutput_colorindex.onnx`
- sha256: `99ac3a1f449e0108e4d700e786fef25918e8fbfeabefaef936dca2baa6afeee4`
- Assembly: `Slice` yellow → erosion `Conv`(cross5,pad1) → `interior=Greater(er,4.5)` → `border=yb∧¬interior` → row-block split (`ReduceSum`/`CumSum`/`Equal`) + area compare (`Less`) → color-index plane `{0,1,2,4}` (INT32 `Where` chain) → `Equal` vs colorvals → one-hot `[1,10,10,10]` → `Cast` U8 → `Pad` → output `[1,10,30,30]` U8. NO QLinearConv.
- Numerically verified equal to the live graph on 5 rectangle configs (small-top / small-bottom / equal / wide) BEFORE gating.
- **Gates: `[visible] 265/265 (100.00%) PASS` (fully correct). `profile_real_cost --arcgen 266`: static=6311 profile=6311 (memory 6256 + params 55; 265 real inputs).**
- **`ng_graph_cost_diff` vs live: NET COST DELTA +5320 → candidate is +6.5× MORE EXPENSIVE (regression).** params −201 (no 180-weight conv / no bias / no codes) but memory **+5521 B**.
- Full battery (truth/ab/battery) deliberately NOT run — gate hygiene forbids `--full` on a candidate that loses on cost (a `--full` on a 6.5× loser is pure burn).

WHY it loses (the forced-component argument, now empirical): the live `QLinearConv` fuses four things into ONE op whose U8 output is the *uncharged* graph output — (a) quantized-feature read, (b) 3×3 border/interior detection, (c) 10-color-channel expansion, (d) 30×30 spatial placement. Any decomposition MUST materialize at least one of these as a charged intermediate: the color-channel one-hot (`[1,10,10,10]` ≥1000 B, here ×2 for BOOL+U8), an F32 `Conv` output over 30×30×10 (≥36,000 B), or a 30×30 canvas (≥9000 B). Even a maximally-lean free-output render floors at ~2000 B (slice 400 + erosion 400 + onehot 1000+) ≫ 977. The QLinearConv fusion IS the floor.

**Verdict: WALL.** Cheapest correct formulation = the live 977 QLinearConv (already team-best/shipped). No strict win below 977 exists by any route: the named FLOAT16 prepad is already dissolved (DROP premise), the free-output alternative is empirically 6.5× costlier, and every charged byte of 977 is forced.
