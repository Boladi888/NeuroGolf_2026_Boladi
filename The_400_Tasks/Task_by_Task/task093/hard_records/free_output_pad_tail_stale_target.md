# task093 — FreeOutputPadTail stale-target DROP (2026-07-10)

- Live/team-best bar: **1715** = memory 1474 + params 241.
- Live artifact: `Team_Best_Onnx/task093.onnx`.
- Live SHA-256: `537c5e31e24eecf7441ec291454132816629199b3c8868398722d2ec40a3a4ee`.
- Authoritative source: `tools/ng_current_bar.py --task 93 --json`; manifest, workbook, and Team_Best all agree and report no stale task093 source. `Team_Submission/Staged_Boladi` has no task093 JSON/ONNX or log row.
- Attempted route: delete the prompt's old F16 `[1,10,14,14]` pre-pad carrier and replace it with one predicate/index plane plus a final `Equal`/`Where` output tail.

## Exact failed pre-build gates

1. **Charged-component presence gate: FAIL.** The 1715 graph contains no `Pad`, no `Where`, no 14x14 decoded carrier, and no 900-byte canvas. It ends with `Einsum(...)->output [1,10,30,30]`. The advertised 3920-byte carrier was already removed before this prompt.
2. **Fast price gate: FAIL (>490).** A literal static one-channel predicate/index at the required 30x30 ABI costs at least `30*30*1 = 900` charged bytes as BOOL/U8 before producer tensors or params. A 14x14 U8 index followed by Pad then Equal charges `196 + 900 = 1096` bytes; Equal before Pad charges a `10*14*14 = 1960`-byte BOOL one-hot. Therefore this representation family cannot meet the requested ~490 target.

Per the prompt's 2026-07-10 workflow update, a missing charged target or priced wall is an immediate **DROP**. No candidate was built or staged, and no full truth/A/B gate was burned on a non-win.

## Refreshed reference proof

- `python -c "from tools._score_v2 import score_network_v2; print(score_network_v2(r'Team_Best_Onnx/task093.onnx'))"` -> `(0, 1474, 241)`.
- `python tools/agent_helpers/profile_real_cost.py --task 93 --cand Team_Best_Onnx/task093.onnx --arcgen 266` -> `task093 task093.onnx: static=1715 profile=1715 (memory 1474 + params 241; 265 real inputs)`.
- `python tools/ng_gate.py visible --task 93 --cand Team_Best_Onnx/task093.onnx` -> `[visible] 265/265 (100.00%)  PASS`.
- `python tools/ng_safety.py model --task 93 --cand Team_Best_Onnx/task093.onnx --json` -> `PASS`, 0 failures, 0 warnings.
- Incorporated manifest proof (not rerun): truth `200000/200000 (100.000%)`; A/B `match 200265/200265 | divergences=0 | byte_equal=True`.

## Prior boundary evidence

- `task093_v2_signed14.onnx`: cost 1405, but official visible `0/265` because a 14x14 graph output violates the required 30x30 ABI.
- `task093_v3_signed14_pad30`: cost 5334 because its pre-Pad F16 one-hot costs 3920 bytes.
- `task093_aud_orientation_from_topmax.onnx`: cost/profile 1668 and visible 265/265, but truth `199997/200000`; raw count-derived orientation is not exact.
- Full-input F16 count fold: cost 19229 because `Cast(input)->F16` materializes an 18,000-byte frame.

## One remaining hypothesis (unbuilt, ungated)

Try a fractional gray tag inside the existing 1668 `orientation_from_topmax` structure: set `nz_ch[5] = 17/16`, reuse `top_count30`, and classify vertical with `ReduceMax(top_count30) > 117/16`. Generator structure bounds a horizontal top-half column to at most `7 + 3/16`, while a vertical full-height gray bar gives `7 + 7/16`; the active render counts move by less than the current half-integer decision margin. Expected structure remains 1668 (47 bytes below live). This is only a possible rung shave, not the requested ~490 false-floor route; it still needs visible, profile, fresh truth/A/B, graph-cost diff, and safety before any staging claim.
