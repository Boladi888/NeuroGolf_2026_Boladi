# task122 — compressed history (moved from task122.md 2026-06-07; task233 convention)

Live progression: parent 52517 → v1 10125 → v2-dilated 7586 → v5 6167 → v6 5700 → **v7 4816 (shipped→LIVE)** → **v8 4688 (floor, staged 2026-06-07)**. Current rule, floor proof, and the v5–v8 ladder live in `task122.md`.

## Superseded / failed attempt rows (moved out of the main ladder)

| Date | Candidate | Result | Cost | Notes |
|---|---|---|---|---|
| 2026-05-19 | build_task122_crop21.py | ✗ worse | 68804 (was 67289, Δ+1515) | Crop to 21×21 overhead exceeds savings — LIVE model intermediates already narrow. (Old 67289-era parent.) |
| 2026-05-19 | build_task122_v2_slicepad.py | ✗ worse | 68189 (was 67289, Δ+900) | 21-node Slice+Pad+Where shift rewrite; Where+pad overhead exceeds the Gather approach. (Old parent.) |
| 2026-06-04 | build_task122_v3_ifshift.py | ✗ invalid ORT | score None | `If`-branch shift looked smaller statically (6276) but ORT rejects the branch subgraph's reference to runtime `ch2_b`. Not staged. |
| 2026-06-04 | build_task122_v4_vecorient.py | ✗ invalid ORT | score None | `ReduceMax(BOOL)` rejected by ORT; also widened dynamic gather indices before the invalid load. |

## Stale f16 notes (contradicted; kept for the record)

- 2026-05-20 [OB_HUGE worker]: pattern=f16_whole_graph_with_f16_output delta=-14772 verify=266/266 cand=task122_f16drop_t_78a7c394_on_5835_30 — from the 5835-era graph; superseded.
- 2026-05-20 PR_HUGE worker (attempt failed): tried f16 conversion (robust onnxconverter_common + direct manual converter); NO CANDIDATE; f16 makes worse (+3228). With the current F32-input Conv decode, f16 is dead anyway (input forced F32; casting it is +18000 B) — see task122.md floor proof.
