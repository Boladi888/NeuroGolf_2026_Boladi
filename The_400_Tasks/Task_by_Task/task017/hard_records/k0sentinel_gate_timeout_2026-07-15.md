# task017 — k0-sentinel shave stopped at heavy-gate timeout (2026-07-15)

## Live bar
- Authoritative bar: **3857**, owner Boladi, source `python tools/ng_current_bar.py --task 17 --json`.
- Current-best SHA256: `cda860ddb5c11c923fe1004ae71ee4ad2dbb54721c5fc424c5e77634e1c660a9`.
- Parent used for A/B: `The_400_Tasks/Task_by_Task/task017/staged_candidates/task017_rawmask_honly_cost3857/task017.onnx`.

## Attempted route
`prototypes/_task017_rawmask_k0sentinel_v1.py` builds `prototypes/_task017_rawmask_k0sentinel_v1.onnx`.

Change: fold the current raw-mask detector's p=0 exclusion into `Kfull[0] = -999`, deleting the `excl0`
initializer and the `And(valid, excl0) -> valid_ex` tensor. The rest of the raw signed-mask detector and Sho
residue Einsum tail are unchanged.

Cost evidence:
- Static score: `(0, 3528, 309)` => **3837**.
- `profile_real_cost --arcgen 266`: `static=3837 profile=3837 (memory 3528 + params 309; 266 real inputs)`.
- `ng_graph_cost_diff` vs the 3857 parent: `NET COST DELTA -20`; removed `excl0` (-10 params) and `valid_ex` (-10 memory).
- Candidate SHA256: `3b5f883169837020588b2b2dcad58a1d0b1c6a1695e0895d5ff0cc27f1b7cbc6`.

Gates run:
- `python tools/ng_gate.py visible --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx`
  - `[visible] 266/266 (100.00%)  PASS`
- `python tools/ng_gate.py truth --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx --n 2000`
  - `[truth] vs ARC-GEN ground truth: 1999/2000 (99.950%)`
- `python tools/ng_gate.py ab --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx --n 2000`
  - `[ab] vs submission.zip: match 2247/2266 | divergences=19 | byte_equal=False (first_div=616)`
  - This is the known stale-zip comparator, not the current 3857 bar.
- `python tools/ng_gate.py ab --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx --ref The_400_Tasks/Task_by_Task/task017/staged_candidates/task017_rawmask_honly_cost3857/task017.onnx --n 2000`
  - `[ab] vs explicit: match 2266/2266 | divergences=0 | byte_equal=True (first_div=None)`
- `python tools/ng_gate.py ab --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx --ref The_400_Tasks/Task_by_Task/task017/staged_candidates/task017_rawmask_honly_cost3857/task017.onnx --n 8000`
  - `[ng_gate] wall budget 900s exceeded for ab task017; exiting to avoid orphaned gate work`

Safety:
- `python tools/ng_safety.py model --task 17 --cand The_400_Tasks/Task_by_Task/task017/prototypes/_task017_rawmask_k0sentinel_v1.onnx --json`
- Verdict `PASS`, 0 failures, 0 warnings. Risk op flags inherited from parent: `Cast`, `Mod`.

## Why this is stopped
The candidate is a real local cost/profile shave and passed visible plus explicit-parent A/B over the 2000 window,
but the required Path-A heavy `ab --n 8000` did not complete inside `ng_gate`'s default shared wall budget.
Per worker protocol, do not relaunch this candidate with a larger `--max-seconds`, and do not stage it as
LOCAL_GOLD/reviewer-ready from this worker pass.

## Remaining hypothesis
A reviewer/captain with an appropriate gate window can rerun the explicit-parent `ab --n 8000`. If it proves
byte-equal, this should become a small Path-A inherited shave (`3857 -> 3837`) with no profile gap.
