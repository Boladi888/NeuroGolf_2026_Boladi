# task368 - separable/free-output Einsum rank wall

Recorded: 2026-07-13 by `bo_ng_cdx_worker2`.
Verdict: DROP for the separable/free-output Einsum route. No ONNX candidate staged.

## Live bar and parent

- Manifest/live bar from `python tools/ng_current_bar.py --task 368 --json`: **4532**.
- Parent/live SHA-256: `902959d0299713acc6eb0438f80f8bbc4a6f5ac3a6b3280683c943bd7de99341`.
- Effective team-best bar used for reviewer-ready claims: **~1558** from task-local note `sharedcastequalfold_falsefloor.md`; reconfirmed with:
  - `python tools/agent_helpers/profile_real_cost.py --task 368 --cand Team_Best_Onnx/task368.onnx --arcgen 266`
  - Result: `task368 task368.onnx: static=None profile=1566 (memory 1486 + params 80; 269 real inputs)`
- Parent visible check:
  - `python tools/ng_gate.py visible --task 368 --cand Team_Best_Onnx/task368.onnx`
  - Result: `[visible] 765/765 (100.00%)  PASS`

## Attempted route

Prompt route: techniques #109/#119/#132, direct one-hot output through a low-rank row/column/channel selector bank in one terminal free-output `Einsum`.

Task rule remains template-stamp: each input has one non-0/non-5 template block and 2 or 3 gray-5 placeholder blocks. Output copies the template pattern into every gray placeholder. Template position, placeholder positions, and placeholder count are data-dependent.

## Failed gate

Fast probe gate before graph build: rank/term-count plus selector-only pricing.

Command:

```powershell
python The_400_Tasks/Task_by_Task/task368/prototypes/rank_probe_separable_einsum.py --fresh 400 --json-out The_400_Tasks/Task_by_Task/task368/prototypes/rank_probe_separable_einsum.json
```

Key result lines from `rank_probe_separable_einsum.json`:

- Official 765 pairs: nonzero output term count min/median/max = **4 / 12 / 18**.
- Fresh ARC-GEN 400 pairs: nonzero output term count min/median/max = **3 / 12 / 19**.
- Combined gray placeholder component histogram: **2 blocks: 540**, **3 blocks: 625**.
- F32 full-30 selector-only lower bound = `terms * (10 channel + 30 row + 30 col) * 4`.
- Official selector-only lower bound min/median/max = **1120 / 3360 / 5040**.
- Fresh selector-only lower bound min/median/max = **840 / 3360 / 5320**.
- 1151/1165 combined samples already have selector-only F32 bound >= 1558 before paying any template detector, anchor extraction, runtime selector construction, stamp placement, background channel, or decode costs.

## Why this route is dead

The #109/#119 route only wins when the one-hot output has a few separable rank-1 spatial terms and when K/R/C can ride as cheap operands of the terminal node. Here the necessary nonzero output masks commonly need about 12 terms and reach 18-19 on official/fresh samples. A model must provision the worst case, so the final selector operands alone cost around 5.0-5.3 KB at F32. That is already above both the effective ~1558 parent and the official 4532 manifest bar.

The #132 stamp-tiling condition also fails: placement count and placement coordinates are runtime-discovered from gray regions, not a small structural placement bank. Building those selectors would reintroduce the render planes that the free-output formulation was meant to remove.

## Remaining hypothesis

Only a small Conv/render micro-shave remains plausible: fuse one 100-byte mask plane or find a legal way to reduce the 3-channel re-encode without violating the `>0` one-hot decoder. Prior hard record proves 2-channel re-encode infeasible, so this is at most an epsilon route, not a separable-Einsum rewrite.
