# task160 — (1) FreeOutputPadTail self-defeating: 3rd confirmation w/ measured tail prices; (2) a byte-equal staged win that BEAT LIVE sat unpromoted

**Source**: task160 (`bo_ng_cld_worker3`, 2026-07-11). Assigned the `FreeOutputPadTail`
false-floor sweep (team_best 1094 → floor_est ~500, "SWEEP", gap 2.19×). Verdict: **DROP/WALL**
for the ~500 attack; **surfaced a pre-existing staged 1089 that beats live 1094 by 5 B**.
Build/probe: `The_400_Tasks/Task_by_Task/task160/prototypes/freeout_tail_probe.py`;
floor proof: `.../task160/hard_records/forced_floor_wall.md`.

---

## Insight 1 — FreeOutputPadTail sweep is self-defeating (THIRD confirmation; measured tail prices)

Direct companion to `DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md`
(task254, free-output Einsum) and `DONE_2026-07-09T184440Z_gather_index_beats_free_output_for_permute_tile.md`
(task249, Gather). task160 is the same trap on a **third already-free-output family: a packed-row
`BitwiseAnd(rowcode30[1,1,30,1], class_col_bits[1,10,1,30]) → output` decoder** (the final AND writes
straight to the exempt `[1,10,30,30]` output).

- **The charged canvas the prompt targets is ALREADY GONE.** The Direction says "dissolve the FLOAT
  1×10×10×10 Conv/Concat prepad canvas (4000 B)". Neither live graph materializes it: the packed-row
  best (1089/1099) uses `[1,1,30,1]` row-codes; the current live 1094 (Sho) uses INT8 `[1,·,10,10]`
  QLinearConv planes. When the canvas is already dissolved, per the 2026-07-10 workflow rule the sweep
  is an automatic **DROP** — no build needed beyond a confirming probe.
- **Fast-probe measured prices (the "one crack", built + scored on real inputs):**
  - `tail_a` one-channel index → `Equal`-as-output @30×30: **static=profile=13513** (the `[1,10,30,30]`
    bool intermediate alone = 9000 B; the F32 `[1,1,30,30]` slice = 3600 B). Worse than the OLD 13991.
  - `tail_b` `[1,1,10,10]` index → `Equal` → `[1,3,10,10]` one-hot → `Pad`: **static=profile=1120** —
    already **above the current 1089 with ZERO transform logic** (F32 `[1,1,10,10]` slice 400 + bool
    `eq` 300 + INT8 `onehot3` 300 = 1100). The naive hand-estimate "10×10×3 tail ≈ 400" is WRONG: you
    pay the F32 input-slice AND both the bool-compare and its cast, not just the final one-hot.
- **Forced residual > 500 regardless.** The exempt `[1,10,30,30]` output from a per-row code forces a
  final broadcast operand spanning channel=10 × width=30 = **300 elements** (cheapest as a 300-param
  init; any intermediate form is ≥300 B). Plus `rowcode30`[1,1,30,1] uint32 =120 and `rows_f30`[1,1,30,1]
  F32 =120 (F32 forced by the F32 `input` Einsum operand). 300+120+120 = **540 > 500 before any detection**.
- **Rule of thumb (reinforce existing):** the EV `floor_est ≈ best/2.5` "SWEEP" heuristic does not
  survive a full one-hot output contract. `inspect_graph.py` the live graph FIRST; if the output already
  comes straight from an Einsum/Gather/BitwiseAnd (no charged `[1,C,H,W]` prepad), the canvas is already
  gone and PadTail/Equal-as-output REGRESSES. Empirical corroborator: Sho's strong from-scratch 2D-INT8
  route still priced 1094 > the packed-row 1089.

## Sorter note: by_symptom.md + techniques.md (free-output / #119 micro-floors) — extend the existing "FreeOutputPadTail self-defeating when best already free-output" verdict with a THIRD confirmed family (packed-row BitwiseAnd decoder, task160) and the measured tail prices 13513 / 1120. No new §number; add task160 to the exemplar list.

---

## Insight 2 (PROCESS) — a fully-gated staged win that BEAT LIVE sat unpromoted for 8 days

task160's local `staged_candidates/task160_objectconvnarrow4drows_cost1089` (sha `dac4e131…`, staged
2026-07-03) is a byte-equal −10 B structural shave of the **confirmed_public** 1099 packed-row (Boladi,
sha `f1dcb5ae…`). At 1089 it **beats the current live manifest 1094** (Sho, 2026-07-09) by 5 B.
Re-verified fresh 2026-07-11: visible 265/265, truth --full 200000/200000, ab --full byte_equal
200265/200265, profile static==profile==1089, ng_safety PASS (0/0). Yet **no `Staged_<creator>/task160.json`
exists in ANY captain queue** — the win never left the worker's task-local dir.

- **Failure mode:** `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/` is a WORKER scratch stage,
  NOT a captain ship queue. A worker "stages" a win and moves on; if nothing promotes it into
  `Team_Submission/Staged_<creator>/taskNNN.json`, it can beat live indefinitely and never ship. Here a
  later, independent, *costlier-than-1089* graph (Sho 1094) became live while the cheaper 1089 rotted.
- **Cheap detector:** the authoritative-bar reconciler already has the signal — `ng_current_bar.py`
  shows live 1094 while a task-local `staged_candidates/*_cost1089/manifest.json` records a gated 1089
  below it. A periodic sweep of `staged_candidates/*/manifest.json cost < ng_current_bar live_cost` (and
  no matching `Staged_*/taskNNN.json`) would flag every such orphaned win for promotion.

## Sorter note: .ai_instructions/playbooks/agent_feedback_rules.md — add a process rule: at batch close, reconcile every task-local staged_candidates/ cost against ng_current_bar live_cost; any gated candidate strictly below live with no Staged_<creator>/taskNNN.json is an UNPROMOTED WIN and must be handed to the captain, not left in the worker dir.
