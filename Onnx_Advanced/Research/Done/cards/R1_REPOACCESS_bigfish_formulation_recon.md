# Research Card R1 (REPO-ACCESS) — Big-fish formulation-recon worklist

**Type**: repo-access research/analysis (NOT an internet-only card; NOT a single-task cost-drive). Run inside the repo with full tool access. **Deliverable**: a ranked worklist written to `Logs/research/R1_bigfish_formulation_recon.md` + a machine-readable `Logs/research/R1_bigfish_worklist.csv`. Do NOT stage candidates or submit — this card produces the *decode + plan* that per-task workers then execute.

## Why this is high-value

The current frontier is ~100 "big-fish" tasks (LIVE cost ≥10k, boladi team-best per `task_state.csv` `boladi_workable=Y`). Per-task workers currently each spend their first ~20 min re-decoding the rule and re-discovering the cost structure from scratch — and 3/20 last batch were misled by wrong family labels (traps §FAMILY-LABEL). This card front-loads that work ONCE across the whole tier, so each per-task worker starts from a decoded rule + a recommended formulation + an estimated floor instead of a blank page. It is the force-multiplier for batches 17+.

## Read first
`Onnx_Advanced/WORKER_PROTOCOL.md`, `INDEX.md`, `techniques.md` (esp. the free-output catalog #67/#78/#81, einsum-final #68/#89, placer/block #80/#84, flood #88, decode #82/#90), `traps.md` §FAMILY-LABEL + §CLASS-CHECK + floor doctrine.

## Scope
Every task with `boladi_workable=Y` AND `current_cost >= 10000` in `The_400_Tasks/task_state.csv` (~100 tasks; start with the highest-cost 40, the batch-17 set 264/365/349/044/046/253/163/390/019/074/243/289/278/310/101/371/063/169/059/183 first). Skip tasks already at a proven floor (live_costs note contains floor/closed/proven) and the legacy whales already owned by the deferred-tail compaction program (taskNNN.md ≥20 KB) — list those as "compaction-owned, skip."

## Method (per task, ~5–8 min each — this is breadth, not depth)
1. **Decode the rule from the generator, not the label**: read `external/ARC-GEN/tasks/task_*.py` (the generator `context_pack --task N` names) + 15–20 official pairs. One-sentence rule. Flag if the DeepSeek `family`/`transformations` label is a misnomer.
2. **Profile the LIVE cost structure**: load the LIVE graph (from `submissions/submission.zip::taskNNN.onnx`), list the top 3 charged tensors (shape×dtype×bytes) and the param tensors. What dominates the 10–35k?
3. **Map to a formulation**: name the cheapest library formulation that fits the decoded rule (free-output tail? einsum-final separable contraction? block-upsampler ConvTranspose? flood? linear-map per M3?) and a rough floor estimate (skeleton bytes).
4. **EV rank**: estimate point-gain = `25 - ln(floor_est)` minus current points; flag confidence (decoded-clean / uncertain).

## Output schema (`R1_bigfish_worklist.csv`)
`task,current_cost,generator_id,decoded_rule_1liner,label_misnomer(Y/N),dominant_cost_tensor,recommended_formulation,floor_estimate,est_point_gain,confidence,notes`

## Verification / discipline
- Decode from generator source — every `label_misnomer=Y` must cite the generator behaviour that contradicts the label.
- Floor estimates are skeleton-level (mandatory-tensor bytes), explicitly NOT built/gated — mark them ESTIMATE.
- Do not stage or submit. The per-task worker (or captain) consumes this worklist.
- End the prose report with: the 10 highest-EV tasks, any cross-task pattern (N tasks share one formulation = a batch-transfer opportunity), and any newly-suspected family-label misnomers to add to traps §FAMILY-LABEL.
