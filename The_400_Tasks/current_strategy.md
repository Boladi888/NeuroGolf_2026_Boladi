# current_strategy.md - NeuroGolf retrospective strategy

**Last update**: 2026-07-15 cleanup retrospective. The Kaggle competition is finished (final owner-reported score: 7311.74 on 2026-06-21). This file is no longer a live queue log. It preserves the reusable doctrine a future NeuroGolf-like project should read first.

**Doc-hygiene rule (owner 2026-06-01)**: never hardcode a living leaderboard total in a living doc. It goes stale. Historical final scores and dated session records are fine; current values should come from the authoritative ledger or manifest for the active project.

**Archived dated log**: the old append-only body, including batch-9 through batch-40-era records and superseded queue notes, is archived at `_archive/current_strategy_session_log_2026-07-15.md`.

## §TEAM-SPLIT - Source Of Truth

During the competition, task ownership and bars came from the OnnxlyFans workbook plus team manifest, not stale local task docs. For a future project, preserve the same rule:

- Treat the current workbook/manifest as the authority for ownership, current best cost, and submit eligibility.
- Treat `Task_by_Task/taskNNN.md`, `hard_records/`, and any local generated status as tactical context that can lag.
- Before writing prompts, refresh from the authoritative workbook/manifest and call out any mismatch instead of silently reconciling it.
- Do not point workers at sibling repo payloads unless the prompt explicitly grants that scope.

## §NEXT-STAGE #2 - Prefer Sweeps When Headroom Is Thin

When many tasks have small remaining headroom, prefer one shared-lever sweep over many one-off prompts. Good sweep candidates are repeated structural patterns such as free-output Pad-to-output, dtype narrowing, color-index collapse, rank/basis factoring, or common gate/profile anomalies. A sweep earns its place only when the method cost is amortized across several related tasks.

## SHIP DECISION REFERENCE (§ship-decision)

The reusable decision model is cost reduction under correctness, not leaderboard folklore.

### Path A - Byte-Equal / Behavior-Equivalent Compression

Use this when the current graph is already behaviorally correct and the target is lower credited cost.

- Gate against the exact shipped parent, not the latest prototype label.
- Byte-equal A/B is the gold gate when the task is deterministic.
- If a candidate is not byte-equal, attribute every divergence with dual-truth before treating it as a bug or a win.
- A worse-cost sibling behind a byte-equal cheaper primary almost never earns a submit slot.

### Path B - From-Scratch Rule Rewrite

Use this when the live graph structure is the cost wall and a new algorithm can compute the same rule cheaper.

- Do not require byte-equality to the old graph if the old graph is merely one implementation of the rule.
- Gate against the true task rule: official visible pairs, verified generator/truth corpus, and class checks.
- For ambiguous or filtered-clean tasks, compare the candidate's misses to LIVE's misses over the same window.
- Submit-first was a competition tactic because Kaggle kept highest; do not cargo-cult it outside a similar scoring system.

## Mandatory Gates

For future active work, the gate order that survived the competition was:

1. Visible gate first. One official visible miss means certain zero unless the owner is deliberately running a distribution probe.
2. Class-check the LIVE graph with truth on the same corpus before judging a non-100% truth candidate.
3. A/B against the current shipped parent for byte-equal candidates.
4. Truth on both LIVE and candidate when divergences need attribution.
5. Score with the same cost oracle the grader uses, then profile real inputs; credited cost is `max(static, runtime)` per tensor.
6. Record durable floors and dead routes in `Task_by_Task/taskNNN/hard_records/`, not in prompt logs.

## Failure Modes F1-F6

### F1. Cross-Bundle ONNX Swap

Swapping a competitor graph into the bundle was unreliable. Mine public notebooks for techniques, not weights or entire graphs.

### F2. Novel Op Chain Myth

The risk was not novelty by itself. The discriminator was hidden correctness and credited cost. Novel chains failed when they computed the wrong rule; from-scratch rewrites succeeded when full-corpus behavior matched the rule.

### F3. Visible-Fit Shortcuts

Visible-fit shortcuts can work on solved tasks when visible-family coverage is high, but they are dangerous on unlock/rewrite tasks where hidden tests probe the missed rule. Use class checks and independent truth, not confidence language.

### F4. Stale `solved_status`

CSV status flags lagged reality. Ship outcomes and the authoritative manifest beat stale local status columns.

### F5. Oracle Independence

A synth oracle derived from the same candidate implementation only proves self-consistency. Independent generator/rule checks are required before calling a rewrite safe.

### F6. Different Rule, Different Cost

A structurally cleaner candidate that costs more is still worse if both solve hidden. Cost is the point once correctness is held fixed.

## §captain-session-2026-06-01 Meta-Lessons

- If the cheapest staged probe solves, the floor is below the spread; probe cheaper before retiring the task.
- A declared floor is usually a hypothesis. Verify grid dimensions, dtype/op constraints, and at least one cheaper formulation before promoting it to a wall.
- Path-B/op-swap safety is decided by full-corpus correctness, not whether the op vocabulary looks familiar.
- Every task except task000 was ultimately treated as scoring; the zero-weight family theory was rejected.
- The 2026-06-01 lesson is reusable, but the dated ship ledger is not first-read material; use the archive when provenance is needed.

## Named Doctrine Anchors

### §CLASS-CHECK

Run truth on LIVE first. If LIVE itself is below 100% on the same corpus, the task is ambiguous/filtered-clean and byte-equal-vs-LIVE or dual-truth attribution matters more than raw truth percent.

### §GRADER-GAP

Dynamic or exotic local costs can be fiction. The credited cost was the maximum of static and runtime-profiled tensor cost on real inputs. Always build a boring-static sibling when testing an exotic graph.

### §VISIBLE-FAIL

Official visible pairs are part of the effective hidden contract. One visible miss is a zero-score candidate unless it is intentionally being used as a probe.

### §DUAL-TRUTH

An A/B divergence is not automatically a candidate bug. Run truth on LIVE and candidate over the same window; if the candidate fixes LIVE misses, the divergence can be a win.

## Documentation Hierarchy

| Place | Durable role |
|---|---|
| `Onnx_Advanced/WORKER_PROTOCOL.md` | Active worker contract and gate order |
| `Onnx_Advanced/traps.md` | Reusable traps and doctrine anchors |
| `Onnx_Advanced/techniques.md` / catalog | Reusable technique library |
| `The_400_Tasks/Task_by_Task/taskNNN.md` | Current per-task rule, conclusion, floor, and don't-do notes |
| `The_400_Tasks/Task_by_Task/taskNNN/hard_records/` | Dense proof records for walls and dead routes |
| `The_400_Tasks/Task_by_Task/_archive/` | Provenance and long histories, not first-read guidance |
| `The_400_Tasks/Task Prompts/71/` | Preserved prompt-wave example after cleanup |
| `Logs/submissions_ledger.csv` | Historical submit outcomes |

## Reuse Checklist

For a future similar project:

1. Establish the authoritative manifest/workbook before writing prompts.
2. Keep prompt waves small and delete run-history after extracting durable lessons.
3. Promote a floor only when the forced components are named and falsifiable.
4. Keep one representative staged-candidate proof record, not every ladder rung.
5. Archive dated queue logs quickly; stale strategy prose is worse than no strategy prose.
