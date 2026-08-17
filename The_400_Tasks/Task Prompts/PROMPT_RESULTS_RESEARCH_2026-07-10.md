# Prompt results research - 2026-07-10

Purpose: identify which late-stage prompt types are still producing useful NeuroGolf results, which are mostly proving walls, and how to adjust future Prompter waves.

Scope read:

- `Agenters/Prompter/README.md`
- `Agenters/Prompter/PROMPT_TEMPLATE.md`
- `The_400_Tasks/Task Prompts/README.md`
- current prompt folders `56`, `57`, `58`, `59`, and `By hand`
- archived prompt folders for recent ran batches, especially `46`, `47`, `52`, `53`, `55`, and the `ran_52-55_repack_20260709T214139Z` bundle
- closeout and monitor files under `Handoff/logs/`
- live/consumed captain packet JSONs where available

## Executive read

The late game is not out of gains, but the easy broad levers are mostly gone. The prompts that still get results are not generic floor-break prompts. They are prompts that name a concrete formulation change, prove why the current rung is a false floor, and give the worker a cheap kill test before a full build.

Best current pattern:

1. Start from team-best/workbook+manifest, not home cost.
2. Size the prize against a forced-component `floor_est`, not `rung - epsilon`.
3. Name one expensive tensor/op and the cheaper primitive that dissolves it.
4. Name the prior floor/wall claim as a tried path, then specify the different formulation.
5. Use class-aware gates and packet fields so a small real gain can move downstream cleanly.

Carry-forward rule from the retired Boladi work plan: a floor claim reports
what was tried, not what is impossible. Treat only forced-component or
information-theoretic arguments as real floors; otherwise require a different
formulation and a cheap kill test.

Worst current pattern:

1. Clone a sweep label over many tasks without per-task forced-floor evidence.
2. Ask a worker to "drive below current" or "prove floor" without a new route.
3. Burn high-reasoning workers on epsilon nibbles or stale EV from local/home costs.
4. Treat review/parse failures as prompt success. They are usually formatting/overload tax.

## Recent scorecard

These are local observed runner/captain outcomes, not Kaggle-bank guarantees.

| Batch / source | Prompt family | Outcome | Takeaway |
|---|---:|---:|---|
| Codex p52 | targeted structural graph surgery, U8/dtype/crop-at-source, detector collapse | 13 SHIP / 4 DROP | Strongest recent shape. Specific named rewrites still pay. |
| Codex2 p55 | mixed structural/rule-specific probes | 14 SHIP / 14 DROP, plus 2 review failures | Good throughput, but many wins are tiny. Needs stricter EV sorting. |
| Claude p56 | mostly `FreeOutputPadTailSweep` plus 2 specific probes | 2 SHIP / 8 DROP | Sweep still alive, but thin. Do not fill a whole lane with marginal cloned sweep tasks. |
| Codex2 p57 latest | `FreeOutputPadTailSweep` current run | as of 2026-07-10T15:42Z: 1 SHIP, 17 DROP, 8 review_parse_failed, 1 worker_failed, 1 worker_running | This family is over-mined unless each task has fresh, task-local evidence. |
| Claude p53 ceiling hunt | mostly `ZeroopsweepCeilingAudit` | 3 SHIP / 47 DROP across slices per closeout supplement | Good for bounded crack/wall audit, bad as a large default wave. |
| Codex p47 | older mixed near-floor prompts | 0 SHIP / 9 DROP / 2 parse failed | Stale or underspecified near-floor probes waste lane time. |
| Claude p46 | targeted subset of folder 46 | 3 SHIP / 3 DROP after exclusions | Narrowed, evidence-backed prompts can still work even on hard tails. |
| Claude p50 | 10-task targeted lane | 4 SHIP / 6 DROP | Strong individual prompts beat generic waves. Task394 and task201 were meaningful. |

Representative packet deltas:

| Task | Prompt title | Batch | Result | Delta |
|---|---|---:|---|---:|
| task052 | `ZeroopsweepCeilingAudit` | p53 | SHIP | cost 40 -> 30, +0.288 pts |
| task343 | `DynamicGatherFreeHashBank` | p52 | SHIP | cost 324 -> 284, +0.132 pts |
| task011 | `FreeOutputPadTailSweep` | p57 | SHIP | cost 318 -> 282, +0.120 pts |
| task396 | `U8CropAtSource` | p52 | SHIP | cost 3694 -> 3378, +0.089 pts |
| task050 | `FreeOutputPadTailSweep` | p56 | SHIP | cost 172 -> 164, +0.048 pts |
| task286 | `BitParallelTransitiveClosureFloor` | p52 | SHIP | cost 21558 -> 20630, +0.044 pts |
| task250 | `NibbleOrColorIndex` | p55 | SHIP | cost 2054 -> 1967, +0.043 pts |
| task361 | `SymmetryRenderWithoutGridSampleTable` | p52 | SHIP | cost 4839 -> 4645, +0.041 pts |
| task158 | `SparseMarkerBitRenderer` | p56 | SHIP | cost 23201 -> 22550, +0.029 pts |

The p56 closeout explicitly says its 2 packets were held, not staged, due grader flake. Treat this report as prompt-selection evidence, not final banked-score accounting.

## What is getting results

### 1. Concrete structural rewrites against a false floor

This is the best late-game shape. The prompt says the current graph is not the floor because a specific charged structure is avoidable.

Examples:

- `BitParallelTransitiveClosureFloor`
- `DynamicGatherFreeHashBank`
- `U8CropAtSource`
- `SymmetryRenderWithoutGridSampleTable`
- `FactorizedFinalGraphSurgeryTo`
- `CompactDetectorCastDelete`
- `MomentShapeFillCollapse`

Why it works:

- It gives the worker a route, not a wish.
- It usually has task-local evidence in `taskNNN.md`, `hard_records/`, `prototypes/`, or `staged_candidates/`.
- It targets the forced component floor, not a one-byte shave.
- It creates a buildable candidate that a reviewer can classify Path A/Path B.

Prompting rule:

Use this as the default high-value lane. A structural prompt should include:

```text
Prior floor claim: <old path> only proves that formulation.
New formulation: <specific replacement primitive>.
Forced floor: <component every correct solution still pays>.
Fast probe: price/falsify <specific subgraph>; kill if above <T>.
```

### 2. Dtype/carrier/source-crop reductions with exact byte-equal reference

These are smaller than a new formulation, but still productive when the live graph has obvious waste.

Examples:

- `U8CropAtSource` on task396, +0.089 pts
- `TernaryCarrierSweepMember` on task294, +0.018 pts
- `RingScanU8` on task346, small but real
- `RectangleScoreWithoutLogQLinear` on task355, tiny but valid

Why it works:

- Byte-equal compression is reviewer-friendly.
- It can be proven against the live manifest graph.
- It avoids hidden-set risk if the parent/source/hash are recorded.

Prompting rule:

Use this when the prompt can point to the live graph's charged F32/INT64/full-canvas carrier and name the legal narrower carrier. Require parent source/hash, `ab --full` or equivalent byte-equal proof, and `score_v2` plus full `--arcgen` profile.

### 3. Small, high-confidence sweep members

Sweeps still produce occasional wins, but the broad untapped sweep phase is over. The p56/p57 evidence says `FreeOutputPadTailSweep` is now thin:

- p56: 2/10 SHIP
- p57 latest: 1 SHIP among 28 non-todo statuses, with many drops/parse failures

The wins are still real:

- task050 172 -> 164, +0.048
- task011 318 -> 282, +0.120

But many sweep members are now proving true forced floors or stale targets. A cloned sweep prompt should not get a slot unless it has task-local evidence that the charged pad/output/canvas still exists.

Prompting rule:

Use sweeps as a small, filtered tranche:

- require a visible charged component in the current team-best graph
- require `floor_est` at least 0.15 pts below team-best, unless it is part of an explicit amortized sweep
- require a task-specific kill rule
- skip if task-local notes already show the pad/canvas was removed or the target is below forced bytes

### 4. Ceiling/zeroop audits as bounded crack tests

The p53 ceiling hunt looks bad by hit rate but still found a big relative win:

- task052 40 -> 30, +0.288 pts
- task211 96 -> 92, +0.043 pts

This is useful because near-perfect tasks are logarithmically sensitive. But it is also mostly wall-confirmation work.

Prompting rule:

Keep ceiling audits small and explicit. The prompt must say exactly what crack is being tested:

```text
This is a 30-minute crack audit, not an open-ended rebuild.
Try <one compression identity>. If it does not price below BEST, write the floor note and stop.
Never stage a tie.
```

Do not run 50 generic ceiling audits unless the goal is explicitly wall certification rather than point gain.

### 5. Rule/generator decode only when the generator clue is real

Rule-decode prompts in p55 produced some packets, but most were tiny:

- task105 `RuleDecodeNoLookup`: +0.001
- task107 `GeneratorBoundTinyPatch`: +0.0004
- task182 `OfficialGuardStructural`: +0.016

These are not bad if they are safe, but they should not displace bigger structural work. They are worth a slot when the prompt can cite a generator/source rule or a live template lookup that can be replaced cleanly.

Prompting rule:

Require the task JSON/generator route in recon, and require an early falsification test on fresh ARC-GEN pairs. If it is a visible-fit or grader-gap risk, packet status should be `PROBE_READY`, not `LOCAL_GOLD`.

## What is not getting good results

### 1. Whole-folder cloned sweep waves

Batch 57 is the caution sign. One `FreeOutputPadTailSweep` win exists, but most of the lane is drop/review-wait/parse-fail. This suggests the prompt family is being applied past the point where the per-task evidence supports it.

Fix:

Before a sweep prompt is written, run a cheap author-side audit:

- Does the live graph still pay the charged 10-channel pad/canvas/output tensor?
- Is the `floor_est` physically reachable after forced bytes?
- Has `taskNNN.md` or a staged sidecar already shown this exact route dead?
- Is the target below current team-best by more than a rounding/tie nibble?

If no, do not include it.

### 2. Floor-proof prompts without a different formulation

Prompts that ask a worker to verify a floor can still be useful, but only when bounded. Otherwise they spend a worker proving the same false floor again.

Fix:

Always include:

```text
Old floor claim: <path A>.
Why it may be false: <path A left charged component X>.
New path: <path B>.
Kill rule: if path B still costs >T or fails <test>, stop and report wall.
```

### 3. Epsilon targets and stale rung chasing

Several late packets are +0.0003 to +0.004 pts. Some are valid if they are safe queue filler, but they should not drive selection. A one-byte strict-below target is only useful after a task is already known to be at the forced floor.

Fix:

Selection should rank by `ln(team_best / floor_est) * confidence`. If the expected gain is below 0.15 pts and it is not a high-confidence sweep member or near-perfect crack audit, skip it.

### 4. Prompt bodies that overload reviewers

p57 has many `review_parse_failed` states. Some may be reviewer fragility, but the prompt wave shape likely contributes: many similar low-EV tasks with enough nuance to confuse reviewer classification.

Fix:

For near-floor or sweep prompts, shorten the worker deliverable. Make the verdict easy to parse:

```json
{
  "verdict": "SHIP|DROP|KEEP_WORKING",
  "candidate_class": "PathA_byte_equal|PathB_exact|PROBE_READY|REJECT_LOCAL",
  "cost": 0,
  "live_cost": 0,
  "status": "LOCAL_GOLD|PROBE_READY|RISK_EXPERIMENT|REJECT_LOCAL"
}
```

## Recommended next-wave mix

For a normal point-seeking wave near the end:

| Share | Prompt class | Selection bar |
|---:|---|---|
| 60-70% | STRUCTURAL false-floor rewrites | named new formulation, EV >= 0.15, preferably >= 0.30 |
| 15-25% | High-confidence sweep members | same lever across related tasks, task-local proof that charged component remains |
| 5-15% | Ceiling/zeroop crack audits | one exact crack, hard time cap, near-perfect logarithmic sensitivity |
| 0-5% | PEEPHOLE nibbles | only after forced floor is proven; never filler |

For a ceiling-hunt/perfect-25 wave:

- Use direct-output/perfect-score framing.
- Keep the audit list smaller unless the explicit goal is to certify walls.
- Do not let ceiling-hunt logic leak into floor-break waves.

For a floor-break wave:

- Start from forced floor estimate.
- Do not anchor to current rung minus epsilon.
- Prefer the tasks where the old floor proof names the expensive component, because that proof is a map.

## Prompt author checklist

Use this before writing each prompt:

1. Team-best verified from workbook+manifest.
2. `current_leaderboard_state.json` checked for hard-avoid notes.
3. `Team_Submission/Staged_Boladi/taskNNN.json` checked if present.
4. `Task_by_Task/taskNNN.md`, `hard_records/`, `prototypes/`, and `staged_candidates/` checked.
5. Current graph's expensive component named.
6. Forced floor named in bytes/shape terms.
7. Target set to `floor_est`, not a tie shave.
8. EV computed against team-best.
9. Prompt class chosen: STRUCTURAL, SWEEP, PEEPHOLE, or CEILING_AUDIT.
10. Fast probe and kill rule included.
11. Known dead ends named.
12. Gates are class-aware.
13. Packet fields required.
14. Guardrails included: `DO NOT submit` and `Do not edit/stage into K-NeuroGolf-OnnxlyFans`.

## Suggested prompt wording changes

Add this to structural prompts:

```text
This is a false-floor attack, not a strict-below-rung shave. The old floor claim only proves <old formulation>. The new formulation is <new formulation>. If the fast probe cannot price below <floor_est/target>, stop and report the priced wall.
```

Add this to sweep prompts:

```text
Before building, confirm the current team-best still pays <charged component>. If that component is already gone, DROP with the refreshed floor note. Do not spend the wave proving a stale sweep.
```

Add this to ceiling audits:

```text
Timebox: one crack only. Try <identity/factorization>. If it does not beat BEST, write the exact irreducible component list and stop. Never stage a tie.
```

Add this to reviewer-facing sections:

```text
Verdict must be parseable JSON plus a short note. If SHIP, include candidate_class, parent_source, parent_sha256, candidate_sha256, candidate_cost_static, candidate_cost_profile, proof_results, safety_results, x86_required, kaggle_probe_required, and status.
```

## Bottom line

The remaining gains are mostly in two places:

1. Specific false-floor rewrites where a documented proof exposes a charged component that a different formulation can avoid.
2. Carefully filtered sweep or ceiling cracks where the target is a real forced floor, not a stale rung.

The next prompting improvement is stricter selection, not more prompting volume. Do fewer cloned prompts, require fresher task-local evidence, and make every prompt carry a buildable lever plus a cheap stop condition.
