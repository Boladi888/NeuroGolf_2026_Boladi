# Prompter Prompt Template

Use this for every worker prompt. Keep the final prompt under ~2 KB, but do not omit the baseline, staged-pending status,
dead ends, gates, or the three sizing lines (Prize / forced-floor / fast-probe — below). Low-context workers should be
able to start from the prompt alone AND know the size of the prize before they build.

**Sizing first — the anti-tie-shave rule.** Anchor the target to a **forced-component floor estimate** (`floor_est` — the
bytes every correct algorithm must pay), NOT to `rung − ε`. `gap = team_best / floor_est`; `EV = ln(team_best / target)`
in points. Class it: **STRUCTURAL** (gap ≥ ~2×, EV ≥ 0.30 — the default), **SWEEP** (one lever over ≥3 tasks), or
**PEEPHOLE** (EV < 0.20 — backlog only, cap 5 per 40-wave, never a 1-byte "strict-below the staged rung" target). Reject
EV < 0.15 unless it is a SWEEP member. Full doctrine: `README.md §2`.

## Preflight Checklist

- Task is eligible regardless of `assigned_to` in `NG_OnnxlyFans_TaskStatus.xlsx`; `assigned` is only an older fallback
  alias and all assignment/current-scorer fields are context only.
- Baseline phrase is exact: ``workbook+manifest `taskNNN` = COST``.
- Manifest cost matches workbook cost before writing.
- `current_leaderboard_state.json` and `prompter_batch_preflight.py` were checked for recent rejected, hidden-failed,
  stale, unsafe, or not-submitted tasks.
- `Team_Submission/Staged_Boladi/taskNNN.json` was checked. If present, the staged cost is the CEILING (must beat it, do
  not rebuild/re-vet it) — but the target is still `floor_est` below it, not an epsilon under the staged rung.
- Any floor/wall audit evidence was checked in `Task_by_Task/taskNNN.md`, staged sidecars, and recent orchestrator notes.
  Floor claims are treated as prior failed formulations, not final exclusions. If selected, the prompt names the old claim
  and the different formulation that could break through it.
- Prompt starts at `Onnx_Advanced/INDEX.md`; do not cite moving Onnx_Advanced file paths.
- Prompt includes this exact guard phrase:
  ```text
  Do not edit/stage into `K-NeuroGolf-OnnxlyFans`
  ```
- Prompt says `DO NOT submit`; captain ships.
- Gates are class-aware, not "run everything": Path A byte-equal work final-proves with `ab --n 8000`; Path B exact rewrites
  final-prove with truth/fresh-window policy; fitted/approx/grader-gap work is `PROBE_READY`, not `LOCAL_GOLD`, unless
  captain approves x86/Kaggle proof. All packets still include `score_v2`, full gate-5 profile `--arcgen` (full 266-input
  set), and `python tools/ng_safety.py model --task NNN --cand C.onnx --json`. Bare "profile" is not enough.
- Staging path uses `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<desc>_cost<C>/taskNNN.onnx`.
- Target is anchored to `floor_est` (the forced-component floor), strictly below the effective best (staged cost if
  present, otherwise workbook+manifest cost) — NOT `rung − ε`. `EV = ln(team_best/target)` is computed and ≥ 0.15 (or the
  prompt is a SWEEP member). No 1-byte tie-shave targets; PEEPHOLE class capped at 5 per 40-wave.
- Prompt carries the three sizing lines: a **Prize** line (EV, floor_est, gap, class), a **forced-floor argument** (what
  every correct algorithm must pay), and a **fast go/no-go probe** with a kill rule.
- Prompt names at least one concrete lever and at least one concrete dead end so the worker does not rediscover old work.
- Prompt requires the worker packet fields: `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`,
  `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`,
  `safety_results`, `x86_required`, `kaggle_probe_required`, and `status` (`LOCAL_GOLD`, `PROBE_READY`,
  `RISK_EXPERIMENT`, or `REJECT_LOCAL`).

## Template

```md
# taskNNN - <specific lever> (<team_best> -> <target = floor_est>, EV +<X> pts)

START at `Onnx_Advanced/INDEX.md`; route to <technique/trap concepts>. workbook+manifest `taskNNN` = COST. DO NOT
submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders.

Rule: <one or two sentences describing the ARC rule/output contract>.

Baseline: <owner/current scorer and what the current graph does>. If `Staged_Boladi/taskNNN.json` exists: `Staged_Boladi`
already has <rung>@STAGED pending; do NOT rebuild/re-vet it — target below `floor_est`, not an epsilon under STAGED.

Prize: EV +<X> pts (team_best <B> -> floor_est ~<F>, gap <G>x). Class <STRUCTURAL|SWEEP|PEEPHOLE>. For STRUCTURAL: this is
a from-scratch rewrite to the forced floor, NOT a shave of the current graph.

Forced floor: <what every correct algorithm must pay — the forced F32 marginal read / output-canvas bytes / irreducible
operand>. Aim here; the current rung is a formulation artifact above it.

Direction: <one concrete attack surface: largest charged tensor/op + cheaper primitive>. Use <Onnx_Advanced concept
numbers/names if known>. Preserve <critical correctness invariant>.

Fast probe (BEFORE the full build): <cheapest test that proves the lever can reach floor_est — price only the forced
tensor / falsify the cheap formulation on 3000 fresh ARC-GEN pairs (0% kills it) / fire the cheapest rung>. Kill rule: if
it prices above <T> or falsifies on fresh ARC-GEN, stop and report the dead formulation — do not grind.

Dead ends: Do not retry <known failed/worse routes, profile fictions, hidden-bad ideas, validator-banned ops>. If this
is a documented floor-breaker, state the prior floor claim as "tried path" and name the new path.

Recon: `extract_task.py NNN`; `The_400_Tasks/Task_by_Task/taskNNN.md`; task JSON/generator if needed;
`inspect_graph.py taskNNN`; `largest_intermediate.py --top 20`; staged sidecars if present; BoBrain provider
`context/consistency/search` if it can sharpen the lever.

Gates: first classify `candidate_class`.
- Path A byte-equal / inherited: develop with `ng_gate visible` + `ng_gate ab --n 2000 --early-exit`; final proof is
  `ng_gate ab --task NNN --cand C.onnx --n 8000` against a recorded parent hash.
- Path B exact rewrite: develop with `ng_gate visible` + `ng_gate truth --n 5000`; final proof is `truth --n 8000` or explicit
  fresh-window policy. Do NOT claim `LOCAL_GOLD` unless task-specific exactness and required x86/fresh evidence are in the packet.
- Fitted/approx/grader-gap/risky-ISA: mark `PROBE_READY` unless captain approves x86/Kaggle proof.
Always run `score_v2`; full gate-5 profile `--arcgen` (full 266-input set); and
`python tools/ng_safety.py model --task NNN --cand C.onnx --json`. Add task-specific adversarial probes if rule logic changes.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<desc>_cost<C>/taskNNN.onnx` plus manifest with cost,
profile_cost, sha256, candidate class, parent source/hash, gate results, safety JSON, `x86_required`, `kaggle_probe_required`,
status, and ladder/floor notes.
```

## Short Floor-Break Audit Variant (PEEPHOLE / low-EV — use sparingly; counts against the 5-per-wave cap)

Use ONLY when a task is genuinely near its forced floor (gap < ~1.5×) and one distinct untried formulation is worth a
capped check. This is the low-EV variant, NOT the default — if the task has a real structural gap, use the full template
above and target `floor_est` instead.

```md
# taskNNN - short floor-break audit (BEST -> <TARGET>, EV +<X> pts, PEEPHOLE)

START at `Onnx_Advanced/INDEX.md`. workbook+manifest `taskNNN` = BEST. DO NOT submit. Do not edit/stage into
`K-NeuroGolf-OnnxlyFans`.

Documented floor claim: <specific proof and source>. Treat this as prior-path evidence, not truth. Cap around 30 minutes.
Try only <one distinct formulation/lever>. If it does not price below BEST, stop and report the exact failed idea; do not
stage a tie.

Gates: <same class-aware gates as above, including `score_v2`, full gate-5 profile `--arcgen`, and `ng_safety.py model`>.

Stage only if <BEST: `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<desc>_cost<C>/taskNNN.onnx` plus manifest.
```
