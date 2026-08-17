# Prompter — playbook

**Your job:** pick any eligible tasks with real cost-reduction headroom, find the lever, and write worker
"drive-to-floor" prompts into the queue folders. You do **not** build ONNX (workers do) and you do **not** submit
(the Submit Captain ships). You find the cheapest-graph angle and hand the worker a ≤2 KB shell pointing at it.

The loop, every refill:

> **read the xlsx (team-best costs) → size each task's forced-component floor → rank eligible tasks by EV = ln(best/floor)
> × confidence → find the lever (NEVER accept "floored") → manifest-verify → write the prompt (Prize + forced-floor +
> fast-probe lines) → pack folder 1 to 10, then folder 2, …**

**Read first:** `Agenters/README.md` (the pipeline + the shared source of truth) ·
`Agenters/Prompter/PROMPT_TEMPLATE.md` (the prompt quality checklist) · `Onnx_Advanced/INDEX.md` (the router).
**You feed** the prompt folders → the Orchestrator picks them up for worker waves; the Submit Captain ships what lands.

NeuroGolf scores `points = 25 − ln(cost)` per task; the only lever is **cost reduction**. A task at 700 rewritten to
~90 is still **+2 pts** — "small cost" means "already had work done," NOT "floored."

Nibble rule: high-cost tiny deltas are not a good worker target when a structural rewrite route is untried — that is a
PEEPHOLE (§2), capped at 5 per 40-wave and never a 1-byte tie-shave. Near and below ~1,000 cost, small verified reductions
are useful and can be worth prompting only when the task is already at a proven forced-component floor.

Pre-build budget rule from external task packs: compute `current_cost - target_cost`, then name the tensor family that
can actually pay that reduction before drafting the prompt. If the needed cut is 20k and the idea only demotes a few
initializers or adds F16 wrapper casts, it is not the prompt.

---

## 1. Source of truth — read FIRST, every session

- **`K-NeuroGolf-OnnxlyFans/NG_OnnxlyFans_TaskStatus.xlsx`** → the **`Tasks` sheet**: current columns are `task /
  assigned_to / current_best_by / cost / points / sha12 / updated_utc` plus attempt metadata. Older exports used
  `assigned / owner`; accept those aliases only as fallback. The **`cost` is the TEAM-BEST** (the scorer = the cheaper of
  Boladi/Aditya per task). The `current_best_by`/`owner` column is who built that best. Do **not** route/filter by
  `assigned_to`; assignment and current-scorer columns are context only. The `Dashboard` sheet has the current LB + the contribution split;
  `Progress` has the LB history.
- **`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`** → the authoritative `task,creator,cost,…`, auto-rebuilt
  by `submit.py` from what's actually credited on Kaggle. Use it to **VERIFY a find before writing** (§3).
- **`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/current_leaderboard_state.json`** → current public score plus the
  latest captain note. Parse the `note` before selection; if it says a task was rejected, hidden-failed, stale, unsafe,
  or not submitted, treat that task as a hard avoid until a newer note reverses it.
- **BoBrain / NeuroGolf Provider** → use as compact recall after the workbook/manifest facts are loaded, not as score
  authority. From `Q:\AIz\BoBrain\Providers\NeuroGolf_Provider`: `python -B -m neurogolf_provider.cli context taskNNN`,
  `consistency taskNNN`, `search "concept"`, and `tasks-search "term"` are good worker-direction inputs.
- **`K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/taskNNN.json`** → pending staged team wins not yet in the
  workbook/manifest. Treat staged-pending as the effective team-best CEILING for prompt routing: do **not** rebuild/re-verify
  it; write only prompts that target below the staged rung (at `floor_est` per §2, not an epsilon under it), or skip. Never
  edit this team folder from Prompter.
- Read tasks one row at a time: `python tools/agent_helpers/show_task.py --task N` (never read `task_state.csv` whole — 1 MB).
- Run the cheap batch audit before and after packing:
  `python tools/agent_helpers/prompter_batch_preflight.py --new-folders 44 45 --all-active --expect-counts 10 20 --expect-total 30 --strict`
  (change folder numbers/counts for the wave). Use `--previous-active 2` for a literal "previous two batches" request, or
  `--all-active` when active prompt folders are messy and conservative exclusion is cheaper than debate.

### ⛔ The home-cost trap — the #1 way to waste a prompt
`task_state.csv current_cost` and generated task notes can be the **stale HOME (Boladi) version**, often **10–90× higher** than
the team-best because Aditya already holds a cheap version. **Never compute EV against the home cost.** Always use the
xlsx/manifest team-best. Symptom: an agent reports "task271 7,145 → 1,311, EV +1.70" — but the team-best is *already*
1,311 (Aditya's), so the real gain is **0**. Caught 5 false positives in one dig this way.

---

## 2. Choosing tasks — the EV routing

**Size the prize before you select. Anchor the target to the floor, not to the rung.** The #1 failure mode is the
tie-shave: `target = rung − ε` (e.g. task245 940→939 = **+0.001 pt**, task286 21558→21210 = **+0.016 pt**). At
`points = 25 − ln(cost)` those are invisible, and if the rung is a *false* floor the target blocks its own breakthrough.

1. **floor_est = the forced-component floor** — the bytes *every correct algorithm* must pay (a forced F32 marginal read,
   the output-canvas bytes, an irreducible operand, a deterministic-truth requirement). This is the ONLY true floor (§3).
   A "no route found / at its floor" claim is NOT a floor.
2. **gap = team_best / floor_est** — the false-floor multiple. **Rank candidates by gap** — the fattest gaps are the
   headline prizes, not the tasks that merely have a named peephole.
3. **target = floor_est**, NOT `rung − ε`. If you can only argue partway, target the best-supported structural rung —
   still well below the current rung. **`target ≈ team_best` = a tie = SKIP** (someone already holds it).
4. **EV = ln(team_best / target)** in points. **Reject EV < 0.15** unless the prompt is a SWEEP member.
5. **Class each candidate:**
   - **STRUCTURAL** — route-B from-scratch rewrite; gap ≥ ~2×, EV ≥ 0.30. **The wave is mostly these.**
   - **SWEEP** — one shared lever across ≥3 related tasks (free-output Pad→output, dtype-narrow, color-index collapse).
     One worker, N tasks, method cost amortized. Prefer a sweep over N thin prompts when headroom is thin
     (`current_strategy.md §NEXT-STAGE #2`).
   - **PEEPHOLE** — route-A residual; gap < ~1.5×, EV < 0.20. Backlog filler only, **cap 5 per 40-wave**, only when
     genuinely at the forced floor. A "strict-below the staged rung" 1-byte target is not a valid prompt.
6. **Rank by EV × confidence.** Confidence: HIGH = a proven transfer (a twin task already done that way) or a verified
   structural fact; MED = a plausible named lever; SPEC = a long-shot crack on a near-floor. Highest EV × confidence goes
   in the Alpha lane.
7. **Owner/assignee does NOT gate it.** If the worker can hit `target < team_best` — even beating someone else's
   *ungolfed batch* cost — it's a real team gain. The richest finds are often tasks the current scorer never structurally golfed
   (e.g. task318 739→120, task242 726→120).
8. **Don't pad the folders with no-lever/low-EV tasks.** Documented "floors" are not automatic excludes; they are maps of
   what has already failed. If the wave can't fill with EV ≥ 0.15 tasks, manufacture EV-dense slots with SWEEPs, then
   report the honest EV distribution in your final note — do not pad silently with ties.

---

## 3. ⛔ THE NEVER-FLOOR MANDATE (owner, 2026-06-10)

**NEVER accept "at its floor" / "unable to lower" / "already optimal."** That verdict has been **proven wrong many
times** — task152 (a 1,520 GridSample everyone called floored → 456 via a free-output Einsum), and a single sub-1k sweep
that dissolved ~30 "floored" tasks. A floor CLAIM tells you what's been **TRIED**, not what's impossible. For **every**
task, find the most promising untried lever and prompt it.

The **only** true floor is **forced-component / information-theoretic** — the dominant cost is something *every correct
algorithm* needs (a forced F32 marginal read, an Equal-tail output canvas, an irreducible operand, a deterministic-class
truth requirement). A **per-algorithm "no-lever" claim** (exhaustion, "no found route", a specific-primitive floor) falls
to a **different formulation** — propose the one the lesson didn't try.

**That forced-component floor is your `floor_est` — set `target = floor_est` and size EV against it (§2), never `rung − ε`.**
The gap between the current rung and `floor_est` is the whole prize; a target an epsilon under the rung throws it away.

**Verify before writing (kills the home-cost trap):** manifest-check the find — `python` against `manifest.csv`, confirm
`target < team_best`. If the team-best already ≤ target → SKIP.

---

## 4. The dissolve lens — the #1 lever-finding tool

Find an EXPENSIVE materialized op/tensor and the cheaper primitive that replaces it:

| expensive thing | dissolve to |
|---|---|
| GridSample / materialized canvas / F32 plane / 10-ch one-hot | **free-output separable Einsum** (`output = Einsum('bchw,hH,wW->bcHW', crop, M_row, M_col)` — the big product is in-kernel/free, NO canvas) |
| a built output canvas | **Where-as-output** (`Where(cond, color, input)` — input passthrough is free) / **Equal-tail** (single-channel U8 color-index canvas + `Equal(canvas, chids)` → BOOL output) |
| a dense `Conv[10,10,k,k]` that's 90% zeros | a **single-channel ones-kernel + passthrough**, or a **`Gather(idx[10])`** if the weight is a color permutation (read the weights: `inspect_graph.py`) |
| F32 carriers read only by `Max`/`Equal`/compares | **U8 via QLinearConv** (the #137 carrier-narrow) |
| a 10-channel plane manipulated spatially (flip/crop/rot) | **collapse the channel axis FIRST** (ArgMax → `[1,1,H,W]` U8 color-index), do the spatial op on that (~10× cut), Equal-tail |
| a lookup / template-memorization (stored answer tensors) | the **decoded generator rule** (read `external/ARC-GEN/tasks/task_*.py`) |
| a stored periodic template | **`Pad(mode='wrap')` @opset19** or a position-basis tile |
| a big intermediate | check if it's **separable (row⊗col)** and fold it |

The technique numbers (#67/#78/#81/#89/#108/#137 …) live in `Onnx_Advanced/` — **route through `Onnx_Advanced/INDEX.md`**.
The INDEX is the stable entry point; use the catalog/by-symptom routes and grep by concept instead of hardcoding moving paths.

---

## 5. The prompt quality bar

Use `Agenters/Prompter/PROMPT_TEMPLATE.md` for every prompt. Before a prompt enters the queue, it must pass this
preflight:

- No assignee filter was applied. `assigned_to`/`assigned` and `current_best_by`/`owner` were read as context only.
- Exact baseline phrase: ``workbook+manifest `taskNNN` = COST``.
- Workbook cost and manifest cost agree.
- `Team_Submission/Staged_Boladi/taskNNN.json` was checked. If present, the staged rung is the CEILING (must beat it, do
  not rebuild/re-vet it) — the target is still `floor_est` below it, not an epsilon under the staged rung.
- Floor/wall evidence in `Task_by_Task/taskNNN.md`, staged sidecars, and recent run notes was checked. Treat floor claims
  as "what was tried," not final truth. Only recent leaderboard/captain evidence of hidden-failed, rejected, stale
  regression, unsafe, or not-submitted probes is a hard avoid. If the prompt targets a documented floor, it must name the
  old proof and the new formulation that differs from it.
- The prompt starts at `Onnx_Advanced/INDEX.md`, says `DO NOT submit`, includes task-local `staged_candidates/`, and includes this
  exact guard phrase:
  ```text
  Do not edit/stage into `K-NeuroGolf-OnnxlyFans`
  ```
- Gates are class-aware. Do not ask workers to run both `truth --n 8000` and `ab --n 8000` by habit:
  Path A byte-equal/inherited candidates final-prove with A/B `--n 8000` against a recorded parent hash; Path B exact rewrites
  final-prove with truth/fresh-window policy plus task-specific exactness reasoning; fitted/approx/grader-gap candidates
  stay `PROBE_READY` unless x86/Kaggle/captain proof clears them. Every packet still includes `score_v2`, full gate-5
  profile `--arcgen` (full 266-input set), and `python tools/ng_safety.py model --task NNN --cand C.onnx --json`.
- Target is anchored to `floor_est` (the forced-component floor), strictly below the effective best — NOT `rung − ε`.
  `EV = ln(team_best/target)` ≥ 0.15 (or a SWEEP member). No 1-byte tie-shave; PEEPHOLE capped at 5 per 40-wave.
- Prompt carries the three sizing lines: **Prize** (EV, floor_est, gap, class), a **forced-floor argument**, and a
  **fast go/no-go probe** with a kill rule.
- Prompt requires the worker packet fields needed for local-gold staging:
  `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`,
  `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`,
  `kaggle_probe_required`, and `status` (`LOCAL_GOLD`, `PROBE_READY`, `RISK_EXPERIMENT`, or `REJECT_LOCAL`).
- Direction names a concrete lever and concrete dead ends. No generic "try to improve" prompts.
- Prompt stays under ~2 KB unless the extra bytes are buying necessary wall/staged context.
- `prompter_batch_preflight.py` passes for the selected folders/tasks: no duplicate ids, no overlap with the requested
  previous active folders, no hard avoid ids from the current leaderboard note/manual flags, no assignee filtering, and
  workbook/manifest costs agree.

## 6. The prompt format (≤2 KB shell — the work is the Direction block)

```
# taskNNN — <dissolve angle in a phrase> (~<team_best>→~<target>)

**Protocol FIRST:** START at `Onnx_Advanced/INDEX.md` (the stable router; it points to the protocol + the technique/trap
you need). **DO NOT submit — captain ships.**

**Why:** <assignment/current scorer context; why the current team-best is still breakable> ≈ <team_best> (re-measure: `extract_task.py NNN` +
`tools/_score_v2.score_network_v2`).

**Prize:** EV +<X> pts (team_best <B> → floor_est ~<F>, gap <G>×). Class <STRUCTURAL|SWEEP|PEEPHOLE>. For STRUCTURAL: a
from-scratch rewrite to the forced floor, NOT a shave of the current graph.

**Forced floor:** <what every correct algorithm must pay>. Aim here; `<team_best>` is a formulation artifact above it.

**Target:** ~<target = floor_est> (Path A byte-equal / Path B from-scratch rewrite).

## Direction
<the lever: name the expensive materialized tensor + the cheaper primitive that replaces it; cite the technique by CONCEPT
+ number; list the dead-ends/don't-dos from taskNNN.md so the worker doesn't re-walk them>

## Fast probe (before the full build)
<cheapest test that proves the lever can reach floor_est: price only the forced tensor / falsify the cheap formulation on
3000 fresh ARC-GEN pairs (0% kills it) / fire the cheapest rung>. **Kill rule:** if it prices above <T> or falsifies on
fresh ARC-GEN, stop and report the dead formulation — do not grind.

## Recon
`extract_task.py NNN` · `largest_intermediate.py --top 20` · history `taskNNN.md` (READ FIRST) ·
`inspect_graph.py taskNNN` (if weights decide it) · the generator source (if decode-first).

## Gates
First classify `candidate_class`, then run the one right heavy proof. Path A/inherited: `visible` + small `ab` during
development, final `ab --n 8000` against a recorded parent hash. Path B exact rewrite: `visible` + bounded truth during
development, final `truth --n 8000` or explicit fresh-window policy plus task-specific exactness evidence. Fitted/approx or
grader-gap risk stays `PROBE_READY` unless x86/Kaggle/captain proof clears it. Always include `score_v2`,
full gate-5 profile `--arcgen`, and `python tools/ng_safety.py model --task NNN --cand C.onnx --json`.

## Staging (captain ships — you NEVER submit)
`The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<desc>_cost<C>/taskNNN.onnx` + manifest (cost, profile_cost,
sha256, candidate class, parent source/hash, proof results, safety JSON, x86/Kaggle requirement, status). Report the ladder
+ floor verdict to the captain.
```

**Reference live sources, never hardcode moving numbers.** "team-best ≈ X (re-measure)" not "cost is X"; "EV +X" is fine
as a target. The worker should be able to re-derive every number with the cited command. Live examples: the current
 prompts in the numbered `The_400_Tasks/Task Prompts/` queue folders.

---

## 7. Packing the queue folders

- Use the numbered queue folders requested for the current wave, e.g. `The_400_Tasks/Task Prompts/1`, `/2`, ... .
  **Default house wave is 10 prompts in the first/Alpha folder and 20 prompts in the second/Beta folder unless the user
  asks for a different split.** Older waves may have used 20/20 or 10/10.
- **Fill the first requested folder to its requested count BEFORE putting any in the next folder**, then continue in order.
  Dense, NOT spread. EV-dense: earlier folders = the highest-EV prompts.
- Filename: `Prompt_<UTC-ISO>_TaskNNN_<ShortDesc>.md` (the `_TaskNNN_` token is how a re-pack script finds it).
- Before a refill, **archive ran/rejected prompts**: `mkdir _archive_ran_<date>` and `mv` the old `*/*.md` into it
  (they ran or were rejected — don't leave them mixed with fresh ones).
- The wave is maxed at the requested total (often 40) — nothing more goes in until workers drain it; keep the overflow
  finds in `The_400_Tasks/current_strategy.md` or the Boladi work plan as a ranked backlog.
- Before calling the wave done, run the batch preflight helper with `--strict` and record any intentional overrides in your
  final note. Mechanical exclusions are a script, not a model debate.

---

## 8. Gates to cite in every prompt

- **A DISSOLVE that reproduces LIVE's output exactly → byte-equal → gate `ab`:**
  develop with `ng_gate visible` + `ng_gate ab --task N --cand C.onnx --n 2000 --early-exit`; final proof is
  `ng_gate ab --task N --cand C.onnx --n 8000` against a known-good parent. Record `parent_source` and `parent_sha256`.
- **A RULE-REBUILD that replaces a lookup/template with the generator rule (may generalize beyond LIVE) → gate `truth`:**
  develop with `ng_gate visible` + `ng_gate truth --task N --cand C.onnx --n 5000`; final proof is `truth --n 8000` or explicit
  fresh-window policy. **Do not claim `LOCAL_GOLD` for non-byte-equal rewrites unless the packet explains why the
  construction is exact for that task class and clears any required x86/fresh evidence.**
- **Fitted schedules, visible-example repairs, approximations, or grader-gap graphs** stay `PROBE_READY` unless captain
  approves x86/Kaggle proof.
- Then always: **`score_v2`** (static prices exact) → **gate-5 real-input profile over the FULL arc-gen set** (`--arcgen`,
  266 inputs; the 4-input default undercounts dynamic-crop graphs — task029 6,796→19,436) → **`ng_safety.py model`**.
  Record `profile_cost` and safety verdict. Runtime integer `Div`/`Mod` warnings mean x86 proof is required before
  local-gold staging, even if official-board denominator probing is clean.

---

## 8. Assessing at scale — read first; fan out only to PRODUCE

**⛔ Cheap-first rule (owner, 2026-06-12 — a 4-agent EV-rank of 48 already-documented tasks burned ~460K tokens to
re-state facts that were already written down and largely re-confirm a triage the owner had just handed over).** The
EV-relevant facts almost always ALREADY EXIST: the `taskNNN.md` floor proofs + measured-alternative tables, the staged
sidecar costs, and the xlsx/manifest team-best. **Read them directly** (a script + a few targeted reads) before spawning
anything. A floor proof is a map for a breakthrough attempt, not a stop sign. Mechanical work — which prompt ran, who a
task is `assigned` to, baseline-vs-team-best drift, packing/repacking —
is a **SCRIPT, never an agent**.

**Fan out to PRODUCE, not to ASSESS.** A general-purpose (= main-model, expensive) agent fan-out is justified only when it
creates an artifact that doesn't already exist: inventing a NEW lever on a documented floor/breakthrough candidate, a task
with no prior floor analysis, or writing/repairing a batch of prompts. Do **not** spawn a fan-out to merely rank,
re-triage, or re-assess tasks that already carry documented floor analysis — that re-derives known facts at high cost.
Ranking by EV is `team_best` (xlsx) ÷ the proposed floor-break target, with the old documented floor used as evidence
about what not to repeat.

**When the owner hands you a triage/breakdown, ACT on it** (archive, edit, repack) — do not re-litigate it with a fan-out.

If a producing fan-out IS warranted (a real band of un-analyzed candidates), keep it tight — fewer, scoped agents, capped ask:
1. Extract the candidate list from the xlsx (eligible tasks in a cost band; no assignee filter), split into batches of ~10.
2. Spawn one agent per batch only if the owner allows fan-out (for example, "up to 3 subagents"); give each the
   **never-floor mandate + the dissolve lens**; tell it to **read `taskNNN.md` +
   cross-reference onnxadv via `INDEX.md`**, output one compact block per task
   (`task | cost | approach | LEVER | target | EV | confidence`), and **compute EV against the GIVEN team-best — not any
   home/`live_costs` number** (the trap).
3. Collect, rank by EV × confidence, **manifest-verify the top finds** (target < team-best), write prompts, pack folders.

---

## What NOT to do

- ❌ Build ONNX or submit to Kaggle — that's the worker / the Submit Captain. You write prompts only.
- ❌ Compute EV against the home/`live_costs` cost (the trap). Always the xlsx/manifest team-best.
- ❌ Accept any "floored" verdict at face value (§3). Find the lever.
- ❌ Anchor the target to `rung − ε` or stage a 1-byte tie-shave. Target `floor_est` (§2); reject EV < 0.15; PEEPHOLE
  capped at 5 per 40-wave. Every prompt carries a Prize + forced-floor + fast-probe line.
- ❌ Treat a documented floor as an exclusion. Exclude only if there is no named different formulation, or if current
  leaderboard/captain evidence says the task/probe is hidden-failed, rejected, stale regression, unsafe, or not submitted.
- ❌ Spawn an expensive agent fan-out to rank/assess tasks whose floor analysis already lives in `taskNNN.md` / staged
  sidecars (§8). Read them directly and compute EV inline. Fan out to PRODUCE (new levers, prompts), not to re-derive
  documented facts. Mechanical classification/packing is a script, not an agent.
- ❌ Hardcode numbers that move — cite the re-measure command.
- ❌ Reference specific onnxadv file paths — cite `INDEX.md` (files can move; the INDEX is stable).

## Related

- **Boladi work plan** (the live routing + standing rules + backlog): `The_400_Tasks/Task Prompts/_WORK_PLAN_Boladi_2026-06-10.md`.
- **Downstream — the Submit Captain** (what happens to the workers' candidates): `Agenters/Submit_Captain/README.md`
  (vet → batch-validate own-stack every ~5 → stage to the team).
