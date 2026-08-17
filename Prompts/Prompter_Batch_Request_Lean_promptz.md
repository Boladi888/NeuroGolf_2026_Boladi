You are the Prompter for NeuroGolf. Follow `Agenters/Prompter/README.md` + `Agenters/Prompter/PROMPT_TEMPLATE.md`.

## Read this first — why the last batches underperformed
The recent waves degenerated into **tie-shaves**: the target was set an epsilon below the current rung instead of at the task's floor. Look at the batch 46/47 output — task245 `940 → strict-below 939` (**+0.001 pt**), task286 `21558 → 21210` (**+0.016 pt**), task260 `→ strict-below 866` (already staged). At `points = 25 − ln(cost)` those are invisible on the leaderboard, and when the current rung is a *false* floor the "strict-below rung" target actively **blocks its own breakthrough**. That is why we stay far from the leader and why the light at the end of the tunnel is invisible: the worker is aimed 1–2% below the rung, not at the 5–50× structural floor.

**This batch inverts that.** Size the prize in points first. Anchor every target to a forced-component floor ESTIMATE, not to rung−ε. Fill the wave with false-floor structural breaks; hard-cap epsilon-shaves.

## Output modes
Default fresh-wave mode:

Create the next two numeric prompt folders under:
`<REPO_ROOT>\The_400_Tasks\Task Prompts`
Determine the numbers from the existing numeric folders. First new folder = **Alpha lane, exactly 10 prompts** (the highest-EV, highest-confidence structural breaks — Alpha stages to the team). Second = **Beta lane, exactly 30 prompts** (the speculative false-floor portfolio + cross-task sweeps that feed Alpha). **40 unique tasks, EV-dense: the biggest prizes go in Alpha.**

Existing-folder maintenance mode:

If the user names existing folders, do **not** blindly create the next pair. Audit those folders in place. Move any "by hand" loose prompts into the requested numeric folder, archive prompts that have already run or whose target no longer beats the manifest/effective best, then repack the named folders by priority. Put the hardest/highest-EV structural prompts in the earliest folders and split lower-priority sweep/near-floor work across the later folders. Create extra prompts only when a better target has fresh task-local evidence and is not already active, already run, or current-LB hard-avoid.

## Sources of truth
- Workbook: `<TEAM_REPO>\NG_OnnxlyFans_TaskStatus.xlsx` — `Tasks` sheet, `cost` = TEAM-BEST. Do **not** filter by `assigned_to`; assignment/current-scorer fields are context only.
- Manifest: `...\Team_Submission\Backend\manifest.csv` — verify `target < team_best` before writing (kills the home-cost trap).
- Leaderboard: `...\Team_Submission\Backend\current_leaderboard_state.json` — parse the note; hidden-failed / rejected / stale / unsafe / not-submitted = hard avoid until a newer note reverses it.
- Per task: `staged_candidates/`, `Task_by_Task/taskNNN.md`, `Staged_Boladi/taskNNN.json` (staged = effective best; do not rebuild it — target strictly below it AND below the structural floor).

If workbook and manifest disagree, do not pretend strict preflight passed. Use the manifest plus the current leaderboard note as the live team-best for prompt triage, source-note the mismatch in the prompt, and report that strict preflight remains blocked until the workbook is refreshed. If the manifest/effective best is already <= target, archive/drop the prompt.

## Sizing — do this for EVERY candidate BEFORE selecting (the core change)
1. **floor_est** = the forced-component / information-theoretic floor: the bytes *every correct algorithm* must pay (a forced F32 marginal read, the output-canvas bytes, an irreducible operand, a deterministic-truth requirement). This is the ONLY true floor. A per-algorithm "no route found / already optimal / at its floor" claim is **not** a floor — it is a map of what was tried.
2. **gap = team_best / floor_est** — the false-floor multiple. Rank by this. The headline candidates are the **fattest gaps**, not the tasks that happen to have a named peephole.
3. **target = floor_est** (a from-scratch rewrite of the decoded rule), NOT rung−ε. If you can only argue partway, target the best-supported structural rung — still well below the current rung.
4. **EV = ln(team_best / target)** in points.
5. **class**:
   - **STRUCTURAL** — route-B from-scratch rewrite; gap ≥ ~2×; EV ≥ 0.30. **The wave is mostly these.**
   - **SWEEP** — one shared lever across N≥3 related tasks (free-output Pad→output, dtype-narrow family, color-index collapse). One worker, N tasks, method cost amortized. **Prefer a sweep over N separate thin prompts when headroom is thin** (`current_strategy.md §NEXT-STAGE #2`).
   - **PEEPHOLE** — route-A residual shave; gap < ~1.5×; EV < 0.20. Backlog filler only, **hard cap 5 of 40**, and only when the task is genuinely at its forced floor. **A "strict-below the staged rung" 1-byte target is not a valid prompt — cut it.**

**Reject EV < 0.15** unless it is a SWEEP member. If fewer than 40 tasks clear the bar, manufacture EV-dense slots with SWEEPs before falling back to the best remaining PEEPHOLEs — and state the honest EV distribution in the final note rather than padding silently.

Rank by **EV × confidence**. Confidence: HIGH = a proven transfer (a twin task already done this way) or a verified structural fact; MED = a plausible named lever; SPEC = a long-shot crack on a near-floor.

## Scout brief — hunt false floors, don't re-list them
Do not spawn agents for mechanical re-ranking, stale-target checks, folder packing, or re-stating facts that already exist in `taskNNN.md`, staged sidecars, hard records, or prototypes. Script those.

Use up to 3 compact scouting subagents only when new prompts require genuinely new levers or un-analyzed candidates. Give each a candidate slice. Require ONLY this table, one row per task, no prose:

`task | team_best | floor_est | gap | class | untried_structural_lever | fast_probe | evidence_source | target | EV_pts | confidence`

- Rank your slice by **gap descending**. `floor_est` must be a forced-component argument, not "what's been tried." If you cannot argue below the current rung, mark gap≈1, class=PEEPHOLE (it will likely be cut).
- **fast_probe** = the cheapest test that proves the lever can reach the target BEFORE a full build — e.g. "build only the forced tensor and price it", "falsify the cheap formulation on 3000 fresh ARC-GEN pairs (0% kills it)", "fire the cheapest rung — if it clears hidden the floor is below your spread." This is the worker's go/no-go in minutes.
- No active-prior-folder tasks. No current-LB hard avoids. **Compute EV only against team_best, never home/`live_costs` cost.**
- Scouts do NOT write final prompts. You are the single writer: merge the tables, verify against workbook+manifest+leaderboard, then write.

## Every worker prompt (existing batch style, plus 3 mandatory lines)
Keep everything the current template requires — exact `workbook+manifest` baseline phrase, `Onnx_Advanced/INDEX.md` route, BoBrain/provider + library lookups, visual-first inspection commands, the documented floor/wall framed as "what has been tried", concrete dead ends so the worker doesn't repeat old work, gates incl. full `--arcgen` profile + `score_v2`, `DO NOT submit`, `Do not edit/stage into K-NeuroGolf-OnnxlyFans` — and ADD:
- **Prize:** `EV +<X> pts (team_best <B> → floor_est ~<F>, gap <G>×). Class <STRUCTURAL|SWEEP|PEEPHOLE>.` For STRUCTURAL say it plainly: *this is a from-scratch rewrite to the forced floor, NOT a shave of the current graph.*
- **Forced-floor argument:** name what every correct algorithm must pay, so the worker aims at the floor, not at rung−ε.
- **Fast probe / go-no-go:** the cheapest test that confirms the lever can reach the target, run BEFORE the full build, with a kill rule: *if it prices above <T> or falsifies on fresh ARC-GEN, stop and report the dead formulation — do not grind.*

## Verify

Fresh two-folder wave, from repo root — discovers the two newest numeric folders, then checks counts:

    $folders = Get-ChildItem 'The_400_Tasks\Task Prompts' -Directory | Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [int]$_.Name } | Select-Object -Last 2 -ExpandProperty Name
    python tools\agent_helpers\prompter_batch_preflight.py --new-folders $folders[0] $folders[1] --all-active --expect-counts 10 30 --expect-total 40 --strict

Existing-folder maintenance wave — use the exact folders and counts you just produced:

    python tools\agent_helpers\prompter_batch_preflight.py --new-folders 58 59 60 61 --all-active --expect-counts <c58> <c59> <c60> <c61> --expect-total <total> --require-assigned "" --json

Run `--strict` only when workbook and manifest costs agree. If strict fails solely because the workbook is stale while the prompt uses the current manifest/effective best, say that explicitly and include the stale task ids.

Final facts must be: folder numbers, exact counts, selected task ids, no duplicates, no overlap with the protected active folders, no current-LB hard-avoids, no assignee filter applied, workbook/manifest drift disclosed, and every active prompt carries a Prize + forced-floor + fast-probe line. For a fresh Alpha/Beta wave, also report median Alpha EV ≥ ~0.5 and ≤5 PEEPHOLE across the 40 prompts.

Finish with the folder numbers, the selected task ids with per-task EV and class, the class mix (how many STRUCTURAL / SWEEP / PEEPHOLE), and the exact verification result.
