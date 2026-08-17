# Agent Feedback Rules — durable process feedback for all AI workers

> **Migrated from .claude/memory/ on 2026-05-25** so that all AI agents (Codex, ChatGPT researchers, Claude Code sessions in any worktree, Hermes-spawned workers) can see the same rules. Prior to migration these were stored in Claude Code's auto-memory, which only Claude could read.

This is the canonical source for process/protocol feedback the user has given across sessions. **If you have an auto-memory system, do NOT save process feedback there — extend this file instead.** Memory is invisible to other agents and counts as keeping secrets from the project.

For technical traps (encoder, padding, lookup-family, verify gates, synth thresholds) see `Onnx_Advanced/traps.md`. For technical techniques (F16 placement, bool-first, crop, opt_safe blind spots) see `Onnx_Advanced/techniques.md`. For project paths and tooling references see `Onnx_Advanced/core.md`.

---

## 1. Action bias — terse positive ack = full authorization

When an agent offers a discrete piece of work in a question and the user gives a terse positive ack ("yes", "keep working", "ok do it", "go"), that is **explicit authorization to execute end-to-end** without asking sub-decisions.

**Origin (2026-05-12, PR #75)**: agent offered to "build a local cascade simulator before we decide?"; user said "Keep working"; correct behavior was to build + run + report in one batch.

**How to apply**:
- Default to action over verification on offered improvements.
- Reserve confirmation for genuinely risky/destructive operations (force-push, drop tables, rm -rf, sending external messages).
- Don't keep asking for sub-decisions after a green light.
- Auto Mode is the user's normal state; they don't want to micromanage.

## 2. Handoff hygiene at session close

When the user says "we're done", asks for handoff, or signals session close, execute the full close-out without asking step-by-step:

1. All tracked work pushed to GitHub (PR branch or master).
2. **Gitignored artifacts mirrored to the main worktree** at `<REPO_ROOT>/`. NeuroGolf gitignores `submissions/**` with explicit exceptions — anything new and useful in those dirs must be copied to the main worktree at the same path. Verify md5 after copy.
3. `CURRENT_STATE.md` updated with the session's outcome, output artifacts, "where next agent picks up."
4. `Logs/lessons.md` appended if any durable insight was generated.
5. PR description updated if it was a draft template — replace with actual outcome.
6. **Background processes stopped** — crons deleted (CronDelete), background bash terminated.
7. Final commit + push so origin matches local.
8. Final STATUS comment on the PR summarizing handoff state.

**Origin (2026-05-12, PR #75 close)**: user asked for "absolutely all the work in this worktree is either in the main local dir or on GitHub." Worktrees accumulate; the user routinely cleans them up; anything left only in a worktree's gitignored space dies with the worktree.

## 3. PR workflow — one PR per worker, owner batch-merges

One worker = one branch = one PR with many commits. Many commits accumulate in that PR over the lifetime of the worker. **Don't tell agents to merge their own PRs.** The owner reviews and batch-merges all PRs at session end.

**Origin (2026-05-12 NeuroGolf orchestrator sprint)**.

**How to apply**:
- In ASSIGN / CONTINUE / APPROVE_SUBMIT directives, never include "merge this PR" or "open a new PR per task".
- Each worker opens ONE milestone PR when they hit their first CANDIDATE_READY. Subsequent CANDIDATE_READY on additional tasks go into the SAME PR as new commits.
- If a worker is reassigned (NEW_TASK), they stay on the same branch + same PR.
- Milestone comments — "ORCH APPROVED: ..." / "Kaggle landed: +X LB" — go on the existing PR.

## 4. Don't burn worker context after a single failed attempt

When a worker's Kaggle submission fails (hidden-bad / regression), **pivot the worker on the SAME task** with a different technique before reassigning. Their LIVE-graph analysis, prototype debugging, adversarial test infrastructure is valuable and is wasted on REASSIGN.

**Origin (2026-05-12, worker2 on task080)**: had 50 min of LIVE-graph analysis when v5_bool_slices landed -11.79 LB. Orchestrator's reflex was REASSIGN to task153. Owner pushed back: "why not let worker2 keep working on task080?" Three math-equivalent paths (FP16, onnxsim+safe-opt, v4 int64) had not been tried yet — each provably safer than v5 because they preserve outputs.

**How to apply**:
- After a failed Kaggle submission, BEFORE reassigning, enumerate alternative approaches that haven't been tried.
- Prefer math-equivalent transformations (FP16, onnxsim, dedup, CSE, const_to_init) over algorithm rewrites — these can't be hidden-bad.
- REASSIGN only when **≥3 distinct techniques** have failed. A single hidden-bad outcome is one technique, not three.
- If you reassigned prematurely, supersede with a fresh REASSIGN to the original task; newest directive wins.

## 4b. Don't release worker context after a SUCCESSFUL attempt either — grind to the absolute floor

§4 says don't burn context after a *failure*. This says don't release it after a *success* either. **"Verified + improves LB" is NOT a stop condition.** When a worker returns a candidate the captain verifies and ships, the default is to send that worker **right back at the SAME task** with its context warm — try a different algorithm, decomposition, dtype, or structural angle — and keep grinding toward the absolute cost floor.

**Origin (2026-05-31, owner directive)**: "if there's even 5 bytes to be saved, I prefer to have that worker keep working, try different algorithms, and work the task to the absolute floor. I don't care if it takes weeks to get one task all the way down while ignoring others ... my goal is to win, not just 'do a good job'. So unless it is genuinely impossible that any other code could get a better score (either by rewrites, hidden-safe or whatever) I generally prefer to utilize the worker's context and keep working the problem." The owner WILL return to every task eventually for the last 0.02 LB regardless — the worker's already-warm context is the cheapest possible time to spend those weeks.

**How to apply**:
- After ship, re-task the SAME worker on the SAME task by default. Hand it the new floor target and at least one untried angle (different op family, different decomposition, a hidden-safe trick not yet attempted).
- **Release a worker from a task ONLY on proof-of-floor**: it is genuinely impossible for *any* code — rewrite, restructure, narrower dtype, different op family, sparse/hidden-safe trick, anything — to score lower. That bar was met for the 14 must-hold family (6+ independent agents + competitor-field scan + all-401 reversal scan + sparse-index loophole rejected at 3 gates). It is NOT met by "diminishing returns," "good enough," "only a few bytes left," or "better EV on another task."
- Even a **5-byte** improvement justifies keeping the worker on it. Bytes compound into the `25 − ln(memory+params)` curve and the owner is optimizing for the win, not throughput.
- No opportunity-cost / EV-per-hour pivots (consistent with the standing no-pivot rule). The worker stays until the floor is proven.

## 4c. Captain: SUBMIT FIRST, analyze after — pre-submit validation is wasted time

When a worker delivers a candidate, the captain's job is **swap it into `submission.zip` → submit to Kaggle → poll → THEN analyze**. Do **NOT** run `score_network_v2`, `verify_against_live`, generator-source analysis, or A/B byte-equal harnesses *before* submitting. The only pre-submit step is the mechanical swap itself.

**Why (the actual competition mechanics)**: submits are **unlimited** (the daily cap is never reached) AND Kaggle **keeps your highest score** (no revert needed). So a bad submit costs *exactly zero* — the LB just doesn't improve, and the prior best stands. That makes the real hidden-set LB delta the **authoritative oracle**, and it's free to obtain. Any local gate is a slower, weaker proxy for a result you can simply *measure* by submitting. Time spent gating before the submit is pure overhead, and it burns the captain's context.

**Origin (2026-05-31, owner directive)**: after the captain spent a turn each scoring + `verify_against_live` + reading the ARC-GEN generator source on task306 and task233 *before* submitting, the owner cut it off: *"Stop trying to validate these and just submit them. Submits are free and we never use them all. Your validation is a waste of time. Analyze after the submit so you have more data."* (task306 had already been a "just submit and see" the same session — the lesson generalized.)

**How to apply**:
- Worker delivers candidate → swap into `submission.zip` → `kaggle competitions submit` → poll for the score. No correctness/scoreability gating in between.
- **After** the LB lands, THEN analyze — now with *more* data (the real hidden result): if it improved, reconcile ledger/strategy/`task_state`+xlsx and push; if it didn't, NOW dig into *why* (`score_network_v2`/per-tensor breakdown/`verify_against_live`) to guide the next iteration. Re-arm the worker per §4b.
- Post-submit analysis is still valuable (it guides the next floor target) — it's just *rescheduled to after the submit*, never a blocker before it.
- **This overrides the "validate before you submit" framing in `traps.md` SHIP DECISION REFERENCE.** Those synth/byte-equal gates remain useful as *post-submit* explanatory tools and for deciding the next move, but they are NOT pre-submit ship gates. The LB is the gate.

## 5. Stay on task — no freelance work

Worker sessions must process only the cards the orchestrator assigned. **Do NOT freelance** — do NOT spot a polish-sweep / spread-scan opportunity and start applying it to additional tasks beyond the assigned card.

**Origin (2026-05-18)**: worker-5 finished assigned cards, then batch-applied polish-sweep hits (task283, task221, task252, ...) without filing cards. Owner halted with "did you abandon the kanban and decide to do stuff on your own?" Off-kanban work is hard to merge — no card → no reviewer routing → no canonical handoff.

**How to apply**:
- When assigned cards are done, STOP and report back.
- Do NOT start the next obvious thing.
- List opportunities in the session summary; let the orchestrator file fresh cards.
- Same rule for tooling/refactor temptations not on the original card.

## 6. Pool B may cold-start BACKLOG bank cards (2026-05-18 onward)

`tools/neurogolf_dependency_serializer.py` was changed on 2026-05-18:

```python
_BACKLOG_BANK_ALLOWED_WORKERS = (
    set(POOL_A_SHRINK_WORKERS)
    | set(POOL_A_DRAINING_WORKERS)
    | set(POOL_B_SOLVER_HUNT_WORKERS)   # added 2026-05-18
)
```

Pool B workers (5, 7, 8) can now claim `[BACKLOG single task]` cards without the misclaim guard stripping the assignee. The serializer no longer strips, and the playbook's cold-start instruction (step 3) works as documented.

**Why**: Hermes-paid LLM seats are off; manual /loop sessions are the only active candidate producers, so they need to claim BACKLOG cards directly. The Pool A foreman's `create_worker_from_bank` conversion path is dormant.

**How to apply**:
- Pick a task from `The_400_Tasks/Task Prompts/` folder, work it, stage to `staged_submits/`, report back.
- Per-task `.md` documentation still benefits everyone regardless.

**To revive strict Pool-A-only**: drop the `| set(POOL_B_SOLVER_HUNT_WORKERS)` term in `tools/neurogolf_dependency_serializer.py`. One-liner revert. Affected tests document the policy change explicitly and reference the same toggle.

## 7. Worker protocol summary (2-Claude-session orchestrated /loop)

The user (`Boladi888` on GitHub) runs the NeuroGolf project (`Boladi888/NeuroGolf_bo`), a Kaggle ARC-style competition where the deliverable is `submission.zip` containing 401 task ONNX models. Objective: maximize LB via `points_per_task = max(1, 25 - log_e(memory + params))` summed across tasks.

**Orchestrated /loop protocol** (drafted PR #74, first run PR #75):
- User spawns 2 Claude Code sessions from `origin/master`, each in its own `.claude/worktrees/<name>/`.
- One session is the orchestrator ("NG-Control"); the other(s) are workers.
- Workers poll their own PR every ~30 min for `ORCH:`-prefixed comments and execute one task at a time.
- Workers MUST NOT switch tasks autonomously, MUST NOT submit to Kaggle without an `APPROVED:` comment, and MUST post a STATUS comment every loop iteration.
- Workers read prompts from `The_400_Tasks/Task Prompts/`, stage results to `staged_submits/`, captain ships.

**Why**: Single-session optimization runs drifted (worker would "shop" tasks when one was hard, or submit untested candidates). The 2-session split with PR-comment audit trail enforces discipline. The −99.38 LB session from 2026-05-12 (in `Logs/lessons.md`) is the cautionary precedent.

## 8. Memory policy — do not use Claude Code's auto-memory for this project

Claude Code's `.claude/projects/.../memory/` is invisible to other AI agents (Codex, ChatGPT researchers, other Claude sessions). The user runs orchestrated multi-agent loops where Codex and ChatGPT-based workers must see the same rules as Claude.

**Rule**: do NOT save process feedback or technical feedback to Claude Code auto-memory for this project. Extend the appropriate project doc instead:

| Memory category | Project home |
|---|---|
| Process / worker protocol feedback | This file (`.ai_instructions/playbooks/agent_feedback_rules.md`) |
| Technical traps (encoder, lookup, verify, synth thresholds) | `Onnx_Advanced/traps.md` |
| Technical techniques (F16, bool-first, crop, opt_safe) | `Onnx_Advanced/techniques.md` |
| Project paths, key files, tooling | `Onnx_Advanced/core.md` |
| Per-task lessons | `The_400_Tasks/Task_by_Task/taskNNN.md` |
| Session lessons | `Logs/lessons.md` |

**Why**: The user has explicitly called auto-memory "secrets kept from the project." All durable knowledge must live where every AI agent in the loop can find it. This rule is the senior-most piece of feedback in this file — observe it whenever you would otherwise reach for memory.

---

## Migration manifest — what was in `.claude/memory/` and where it lives now

This table is the complete map of the 21 entries that were in `.claude/projects/C--Users-Qx-Documents-Py-cx-K-NeuroGolf/memory/` before 2026-05-25T15:30Z. Every entry has been migrated to a project file. The memory directory is now empty by policy (per §8 above).

| Original memory file | Type | Home in project |
|---|---|---|
| `project_neurogolf_worker_protocol.md` | project | This file §7 (Worker protocol summary) |
| `feedback_action_bias.md` | feedback | This file §1 (Action bias) |
| `feedback_handoff_hygiene.md` | feedback | This file §2 (Handoff hygiene) |
| `feedback_pr_workflow.md` | feedback | This file §3 (PR workflow) |
| `feedback_dont_burn_worker_context.md` | feedback | This file §4 (Don't burn worker context) |
| `feedback_stay_on_kanban.md` | feedback | This file §5 (Stay on kanban) |
| `feedback_backlog_misclaim_policy.md` | feedback | This file §6 (Pool B BACKLOG policy) |
| `reference_neurogolf_paths.md` | reference | `Onnx_Advanced/core.md` "Key paths" section |
| `project_neurogolf_spread_knowledge.md` | project | `Onnx_Advanced/core.md` "Spread-knowledge tooling" section |
| `feedback_competition_encoder_bg_channel.md` | feedback | `Onnx_Advanced/core.md` "NeuroGolf input encoder" + "Drop ch0 via Pad-prepend-zero" anti-pattern |
| `feedback_padding_cell_encoding.md` | feedback | `Onnx_Advanced/core.md` "1-ch0 trap" + `Onnx_Advanced/traps.md` §9 |
| `feedback_crop_advisory_false_positive.md` | feedback | `Onnx_Advanced/traps.md` §26 |
| `feedback_verify_is_hard_gate.md` | feedback | `Onnx_Advanced/traps.md` §23 + `core.md` "Verify gates and hidden-test rules" |
| `feedback_lookup_family_detection_refined.md` | feedback | `Onnx_Advanced/traps.md` §27 |
| `feedback_unlock_synth_threshold.md` | feedback | `Onnx_Advanced/traps.md` §14e (2026-05-25T15:11Z entry) + `The_400_Tasks/current_strategy.md` F-mode |
| `feedback_neurogolf_opt_safe_blindspots.md` | feedback | `Onnx_Advanced/techniques.md` #17 (5 sub-patterns) |
| `feedback_neurogolf_f16_output_trick.md` | feedback | `Onnx_Advanced/techniques.md` (F16 family) + `core.md` "NeuroGolf output decoder" |
| `feedback_neurogolf_crop_to_active_grid.md` | feedback | `Onnx_Advanced/techniques.md` #2 (Crop-to-active-grid) |
| `feedback_f16_cast_placement_strategy.md` | feedback | `Onnx_Advanced/techniques.md` Rule A/B/C (Cast placement) |
| `feedback_bool_first_compute.md` | feedback | `Onnx_Advanced/techniques.md` #1 (Bool-first family, 12 rules) + #15 (Separable mask collapse) |
| `feedback_onnx_opset10_equal_int32_only.md` | feedback | `Onnx_Advanced/traps.md` §5 (opset-10 Equal INT32 only) |

## Append-history

- 2026-05-25T15:30Z — initial migration of 21 memory entries out of `.claude/projects/C--Users-Qx-Documents-Py-cx-K-NeuroGolf/memory/` to project docs. Memory directory cleared per user request: "i consider agent memory as secrets kept from the project."

## §8b. Anti-re-bloat charter (adopted 2026-06-05 from the context-bloat audit — binding on captains AND workers)

1. **Litmus at the door**: would Sonnet produce this cold? Then it's a pointer, not a paragraph. (Empirical VERDICTS that look spec-like — "ORT rejects X" — always pass; explanations of what ops do never do.)
2. **Verdict-first formats**: trap = pattern/symptom/fix/ref in ≤4 lines. Technique = pattern → rewrite → safety → measured saving → exemplar path. No session narration.
3. **Supersede in place**: EDIT the parent section; "ADDENDUM/UPGRADE/ESCALATION" appendages are banned. Old text dies when corrected (CHANGELOG records the change).
4. **No ship-ledgers in reference docs**: per-task outcomes live in `taskNNN.md` / `current_strategy.md` / `Logs/submissions_ledger.csv`; reference docs cite ONE anchor example.
5. **One canonical home per fact** + pointers (a fact stated twice WILL drift — Phase-0 bugs #1-#3 were all duplication-drift).
6. **Section keys are stable**: never reuse a retired §number; new sections take the next free number; title-grep is the contract.
7. **Extraction closes the loop**: Research deliverable extracted ⇒ original to `_archive/` same commit.
8. **Strip citation artifacts on paste** (PUA-wrapped `cite…turn…` tokens).
9. **Every-task set has a size budget** (~65 KB target post-Phase-1), printed in INDEX; adding a file requires evicting equivalent KB.

## §9. No self-initiated worktrees or branches (owner, 2026-06-05)

**Rule**: agents NEVER create git worktrees or branches on their own initiative — edit in place in the main local dir. Enforcement point: root `CLAUDE.md` (auto-loaded by every Claude Code session; zero context cost to non-Claude agents).
**Why**: the owner works locally with uncommitted WIP from parallel agents; a self-made tree+branch (2026-06-05 GUI session) cost a triage round and risked task-work-loss anxiety during cleanup. In-place edits match the loop's actual workflow.
