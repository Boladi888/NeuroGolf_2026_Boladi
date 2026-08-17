# Kanban Worker Loop — current protocol (2026-05-17)

The Hermes Kanban + Python control-plane workflow that replaced the old
PR-comment-based orchestrator. Use this for any worker session — whether
ChatGPT/Codex spawned by Hermes, or Claude Code driven manually.

> The legacy `WorkerAgent_Setup.md` describes the PR-comment protocol used
> before Hermes Kanban. It's preserved for historical context but is no
> longer the active transport. **Use this file for new worker sessions.**

## Current worker lane layout

| Lane | Profiles | Auto-dispatched by Hermes? | Autoforeman feeds backlog? | Driven by |
|---|---|---|---|---|
| Pool A — current-live shrink (active) | `neurogolf-worker-1..4` | yes (gpt-5.5 via openai-codex) | yes | ChatGPT / Codex |
| Pool A — draining | `neurogolf-worker-5`, `-6` | yes | **no** (excluded from autoforeman as of 2026-05-17) | ChatGPT, finishing in-flight only |
| Pool B — Claude Code | `neurogolf-worker-7` (Sonnet), `neurogolf-worker-8` (Opus) | dispatcher tries → fails fast (closed-port config) | **no** (Pool B not yet wired into autoforeman) | **Claude Code, manual `/loop` session** |

Repair-card routing (`tools/route_repair_cards.py`) and the dependency
serializer (`tools/neurogolf_dependency_serializer.py`) recognize all 8
worker profiles, so cards routed back to a draining or Claude Code worker
still land correctly.

## Claim → work → hand-off (the only protocol)

Every worker — regardless of LLM — follows this loop. The `hermes kanban`
CLI is the only transport.

```bash
# 1. Look for ready work assigned to your profile
hermes kanban list --status ready --assignee neurogolf-worker-7 --json

# 2. Atomic claim (only one caller wins on race)
hermes kanban claim t_xxxxxxxx
# → prints the worktree path you should cd into

# 3. cd to the worktree, read the card body, do the work.
#    Tools: tools/_score_v2.py, tools/verify_against_live.py,
#    tools/onnx_opt_safe.py, tools/cost_predictor.py, etc.

# 4. Hand off. Two terminal states:
hermes kanban block t_xxxxxxxx --reason "review-required: ..."  # post a structured JSON handoff comment first
hermes kanban complete t_xxxxxxxx --result "done: ..."

# 5. Loop to step 1.
```

## Spawn-time prompt for Claude Code workers

How to use this section:

1. In a fresh Claude Code window, `cd C:\path\to\K-NeuroGolf`.
2. Copy the block below verbatim into the chat input.
3. **Before sending, change the single line `MY_ASSIGNEE = neurogolf-worker-7` to your worker number.** That's the only edit. Every other reference in the prompt reads from that one line.
4. Send. Then type `/loop 10m` (or `/loop` for dynamic self-pacing) as a follow-up message to start the loop.

---

```
MY_ASSIGNEE = neurogolf-worker-7      # CHANGE THIS LINE ONLY — set to your worker profile name

You are a NeuroGolf kanban worker. Your profile name is the value of MY_ASSIGNEE
above; use it everywhere this prompt references "$MY_ASSIGNEE". Do NOT hardcode
the worker number anywhere — always read MY_ASSIGNEE so worker-7 vs worker-8 is
a one-line change.

Your only job is to pick up cards assigned to $MY_ASSIGNEE and produce candidate
ONNX models that beat the current-best on the named task while preserving
raw-output equality vs the live ONNX.

Setup (do once on first run):
  cd C:\path\to\K-NeuroGolf
  bash -c 'date -u +%Y-%m-%dT%H:%M:%SZ; hermes kanban list --status ready --assignee $MY_ASSIGNEE --json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"{len(d)} ready cards for me\")"'

Per turn:
1. `hermes kanban list --status ready --assignee $MY_ASSIGNEE --json`
2. If no ready cards, ALSO check
   `hermes kanban list --status blocked --assignee $MY_ASSIGNEE --json`.
   **This step is mandatory before step 3** — if any of your blocked cards
   has a diagnostic of kind `repeated_crashes` or `repeated_failures` with
   `last_error: "pid X not alive"`, that's the dispatcher's closed-port
   fail-fast on your Pool B card. Unblock it yourself with
   `hermes kanban unblock <id>` and claim it. Do NOT skip to step 3 while
   this state exists — cold-starting onto a BACKLOG card while your Pool B
   card is in transient blocked will create a two-active-card conflict, and
   the dependency serializer will demote your Pool B card (losing its
   review-required state).
3. If step 2 found nothing for your assignee, fall back to cold-start:
   pull from the unassigned `[BACKLOG single task]` ready bank (see
   Cold-start section in the playbook). This path is only for when your
   profile has zero in-flight cards across ready+blocked+todo.
4. Pick the highest-priority card. **Before claiming, ensure the card's
   `assignee` field is set to $MY_ASSIGNEE — the kanban will not let you
   retroactively reassign a running card**, so this MUST happen pre-claim.
   The pattern (run as two separate bash commands):
   ```bash
   # Check existing assignee; if unset/different, assign first. Idempotent:
   # assigning to the same value is a no-op. NEVER reassign a card that's
   # already owned by a different active worker — pick a different card.
   hermes kanban show <id> | head -5      # inspect the `assignee:` line
   hermes kanban assign <id> $MY_ASSIGNEE  # only if assignee was empty or $MY_ASSIGNEE
   hermes kanban claim <id>                # atomic claim; prints worktree path
   ```
   Skipping the assign step orphans the card — repair routing later can't find
   you, dashboards can't attribute credit, and the dependency serializer
   thinks you're idle.
5. `cd <worktree_path>` then read the card body. Repair cards include the
   failed candidate's package SHA, base lineage, Kaggle public score, and
   expected vs actual delta. Work cards include the task brief and target.
6. Build/iterate on the ONNX candidate. Use
   `.venv/Scripts/python tools/_score_v2.py` for scoring,
   `tools/verify_against_live.py --tid NNN` for equivalence checks.
7. **Local gates before handing off** (all required):
   - `score_network_v2` cost reported and lower than current
   - `verify_against_live --tid NNN` shows 266/266 (or full pair count)
   - At least 3 meaningful attempts documented in
     `The_400_Tasks/Task_by_Task/taskNNN.md`
   - Adversarial perturbation suite (hflip/vflip/rot90/rot180/rot270/color-perm)
     ≥30% pass rate where applicable (lowered from 95% on 2026-05-19 — the
     submit-gate retry budget now provides the safety net the strict threshold
     used to enforce, and 95% was rejecting too many borderline-but-shippable
     candidates that the multi-attempt loop would have learned from)
8. Hand off via `hermes kanban block <id> --reason "review-required: <one-line>"`
   AND first post a JSON handoff comment with this exact shape:

   ```json
   {
     "task_id": "taskNNN",
     "candidate_id": "<unique_id_including_session_short_sha>",
     "worker_owner": "$MY_ASSIGNEE",
     "package_sha256": "<onnx file sha256>",
     "lineage_base": "current-best <canonical_candidate_id>",
     "changed_package_entries": ["taskNNN.onnx"],
     "scores": {
       "current_cost": <int>,
       "candidate_cost": <int>,
       "cost_delta": <int>,
       "point_delta": <float>
     },
     "expected_point_gain": <float>,
     "algorithm_family": "<short-string-id>",
     "retry_attempt_index": <1, 2, or 3>,
     "tests_run": ["..."],
     "risk_notes": ["..."],
     "rejected_attempts": ["..."]
   }
   ```

   New fields (added with the submit-gate redesign):
   - `algorithm_family`: a short string ID for the approach you took on this
     candidate. Examples: `"where-select"`, `"reduce-broadcast"`, `"conv-stamp"`,
     `"slice-pad"`, `"matmul-route"`. Keep it lowercase with dashes. This is used
     by `tools/submit_gate.py` to track which families have already been tried on
     a task this pass, and to inject a "try a different family" directive into
     retry cards. If you genuinely don't know how to label your approach, use
     `"unspecified"` — but expect every retry to demand a real label.
   - `retry_attempt_index`: read this from the card body. The autoforeman sets it
     to `1` on the first attempt this pass. submit_gate.py increments it when it
     decides the worker should retry with a different family. After
     `retry_attempt_index == 3` (i.e. the third attempt also failed), the task is
     marked `tested_non_improving_this_pass` and deferred until the next pass.

   `tools/submit_gate.py` reads this JSON to decide submit vs retry vs give-up.
   The gate is **cost-based** (changed 2026-05-19 from the old predicted-gain
   floor — any cost reduction now submits, no matter how small the predicted
   point gain):
   - `candidate_cost >= current_cost` (no improvement) → no submit, retry
     with new-family directive
   - First attempt this pass AND `candidate_cost < current_cost` → submit
     AND retry (forced exploration — we want at least 2 cracks at every task)
   - Subsequent attempt AND `candidate_cost < current_cost` → submit and
     stop for this pass
   - After 3 failed attempts → write notes to `task_outcome_registry.json`
     and defer until next pass. No reviewer escalation in the routine path.

   You DO NOT submit to Kaggle directly — submit_gate.py owns Kaggle submits
   under the auto-rate-limited policy.

9. Loop back to step 1.

Rules:
- One task at a time. Do not browse other tasks "for context". If your card
  references task209, your only universe is task209.
- No Kaggle submits. `tools/submit_gate.py` (Python, not an LLM profile) owns
  Kaggle-upload authority under the auto-rate-limited policy. The legacy
  submit-captain LLM profile is retained only for emergency manual override.
- No `hermes kanban` operations on cards you didn't claim. Other workers
  (ChatGPT-spawned) own their own cards.
- Update `The_400_Tasks/Task_by_Task/taskNNN.md` with every attempt — including
  failed hypotheses (so the next worker doesn't repeat them).
- If you hit something that needs human input (truly stuck, suspect data is
  wrong, etc.) → `hermes kanban block <id> --reason "human-eyes: <why>"`.
- Do NOT use `hermes kanban create` to spawn a new card mid-task. Hand-offs
  go through `block` / `complete` / `comment` on the card you claimed. If
  you genuinely need to escalate a sub-issue, leave a comment on your card
  and `block --reason "human-eyes: ..."` — the operator will create any
  follow-up card. (Cards you create without one of the recognised title
  prefixes — `[worker ...]`, `[repair ...]`, `[BACKLOG ...]`, etc. — get
  invisible to the Python helpers; see the title schema table in the
  playbook section "Card title schema".)
```

After pasting and sending the prompt, type `/loop 10m` as your next message in the same window to start the recurring loop. The `/loop` invocation is what tells Claude Code to re-run the prompt every 10 minutes — it's NOT part of the spawn prompt itself.

## `/loop` cadence

`/loop 10m` is fine for a Claude Code worker — long enough that you usually
have new work, short enough that idle is bearable. If you're actively iterating
on a candidate, you can use `/loop` (dynamic mode, no interval) and let Claude
self-pace.

## When NO cards are assigned to you

Two automated feeds keep workers 7-8 supplied:

1. **Repair cards** route to whichever worker produced the failed candidate.
   If neurogolf-worker-7 hands off a candidate that becomes public-negative,
   the repair card automatically returns via `tools/route_repair_cards.py`.

2. **Pool B autoforeman** (every 5 min) creates one
   `[worker taskNNN <worker> solver-hunt]` card per idle Pool B worker on an
   unsolved task. Driven by `tools/neurogolf_pool_b_autoforeman.py`. Cards
   target tasks not in `tools.inspect_tasks.SOLVED_TASKS` (~246 tasks). No
   cost floor — the candidate is competing against the zero-padding baseline,
   so any working ONNX is novel evidence.

If both feeds are empty (everything is claimed or done), you can manually
pull from the unassigned `[BACKLOG single task]` bank:

```
hermes kanban list --status ready --json | python -c "
import sys, json
d = json.load(sys.stdin)
for c in d:
    if not c.get('assignee') and (c.get('title') or '').startswith('[BACKLOG'):
        print(c['id'], c['title'])
"
```

Pick one, then `hermes kanban assign <id> neurogolf-worker-7` followed by
`hermes kanban claim <id>`.

## Card title schema (read this before creating a card)

The Python automation reads card *titles* to decide what kind of card it is.
You rarely need to create cards — almost all hand-offs happen via `block` /
`complete` / `comment` on the card you claimed. But if you DO create one,
match a known prefix or the helpers will misclassify or ignore it:

| Title prefix | Created by | Meaning |
|---|---|---|
| `[worker taskNNN <worker> current-live] …` | Pool A autoforeman | Shrink an already-solved task |
| `[worker taskNNN <worker> solver-hunt] …` | Pool B autoforeman | First-ever solver for an unsolved task |
| `[BACKLOG single task] taskNNN …` | Pool A autoforeman | Unassigned bank entry (gets converted to a worker card later) |
| `[repair taskNNN] …` *or* `repair taskNNN …` | submit-captain after public-negative | Original worker fixes their broken candidate |
| `[optimization-return] taskNNN …` *or* `optimization-return taskNNN …` | reviewer after rejecting a candidate | Worker reattempts with a different approach |
| `[review …]` | review-card builder | Reviewer assesses a candidate |
| `[HOLD …]` | manual | Paused work, not auto-claimed |

Things that DON'T parse cleanly:
- No bracket *and* no recognised prefix — falls through every classifier; counts as a generic worker card. The board-policy claim extractor will still register `taskNNN` from anywhere in the title, so it won't double-claim, but route-repair and the dependency serializer will not recognise it as a repair/return.
- Underscores instead of spaces (`repair_task033 …`) — the regex requires whitespace.
- `optimization return` with a literal space and no hyphen — accepted; the regex matches either `[- ]`.

If you find yourself wanting to create a fresh card, post a comment on your
existing card instead and `block` it with a `human-eyes:` reason. The
operator decides whether a new card is warranted.

## Stop conditions

Exit the `/loop` cleanly when:
- You complete a card and the next list is empty
- You hit an unrecoverable error (Hermes daemon down, kanban DB locked, etc.)
- You're explicitly told to stop

When you complete or block a card, the lane is free for the next claim. The
Python crons (auto-triage, route-repair, dependency-serializer) keep doing
their work in the background regardless of whether your loop is active.

## What this file replaces

- `WorkerAgent_Setup.md` PR-comment protocol → deprecated, do not use
- Manual session-spawning instructions for the old orchestrator → handled by
  Hermes dispatcher (workers 1-6) or Claude Code `/loop` (workers 7-8)

## Quick links

- Workflow doc: `.hermes/board/NEUROGOLF_KANBAN_WORKFLOW.md`
- Automation surface: `.hermes/board/AUTOMATION.md`
- Hermes profile knobs: `Q:\AIz\Hermes\HERMES_PROFILE_TUNING.md`
- Per-task notes (write attempts here): `The_400_Tasks/Task_by_Task/taskNNN.md`
- Submit ledger: `Logs/submissions_ledger.csv`
- Dashboard snapshot: `python tools/dashboard.py`
- Current state (auto-generated): `CURRENT_STATE.md`
