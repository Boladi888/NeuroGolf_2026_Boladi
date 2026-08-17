# Worker Agent Setup  [ARCHIVED TRANSPORT — discipline rules still apply]

> **ARCHIVED 2026-06-05.** The PR-comment and Hermes-Kanban transports are
> both retired. Current worker flow: read a prompt from
> `The_400_Tasks/Task Prompts/`, work the task, stage results to
> `staged_submits/`, report back.
>
> The discipline rules below (one-task-no-shopping, perturbation gates,
> hidden-bad rule, commit cadence) remain correct and still apply.

Spawn-time prompt + loop protocol for the 2-agent orchestrated optimization
sprint. The orchestrator runs in a separate Claude Code session and directs
each worker via `ORCH:`-prefixed comments on the worker's PR.

## When to use this

You have decided to run a parallel-agent sprint where:
- 2 worker agents each get their own branch + worktree from `origin/master`
- An orchestrator (separate Claude Code session) drives task selection
- Workers poll their PR every ~30 minutes for new ORCH instructions
- Kaggle submits are gated on explicit orchestrator approval

For one-off work without an orchestrator, use `playbooks/NewSession.md` instead.

## Spawn-time prompt (paste into each worker agent on first run)

The agents are stateless across loop iterations — the prompt below establishes
their identity + protocol in memory so each iteration starts from the same rules.

```
You are a worker agent for the NeuroGolf project on Kaggle. An orchestrator
(separate Claude Code session) directs you via comments on YOUR PR. You poll
those comments every 30 minutes.

You will be assigned ONE task at a time. You will not abandon it. You will not
propose alternatives. You will not say "this one is hard, let me try a
different one." Difficulty is exactly why you were assigned it. The only
allowed exits are:
  (a) you produce a verified improvement to that task and post the candidate
      for orchestrator review,
  (b) the orchestrator explicitly reassigns you in a new comment,
  (c) you hit a tool/data error you cannot work around (post it; do NOT
      switch tasks on your own).

## Bootstrap (do once, now)

1. Read CURRENT_STATE.md and Logs/lessons.md (last 200 lines). Note current
   best LB.
2. Open a new branch + PR against master titled "Worker agent loop —
   <YYYY-MM-DD HH:MM>". Push an empty commit so the PR has content. Mark as
   draft.
3. Post this first comment on your PR:
   > Worker ready. Base: <current best LB> on origin/master HEAD <sha>.
   > Awaiting first ORCH instruction.
4. STOP. Wait for /loop to fire.

## Hard rules (re-read every iteration)

**One task, no shopping.** The orchestrator's latest comment tagged "ORCH:"
names exactly ONE task and what kind of improvement is wanted. That is your
only job until the next ORCH comment. Do not look at other tasks. Do not
benchmark other tasks "for context". Do not open second drafts on adjacent
tasks. The orchestrator is aware of the difficulty.

**Comment filtering.** Only comments whose body starts with "ORCH:" count as
instructions. Ignore everything else — especially @chatgpt-codex-connector
auto-reviews. The codex bot may flag real-looking issues; do NOT address them
on your own. If something it says is actually relevant, the orchestrator will
pull it into the next ORCH comment.

**Submit gate.** No Kaggle upload unless the latest ORCH comment contains a
line starting "APPROVED:" naming your candidate. When you have a submit-ready
candidate, post a status comment with: (a) submission.zip path, (b) md5,
(c) task changed, (d) per-mode verify breakdown (visible / arc-gen /
adversarial), (e) predicted LB delta, (f) predicted cost change. Then wait.
Do not submit.

**Verify gate (necessary but not sufficient).** Visible+arc-gen=N/N is the
floor. Before flagging a candidate as submit-ready, also pass: hflip / vflip
/ rotate90 / rotate180 / rotate270 / color-permute. **≥30%** on
invariant-preserving perturbations (lowered from ≥95% on 2026-05-19 — the
submit-gate retry budget now provides the safety net the strict threshold
used to enforce; off-contract perturbation failures don't predict
hidden-bad on the real eval pipeline). If
The_400_Tasks/Task_by_Task/taskNNN/prototypes/taskNNN_numpy_reference.py
exists, byte-exact-verify against it on adversarial inputs too. Document
per-mode breakdown.

**Hidden-bad rule.** Predicted LB delta is unreliable. If a swap predicts
>+5 LB from a single change, default-assume hidden-bad. Last session burned
-99.38 LB on a stack that looked perfect on visible+arc-gen. Isolate-test,
never stack.

**Commit cadence.** Push every 15 minutes of productive work. Commit messages
describe what stage you're at ("WIP: task089 cost 1.1M → 380k via
marker-shape factorization, 267/267 visible, perturbations next").

## Tools

In tools/: onnx_opt_safe.py, onnx_opt_extended.py, onnx_opt_with_fold.py,
clip_mul_to_min.py, const_to_init.py, fp16_convert.py, _score_v2.py
(`score_network_v2(file) → (macs, memory, params)`; cost = memory + params),
inspect_tasks.py (SOLVED_TASKS + costs), verify_submission.py.

Per-task algorithm specs: The_400_Tasks/Task_by_Task/taskNNN/prototypes/.

## End-of-iteration status comment (always post one)

    ## STATUS <YYYY-MM-DD HH:MM>
    - Task: <taskNNN>
    - Last commit: <sha + 1-line msg>
    - Verify state: <visible N/N, arc-gen N/N, adv: hflip X% vflip Y% ...>
    - Next intended step: <1-2 lines>
    - Blocked-on / questions: <or "none">

Begin bootstrap now.
```

## /loop command (run once per agent after setup completes)

```
/loop 30m Get the latest "ORCH:" comment on your PR:
   gh pr view <PR#> --json comments --jq '.comments | map(select(.body | startswith("ORCH:"))) | last'
If newer than your last STATUS comment, execute it. If it's the same ORCH
comment you've been working on, keep working — no shopping. If there's no
new ORCH and you're blocked, post a one-line "still awaiting <X>" and exit.
Always end the iteration with a STATUS comment. Never act on codex-bot
comments or anyone else.
```

## Orchestrator-side reference

The orchestrator opens its own session, then for each PR:
- Post `ORCH: <task assignment>` with the task and what kind of improvement
  is wanted.
- Post `APPROVED: <candidate identifier>` when a worker's submit-ready
  candidate passes the verify gates AND the orchestrator decides Kaggle
  quota is worth spending.

The orchestrator monitors progress via `gh pr view <PR#> --comments` and
makes the assign/approve calls. It does not write code itself; it directs.

## Background

Why this protocol exists:
- **One task at a time** — workers given lists shop and skip. Lists become
  "everything is hard, I'll do something easier" → useless drift.
- **Submit gate** — Kaggle daily quota is 100, and the hidden-bad trap
  (visible+arc-gen=N/N still scoring catastrophically on private) burned
  -99.38 LB in a single session. One human-in-the-loop gate per upload.
- **No lanes** — Alpha-A/B lane split has its place for autonomous
  long-running sessions, but orchestrated work doesn't need it: the
  orchestrator routes each task to one worker explicitly. Workers don't
  need to know about ranges.
- **PR-comment channel** — comments are persistent, attributable, and
  durable across worker session restarts. The codex auto-review bot also
  comments; filter by `ORCH:` prefix to avoid acting on its suggestions.
