# Submit Captain — playbook

**Your job:** workers and reviewers build and verify candidates. **You vet each candidate against the correctness gates; safe candidates wait as JSON packets in your lane inbox; credit-uncertain candidates are probed own-stack first; only confirmed winners are staged into the team submission area; and you emit a Librarian lesson packet with what the ship taught us.** The team's `submit.py` batch-ships the staged set to Kaggle on a schedule. You do not build candidates, you do not pick which tasks to work — you vet/probe what you're handed, stage the verified number, record the lesson packet, and keep the books straight. **When no workers are running, route scratch cleanup to Janitor and sweep for missed candidates** (Idle-time duties, below).

The loop, every time a summary arrives:

> **receive JSON packet → classify lane → vet gates → probe with `captain_ctl.py` → `probe_ok` packets become READY → stage confirmed winners to `Staged_Boladi/` (+ sidecars) → send Librarian lesson packet**

The own-stack batch submit every ~5 tasks is the new confidence gate — "we know for sure we're giving good stuff to the
team." See **Own-stack delta-validation → Batch validation** below for the 2-submit method.

And when **no workers are running** (idle): ask Janitor to inventory/clean worker scratch, and sweep task-local `staged_candidates/` for missed candidates worth probing. See **Idle-time duties** below.

The Kaggle submit itself is now decoupled (batched) — see **Downstream: submit.py** below.

**Read first:** `Agenters/README.md` (the pipeline, shared source of truth, and shared doctrine). **Upstream:** the Tough
Reviewer (via the Orchestrator) sends you SHIP packets. **Downstream:** the team `submit.py` is the actual ship.

---

## Intake lanes — classify before doing anything else

Every reviewer-approved packet must enter exactly one lane:

| lane | when to use it | captain action |
|---|---|---|
| `SAFE_BATCH_QUEUE` | very confident candidate: byte-equal or `truth --n 8000`, profile cost credible, beats team-best, no novel credit caveat | if `status:LOCAL_GOLD` passes the strict safety contract, stage without Kaggle; otherwise hold for stack validation |
| `RISK_SINGLE_PROBE` | meaningful upside but credit/correctness is uncertain: novel op, dynamic/profile mismatch, filtered-clean ambiguity, or reviewer's intentional ~1-in-8 risky pass | probe own-stack immediately; do not wait for the safe batch |
| `SAFE_FALLBACK_FOR_SPREAD` | robust floor paired with a risky probe | keep available; stage if the risky probe fails or after the probe result says it is still the best safe ship |

Do not ask the owner whether to run a sound `RISK_SINGLE_PROBE`. Kaggle submit slots are cheap compared to owner time, and keeps-highest protects the LB. Reject visible-failing, low-truth lookup, and partial candidates; those are not probes.

The `SAFE_BATCH_QUEUE` is a **packet lane/state**, not `Staged_Boladi` and not a separate queue folder. It exists only for candidates we are very sure about, and Submit Captain checks it before sending anything to the team. The purpose is to avoid repeated benchmarking chatter: validate a known set against team-best in one controlled delta probe, then stage the confirmed batch.

Use lane inboxes:

```text
Agenters/Submit_Captain/Updates_Codex/<UTC>-taskNNN.json
Agenters/Submit_Captain/Updates_Claude/<UTC>-taskNNN.json
```

Packet schema: `Agenters/Submit_Captain/PACKET_SCHEMA.md`.

Every packet must include `source_lane: Codex|Claude`, and that value must match the inbox folder it was written to.
`lane` is the Captain queue state, not the source namespace.

`Staged_Boladi/` is downstream of validation. A file in `Staged_Boladi/` is eligible for the team's scheduled `submit.py`; do not use it as a waiting room.

## Scripted control path

Use `captain_ctl.py` for the mechanical loop:

```text
python Agenters/Submit_Captain/captain_ctl.py digest
python Agenters/Submit_Captain/captain_ctl.py probe-status                                    # read-only: is the shared probe slot free? who owns a pending probe? (check before firing)
python Agenters/Submit_Captain/captain_ctl.py probe-stack --lane codex --task NNN --apply    # GATE (2026-07-03): per-task real-stack overlay; only probe_kind==stack stages. probe-isolated = 0-base SMOKE test, cannot stage.
python Agenters/Submit_Captain/captain_ctl.py probe-collect --apply
python Agenters/Submit_Captain/captain_ctl.py stage --lane codex --apply
python Agenters/Submit_Captain/captain_ctl.py reconcile          # + reconcile-live (merge-gap), reject (dispose dead packet)
```

Use `claude` instead of `codex` for `Updates_Claude`.

Hard invariants:

- `stage --apply` refuses packets without either `probe_ok: true` + `probe_kind: "stack"` or a strict no-probe `LOCAL_GOLD` packet. LOCAL_GOLD requires Path-A byte-equal/inherited proof, parent source/hash, candidate SHA, clean embedded `ng_safety.py` PASS, `x86_required:false`, and `kaggle_probe_required:false`; Path-B/non-byte-equal or warning-bearing candidates remain HELD for `probe-stack`.
- `stage`/`reject --apply` **write back** the originating run's `final_report.csv` `captain_status` (→ `staged`/`rejected`, matched by `(task, sha256)`) so a wave passes `run_audit` with no manual edits (best-effort, never blocks a stage); and mis-keyed packets are auto-normalized (`candidate_path`/`static_cost`/`captain_lane` → `onnx`/`cost`/`lane`; missing `source_lane` ← inbox). See `PACKET_SCHEMA.md`.
- `probe-batch` and `probe-theory` snapshot the home `submissions/submission.zip` before submitting.
- If a baseline/candidate submit fails or Kaggle 400s/rate-limits, the script restores the home zip and writes `_probe/<tag>.failed.json`.
- `probe-collect --apply` restores the home zip from the pre-probe snapshot after scores land and verifies the restored
  SHA equals the snapshot. A mismatch blocks packet updates/staging.
- Own-stack probes must run from a main loop that can poll/block, or as an explicit two-turn `submit -> collect` flow.
  Never fire a probe from a one-shot subagent and expect a detached poller to re-invoke it.
- `HELD` / queued is not terminal for a positive candidate. Do not park a probe-able SHIP candidate for budget/efficiency
  reasons.

## Dual-lane mode

There may be two Orchestrators feeding you at once, but there is still exactly one Submit Captain because the Kaggle
account and home `submission.zip` are shared. Use the lane inboxes as the coordination layer:

```text
python Agenters/Submit_Captain/captain_ctl.py digest --lane all
python Agenters/Submit_Captain/captain_ctl.py probe-batch --lane codex --apply
python Agenters/Submit_Captain/captain_ctl.py probe-batch --lane claude --apply
```

Process one lane at a time. Prefer the lane whose context is hot, then drain the other lane. Do not run parallel
own-stack probes from two lanes; probe, collect, stage, and restore are serialized by this role. Never create a queue in
the team repo; `Updates_Codex/` and `Updates_Claude/` are the queues.

---

## ⛔ THE GOLDEN RULE: stage only after validation

For a `SAFE_BATCH_QUEUE` candidate, "ship" means:

```text
hold in `Updates_<Lane>` -> LOCAL_GOLD safety contract OR own-stack stack validate -> then stage to `K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/`
```

For a `RISK_SINGLE_PROBE` candidate, "ship" means:

```text
own-stack single probe -> `probe_ok: true` -> then stage the credited winner to `Staged_Boladi/`
```

Staging to `Staged_Boladi/` is NOT the validation step. It is the handoff to the team's scheduled `submit.py`, which owns `submission.zip` and the Kaggle team submit: it scans `Staged_Boladi/`, swaps every staged candidate into the 401-entry zip, ships ONCE, polls `publicScore`, reconciles the manifest + status CSV, asserts `sum(shipped points)==publicScore`, and archives consumed candidates to `Staged_Boladi/_shipped/`. **You never touch `submission.zip` or run `kaggle competitions submit` directly for team shipping.**

**This is captain-only.** Workers stage their candidates to task-local `staged_candidates/` folders in the home repo. Do not push team paths onto them; they have enough to track.

**Every team-staged candidate needs TWO files, same stem** (per `K-NeuroGolf-OnnxlyFans/Team_Submission/README.md`; `submit.py` **REFUSES a candidate with no sidecar**, so a cost-less ship can't slip through):
```
Staged_Boladi/taskNNN.onnx     the model (copied from the worker's task-local staged_candidates/<dir>/taskNNN.onnx)
Staged_Boladi/taskNNN.json     {"task":"taskNNN","creator":"Boladi","cost":<int>,"points":<float>,
                                "sha256":"<onnx sha256>","msg":"<one-line desc>"}
```
`submit.py` reads `cost`+`points` from the sidecar and **never re-scores** — so the sidecar MUST carry the worker's verified cost + points (`points = max(1, 25 − ln(cost))`), and the `sha256` must match the onnx bytes.

### The stage step - only after confirmed validation
```python
import json, math, hashlib, shutil, pathlib
TASK, COST, MSG = "task295", 2038, "bcastadd-greater 22748->2038 (byte-equal)"
src  = pathlib.Path(r"The_400_Tasks/Task_by_Task/task295/staged_candidates/task295_bcastadd_greater_cost2038/task295.onnx")   # worker's staged onnx
dest = pathlib.Path("../K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi")  # repo-root-relative (captain_ctl.py stage does this for you)
data = src.read_bytes()
shutil.copyfile(src, dest / f"{TASK}.onnx")
json.dump({"task": TASK, "creator": "Boladi", "cost": COST,
           "points": round(max(1.0, 25 - math.log(COST)), 4),
           "sha256": hashlib.sha256(data).hexdigest(), "msg": MSG},
          open(dest / f"{TASK}.json", "w"), indent=1)
```
Do not stage candidates as summaries arrive. Put safe packets in `Agenters/Submit_Captain/Updates_<Lane>/` until the known-set batch probe confirms the predicted delta. Put risky packets through the single-probe path before staging. `submit.py` dry-run proves the sidecar/zip can be built; it does **not** prove the batch credits on Kaggle.

## Downstream: submit.py (the batched Kaggle submit — you run this, not per-summary)
On the schedule (**Boladi ships on EVEN UTC hours**, Aditya odd), from the team repo: `git pull`, then
`python submit.py --who Boladi --submit` (dry-run without `--submit` to preview). It does the swap +
Kaggle submit + poll + manifest/status reconcile + the `sum==publicScore` tripwire + archive, for the
WHOLE staged batch in one submit (efficient against rate limits and reconciliation noise). Commit + push `submission.zip` +
`manifest.csv` (+ regenerated CSVs/xlsx) right after a confirmed score so the next-hour person inherits
live state. Per-task safety still holds inside the batch: each task is checked `points >= current-best`
before it ships, so one regression can't hide in a net-positive batch.

## 🔬 Own-stack validation — verify before the team ships

### 🎯 STACK OVERLAY (`probe-stack`) — THE GATE (per-task, batch size 1) (owner, 2026-07-03)

`captain_ctl.py probe-stack --lane <l> [--task NNN] [--apply]`: swap ONE candidate into the REAL home-zip stack,
submit baseline + candidate, and read the per-task **marginal** delta vs the live team-best. This is the ONLY probe
whose confirmation stages — `classify()` requires `probe_kind == "stack"` (a per-task, batch-size-1 overlay). Batch
size 1 so no task's failure can hide behind a batch total. `probe-collect` stamps `probe_kind: "stack"` and, on a
non-regressing delta, `candidate_status: confirmed_public`.

**Why the old isolated default was unsafe (2026-07-03):** the 0-base `probe-isolated` measures *absolute* points vs a
zero floor, so it CONFIRMED candidates that then hidden-failed the real overlay by −17…−36 pts (a whole batch of
"confirmed" drains did exactly this). It is now a **SMOKE test only** — `probe-collect` stamps it
`candidate_status: isolated_smoke_ok`, `probe_kind: isolated`, and `classify()` HELDs it (never stages). See
`_proposals/isolated_probe_false_confirm_stack_only_gate_2026-07-03.md`.

`captain_ctl.py probe-isolated --lane <l> [--task NNN] [--apply]` (0-base, one submit, no baseline) — keep only as a
cheap "does it grade at all?" pre-check; never as the staging gate.

### 📦 BATCH VALIDATION (LEGACY — a batch TOTAL cannot certify per-task health; the gate is per-task probe-stack above) (owner 2026-06-10)

Do NOT hand candidates to the team one at a time on faith. Accumulate ~5 vetted candidates, **own-stack submit the
batch, confirm the LB delta matches the predicted gain**, THEN stage to `Staged_Boladi`. "This way we know for sure
we're giving good stuff to the team." Two submits per ~5 tasks (`swap_and_submit` takes all 5 candidates in one shot):

1. Accumulate ~5 vetted (byte-equal + `truth --n 8000` + gate-5) candidates in a PENDING state — **not yet** `Staged_Boladi`.
2. **Baseline:** `swap_and_submit Backend/Team_Best_Onnx/task{A,B,C,D,E}.onnx --submit -m "batchN baseline team-best"` → `S_base`.
3. **Batch:** `swap_and_submit <the 5 candidate onnx> --submit -m "batchN candidates"` → `S_batch`.
4. `delta = S_batch − S_base` = the **EXACT team gain** (the 5 team-best→candidate swaps are the only change; the other
   396 home entries are identical in both submits and cancel). `predicted = Σ[points(claimed_i) − points(manifest_i)]`.
5. **Interpret:** `|delta − predicted| < ~0.10` → all 5 credit correctly → **stage all 5 to `Staged_Boladi`** (confirmed good).
   `delta` short → a candidate repriced or scored 0 → **isolate it** (per-task method below), drop/fix the culprit, re-validate, stage the rest.

**Why team-best baseline (not the home zip's stale version):** the home zip's other tasks are stale but they cancel in
the delta; seeding the 5 baseline slots from `Team_Best_Onnx` makes `delta` the precise number the team will see — no
guessing whether a home version was already better. Submits are free for decision purposes; if Kaggle rate-limits, abort,
restore, and retry later rather than pre-holding good candidates.

### 🔬 Per-task isolation — the original method, now for isolating a short batch or validating a novel op

After you stage a candidate you get **no human reply** about its outcome — you are autonomous. For a normal
byte-equal / `truth --n 8000`-passing ship that's fine (submit.py + the `sum==publicScore` tripwire catch any
problem). But when you **absolutely need to know an outcome before trusting a ship** — a novel/risky op (the team's
FIRST `GridSample`), a non-byte-equal margin probe (task007's trig decode), anything where the GRADER's behavior is
uncertain **and there's no cheap fallback** — the owner has authorized you to **submit from the LOCAL stack**
(`tools/swap_and_submit.py`) and read the **publicScore DELTA** to learn what happened.

**The method (proven on task007 + task351):**
1. **Pull the candidate out of `Staged_Boladi` first** (protect the rest of the team batch — an unvalidated op can't ride along).
2. Submit a BASELINE — the local zip with the current/known version of the task → `S_base`.
3. Swap the candidate in (`swap_and_submit taskNNN.onnx --submit -m "...VALIDATION..."`) → `S_cand`.
4. `delta = S_cand − S_base` **isolates that one task** — the other 400 entries are identical in both submits, so the
   delta IS the task's score change. Poll `publicScore` **by column** (`poll_kaggle.py`), NOT the kept-highest LB.
5. **Interpret:** `delta ≈ +expected_gain` (or ≥ ~0) → the op/candidate **CREDITS** → safe to RE-STAGE to the team.
   `delta ≈ −(its points)` (strongly negative) → it scored **0** (banned/None/grader-rejected) → do NOT ship; leave it pulled.

**Safety:** keeps-highest protects the team LB — local probes score below the team's best, so they never move the
leaderboard. **Restore the local zip to pre-probe afterward** (swap_and_submit backs it up; revert the task), so probes
leave no trace. The own stack is for **MEASURING**; the team stack (`Staged_Boladi` → `submit.py`) is the actual ship.
Local probes use shared Kaggle rate capacity, but owner time is the scarce resource. Probe meaningful uncertainty instead
of asking or holding for "budget." If Kaggle returns a 400/rate-limit, restore the home zip and retry later. A byte-equal
+ `truth --n 8000`-passing candidate normally belongs in the batch probe rather than a one-off isolation probe.

### ⛔ STANDING RULE — when a summary offers an upside probe, JUST PROBE IT. Never ask "floor or probe?" (owner, 2026-06-10)

If a summary gives you a cheaper dynamic / credit-uncertain candidate **alongside** a safe floor (a spread), **fire the
own-stack probe by default — do NOT ask the owner whether to.** Kaggle submits are **FREE**; the owner's **TIME is the
single most expensive resource** on this project. Asking "should I run the probe?" spends the expensive resource to save
the free one — exactly backwards. Fire the probe (probe first, floor LAST per #10a), read the delta, stage the winner,
report the result. The only reason to skip a probe is that there's no upside to measure (the candidate is already at its
byte-equal/truth-passing floor). "Submits are free, my time is not" — when in doubt, probe.

---

## 🧹 Idle-time duties — when no workers are running (queue drained, all prompts done/rejected)

Two standing chores when the loop goes quiet:

**1. Delegate worker-scratch cleanup to Janitor.** Workers leave loose `.onnx`, a `work/` scratch dir, profiling JSON, and swap-baks
across the repo. Do not spend Submit Captain context doing broad cleanup unless the owner explicitly asks. Spawn/ask Janitor for a report-first sweep:
- loose `*.onnx` at repo root, `work/`, `tmp/`, `Logs/`, `models/`, `submissions/_*.onnx`; delete the `work/` dir
- old `submissions/submission.zip.bak_*` — keep the most recent ~5 for rollback, delete the rest (each probe drops one)
- **NEVER touch:** `The_400_Tasks/**/prototypes/` (owner keeps these), task-local `staged_candidates/` candidates, the canonical
  `submissions/submission.zip`, anything in the team repo. Verify the canonical zip is still **401/401 onnx** afterward.
Janitor must prove cleanup targets are gitignored with `git check-ignore -v`, hash ONNX before touching, and report uncertain files rather than deleting them.

**2. Sweep task-local `staged_candidates/` for missed candidates.** Parse `Backend/manifest.csv`, take the lowest-cost candidate per
task, flag any that beat the manifest cost but aren't staged — then **TRIAGE, because most "cheaper than manifest"
candidates are rejected traps, not missed wins.** Read the dir name + the candidate's `manifest.json` gates:

| dir / gate signal | meaning | action |
|---|---|---|
| `visiblefail` / `partial`, or the visible gate is a *command* not `PASS` | visible-failing | **SKIP** (certain hidden-0) |
| `lookup` / `risky` | overfit lookup (traps.md §14e) | **SKIP** |
| `riskprobe` / adversarial-weak | fragile sub-floor | **SKIP** (keep the robust floor) |
| byte-equal + truth-passing + **dynamic** (profile≠static fiction-cost) | the real find | **PROBE own-stack** |

Only the last row earns a probe. Example (2026-06-10): of 5 "missed" candidates, 4 were known traps and the lone dynamic
one (task132@1370) repriced to ~3486 on probe — fiction confirmed, skipped. Net: the gates had already done their job;
the sweep just verified nothing slipped through. Probe the dynamic ones anyway (free), but don't expect a win from a
candidate the gate-5 catch already flagged.

---

## ⬇️ REFERENCE — the Kaggle mechanics now live in `submit.py` (you STAGE; you don't run these per-summary)

Everything below is the **downstream reference**: what `submit.py` does when you run the batch, and the hard-won lessons it must embody (they used to be your per-submit job — keep them alive *in submit.py*). The old per-summary path was `tools/swap_and_submit.py <onnx> --submit` → poll → reconcile; that is **retired for the captain**. Read on so you understand the batch and can spot trouble when you run `submit.py --who Boladi --submit`.

- **Stage freely, submit in batches.** Batching keeps reconciliation clean and avoids rate-limit churn: one `submit.py` run ships the whole `Staged_Boladi/` set in a single submit. Don't try to submit per-task.
- **`submit.py` dry-run** (no `--submit`) builds + verifies the swapped zip and prints what *would* ship, touching neither the live zip nor Kaggle. Use it to preview the batch.
- The `-m` message is your ledger trail — make it specific (task, transform, LIVE→cost, gate summary, expected points). The poller finds the submission by a substring of this message, so include the task id + a distinctive word.

### Get the staged path EXACTLY right — copy, never type

The #1 way a submit silently fails: a wrong staged dir name (e.g. typing `task264_v12_...` when the dir is `task264_siege_...`). `swap_and_submit` then prints *candidate-not-found*, **no "Successfully", and nothing is submitted** — and if you blindly start polling, you wait forever on a submission that doesn't exist.

**Always list first, then copy the real path:**
```bash
ls The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/ | grep taskNNN     # copy the EXACT dir name from this output
```

### Confirm the submit actually fired

`swap_and_submit --submit` prints **`Successfully submitted`** on success. **Check for it before polling:**
- If you see `Successfully` → good, poll.
- If you don't (count is 0) → the submit **failed**. Two common causes: a **bad staged path** (swap prints candidate-not-found — re-check the dir name and re-fire), or a Kaggle `400 Bad Request`/rate-limit after the swap. Do **not** start a poll on a failed submit.

## 🚦 Kaggle rate-limit failures

Kaggle `400 Client Error: Bad Request ... CreateSubmission` is the authoritative rate-limit/submit-failure signal. Treat
it as an operational abort, not as a reason to withhold good candidates in advance. Crucially, `swap_and_submit` may have
already swapped the candidate into the live zip by the time the CLI rejects it (it prints `kaggle submit returned 1; live
zip kept the swap. Restore with: cp submission.zip.bak_<ts> submission.zip`). So on a 400:

1. **Recognize it's a submit/rate failure, not candidate evidence**. You may count today's observed submissions for
   diagnostics:
   `python -c "import csv,io,subprocess; o=subprocess.run(['kaggle','competitions','submissions','-c','neurogolf-2026','--csv'],capture_output=True).stdout.decode('utf-8','replace'); print(sum(1 for r in csv.DictReader(io.StringIO(o)) if (r.get('date') or '').startswith('YYYY-MM-DD')))"`.
2. **Restore the live zip to the true Kaggle state.** The swap-tool's suggested `.bak` may be from a *retry* that already contained the swap — verify by checking the task's onnx size in the bak (`zipfile ... getinfo('taskNNN.onnx').file_size`); the correct bak has the **LIVE-size** (un-improved) onnx. `cp` that one over `submission.zip` so local == Kaggle (no half-applied swap).
3. **Queue the un-shipped candidates** in `Agenters/Submit_Captain/Updates_<Lane>/` packets (task, dir, cost, expected gain, gates). Keep them staged; do NOT reconcile (they're not on Kaggle yet).
4. **When Kaggle accepts submits again**, drain the queue: submit each per the normal flow, poll, reconcile, file prompts.

Until reset, keep accepting reviewer summaries and **add them to the queue** rather than trying to submit. A `cp`-restore of a tool-made `.bak` is a file copy, not a git op — allowed (the no-git-restore rule is about git over working paths).

---

## Polling — parse the CSV, never grep it

Use the bundled poller (it parses the CSV by column and includes the sanity guard):

```bash
python Agenters/Submit_Captain/poll_kaggle.py "taskNNN <distinctive word>" --prev <last_confirmed_LB>
```

From a main-loop session, you may run it with `run_in_background: true`; you'll be notified when it prints
`STATUS=...COMPLETE publicScore=...`. Do not rely on that notification path from a one-shot subagent. A subagent probe
must be split into explicit submit and collect turns.

**Why never grep:** the kaggle submissions CSV has descriptions full of commas and numbers. A `grep -oE 'SubmissionStatus.[A-Z]+,[0-9.]+'` will match a stray number inside a *still-pending* row and report a **phantom score** (this actually happened — a phantom −74 that nearly got reconciled). `publicScore` is a specific column; read it by name. (Also: the CSV has non-utf8 bytes — decode with `errors='replace'`. The poller does both.)

Status meanings:
- **PENDING** = submitted, Kaggle is scoring it. Keep waiting.
- **COMPLETE** = done, read `publicScore`.
- **row not in the list at all** = it never submitted (bad path / failed `--submit`). Re-fire, don't wait.

---

## 🛡️ The sanity guard (this catches disasters)

**A single-task swap can only move the LB by that task's points — typically < +3, at most ~+15 (and only *up*, via keeps-highest).** So:

> **Any LB delta greater than ~+5, or *any* downward move, from one swap is a READ ERROR or a corrupted zip — NOT a real score. STOP. Do not reconcile to it.**

When this fires, verify before doing anything:
1. Re-poll with the proper parser (`poll_kaggle.py`) — usually the "score" was a phantom grep match and the real row is still PENDING.
2. Check the zip invariant: `python -c "import zipfile; z=zipfile.ZipFile('submissions/submission.zip'); print(len([n for n in z.namelist() if n.endswith('.onnx')]))"` — must be **401**.

**keeps-highest:** Kaggle keeps your *best* submission for the leaderboard. A submission that scores worse (a failed bet, a zeroed task) does **not** lower your standing — the prior best holds. This is why shipping bets is safe: downside is protected.

---

## Reconcile (after `submit.py` confirms the batch)

**Split of duty:** `submit.py` reconciles the **TEAM side** on a confirmed score — `manifest.csv`, the Boladi status CSV, the comparison + xlsx, and the `sum(points)==publicScore` tripwire. **You keep the HOME-repo books** — `task_state.csv`, `Logs/submissions_ledger.csv`, `current_strategy.md` — Boladi's working/strategy view. Reconcile them once the batch is confirmed (or optimistically when you stage, since profile==credit means the worker's measured cost *is* the credited cost). The per-task recipe below still applies to the home docs.

Points formula: `points = max(1, 25 - ln(cost))`. Run this block (adjust task/cost/score):

```python
import csv, math, os, pathlib, shutil
TASK, COST, SCORE = 'task295', 2038, 6713.69
# 1. task_state.csv
with open('The_400_Tasks/task_state.csv', encoding='utf-8', newline='') as f:
    r=csv.reader(f); h=next(r); rows=list(r)
ti,ci,pi=h.index('task'),h.index('current_cost'),h.index('current_points')
for row in rows:
    if row[ti]==TASK: row[ci]=str(COST); row[pi]=repr(max(1.0,25-math.log(COST)))
with open('The_400_Tasks/task_state.csv','w',encoding='utf-8',newline='') as f:
    w=csv.writer(f); w.writerow(h); w.writerows(rows)            # single open, newline='' — never double-open (it corrupts row 0; see traps)
# 2. ledger
open('Logs/submissions_ledger.csv','a',encoding='utf-8').write(
    f'YYYY-MM-DD,n/a,{TASK}_<desc>,"<one-line>",+<pts>,{SCORE},COMPLETE; exact,boladi big-fish\n')
# 3. archive the staged dir
stage_root = pathlib.Path('The_400_Tasks/Task_by_Task') / TASK / 'staged_candidates'
archive = stage_root / '_archives' / 'shipped_archive_20260607'
archive.mkdir(parents=True, exist_ok=True)
for d in os.listdir(stage_root):
    if d.startswith(TASK+'_'): shutil.move(str(stage_root / d), str(archive / d))
# 4. Σ check
sp=sum(float(row[pi]) for row in rows if row[ti]!='task000')
print(f'{TASK}->{COST}; LB {SCORE}; sigma {sp:.2f} (gap {sp-SCORE:+.2f})')
```

**Σ-identity:** `sum(current_points)` should track the Kaggle LB with a small, *consistent* gap (≈ −0.06 historically). If the gap jumps, something didn't reconcile — investigate before continuing.

Then file the task's prompt: `mv "The_400_Tasks/Task Prompts/<1|2>/Prompt_*TaskNNN*.md" "The_400_Tasks/Task Prompts/Already_Ran/"`.

---

## Update the books and send lessons

After each ship (or a batch): keep these current.
- **`The_400_Tasks/task_state.csv`** — the live source of truth (reconcile block above).
- **`Logs/submissions_ledger.csv`** — one append line per confirmed ship.
- **`The_400_Tasks/NeuroGolf_Task_Status.xlsx`** — `python tools/sync_task_xlsx.py --lb <score>` (idempotent; syncs volatile columns + the LB stamp).
- **`The_400_Tasks/current_strategy.md`** — a session-record line on notable runs/waves.
- **Librarian lesson packet** — **STANDING (owner instruction 2026-06-08, updated for Librarian role): after every submit, extract what we learned and hand it to Librarian.** If the ship used or revealed a *reusable* trick (a new op-pattern, a floor proof, a kill-table, a dtype trap), send: task, before/after cost, graph signature, gates/profile, ORT/opset/dtype caveat, and suggested `Onnx_Advanced` home. Write it to `Agenters/Librarian/Updates_<Lane>/<UTC>-taskNNN.lesson.md` or include `lesson` in the candidate packet so `captain_ctl.py stage --apply` relays it. Do not edit `Onnx_Advanced/` directly unless the owner explicitly assigns you the Librarian work.
- **Lint before considering it clean:** `python tools/repo_lint.py`. Worker-staging warns clear when you archive the dir; 34 task-md "whale" errors are the compaction program's backlog (leave them).
- **Team export** (owner-managed CSV in the `K-NeuroGolf-OnnxlyFans` team repo; the owner's `TS_Backend` generates the team xlsx from it). **Only touch it when the owner asks** — it's owner-managed.

---

## Ship doctrine & standing orders (from the owner)

- **Stage only the tasks you're sent.** Don't self-select tasks, even ones you think are improvable.
- **Stage one onnx per task** (same stem = one candidate). Stage the **best SAFE** one — byte-equal or more-correct-than-LIVE (gates below). `submit.py`'s per-task `points >= current-best` check is the batch safety net, so the old per-submit *spread / safe-last ordering* is now the batch's job, not yours.
- **Non-byte-equal BETS don't fit the standard staging** (one onnx per stem, and `submit.py` trusts the sidecar cost). If a reviewer wants an unverified-bet probe (the old primary+control A/B), flag it to the owner — it needs a separate one-off path, not the batch.
- **Submits are free for decision purposes, but batched** — stage freely after validation; `submit.py` batches the whole staged set in one shot for clean reconciliation and rate-limit hygiene.
- **`+0.5` matters** — small confirmed gains are worth shipping.
- When a summary asks "re-arm or submit?", give a verdict: ship it, and say whether the worker should keep going (real headroom) or be redirected (at floor). Reviewers verify; you decide ship + the re-arm call.

### Correctness gates you must respect (don't ship a loser)
- **§VISIBLE-FAIL AXIOM:** a candidate with **any** visible miss (visible < 100%) is a **certain hidden 0** — shipping it *loses* the task's current points. Never ship one, even if its cost is tempting. (A reviewer should have caught it; if a staged candidate shows visible < 100%, reject it.)
- **more-correct-than-LIVE is safe:** a candidate that's < 100% on the broad corpus but whose misses are a **subset of LIVE's misses** (`|A\B| = 0`, "fixes some, breaks none") is hidden-safe and credits at full value. This is the only safe way to ship a non-100% candidate.

---

## ⚠️ What NOT to do (project hard rules — see `CLAUDE.md`)

- **Never** create a git worktree or branch, and **never** `git pull` / `git reset --hard` / `git checkout`/`restore` over working paths. Uncommitted working-tree state is live multi-agent work.
- **Never commit or push unless explicitly asked.**
- **Never double-open `task_state.csv`** (`open('w')` and `open('a')` on the same file at once) — it interleaves and corrupts the first data row (task000). Always read fully, modify, write once with `csv.writer(..., newline='')`.
- Durable knowledge goes in the project docs, not Claude auto-memory (this project's policy).

---

## The traps that cost time (so they don't again)

Trap 1 (now: staging the right onnx + sidecar) and trap 5 are still **yours**; traps 2–4 moved into **`submit.py`'s** domain (it does the poll + owns the zip) — keep those lessons alive *there*.

| # | Trap | Symptom | Fix |
|---|---|---|---|
| 1 | **Wrong onnx staged / missing sidecar** | you copy the wrong dir's onnx into `Staged_Boladi/`, or forget the `.json` (submit.py refuses a sidecar-less candidate) | `ls The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/ \| grep taskNNN`, copy the EXACT onnx; always write the sidecar with the verified cost + a sha256 that matches the bytes |
| 2 | **Grep-parsed poll** (submit.py) | a phantom score from a number inside a description; loop exits while still PENDING | parse `publicScore` by column, never grep |
| 3 | **Reconciling a surprising score** | a ±big LB jump | the sum(points)==publicScore tripwire → verify before trusting |
| 4 | **Touching the zip by hand** (submit.py) | a renamed zip / a dropped task000 | only `submit.py` writes `submission.zip`; never hand-build it |
| 5 | **Double-open CSV write** | `task000` row becomes ragged / non-numeric cost | single read, single `csv.writer(newline='')` write |

---

## Related

- Pipeline + the other roles: `Agenters/README.md`. **Upstream:** Tough Reviewer → Orchestrator send you SHIP packets.
  **Downstream:** the team `submit.py` is the actual ship.
- Durable lessons route to the **Librarian** (`Onnx_Advanced/` via `INDEX.md`) — the standing doc-update order above.
