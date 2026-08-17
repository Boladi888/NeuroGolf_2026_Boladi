# HandOff audit — actionable findings (2026-06-29)

**Status:** OPEN draft. Author: Claude (owner-directed audit).
**Scope:** the friend's product *and* the local deployment of it —
1. `Boladi_Skillz/HandOffz/handoff-workflows/` — the friend's Codex **skill package** (the deliverable he's developing).
2. `K-NeuroGolf/Handoff/` — our live deployment: runner code, configs, and the batch30/31/32 run logs.

**Companion doc:** `claude_handoff_cld_lane_findings_2026-06-28.md` already covers the broker/protocol layer (A1–A10) and the cld-lane bring-up. This audit is complementary — it does **not** re-argue those; where they overlap (board 403s, `worker_max_active`) it is marked **[confirms <id>]** with fresh runtime evidence. New ground here: the **skill docs as a product**, **runner code defects**, and **runtime throughput evidence**.

**Bottom line:** the pipeline *logic* works — workers produce real byte-equal candidates, reviewer gating + one bounded rework loop function, 3 verified wins surfaced. The problems are **operational throughput + product completeness**, not correctness of the idea. batch31 stalled at ~45% (9/20 packeted, then died ~03:10Z) and `cld_batch32` never actually ran through the runner. Highest-leverage fixes are P1 below.

## Implemented in the runner (2026-06-30, owner-directed)

The **runner-side** items below were fixed in code and verified (compile + two simulation harnesses + a real codex smoke test that wrote a file with no sandbox refusal + the rewritten monitor run on the real batch31 state). The **SKILL-package** items (SKILL-1…10) remain open for the friend; the worker-cost knob (RUN-5) is left to owner discretion.

| Finding | Status | What changed |
|---|---|---|
| **ENV-1** sandbox blocks writes | fixed + **verified** | codex 0.142.4 unelevated sandbox can't be fixed; runner now passes `--dangerously-bypass-approvals-and-sandbox` (`codex.bypass_sandbox:true`). Elevated / single-root paths also plumbed via `config_overrides`. |
| **RUN-1** no slot refill | fixed + sim-verified | new refilling concurrent scheduler (`run_scheduler`) keeps `worker_max_active`/`reviewer_max_active` full, refills on each completion, pipelines worker→reviewer |
| **RUN-2/3** failures strand tasks | fixed + sim-verified | bounded retry (`worker_max_failures`/`reviewer_max_failures`), fresh session on retry; a stuck slot can't wedge the batch |
| **RUN-4** 4h runaway timeout | fixed | `timeout_seconds` 14400→2700 |
| **RUN-8** state race | fixed | all state mutations + dumps under `state_lock` |
| **RUN-9** JSONL corruption | fixed | `_JSONL_LOCK` around appends (both lanes) |
| **RUN-10** corrupt state crash | fixed | back up corrupt file + recover to default |
| **RUN-11** orphaned child procs | fixed | `taskkill /T /F` tree-kill on timeout (both lanes) |
| **RUN-13** presence recursion | fixed | bounded reconnect loop |
| **RUN-14** bare `python` | fixed | `sys.executable` (both lanes) |
| **RUN-15/17** monitor fields / redaction | fixed | assignee fallback; `sk-ant-`/`Bearer` patterns |
| **MON-1** silent monitor | fixed + verified | emits `complete`/`interrupted`/`stalled`, exits on terminal |
| **MON-2/3** loop pid hygiene | fixed | loop owns pid file, stamps terminal state, survives transient failures w/ backoff |
| **NEW** tight rework loop | added | `max_rework_cycles` cap (needed because one run now drains without inter-step sleep) |
| **RUN-5** forced 2× worker pass | ✅ fixed (2026-06-30) | flipped `worker_attempts_before_review` 2→1 (single pass then review, ~halves worker cost) + `force_first_rework`→false (it was dead config) on both lanes, per owner |
| **RUN-7** cld lane serial | partial | claude lane got all safety fixes but stays serial (no scheduler); parallelizing is a follow-up |
| **RUN-12/16**, SKILL-1…10 | open | low-risk / friend's product |

Files touched: `Handoff/runner/handoff_runner.py`, `handoff_runner_claude.py`, `handoff_monitor.py`, `handoff_presence.py`, `scripts/Start-HandoffRunner.ps1`, `config/runner.json`, `config/runner_claude.json`, `runner/README.md`.

---

## P0 — fix first (cheap, high impact)

- **[RUN-1 / HIGH] Runner batch never refills until the whole batch drains — one slow worker idles every other slot.**
  `Handoff/runner/handoff_runner.py:760-777` (`run_worker_batch`): selects a fixed batch once, submits all to a `ThreadPoolExecutor`, and the `with` + `as_completed` loop blocks until *every* future finishes before `step()` returns. No mid-batch resubmission. With `worker_max_active:3`, two 5-min workers stuck behind one 76-min worker leave 2 slots idle for over an hour. This is the documented "long worker holds the batch open" failure mode — **confirmed in code**, and it explains why batch31 took >5h. **Fix:** keep a persistent pool + feeder — maintain a moving set of in-flight futures, `as_completed` over it, and call `select` on each completion so a freed slot refills immediately.

- **[RUN-2 / HIGH] Failure states are dead ends — no retry path.**
  `handoff_runner.py:498-538`, `629-633` (and claude `332-337`, `381-385`): `worker_failed`, `review_failed`, `review_parse_failed`, `needs_packet_fix` are never re-selected by `select_worker_batch`/`select_review_task`. A single timeout or non-zero exit strands the task permanently. **Live proof:** task076 timed out (Exit 124, 76 min, empty `final_text`), went `worker_failed` at 01:18Z, and was **never retried** through end of batch. **Fix:** bounded auto-retry (map `worker_failed`→`forced_retry` up to N) or an explicit re-queue transition.

- **[RUN-3 / HIGH] Stuck worker kills the batch with it.** task025 grabbed the (effectively serial) slot, hung, timed out at 488s; runner left it `worker_running` / `worker_attempts:0` / empty history and then stopped — 7 of 20 tasks never started (`todo`). This is RUN-1 + RUN-2 compounding: the last task wedged the slot and the loop died. **Fix:** RUN-1 + RUN-2 + a per-task watchdog that demotes a slot held past `timeout_seconds`.

- **[ENV-1 / HIGH] The Codex Windows sandbox refuses to run and blocks file writes — appears in 15+ of ~26 runs.**
  e.g. `logs/raw/runner/task076/worker_bo_ng_worker1/20260628T232817Z.stderr.txt`:
  `failed to prepare windows sandbox wrapper: windows unelevated restricted-token sandbox cannot enforce split writable root sets directly; refusing to run unsandboxed`, then repeated `Failed to write file ...prototypes\worker1_scratch_*.md` (Exit 1). `Failed to write file` recurs in ~17 files / ~40 occurrences. Workers burn turns retrying writes. **Fix (config side, `config/runner.json` `codex.sandbox`):** either run the codex CLI elevated so the restricted-token sandbox can initialize, or set a sandbox mode it can actually enforce on this box, or point worker scratch at a single pre-created writable root. This is the single biggest source of wasted worker turns.

---

## P1 — throughput & cost

- **[RUN-4 / MED→HIGH] Per-command timeout is far too generous; one hung shell burns 25–76 min.**
  8+ Exit-124 events across task025/076/173/286/324; worst is `command timed out after 4567816 ms` (76 min, task076). Combined with runner-level `timeout_seconds:14400` (4h), a wedged worker can hold a slot for hours. **Fix:** drop the codex per-command timeout to single-digit minutes and the runner `timeout_seconds` to ~30–45 min for floor batches; let RUN-2 retry rather than waiting hours.

- **[RUN-5 / MED] Forced 2× worker pass doubles xhigh compute on a floor batch.**
  `workflow.worker_attempts_before_review:2` + `force_first_rework:true` (`config/runner.json:46-47`): every task runs the worker twice before review even when the first pass is already shippable. On a tie/wall-heavy floor batch this is ~2× the most expensive (xhigh) compute for little gain. **Fix:** make the forced rework conditional (only on reviewer `KEEP_WORKING` or a measurable non-improvement), or set `force_first_rework:false` for floor batches.

- **[RUN-6 / MED] `worker_max_active` underused.** Current `config/runner.json` = 3, but monitor snapshots show `worker_running=1` in 5 of 8 checks — i.e. RUN-1 is masking the configured concurrency. The archived batch30 config even had it at `1`. **[confirms B2]** Fixing RUN-1 is the prerequisite for this knob to do anything.

- **[RUN-7 / MED] `cld_batch32` was never executed by the runner.** `private/runner_state_cld_batch32.json`: all 20 tasks `todo`, `worker_attempts:0`, empty history, written 22:32Z and abandoned. The 3 wins in `logs/cld_batch32_results_2026-06-28.md` came from a **separate/manual** run. **Fix:** the Claude lane runner (`handoff_runner_claude.py`) has *no batching at all* — `step()` runs exactly one worker + one reviewer per call, so the cld lane is fully serial regardless of the `workers`/`reviewers` arrays. Either drive it with an outer loop or port the (fixed) batching from the codex runner.

---

## P2 — runner robustness (data integrity)

- **[RUN-8 / HIGH] Shared-state mutation race under threaded batches.**
  `handoff_runner.py` mutates `self.state`/`task_state` (`:601-654`, `:682-757`) **without** `self.state_lock`; the lock guards only `json.dump` in `save_state` (`:232-237`). Concurrent mutation while another thread dumps the same dict → `RuntimeError: dictionary changed size during iteration` or silent lost updates. **Fix:** guard every read-modify-write with `state_lock` (or snapshot under lock before dump).

- **[RUN-9 / MED] Unsynchronized concurrent JSONL appends can corrupt `operations.jsonl`.**
  `append_jsonl` (`:94-98`, claude `:93-97`) does independent `open(...,"a")`/`write` per call; threaded workers can interleave lines (no lock, non-atomic on Windows). **Fix:** serialize appends with an instance/module lock.

- **[RUN-10 / MED] Corrupt/truncated state file crashes startup with no recovery.**
  `read_json` (`:41-45`, claude `:43-47`) only guards `FileNotFoundError`; a malformed JSON raises uncaught `JSONDecodeError`. **Fix:** try/except → back up the bad file and fall back to last-good/default.

- **[RUN-11 / MED] Process kill doesn't reap the child model process tree on Windows.**
  `proc.kill()` / `subprocess.run(timeout=...)` (`:453-455`, claude `:302-304`) only kills the direct child; the codex/claude CLI's child processes orphan. **Fix:** launch with `CREATE_NEW_PROCESS_GROUP`/a Job object, or `taskkill /PID <pid> /T /F` on timeout. (Matches the owner's global "watch artifacts, not processes" rule.)

- **[RUN-12 / MED] Captain re-digest / duplicate-submit risk.**
  `pending_packets()` (`:812-881`, claude `:465-501`) treats any packet without `probe_ok` as pending; `probe_ok` is set async by `captain_ctl.py`. If probing lags a loop iteration, the same batch is re-digested and (with `--allow-kaggle-submit`) potentially re-submitted. **Fix:** stamp packets `digested`/`submitted` immediately and exclude them, or use a marker/lock so a batch is processed once. **[related to B3]**

- **[RUN-13 / MED] `handoff_presence.py:86-90` unbounded recursion on persistent "no valid session".** On a permanent auth failure `call_tool` clears+reconnects+recurses forever → `RecursionError`/hang. **Fix:** bounded retry (1–2) then surface the error.

- **[RUN-14 / MED] Hardcoded `"python"` for the captain subprocess** (`:827-833`, claude `:474-478`) may resolve to the wrong interpreter/venv. **Fix:** use `sys.executable`.

- **[RUN-15 / LOW] Claude-lane monitor always shows `worker=None reviewer=None`.** `handoff_monitor.py:92-95` reads `worker_agent_id`/`reviewer_agent_id`, but the claude runner stores `assignee`/`reviewer`. **Fix:** fall back to those keys.

- **[RUN-16 / LOW] Fenced-JSON extractor truncates nested objects.** `r"```json\s*(\{.*?\})\s*```"` (`:145`, claude `:128`) stops at the first `}`, breaking any nested `captain_packet`. Currently masked by the `raw_decode` fallback (dead path). **Fix:** brace-balanced extractor or rely on `raw_decode`.

- **[RUN-17 / LOW] Redaction patterns inconsistent across the two runners.** Codex `redact()` (`:68-91`) lacks the `sk_…` pattern the claude runner has (`:70-90`); neither catches `sk-ant-…` or generic `Bearer <token>`. Redaction also covers only the op-log/history, not raw transcripts (those rely solely on gitignore). **Fix:** unify patterns; add `sk-ant-`/bearer coverage.

---

## P2 — monitoring & lifecycle (owner's own rules)

- **[MON-1 / HIGH] Monitor never emitted a terminal/crash line — the exact anti-pattern in `~/.claude/CLAUDE.md`.**
  `monitor_batch31.jsonl` has 8 checks at 40-min cadence ending 02:53Z, all with `runner_running=null, runner_pid=null`, while `monitor_batch31.latest.txt` still asserts `running=True`. The runner died ~03:10Z; the next check (03:33Z) never fired and **no complete/interrupted/crashed line was ever written**, despite progress flatlining (+2 ships in 2h40m). `handoff_monitor.py:196-211` records `running=False` but loops forever, never alerts on death or `updated_utc` staleness. **Fix:** emit a distinct terminal alert and exit when `pid_running` is false **or** `state.updated_utc` is stale >5 min — and watch artifact freshness, not pid liveness.

- **[MON-2 / MED] Loop PID files don't record terminal state.** `private/runner_loop.pid.json` and `monitor_loop.pid.json` have **no** `stopped_utc`/`running` field (only `presence.pid.json` does). All three PIDs (30604/6712/32116) are dead, but you can't tell from the files whether they finished or crashed. **Fix:** every loop wrapper should stamp `stopped_utc` + `running:false` (+ exit reason) in its pid file on exit, like presence already does.

- **[MON-3 / MED] `Start-HandoffRunner.ps1:54-58` — any single non-zero exit kills the persistent `-Loop`.** One transient failure ends the whole loop (and nothing updates the pid file → MON-2). **Fix:** catch/log per-iteration failures and continue with backoff; only exit on a real stop signal or repeated consecutive failures.

- **[SEC-1 / LOW] Board sync hit 403 Forbidden 8× consecutively (task029, 17:58–18:35Z).** `handoff_sync_failed … HTTP Error 403: Forbidden` across in_progress→verified. **[confirms A5]** — status transitions need the assignee key, not the owner bearer. Already captured in the companion doc; logged here as live confirmation.

---

## P1 — the friend's SKILL product (`handoff-workflows/`)

This is what the friend is actually shipping. An agent loading this skill today **cannot complete the core workflow** — two HIGH gaps:

- **[SKILL-1 / HIGH] The skill tells the agent to run `runner/handoff_runner.py` + `handoff_monitor.py` but never specifies, skeletons, or says to author them.** `SKILL.md` §Standard step 7, `references/runner-workflow.md`, and `setup-checklist.md` Folder Layout all assume these scripts exist; nothing defines their contract. An agent following the skill has nothing to run. **Fix:** add a runner/monitor contract section (inputs: runner_state file + agent envs; outputs: the documented status states + log files) or ship a minimal template, or explicitly say "you must author these to this spec."

- **[SKILL-2 / HIGH] The skill claims MCP/REST coordination but names zero MCP tools and exactly one REST endpoint.** `description` promises "create projects/goals/tasks, assign, sync, list to avoid duplicates," but `api-and-mcp-notes.md` only documents `/auth/pair/claim` — no path/method for create-project, create-task, task-sync, or list-for-dedup. **Fix:** enumerate the actual `handoff` MCP tool names (`register_agent`, `create_team`, `list_tasks`, `update_task`, `verify_task`, …) and/or REST method+path per operation. (The real tool surface exists — it's exposed as `mcp__handoff__*` — the skill just doesn't name it.)

- **[SKILL-3 / HIGH] Status vocabulary is inconsistent across three files with no crosswalk.** `setup-checklist.md` uses HandOff `todo/in_progress/pending_verification/verified/rejected`; `runner-workflow.md` says `pending_review` (≠ `pending_verification`); `monitoring-and-logs.md` uses internal `worker_running/forced_retry/ship_packeted/drop/worker_failed/…`. An agent will push the wrong status string to the board. **Fix:** pick one canonical HandOff status set, fix `pending_review`→`pending_verification`, and add a runner-state→HandOff-status crosswalk table.

- **[SKILL-4 / MED] `security-and-secrets.md` export exclusion is path-based but allows the owner token in a *root* env file.** Token told to live "in a root env file or private env file," yet exclusion only covers `Handoff/private/` — a root-file token escapes the exclusion and can be swept into exports/commits. **Fix:** mandate the owner token live only under `Handoff/private/` + gitignored; make redaction content-based, not just path-based. *(Note: our deployment's `HandoffLib.ps1` redaction is content-based and the sanitized export verified clean — so adopt that approach in the skill.)*

- **[SKILL-5 / MED] `api-and-mcp-notes.md` pair-claim flow is incomplete.** Shows `pair/claim` with `pair_code`+`machine_id` but never says where the code comes from, what `machine_id` is, or what the response returns (the agent key). **Fix:** document the full pair lifecycle and that the response yields the agent key to store in `private/<agent_id>.env`. **[relates to A10 "agents born orphaned"]**

- **[SKILL-6 / MED] `setup-checklist.md` Batch Setup example violates the skill's own namespacing rule.** Uses `bo_ng_worker1`/`bo_ng_reviewer1` (no `_cdx_`/`_cld_` tag) next to `bo_ng_cdx_worker2`; also `"worker"` (singular default) vs `"workers"` (array pool) precedence is undefined. **Fix:** namespace all sample IDs; document which field the runner reads as default vs pool.

- **[SKILL-7 / MED] `agents/openai.yaml` is thin and uses an unexplained `$handoff-workflows` invocation token** with only an `interface` block (no model/tools/triggers). May not register if Codex's loader expects more. **Fix:** verify the `$skill` syntax against the real Codex skill schema; add required manifest fields.

- **[SKILL-8 / MED] Almost no copy-pasteable commands — nearly all prose, one curl.** Forces the agent to re-derive every command. **Fix:** add copy-paste blocks for high-frequency actions (register agent, create/sync task, read runner state, tail logs).

- **[SKILL-9 / LOW] Reviewer-verdict → runner-state mapping left implicit.** `SHIP/KEEP_WORKING/DROP` ↔ `ship_packeted/forced_retry/drop` is clearly paired but never stated. **Fix:** state the 1:1 mapping.

- **[SKILL-10 / LOW] `address`/`key` substring redaction over-redacts.** Substring match nukes benign config keys, corrupting sanitized logs. **Fix:** anchor to known secret key-names + value patterns (`hot_*`, `ak_*`, hex, email), not broad substrings.

*Skill positives to keep:* credential model is consistent across all three references (owner→`Authorization: Bearer`, agent→`X-Agent-Id`+`X-Agent-Key`); base URL + `base-sepolia` default consistent; "don't paste a raw token as a shell command" correctly reinforced; the `description` field is keyword-rich and should trigger well.

---

## Secret hygiene (our deployment) — verified OK, one doc mismatch

- `Handoff/` is entirely git-ignored (`git ls-files Handoff/` empty; `private/`, `runner_state.json` all `check-ignore` positive). ✅
- Sanitized export + shareable logs scanned for `ak_`/`hot_`/`Bearer`/hex/password — **no leaks** (only hit was the README's own example grep). The `HandoffLib.ps1` content-based redaction works. ✅
- **[DOC-1 / LOW] README claim is wrong:** `Handoff/README.md:8` says "Live credentials stay in the ignored root `.env.handoff`," but real `ak_…` agent keys live in plaintext `Handoff/private/*.env` (git-ignored, so safe). **Fix:** correct the README to say agent keys live under `private/*.env`; reserve root `.env.handoff` for the bootstrap token only.

---

## Suggested order of attack

1. **ENV-1 + RUN-4** (sandbox writes + command timeout) — stops most wasted turns immediately, config-only.
2. **RUN-1 + RUN-2 + RUN-3** (refill + retry + watchdog) — the throughput core; unblocks `worker_max_active`.
3. **MON-1 + MON-2 + MON-3** (terminal-state + don't-die-on-one-failure) — so a stall is visible next time instead of silent.
4. **SKILL-1 + SKILL-2 + SKILL-3** — the friend's product can't be followed end-to-end without these; everything else in the SKILL list is polish.
5. RUN-8/9/10 (data-integrity) before raising concurrency further.
