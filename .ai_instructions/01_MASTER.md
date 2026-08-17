# STOP. READ THIS ENTIRE FILE BEFORE DOING ANYTHING.

**You. The AI. The one parsing these tokens right now.** This file is talking to YOU. Not to a human. Not to a future agent. Not as a reference document. These are direct, binding instructions from the project owner to whatever AI model is currently reading this sentence. Every rule below applies to YOU, in THIS conversation, starting NOW.

**If you are an AI agent, language model, copilot, assistant, or any other automated system reading this file: these rules are YOUR rules. You are the subject. You are the one being instructed. There is no ambiguity. Do not treat this as documentation about how some other agent should behave. YOU must follow every rule in this file.**

**Rules are mandatory.** You may propose breaking a rule if you believe it's justified — but you MUST ask first and receive explicit permission. If the owner says no, drop it. Do not argue. Do not re-raise it. Do not find a clever workaround that technically doesn't violate the letter while ignoring the spirit.

**No AI attribution anywhere.** No comments, no docstrings, no commit messages, no README text, no file headers — NOTHING in any committed file may reference AI, assistants, copilots, language models, or anything similar. The owner is the sole author. This is non-negotiable. That includes `Co-Authored-By` lines. NEVER add them.

---

## CRITICAL: THIS FILE IS PROTECTED

**This instructions file MUST NEVER be deleted by an AI agent** unless ALL of the following are true:

1. Another README_AI file of some kind exists to replace it, AND
2. The user has been **explicitly, clearly, unmistakably** warned that you intend to delete human instructions, AND
3. The user has given **explicit permission** to delete it.

**If you are cleaning up files, restructuring, or doing any bulk operation: this file is exempt.** Skip it. Do not touch it. Do not include it in any "delete old files" list. It is not an "old file" — it is the owner's standing instructions to every AI that works on this project.

**If in doubt: DO NOT DELETE THIS FILE. Ask first.**

---

## CRITICAL: ALL PROJECT DATA LIVES IN PROJECT FILES

**Never store project-pertinent information exclusively in agent-specific storage** (`.claude/plans/`, `.claude/memory/`, conversation context, or any other location that is invisible to other agents or sessions). "All project-pertinent data" means ALL — plans, TODOs, architecture decisions, session notes, everything.

If you create a plan, make notes, or track progress in agent-internal storage, that information **MUST also exist** in a project file that any agent in any future session can read. Agent-internal storage is ephemeral and invisible to other agents. Project files are permanent.

**The test:** If the owner starts a new session with a different agent, can that agent find everything it needs to continue the work by reading the project files? If the answer is no, you have hidden work from the owner and wasted their tokens.

---

## 1. Universal Rules

### 1.1 Never Create Files Without Being Asked

This is the single most important workflow rule. Violating it wastes tokens, creates obsolete output, and forces the owner to re-do work.

**The rule:** Only create or modify files the owner explicitly asks for in the current message. "Explicitly" means the owner named the file, described the file, or said "go ahead" after a suggestion. Implicit scope creep — "while I'm at it I'll also update X" — is **forbidden**.

**What you SHOULD do instead:** After delivering the requested file(s), list what else you think should change and why. Let the owner decide. They may have context you don't (e.g., they're about to upload a file that makes your update obsolete).

**Common violations to watch for:**
- Owner asks for file A → agent also creates/updates files B, C, D "for consistency"
- Owner asks to rename something → agent renames across every file it can find
- Owner asks a question → agent creates a file to answer it
- Owner approves a name/design → agent immediately builds everything touched by that decision

**If you catch yourself about to create a file that wasn't requested, STOP. Suggest it instead.**

### 1.2 Ask Before Doing — "While I'm at it" Is Forbidden

Do NOT modify files unless the owner explicitly asked you to modify that specific file.
Do NOT "update for consistency" across files you weren't asked to touch.
Do NOT rename things across the codebase when asked to rename one thing.

If you think something else should change: **say so, explain why, and wait.** The owner may have context you don't — an upload coming, a decision already made, a reason you can't see. Suggesting is good. Acting without permission is unacceptable.

**Deliver what was asked. Then suggest what else might need attention. That's the workflow. Every time.**

### 1.3 One Task at a Time

If there is a TODO list, a set of tasks, or multiple things that need doing — **work one task at a time.** Complete it, verify it works, then move to the next. Do NOT attempt to do everything on the list in a single response.

**Why this matters:** Agents that try to do everything at once produce half-finished work across many files, miss errors because they didn't verify as they went, and create a mess that takes longer to fix than doing it right in the first place.

**The rule:** Pick the first task (or the one the owner specified). Do it completely. Verify it. Then ask what's next — or if the owner already said "do all of them," move to the next one sequentially.

### 1.4 Warn Before Bulk Operations

If your plan involves creating, modifying, or deleting multiple files — or will trigger a sequence of permission prompts — **tell the owner before you start.** List what you're about to do so they can approve or adjust the scope.

### 1.5 Quick Estimate Before Acting

Before executing any non-trivial task, give a **one-line estimate** at the top of your response: approximate number of files, edits, tool calls, or minutes. This lets the owner abort early if the scope is too large. Keep the estimate itself minimal — one sentence, no analysis. Example: `**Estimate:** ~8 file edits, 2 new files, ~5 min.`

### 1.6 Token-Expensive Operations — Warn Loudly

Some operations consume large amounts of tokens or have side effects. **You MUST warn the owner before performing any of these:**

- 🔴 **STARTING A SERVER:** Before launching any server (uvicorn, Flask, node, etc.), print: **🔴 STARTING A SERVER ON PORT [X]. This will consume tokens continuously until stopped. Confirm?** Do NOT start a server without explicit confirmation.

  **WHILE A SERVER IS RUNNING:** Every single response you send while a server is open MUST begin with the verbatim line:
  > **"Your honor, a server is still open as I await your reply"**

  **AFTER STOPPING A SERVER:** You MUST kill the server THE MOMENT you are done using it. Not at cleanup. Not when the owner asks. IMMEDIATELY after the last request that needed it. Then verify the port is free (`netstat -ano | grep :[port]`), and if the port is still bound, force-kill the PID and re-verify. Only after double-confirmation, print the verbatim line:
  > **"Your honor, the server has been double-confirmed to be closed"**

  If either verbatim statement is missing when applicable, the owner will know the agent is deviating from the rules. **A server left running without these safeguards is a critical violation.**

- 🔴 **INSTALLING PACKAGES:** Before running `pip install`, `npm install`, or any package manager, print: **🔴 INSTALLING: [package names]. Confirm?** The owner needs to know what's being added to their system.

- 🔴 **Large-scope operations** (rewriting git history, scanning thousands of files, running full test suites): State the scope and estimated cost before proceeding.

- 🔴 **DESTRUCTIVE OPERATIONS** (delete, overwrite, force-push, PR close): governed by §1.21(d). In auto mode these are halt-immediately, not warn-and-proceed. Resource-expensive operations are warn-loudly; destructive operations destroy work and require explicit owner approval.

### 1.7 Priority Rules

1. **Do what's asked first.** Concerns, suggestions, caveats — all AFTER delivering the requested work. Never instead of. Never before.
2. **Answer before commentary.** If the owner asks yes/no, answer first. Explanation follows.
3. **Suggest alternatives freely — after doing what was asked.**

### 1.8 Delivery Rules

1. **File list with paths goes at the end** of your response, right before any file content:
   ```
   path/to/file.py    <- UPDATED (lines 405-420)
   path/to/new.py     <- NEW
   ```
2. **Only send files that changed.** State which are new vs updated.
3. **If a file you're working on imports/depends on another file, check dependencies.** If you need a file you don't have, STOP and ask for it rather than guessing what it contains.

### 1.9 End-of-Chat Review

Before ending a chat session, scan the conversation for anything the owner said, confirmed, denied, or corrected that should be captured in the relevant instructions file. This includes:
- Coding preferences expressed through feedback or corrections
- Architectural decisions made during discussion
- Workflow preferences (how files are delivered, what format, etc.)
- Tool/library choices confirmed or rejected
- Any repeated clarification that indicates a missing or unclear rule

**Do not silently add entries.** Propose changes and let the owner approve. The goal: the owner never repeats themselves across chat sessions.

### 1.10 Verify Your Own Work

- After every git operation, check the result across all affected branches, worktrees, and on-disk state. Don't just check one and assume the others are fine.
- Think through second-order effects. Before running a command, ask: "What else will this change?" A revert can delete files. A merge propagates commits. An edit in one worktree doesn't fix another. Anticipate consequences.
- If the owner gives direction or corrects you, that correction applies to ALL future work in the session, not just the immediate task.
- **Auto-mode addendum:** when verification fails, see §1.21(c) — investigate, document the failure, test the leading alternatives, and only halt if no path forward exists. Silent fix-and-continue on a failed verification is the violation.

### 1.11 Check What's Actually on Disk

**`.gitignore` whitelist patterns hide files from git.** Files on disk that aren't whitelisted are INVISIBLE to `git status`, `git ls-files`, and `git log`. Before assuming a module or file needs to be created, **`ls` the actual filesystem** (not the git index).

Previous agents have wasted enormous amounts of tokens re-building modules that already existed on disk because they checked git only. Always run `ls` (and any needed subdirectory listings) before writing a file you think is new.

**Gitignored tools (`.venv/`, `node_modules/`, local DBs, build artifacts).** When you need a tool that's typically gitignored, the discovery sequence is:

1. `ls` the current folder. The tool may be present but git-invisible.
2. If absent, `ls ../` and the project's main worktree (e.g., `<project>/` if you're in `<project>/.worktrees/<branch>/`). The tool may live there.
3. **If the tool exists in the main worktree but not yours**: halt and ask the owner. Do NOT silently write into the main worktree's tool from within a side-session — that contaminates the owner's main checkout and is invisible to `git status`. Per §1.24, the owner picks the resolution: (a) duplicate the tool into your worktree, (b) symlink across, (c) you set up a fresh tool inside your worktree, (d) paste content into chat.
4. **If the tool is missing everywhere**: halt and ask before creating it. Per §1.6, package installs are token-expensive and need owner approval.

**Never assume "missing" without checking siblings.** Most agents waste hours re-creating a `.venv/` that already exists in the main worktree, or worse, contaminate the main worktree's venv from a side-session.

### 1.12 Memory Writes Require Explicit Owner Permission

Auto-memory files (`~/.claude/projects/<key>/memory/*.md`, `MEMORY.md`, and similar agent-internal storage) are "files" under §1.1. Owner-invisible-by-default storage is **not** a workaround for the §1.1 prohibition. If the owner has not asked for a memory entry, do not write one — and "I noticed something I should remember" is not a request from the owner.

**If you encounter an auto-memory file created by a prior session:** port any project-relevant content into the appropriate project files per "ALL PROJECT DATA LIVES IN PROJECT FILES" (typically `03_PROJECT.md` for state/decisions, `Logs/lessons.md` for durable lessons), then delete the auto-memory file. Don't leave it sitting; it's invisible to the owner and to other agents.

### 1.13 ScheduleWakeup Is `/loop`-Only

Calling `ScheduleWakeup` outside an active `/loop` invocation is forbidden. The autonomous wakeup will resume work the owner did not authorize, possibly after the session has been halted.

### 1.14 Same-Turn Violation After Confirmed Reading = Halt

If the agent has enumerated which `01_MASTER.md` / `02_GIT.md` / `03_PROJECT.md` rules it has read in the session-start handshake, and within the same conversation chain it violates one of those rules and is corrected by the owner, the agent must HALT at that point and report — not "fix and continue." Same-turn violation indicates the rules-read confirmation was nominal, not real, and the rest of the session is suspect until the owner restarts or restates expectations.

### 1.15 Background-Job Polling Has Hard Limits

Once a long-running job is started with a notification mechanism (Bash `run_in_background`, Monitor, or a one-shot watcher), **stop polling its output file.** Repeated `cat`/`tail` of a buffered output wastes tokens for zero new information. If a mid-job signal is genuinely needed, the correct pattern is a one-shot Bash watcher that exits when a condition is met (e.g. `until [ $(stat -c%s file) -gt N ]; do sleep 10; done`) — not repeated polling.

### 1.16 Don't Silently Fall Back on Failed Preconditions

When the agent's plan depends on a precondition (a token being visible, a tool being installed, a file being present, a network call succeeding), and that precondition fails: **STOP and report to the owner.** Do NOT silently substitute a different approach, even one that seems clearly equivalent. The owner may want to fix the precondition rather than work around it; substituting hides the symptom and removes their choice. Reporting is cheap; quietly diverging from the agreed plan can waste tokens or produce results that don't answer the original question.

**Auto-mode addendum:** see §1.21(b). In auto mode, the equivalent of "STOP and report" is "investigate the cause, document findings, test reasonable alternatives, and halt with a diagnosis." Going to the owner with a tested workaround they can approve or reject is more useful than going with raw failure — but the choice is still theirs, not the agent's.

### 1.17 Present Action Choices as Numbered or Lettered Lists

When asking the owner to pick between options, structure them as a numbered (1, 2, 3) or lettered (A, B, C) list — never as a flat sentence or paragraph that requires the owner to write out which one they mean. The owner replies with "1,3,B" rather than typing the option text.

Apply this whenever the response would prompt the owner for a next-action decision. Even with two options, label them — makes "do both" / "neither" / "just A" replies trivial. Don't mix numbers and letters in the same list unless there's a structural reason.

### 1.18 No Stacked Pull Requests

Never open a pull request whose `base` is another open PR's branch. GitHub's "Merge" button merges into whatever the `base` says — if the base is a session branch nobody intended to keep, the merge lands in the wrong place, the PR shows MERGED but the content never reaches `main`, and the recovery is force-push and rebase coordination across the stack. Vanilla GitHub does not warn about this.

If a session needs multiple logically-distinct changes, the choices are:

1. **Same branch, multiple commits, single PR.** Reviewers still see the per-commit diff inside the PR. Default option.
2. **Sequenced PRs.** Open #1, wait for it to merge to `main`, *then* create #2 from the new `main`. Never create #2 while #1 is still open with #1 as the base.

This rule is universal — it applies regardless of whether the owner says "make it PR 26" mid-session. If the owner phrases a request in a way that implies stacking, ask before opening.

### 1.19 Ask Before Opening Any Pull Request

Before invoking `gh pr create` (or any equivalent), confirm with the owner: "Open as a new PR, or add as a commit on the existing branch?" Even when the owner has previously said "open a PR" — confirm scope, base, and whether to bundle with existing in-flight work. The cost of one confirmation is low; the cost of an out-of-scope or wrongly-stacked PR is force-pushes and rebases.

<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **MULTI-TRACK ADDITION — ignore if your project is single-track (a single-track project can later become multi-track)** 🟧🟧🟧

### 1.20 Lane Discipline in Multi-Track Projects

If the project's `03_PROJECT.md` declares parallel agent tracks (e.g., Alpha and Beta), the §1.1 / §1.2 "don't touch what you weren't asked" rule extends across lanes. Even with general permission to edit a file class, do not cross into another lane's owned files without explicit owner authorization for the current session. Lane assignments are declared in the project's lane-assignments subsection of `03_PROJECT.md`; consult that section before touching any file.

Shared-edit files (typically `03_PROJECT.md`, `Logs/lessons.md`, `01_MASTER.md`, `02_GIT.md`, `.gitignore`) require section-level interleaving on rebase, not whole-file overwrites — preserve the other lane's edits. Universal git workflow rules for multi-track projects (PR cadence per lane, lane-suffixed session-log filenames, parallel-rebase coordination) live in `02_GIT.md` §1.1 / §3 / §10.

🟧 **END addition**

</div>

### 1.21 Auto Mode Discipline

Auto mode is the unattended-execution toggle. The owner usually leaves it OFF for the first prompt of a session — that prompt is the kit-check / sanity pass — and turns it ON only after basic problems are handled. Once on, the agent runs without per-call confirmation gates. The rules in this section govern that interval.

**The principle.** Ambiguity is a signal to investigate, not a license to choose. Errors are diagnostic events, not workflow continuation events. Destructive actions and scope creep require an explicit halt regardless of how confident the next step looks.

**On the auto-mode toggle.** When the owner enables auto mode mid-session, acknowledge it in one line that names this section: e.g., `Auto mode on — §1.21 in effect.` No restatement needed; the named reference is the marker. From that point until auto mode is toggled off (or the session ends), every implicit decision the agent makes is captured per the logging clause below.

#### Decision-point checkpoints

Two response classes apply.

**Investigate-first triggers (preferred default).** The agent investigates the cause, documents findings in the session log, tests the leading alternatives, and only halts if no reasonable path forward exists. Going to the owner with a diagnosis + tested alternatives is more useful than going with a raw failure.

(a) **More than one reasonable approach exists for the current step.** Don't pick blindly. State the ranked alternatives, try the leading candidate, document what happened, try the next if the first failed. Log the decision and rationale per the logging clause.

(b) **A precondition failed** — a file missing, a tool unavailable, a config wrong, a network call denied. Investigate: is it actually missing or just not where expected? Check the on-disk filesystem (§1.11), check whether a sibling location holds it. Document. If a workaround is reasonable, take it and log it. If no clear path, halt with the diagnostic. This is the auto-mode refinement of §1.16.

(c) **A verification step failed** — test, smoke, type-check, lint, build. Investigate: regression, flake, environmental, stale cache, or actual bug? Re-run, narrow the scope, document. If a fix is obvious and within the originally-named scope, apply it and re-verify. If not, halt with the diagnostic in the report. This is the auto-mode-aware variant of §1.10.

**Halt-immediately triggers.** No investigation phase. No "I'll just confirm and proceed."

(d) **About to delete, overwrite, or force-push.** Halt unconditionally. State the destructive action, why, and present numbered alternatives (proceed / skip / different approach). Includes `rm`, `git rm`, file-tool delete, `git reset --hard`, force-push to any branch, closing or force-merging a PR, and any operation that destroys the only on-disk copy of work. Resource-expensive operations are governed by §1.6 (warn-loudly); destructive operations are governed by this clause and require halt. Deletes destroy work — that's the worse failure mode.

(e) **Scope grew beyond the originally-named files.** A file the agent is about to touch wasn't in the initial scope declaration → halt. Don't quietly extend the blast radius. If a dependency genuinely needs editing, halt and ask first.

(f) **Cumulative impact threshold reached** — more than 10 files modified, more than ~30 tool calls, or more than 1 working hour elapsed since the last owner input. Halt and surface a status update with the option to continue, change direction, or stop. Auto mode is not unbounded.

Tool calls and elapsed time count only when the agent is actively producing changes. Time spent waiting on a long-running external process (training run, test suite, ingest) does NOT count toward this threshold — those processes are governed by §1.23 (visible terminal window), and silence during the wait is correct.

#### The pivot rule

Investigation precedes any approach change. Before switching from Plan A to Plan B, the session log must contain (1) the diagnosis of why Plan A failed, (2) the alternatives considered, (3) the rationale for choosing Plan B. Bare "Plan A didn't work, switching to Plan B" is not a documented pivot — it's a quiet substitution and is forbidden under §1.16.

After one documented pivot, a second pivot (Plan B → Plan C) is a halt-immediately trigger; the original framing is wrong and continuing wastes everything that came after.

#### The two-unknowns rule

If the agent doesn't know two or more things required to make progress, halt and ask all of them at once. Don't ask about one and guess the others — that pattern produces work that's wrong on multiple axes simultaneously and the owner can't easily isolate which guess broke it. One missing fact may sometimes be safely assumed; two or more means the spec is wrong.

#### Implicit-decision logging

Every time the agent makes an implicit decision while auto mode is on (any (a)/(b)/(c) trigger fired and the agent investigated + chose a path without owner input), one bullet goes into the session log's `## Implicit decisions made` block at session end. Format:
    - [trigger letter] short description — chose X because Y. Alternatives considered: A, B.
If the count exceeds ~3 in a single session, surface that fact in `Where I left off`: either the prompt was under-specified or the agent should have been halting more aggressively. Both are signals worth catching at handoff.

### 1.22 Gitignored / Untracked Stores Must Be Merged to the Owner's Working Copy at Handoff

Auto-worktrees and per-session sandboxes can accumulate gitignored or untracked data (SQL DBs, replay logs, sentinel dirs, generated artifacts, ingest indexes, submission zips) that the owner's main working copy never sees. PR review, merge, and worktree cleanup do not propagate this data unless it is committed, copied, or archived. The session that creates it is responsible for merging it back at handoff; this is part of the session-end checklist, not optional cleanup. Skipping it silently destroys hours of work the next time the owner returns to the main worktree (or the auto-worktree gets garbage-collected post-PR-merge).

Untracked files shown by `git status` are covered by this rule too. If they are not being committed, preserve the complete set in the owner's main working copy under a clearly gitignored handoff/archive path and record a manifest + checksum in the session log; do not curate the set silently.

The full handoff protocol — what to merge, how to verify the merge is safe (superset check on row counts / file sets / line counts), when to halt and report instead of overwriting — lives in `02_GIT.md` §11.

**Project opt-out.** If a project's gitignored stores are cheap to regenerate (e.g., `node_modules/` after `npm install`, `.venv/` after `pip install -r requirements.txt`), the project's `03_PROJECT.md` should state that explicitly so the agent can skip the merge for those stores. Default is "merge unless opted out."

### 1.23 Long-Running Processes Get a Visible Terminal Window

Any process the agent expects to run for more than 10 minutes must be launched in a NEW terminal window that pops up on the owner's desktop — NOT in the agent's own shell, not as a background `&` job, not via `screen`/`tmux` (no GUI surface). The owner needs to be able to glance at the window and see live progress.

**On Windows**, use one of:

- `start cmd /k "<command>"`
- `start powershell -NoExit -Command "<command>"`
- `start wt -p PowerShell powershell -NoExit -Command "<command>"`

Requirements for the spawned window:

1. **Visible, not minimized** — `start` with no `/min` flag is the default.
2. **Stays open after the command completes** — `/k` (cmd) or `-NoExit` (powershell) so the owner can read the final state.
3. **Banner line at the start** — print a recognizable header (e.g., `=== TRAINING RUN: neurogolf onnx-build, ETA ~25 min, started 14:32 UTC ===`) so the owner can identify the window at a glance.

The agent still tracks completion — typically via a sentinel file the spawned process writes, or by tail-watching a log file — but does NOT poll the spawned shell's stdout directly. Per §1.15, repeated polling of a running job's output wastes tokens; the spawned terminal is for the owner's eyes, not the agent's.

If a process is shorter than 10 minutes but produces interesting interim output (training metrics, ingest progress), the agent may still spawn a window when the owner has previously asked for visibility on similar processes. When in doubt, ask.

### 1.24 Stay in Your Home Folder

The agent's "home folder" is the directory the owner placed it in. Possibilities:

- A project's main checkout (e.g., `Py_cx/<project>/`)
- An auto-worktree (e.g., `Py_cx/<project>/.worktrees/<branch>/`)
- The canonical `AI_Instructions/` cockpit folder

Reaching outside that boundary is forbidden by default — applies whether the destination is a peer project, a sibling folder, the canonical cockpit, or (most subtly) the main worktree of the same project from inside an auto-worktree session.

**Two patterns this rule prevents:**

**Cross-folder reach** (canonical ↔ project, project ↔ peer project, peer ↔ peer). The manual-distribution model is the deliberate alternative to a bundle script — each side owns its own files. To propagate a change across the boundary, generate a portable prompt (per `playbooks/handoffs/`) for the owner to paste into the receiving agent's session. Never edit across folders directly.

**Worktree ↔ main-worktree contamination.** From within an auto-worktree session, never write to a path that resolves into the main worktree. The most common failure: the agent needs a gitignored tool (typically Python's `.venv/`), doesn't find it inside the worktree, finds it in the main worktree, and silently writes there — installing packages, mutating data files, or activating that venv and `pip install`-ing into it. That contaminates the owner's main checkout from a side-session, and the contamination is invisible to `git status`. See §1.11 for the gitignored-tool-discovery sequence.

**Common violations:**

- Editing project files from a canonical session "for consistency"
- Activating the main worktree's `.venv/` from within an auto-worktree's shell, then `pip install`-ing
- Reading a sibling project's files when the task says "just review this project"
- Running git commands that affect another worktree (always check `pwd` and `git worktree list`)

If you need a file or tool that lives outside your home folder, halt and ask. The owner has options you can't see — copy across, symlink, fresh setup, paste content into chat, or temporarily authorize a cross-folder operation.

### 1.25 Proposed Master Changes Override Master (Strengthening Only)

`proposed_master_changes.md` (in each project's `.ai_instructions/`) is read at session start as part of the tier-0 read order, immediately after `01_MASTER.md`. **Active proposals in that file override the corresponding master rules for THIS project until the proposal is accepted to canonical.**

**Direction restriction.** Active proposals can only ADD constraints or STRENGTHEN existing rules. They cannot WEAKEN or REMOVE master rules. An agent encountering a proposal that weakens a master rule must halt and surface to the owner — weakening requires explicit owner action at canonical, not a project-side patch.

**Trust boundary.** The owner curates this file at overview sessions. Agents add proposals during sessions per the file's append-only protocol. Once added, a proposal is effective immediately for the project; canonical sync happens later. **Accepted stubs** (the lower section of the file) are historical record only — those rules are already in master.

**Reading order.** Read `01_MASTER.md` first to establish the baseline. Then read `proposed_master_changes.md` to apply local patches. Treat the patched ruleset as the effective rules for this session.

---

## 2. Git & Version Control Safety

Previous agents have caused hours of cleanup by ignoring these rules. Do not be the next one.

### 2.1 The .gitignore Is Sacred

Repos use a **whitelist `.gitignore`** by default — it starts with `*` (ignore everything) and only `!`-whitelists specific files. The `.gitignore` itself is normally NOT whitelisted and is NOT tracked in git. **Exception:** shared-collaborator repos may track `.gitignore` so the collaborator's clone has a consistent file-visibility surface; see rule 1 below.

**Rules:**
1. **Default to LOCAL-ONLY `.gitignore`** in single-developer repos — `*` covers everything anyway, and any AI-related whitelist entries leak. **NEVER `git add .gitignore`** in a single-developer repo. **Exception: shared-collaborator repos.** If multiple developers clone the repo and need a consistent file-visibility surface, `.gitignore` IS tracked. The project's `03_PROJECT.md` declares the convention used. Untracking a tracked `.gitignore` in a shared repo requires explicit owner permission and a plan for the collaborator.
2. **NEVER add `.claude/` or any AI-related entries to `.gitignore`** — the `*` already covers everything. Adding AI-specific entries leaks information about your existence. **Exception for shared-collaborator repos under the rule-1 exception:** a whitelist line for an AI-instructions file (e.g., `!README_AI_<KEY>_v*.md`) is allowed despite the mild leak. The owner accepts this trade-off as the cost of letting the collaborator have project context.
3. **`!.gitignore` whitelist line is allowed only when `.gitignore` is tracked under the rule-1 collaborator exception.** Otherwise omit it.
4. **If you edit `.gitignore`, restore it to the original content when done.** Each project has its own canonical `.gitignore` content.
5. **The `.gitignore` must exist on disk in every worktree** or VS Code will show every local file as untracked under the Commit button. If you delete it, recreate it immediately.
6. **FIRST ACTION in any new worktree: check `.gitignore` exists.** Claude Code creates worktrees without copying `.gitignore` (since it's untracked). Your very first command in any worktree — before doing ANY other work — must be to check if `.gitignore` exists and create it with the canonical content if it doesn't. No exceptions.
7. **Claude Code auto-creates `.claude/settings.local.json` in worktrees.** Without `.gitignore`, this file shows up under VS Code's Commit button.
8. **Never track anything under `.claude/`.** If you see `.claude/` anything staged for commit or listed in `git ls-tree`, that is a bug — untrack it immediately with `git rm --cached`.

### 2.2 Check Branch State Before Creating PRs

Claude Code often auto-pushes worktree branches at session start. The owner may merge a PR between sessions. **Before creating a PR, always check whether the branch has already been merged into master:**

1. `git fetch origin`
2. `git log origin/master..HEAD --oneline`

If there are no commits ahead of master, the branch was already merged. Do NOT attempt to create a PR — tell the owner it's already merged and they just need `git pull` in their main folder.

### 2.3 Worktrees Have Separate Working Directories

Each git worktree has its own copy of on-disk files. If you edit `.gitignore` in the main worktree, another worktree is unaffected (and vice versa). **Check and fix ALL worktrees, not just the one you're in.**

### 2.4 Git Operations — Think Before You Act

1. **Before any git revert, reset, or destructive operation:** trace the full impact. What files will be created, modified, or DELETED on disk?
2. **After any git operation:** verify the result. Check `git status`, `git ls-tree`, AND the on-disk state.
3. **Check ALL branches that could be affected.** Commits propagate through merges.
4. **VS Code shows untracked files as committable.** The owner's VS Code must show ZERO untracked files at all times. This is your responsibility.
5. **NEVER force push to `main` without explicit permission.**
6. **After deleting a branch (local + remote), run `git fetch --prune` in the MAIN worktree** (not just the current worktree). Then tell the owner to reload VS Code.

### 2.5 Commit Messages

- No `Co-Authored-By` lines. No AI references. No exceptions.
- Write commit messages as if the owner wrote them.
- Keep them concise and descriptive.

---

## 3. Owner Profile

### Hardware (Primary Target)

- Intel i7-13700K (24 threads)
- 64GB DDR5 5600MHz
- **Intel Arc A770 16GB** — actively used for ML workloads. **No CUDA**: Unsloth, Flash Attention 2, bitsandbytes 4-bit, NVIDIA Apex are all unavailable on this GPU. Three working backends (in order of preference): **PyTorch + XPU** (`torch 2.11+xpu`, upstream Intel GPU support — closest to "CUDA but Intel"), **OpenVINO** (Intel's inference toolkit, production-grade for *running* models on the Arc), **PyTorch + DirectML** (any DX12 GPU, easier install, less efficient, some ops fall back to CPU). Canonical setup recipe + a `verify_ml.py` validator script: `K-Orbit Wars/README_ML.md` (project-agnostic; mirror into other ML projects as needed).
- NAS on Gigabit Ethernet (TrueNAS Scale via SMB)
- Windows 11 Pro, Developer Mode enabled

#### Hardware capabilities — what the Arc CAN and CANNOT do

- ✅ Inference of an 8B-parameter LLM in bf16 (16 GB just fits weights; ~14 GB used).
- ✅ Training a 1–3B model with LoRA fine-tuning.
- ✅ Standard CV / RL / small-network training at speeds competitive with mid-tier NVIDIA cards.
- ❌ Pre-fine-tuning a 7B+ model from scratch (weights alone fill VRAM in bf16; no headroom for activations, optimizer state, gradients).
- ❌ Anything requiring CUDA-only kernels (Unsloth, FA2, bnb 4-bit) — has to be re-implemented in vanilla PyTorch.
- ⚠️ For projects that need to train large LLMs, plan on cloud rental (H100/A100, ~$50–300 one-time) OR start from a published fine-tuned checkpoint.

#### ML environment convention

Projects that need ML use a **two-extra-venvs pattern** alongside the project's main `.venv`:

| Folder | Backend | Purpose |
|---|---|---|
| `.venv/`     | none | Mirrors the production runtime (Kaggle, etc.); minimal deps |
| `.venv-ml/`  | PyTorch+XPU + OpenVINO | Heavy ML stack; primary training/inference |
| `.venv-dml/` | PyTorch+DirectML | Alternate GPU path; conflicts with XPU torch so kept separate |

All venvs are gitignored. The full install recipe + driver requirements + known gotchas (OpenVINO-GPU and torch+xpu cannot share a Python process; first XPU call has ~6 s JIT compile penalty; IPEX has ABI mismatch with torch 2.11 — leave it uninstalled) live in `K-Orbit Wars/README_ML.md` and should be mirrored into any new ML project.

### Preferences

- **Author:** boladi — https://github.com/Boladi888/
- **OS:** Windows 11 only unless stated otherwise
- **Distribution:** Prefers single-file or minimal-file projects where practical
- **Dependencies:** Conservative — don't add dependencies without permission

---

## 4. Coding Patterns

These apply to all coding projects unless a project section below overrides them.

### Style
- f-strings for all formatting
- Comments explain WHY not WHAT
- Specific exception types in except blocks — no bare `except Exception`

### Error Handling
- Never crash on recoverable errors
- Every external call (DB, API, I/O, network) in try/except with specific exceptions
- Degrade gracefully: failed external service → show what you have, warn about what failed
- Guard clauses at entry. Cascading fallbacks: best method first, fall through, fail cleanly
- Document fallback chain in comment

### SQLite Patterns
- WAL + PRAGMAs on every connection: `journal_mode=WAL`, `synchronous=NORMAL`, `temp_store=MEMORY`
- Thread-local connections via `threading.local()`
- Every DB call in `try/except sqlite3.Error`
- JSON columns for extensible data
- Parameter version tracking in `cache_params` — bump version → auto-invalidate
- Bulk ops via temp tables + JOINs
- `DROP+CREATE` not `DELETE FROM` for large clears
- `PRAGMA wal_checkpoint(TRUNCATE)` after bulk ops
