# Agenters — the NeuroGolf agent roles

Role playbooks for the multi-agent NeuroGolf cost-golf pipeline. Each role is a folder with a **`README.md`** (the full
playbook) and a **`SPAWN_PROMPT.md`** (paste-to-start bootstrap). Read this index first to see where your role sits.

## The pipeline

```
owner
  │
  ├─▶ Supervisor      scans project state, then starts one Orchestrator for an assigned task batch
  │
  ├─▶ Prompter        picks high-EV Boladi tasks, writes ≤2 KB worker prompts into Task Prompts/ (numbered folders)
  │
  ├─▶ Orchestrator    assigns prompts to workers, routes their output, enforces parallel-write safety
  │      ├─▶ workers          build candidates (task-local files only)
  │      └─▶ Tough Reviewer   attacks the result → SHIP / KEEP-WORKING / DROP
  │
  ├─▶ Submit Captain  vets SHIP candidates, own-stack batch-validates every ~5, stages to the team
  ├─▶ Janitor         cleans local scratch / loose artifacts so workers do not carry repo logistics
  │
  └─▶ Librarian       receives durable lessons, maintains Onnx_Advanced/ (the shared library)
```

The point of the chain: throughput **without** letting false floors, overfit/lookup candidates, dynamic-cost fiction, or
parallel file conflicts reach the team submission.

## Live lane model

Two Orchestrators may run at the same time only when the owner assigns disjoint task sets. Each Orchestrator writes its
own run folder and feeds a lane inbox; the downstream writers stay serialized:

- **one Submit Captain** drains `Agenters/Submit_Captain/Updates_Codex/` and `Updates_Claude/`, runs own-stack probes,
  and stages only `probe_ok` candidates;
- **one Librarian** drains `Agenters/Librarian/Updates_Codex/` and `Updates_Claude/`, integrates durable lessons into
  `Onnx_Advanced/`, and archives consumed lesson packets.

The lane names are inbox namespaces, not a requirement to use a specific model vendor. Do not create team-repo queues or
shared state files for this; role-local inbox files are the coordination layer.

## Orchestration modes

Default long-wave mode is **CLI-orchestrator-native-workers**: Supervisor launches a top-level `claude -p`
Orchestrator with subscription auth, pinned model, and an approved allow-list. Because that Orchestrator is top-level, it
can spawn native worker/reviewer subagents while Supervisor reads only ledgers and finals.

Use **main-session-native** for a short one-sitting wave where the active main session owns the ledger and native
subagents. If a spawned Orchestrator cannot spawn children, use explicit **parent-spawn** recovery or owner-approved
bounded **CLI-worker** fallback.

Do not silently blur these modes. Record the launch mode in the run folder before workers start.

## Run-integrity invariant

No positive expected delta may be left unresolved. Every candidate with positive expected gain must be terminal or
actively owned:

- `DROP`, `PROBE_OK`, `STAGED`, or `REJECTED`;
- `PROBING` with a probe tag and named owner;
- `EXTERNAL_OWNER` with the lane/agent and next action recorded.

`HELD` / queued is **not terminal**. It is only a transient inbox state. A held positive candidate blocks run closure
unless another lane explicitly owns the probe/stage action.

## The roles

| role | one-line job | owns | feeds |
|---|---|---|---|
| **[Supervisor](Supervisor/README.md)** | scan first, then start one Orchestrator | assignment packet + process proposals | Orchestrator |
| **[Prompter](Prompter/README.md)** | find the lever, write the prompt | task selection + the prompt folders | Orchestrator / workers |
| **[Orchestrator](Orchestrator/README.md)** | run worker waves | assignment, routing, parallel-write safety | Tough Reviewer → Submit Captain |
| **[Tough Reviewer](Tough_Reviewer/README.md)** | attack floor claims | SHIP / KEEP-WORKING / DROP verdicts | Orchestrator (→ Submit Captain) |
| **[Submit Captain](Submit_Captain/README.md)** | get good stuff to the team | lane inbox + own-stack validate + stage | the team `submit.py` / Librarian |
| **[Janitor](Janitor/README.md)** | keep local scratch contained | loose files, temp dirs, cleanup reports | Orchestrator / Submit Captain |
| **[Librarian](Librarian/README.md)** | keep the library clean | lesson inbox + `Onnx_Advanced/` maintenance | every agent |

## Model / service policy

Role *contracts* describe capability bands (pick by band, not vendor); the **launch tables** in Supervisor README + Orchestrator SPAWN carry the owner's per-runtime model pins. Honor the pinned model at spawn:

- **Highest-reasoning:** workers on novel ONNX/ORT tasks. Codex workers/reviewers should be `gpt-5.5` + `xhigh` + `service_tier="priority"` when the CLI accepts it. Claude workers/reviewers (owner 2026-07-10): default hard/novel work to Opus fast mode (`claude-opus-4-8` + `--settings '{"fastMode":true}'`), escalate known hard-floor tasks to Fable 5 (`fable`, the heaviest long-autonomous model); sonnet-tier for mechanical. ('Fable Ultracode fast' isn't a real profile — fast mode is Opus-only, not Fable.) The worker is where the solution is invented; do not cheap out there.
- **High-reasoning:** Tough Reviewer, Submit Captain, and major Librarian refactors.
- **Mid/high-reasoning:** Orchestrator when running many agents or resolving conflicts.
- **Low/cheap:** Janitor, but only under a narrow report-first cleanup contract.

Do not spend worker context on local logistics. Workers should solve one task; Janitor handles loose `task*.onnx`,
scratch dirs, stale backups, and cleanup reports. If an agent's context compresses, retire it with a handoff packet and
spawn fresh rather than letting long-context drift mutate the role.

Safety status is now part of the handoff, not an optional note. Submit Captain may stage without a Kaggle stack probe
only when a packet is explicit `LOCAL_GOLD`: Path-A byte-equal/inherited proof, parent source/hash, matching candidate
SHA, clean embedded `ng_safety.py` PASS with zero warnings/failures, `x86_required:false`, and
`kaggle_probe_required:false`. Path-B, fitted/approx, grader-gap, dynamic/profile-uncertain, x86-required, or
warning-bearing candidates are still valuable, but they are `PROBE_READY`.

## ⛔ Shared source of truth — every cost-using role reads this

Task costs come from **`K-NeuroGolf-OnnxlyFans/NG_OnnxlyFans_TaskStatus.xlsx`** (`Tasks` sheet: `task / assigned / owner /
cost / points`). The **`cost` is the TEAM-BEST** (the scorer = the cheaper of Boladi/Aditya per task); `owner` is who
built it. Verify against **`Team_Submission/Backend/manifest.csv`** (authoritative, auto-rebuilt by `submit.py`).

**NOT stale home baselines in `task_state.csv current_cost` or generated task notes** — those can trail team-best, often
**10–90× higher** than the team-best because Aditya already holds a cheap one. Computing EV against them is the
**home-cost trap** (it produced 5 false-positive finds in a single dig). Always route + compute EV against the
xlsx/manifest team-best; a candidate only pays if `target < team_best`.

At this phase, thin real gains matter. A `+0.05` to `+0.20` candidate that beats team-best is worth probing/staging; do
not park it for budget/efficiency reasons.

## ⛔ Shared doctrine — never accept a floor claim

A floor CLAIM tells you what's been **tried**, not what's impossible — proven wrong many times (task152: a 1,520
"floored" GridSample → 456; a sub-1k sweep dissolved ~30 "floored" tasks). The ONLY true floor is forced-component /
information-theoretic (a component every correct algorithm needs). A per-algorithm "no-lever" claim falls to a different
formulation. Every role that touches a floor claim defaults to skepticism.

## Conventions

- **Role-defining files:** each role is `<Role>/README.md` (playbook) + `<Role>/SPAWN_PROMPT.md` (bootstrap). Nothing else is role-defining.
- **Role-local operating files are allowed.** A role may keep its own CSV/JSON/logs/tools inside `Agenters/<Role>/` when that file supports the role's live work, e.g. Orchestrator reviewer leases. These files must have an obvious schema/header or be documented in that role's README.
- **Ownership boundary:** an agent may write its own role folder. It may read other role folders, but must not edit another role's folder or top-level shared `Agenters/` files unless the owner explicitly assigns that cross-role maintenance.
- **Multiple instances:** when two agents hold the same role, each gets a role-local run folder, e.g. `Agenters/Orchestrator/runs/<orchestrator_id>/`. Instance files are write-owned by that instance; sibling instances may read them for collision checks.
- **Shared skeleton** (every README): Mission → You own / you don't → The loop → Read first → Source of truth → Method →
  What NOT to do → Hand-off → Related.
- **Cite `Onnx_Advanced/` via `INDEX.md`** (the router survives the refactor; never cite a moving file path or bare `#N`
  without the INDEX as the entry).
- **Session hygiene (owner rules):** work in place — never create worktrees/branches, never `git pull`/`reset --hard`/
  `checkout` over working paths, never commit/push or submit unless it's your role AND the owner asked. No CLAUDE.md, no
  auto-memory — durable knowledge goes in the shared docs (Librarian owns `Onnx_Advanced/`).
