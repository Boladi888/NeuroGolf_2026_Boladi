run tools/enrich_task_state.py
query ONE row: `python tools/agent_helpers/show_task.py --task N` (task_state.csv is 1 MB / ~272k tokens — NEVER Read it whole)
run cherry13b test on double requested prompts
choose highest lb prompt

## Mandatory library lookup for in-repo prompts

Every repo-access worker prompt must include a compact lookup block before the asset list:

```bash
python Onnx_Advanced/library_db/lib_query.py status
python Onnx_Advanced/library_db/lib_query.py context NNN
```

When the angle changes, add `search "<rule/op/cost keywords>" --limit 8`; before relying on ORT dtype/op lore, add `op <Op> <DType>`; for lesson audits use `facts --task taskNNN` or `facts --status stale`. If `status` reports stale, rebuild `Onnx_Advanced/library_db/build_library_db.py` or fall back to cited Markdown. WebPrompt bodies are the exception: run the lookup author-side before filling RULE/HINTS, but do not paste repo paths into the browser prompt body.
## ⛔ TWO HARD GATES before writing any prompt (2026-06-09 — fixes the two recurring worker failure modes)

1. **GATING — every prompt MUST instruct: run `ng_gate truth --full` before declaring a candidate.** `<99% truth`
   AND not byte-equal to a correct LIVE = a **learning artifact, NOT a submission**. A lookup/memorizer passes 266/266
   *visible* but scores **0 on hidden** — `truth --full` (the generator distribution) IS the hidden proxy. Visible-pass
   is necessary, not sufficient. (traps.md §14e; recurred on task233/task192/task077/396/370.) Never stage a
   <99%-truth candidate as "best."
2. **ROUTING — prioritize by Aditya's manifest cost, NOT a stale flag.** Work tasks where the **team-best (manifest)
   cost is HIGH** (neither side has a cheap version yet — real, reachable LB upside: 382/017/341/295…). SKIP tasks
   **Aditya already covers cheaply** (288/132/059 = wasted ships; he holds the cheap side, our work nets ~0).
   ⚠️ `boladi_workable` AND `team_split` are STALE proxies — they flag this session's biggest WINS (007/022/093/281/
   300/365 are all `N`/`watch` yet shipped real gains). Rank by `Backend/manifest.csv` Aditya cost, not the flags.
   (Where WE hold the cheap side on a watch task — e.g. task158/233, aditya 166k/140k — it's still a real win.)

## ⛔ THIRD GATE (Path B / radical prompts) — preview-led anti-anchoring (2026-06-20)

**Workers inherit the live graph's representation** (10-ch F32 canvases, lookups, wide masks) when they read it first — so on heavily-golfed tasks the cheaper from-scratch formulation never gets found (task366: 198 structural versions in ONE representation). Every Path-B/radical prompt must force anti-anchoring:
1. **Cite `GUI/previews/taskNNN_preview.png` as a FIRST-read asset** (all 400 exist; regen: `python GUI/scripts/render_task_previews.py NNN --arc-gen 4`). Require a **visual-first pass** — rule / output-size relation / object-color roles / ≥2 ARC-GEN falsifications — and a **scratch cost sketch** (2-3 formulations with `memory+params` estimates + dominant intermediate + final-output family) BEFORE the worker opens any existing graph. The existing/staged ONNX is the **LAST** asset, not the first.
2. **Previews only help image-capable workers** → flag `image-capable worker REQUIRED` and route to a vision lane; for text-only lanes, paste a short visual synopsis instead.
3. The preview is a routing / rule-recognition aid, **NOT the oracle** — JSON/generator + `ng_gate truth --full` remain the correctness sources. (Drop these blocks for pure Path-A byte-equal micro-golf.)

## Strategic context (2026-06-01 — applies to EVERY prompt; read first)
- **Every task scores its full points except task000** (proven: reconciliation gap +0.002 + a HiddenBad group cherry13b-probe that dropped the LB by exactly the flagged tasks' points). **`task_state.solved_status` (NotSolved/HiddenBad) is STALE — ignore it.** There are **NO unlocks** — no broken-on-hidden task to "solve" for +full points.
- **The only LB lever is COST REDUCTION** (`points = max(1, 25 − ln(cost))`). Two routes:
  - **(A) Byte-equal compression** — narrow dtypes / crop / fold the LIVE graph; floor capped by its structure. Gate **`ng_gate ab`** (byte-equal-vs-LIVE).
  - **(B) From-scratch rewrite** — a *different, cheaper* ONNX for the same correct output. **Usually reaches the true floor, 10–100× below (A):** task215 69k→**110** (+6.45), task313 75k→2k, task110 76k→24k, task181 64k→850, task324 66k→18k. **Don't stop at byte-equal — if the rule is simple, rewrite it.** The biggest targets (task133 677k, task157 227k) almost certainly need (B).
- **Gating a rewrite (B)**: it is NOT byte-equal to LIVE, so do NOT gate `ab`. Gate on **correctness vs the true rule** — large fresh ARC-GEN (BUT first confirm the `ng_gate truth` generator mapping reproduces the task's *official visible* pairs; it's mis-mapped for some tasks, e.g. task255→`a64e4611`) — or the **hidden LB via submit-first** (captain ships; submits are free, Kaggle keeps highest, so a wrong rewrite costs nothing).

## Prompt-writing rules

**Reference live sources, never hardcode numbers that change.** Prompts written on day N go stale on day N+1 when the LB or costs move. Worker reads a stale "+14 LB" target and gets calibration-misled. Use this pattern:

| ❌ DON'T write in a prompt | ✅ DO write in a prompt |
|---|---|
| "Current Kaggle high: 6166.58" | "Current Kaggle high: see the latest `Logs/submissions_ledger.csv` row (never hardcode — `current_strategy.md` §doc-hygiene)" |
| "Canonical cost: 101,347" | "Canonical cost: extract from `submissions/submission.zip` + score via `tools/_score_v2.score_network_v2`" |
| "Predicted +14.9 LB" | "Predicted LB = `ln(canonical_cost / candidate_cost)`; calibrated to 1.0× per the ledger in `Logs/submissions_ledger.csv`" |
| "Unlock pattern 81.8% (9/11)" | "Unlock pattern rate: see `current_strategy.md` shipped-unlocks count + ledger" |
| "Active workers: task233, task255..." | "Active workers: see `tools/patch_task_status_manual_fields.py` ACTIVE_WORKERS dict" |
| "v17 Path A +0.32 LB" | "Most recent Path A ship for this task: see `task_NNN.md` Attempts table" |

**Number references that ARE safe to hardcode**:
- Calibration constants (Kaggle scoring formula `max(1.0, 25 - log_e(memory + params))`)
- Op-spec details (`QLinearConv-10`, `ArgMax-13`, `Cast` opset rules)
- Banned ops (`Loop, Scan, NonZero, Unique, Compress` — these don't change)
- Visible-pair counts at time of writing (e.g., "266 visible+arc-gen pairs as of YYYY-MM-DD")

**Pattern**: in every prompt, the worker should be able to run a single command to re-derive the numbers themselves. If the prompt body says "extract canonical from submission.zip, score it, then `ln(cost_before / cost_after)`", the worker independently arrives at the right calibration regardless of when the prompt was written.

**Cite `Onnx_Advanced/INDEX.md` as the worker's FIRST read** (the router) instead of dumping a long file list — it routes them to the 2–4 files they actually need. Then list the task-specific files on top of that. Agents don't reliably open subfolders, so name the exact file (e.g. `Onnx_Advanced/scoring_internals.md`), and remember every durable doc now lives flat at the `Onnx_Advanced/` root (the former `Research/` reference files were promoted up 2026-06-01).

**Scoring methodology callout (mandatory in every Path A prompt)**: Kaggle scoring measures `memory + params` from `tools/_score_v2.score_network_v2()` on the ORT_DISABLE_ALL profiler trace — **NOT ONNX file size**. Workers have shipped chains optimized for file size that don't move scorer cost (task029 v6/v7 family produced 60→59-node analysis with file size 1924 bytes but scorer cost 97,875 — wasted 5 days of compression work that gained +0.04 LB instead of the worker's claimed +3.96 LB). Quick check command for any candidate:

```bash
py -c "import sys; sys.path.insert(0,'tools'); from _score_v2 import score_network_v2; print(score_network_v2('YOUR.onnx'))"
# Returns (macs, memory, params). cost = memory + params.
```

## Shared verification tooling — `ng_gate` + the Q: corpus cache (cite as the GATE in every prompt)

There is now ONE shared per-task verification harness — **`tools/ng_gate.py`** (replaces hand-rolled ab_gate / synth_gate / final_battery / argen_verify). It self-throttles to ~80% CPU, is orphan-safe (Windows kill-on-close job), and reads a **deterministic ARC-GEN corpus pre-cached for all 400 tasks on the Q: drive** (`Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache`, env `NEUROGOLF_ARGEN_CACHE`). It resolves task→generator internally, so prompts no longer need to hand the worker a UUID.

Cite these as the GATE in every new prompt (the worker's machine must have the Q: drive mounted):
- `${NEUROGOLF_PYTHON} tools/ng_gate.py visible --task N --cand C.onnx` — vs LIVE on visible+arc-gen pairs (must be all-pass).
- `${NEUROGOLF_PYTHON} tools/ng_gate.py ab --task N --cand C.onnx` — byte-equal vs the shipped parent (Path A gold standard).
- `${NEUROGOLF_PYTHON} tools/ng_gate.py truth --task N --cand C.onnx --full` — ≥99% vs ARC-GEN ground truth (Path B); `--full` uses the 200k Q: cache (auto-extends).
- `${NEUROGOLF_PYTHON} tools/ng_gate.py battery --task N --cand C.onnx` — visible+truth+ab+score in one shot.
- Pre-warm once (long fire-and-forget; give each worker a **disjoint** `--tasks` range — two builds of the same task race the shared `taskNNN.tmp.npz`): `tools/ng_gate.py fill-corpus --tasks A-B --n 200000 --new-window`.
- **Corpus status (2026-06-04): ~396/400 tasks at the full 200k** (last slow fills running: 157/233/264/330). For refills use **`tools/ng_fill_hang_safe.py --all`** instead of raw `fill-corpus`: several generators **infinite-loop on specific RNG seeds** (076/324/018 confirmed — this is what froze earlier fills) and task157 has *healthy* 20–40s slow-tail seeds, so the filler needs the heartbeat watchdog + durable skip-set + time-based checkpoints that tool provides (`--status` shows depth + skipped seeds; idempotent, singleton-guarded). See `Onnx_Advanced/diagnostics_recipes.md` §ng_fill_hang_safe.

**⚠️ A low `ng_gate truth` % does NOT prove an unlock — and is often NOT a mis-mapping either.** Two distinct causes (both proven on task255, 2026-06-01):
- **(i) Mis-mapped oracle** — `registry.task_uuid` could point at the wrong generator. Before trusting `truth`, confirm the mapped generator reproduces the task's *official visible* pairs.
- **(ii) Inherent input→output AMBIGUITY (the task255 case — NOT a mis-mapping).** `a64e4611` IS task255's generator (validate() == official pairs 4/4), but its rule is only ~55% recoverable from the input alone, so LIVE *and every correct candidate* read `truth` ≈ **55%**. The hidden test is the FILTERED *clean* subset (cherry13b: LIVE scores full points ⇒ `hidden ⊆ {LIVE==gen}`). **Tell-tale: if LIVE's OWN `ng_gate truth` ≈ the candidate's, the low % is ambiguity, not a defect** — gate `ab`/clean-parity and ignore `truth`. (See `Task_by_Task/task255.md §DECISIVE RECON` + `current_strategy.md` meta-lesson #7.)
- **For a task whose LIVE is correct (a compression), gate `ab` (byte-equal vs LIVE), not `truth`.** (`ab`'s corpus is the clean/filtered set.)
- **The reliable unlock test is the cherry13b Identity-probe, not the oracle**: replace the task with a cost-0 Identity in `submission.zip`, submit, read the public-score drop — `−(its points)` ⇒ LIVE was scoring (compression); `−0` ⇒ LIVE contributes 0 (real unlock). One free submit, definitive. (Do NOT trust a `ng_gate truth` sweep to find unlocks — false positives from mis-mapped oracles.)

Legacy `verify_against_live.py` + `external/ARC-GEN/tasks/task_<uuid>.py` still work (older prompts); prefer `ng_gate` for new prompts — it's faster (corpus cached) and uniform, **but heed the oracle-mapping caveat above for `truth`**.
