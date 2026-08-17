# NeuroGolf shared test harness — PLAN

Status: **BUILT & TESTED** (2026-05-31). See `README.md` for usage. All 5 modes + battery + fill-corpus implemented in `tools/ng_gate.py` + `tools/ng_harness/`; tests in `tests/test_ng_harness.py` + `tests/test_ng_killjob_orphan.py` (11 pass under pytest & unittest). Decisions taken: N cap 200k, lazy-on-demand corpus + `fill-corpus` for eager all-400, psutil adaptive throttle, `models/` parent registry, JobObject kill-on-close per the Iarc_Recorder reference.

Goal: replace the ~50–80 hand-rolled per-task test scripts with one shared, CPU-efficient
harness that all ~400 agents call. Fixes the duplication and the CPU thrash.

---

## 0. Key paths (decided)

| What | Location |
|---|---|
| Harness code + this plan | `tools/ng_harness/` (repo, in git) |
| **Corpus cache (ARC-GEN pairs)** | **`Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache\`** (main BoBrain domain cache) |
| Parent/reference onnx registry | `models/` (repo) — reuse; convention `models/taskNNN.onnx` = current parent. NOTE: `.onnx` appears gitignored, so parents are a machine-local/shared artifact, not tracked. |
| Repo `data/` | a **Junction** → `<NOGIT_ROOT>\data` (corpus deliberately NOT placed here; goes to Q: per decision) |

⚠️ **Q: is a NEW data home OUTSIDE the existing junction convention.** Per `WHERE_DATA_LIVES.md`, five dirs (`.venv`, `data`, `submissions`, `local_profiler`, `BatchPilotArchive`) junction to `.notongit\NeuroGolf_nogit\`. The corpus on Q: is the first data location not following that pattern, so:
- **Build phase MUST add a row to `WHERE_DATA_LIVES.md`** documenting `Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache\` (agents read that file first; otherwise they won't find / will try to rebuild the corpus).
- Expose it via env var **`NEUROGOLF_ARGEN_CACHE`** (default = the Q: path), following the repo's `os.environ.get(VAR, default)` convention; register in `.env.example`.

Decided settings:
- **All 5 modes** (visible, ab, truth, synth, adversarial) + `battery`.
- **Full corpus** wanted (all 400 tasks eventually cached).
- **Self-throttle** allowed up to **~80% CPU** (including the user's other processes).

---

## 1. The 5 test archetypes agents run today (all hand-rolled, duplicated)

| Mode | Today's examples | Checks |
|---|---|---|
| `visible` | `tools/verify_against_live.py`, `final_battery.py` step 1 | candidate vs the 266 official `taskNNN.json` pairs (vs expected) |
| `ab` | `task324/ab_gate_*`, `task018/ab_test.py`, `argen_verify.py --ab` | candidate byte-equal to a reference onnx (parent / live / variant) |
| `truth` | `task110/final_battery.py` (N=100k), `task018/argen_verify.py` (200k sweeps) | candidate vs fresh ARC-GEN **ground truth** at scale |
| `synth` | `task324/synth_gate_*.py`, `*_task_shaped_synth.py` | candidate vs an independent numpy reimplementation of the rule |
| `adversarial` | `tools/adversarial_test.py` (exists, agents ignore it) | candidate vs live on transformed inputs (rotate/flip/recolor/shift) |
| `battery` | `final_battery.py` | the standard pre-ship combo → PASS/REVIEW + manifest-ready JSON |

Each re-implements the same 5 ingredients: task→generator resolution, grid↔tensor
encode/decode, ORT session setup, ARC-GEN generation, output comparison.

---

## 2. Why they burn CPU (root causes)

1. **ORT thread oversubscription** — only ~1/15 scripts cap threads; ORT defaults to *all cores per session*. N agents × all-cores = thrash. **(biggest, easiest)**
2. **Regenerating the ARC-GEN corpus every run** — generators use unbounded rejection-sampling `while True` loops; nothing cached. 400 agents × thousands of pairs = millions of redundant generations.
3. **Absurd N defaults** — `final_battery` N=100k, task018 200k sweeps. Divergence shows in the first few hundred; the rest is waste.
4. **Parallel seed-window processes** — task018 ran 4 concurrent processes, each all-core ORT. Multiplies (1)–(3).

---

## 3. Design

### 3a. One CLI: `tools/ng_gate.py`
```
py tools/ng_gate.py <mode> --task NNN --cand cand.onnx [--ref PATH] [--n N] [--full] [--json]
```
Modes: `visible | ab | truth | synth | adversarial | battery`.
Auto-resolves generator, official pairs, and parent onnx from `--task NNN` (no hardcoding).
`battery` runs the standard combo and emits PASS/REVIEW + a `neurogolf.staged_submit.v1`-ready JSON block.

### 3b. Library: `tools/ng_harness/`
- `__init__.py`
- `session.py` — **the** ORT session factory: `onnx.load` → `nu.sanitize_model` → `ORT_DISABLE_ALL` → **`intra_op_num_threads=1`, `inter_op_num_threads=1` by default**. All agents get the non-thrashing config for free.
- `encode.py` — canonical grid↔`[1,10,30,30]` encode/decode (wrap `neurogolf_utils.convert_to_numpy` + `(out>0)` decode). Kills the 60/40 hand-roll split. Vectorized batch encode for corpus.
- `registry.py` — `task_uuid(NNN)` via existing `arc_gen_enrich.task_num_to_uuid()` (sorted-UUID order, already correct); `generator(NNN)` (importlib); `parent_onnx(NNN)` (resolve `models/taskNNN.onnx` → else `submissions/submission.zip::taskNNN.onnx` → else `.hermes/board/staged_submits/`); `official_pairs(NNN)`.
- `corpus.py` — cached ARC-GEN corpus (see §4).
- `throttle.py` — machine-wide self-throttle (see §5).
- `modes.py` — the 5 mode implementations + battery, returning structured results.

### 3c. Reuse, don't reimplement
Wrap existing `tools/_score_v2.py`, `tools/verify_against_live.py`, `tools/adversarial_test.py` instead of duplicating.

---

## 4. Corpus cache (the biggest optimization)

ARC-GEN `generate()` is **deterministic given `random.seed(seed)`**, so the corpus is identical
across all agents/runs → generate **once total**, reuse forever.

- File: `Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache\taskNNN.npz` (np.savez_compressed).
- Stored arrays: input + output as **uint8 color-index grids** `[N,30,30]` (0-padded) + `shapes[N,2]` + the seed list. (uint8 grids, not one-hot float → tiny; encode to one-hot on load, vectorized.) Skips BOTH generation and (most) encoding cost.
- **Determinism**: fixed seed scheme (e.g. `seed = SEED_BASE + i`) recorded in the npz so any agent reproduces/extends the exact same corpus.
- **Fill strategy (full corpus)**: lazy-but-complete — a task's corpus is generated on first request (cap **N=200k**, covers the heaviest sweeps), AND a background `ng_gate.py fill-corpus` job pre-generates all 400 tasks. Incremental append if an agent needs more than cached.
- **Disk guard**: estimate ~50–100 MB/task compressed @ 200k → ~25–40 GB for all 400; assert free space on Q: before writing; measure real bytes/pair on task 1 and log it.

Open: confirm **N cap = 200k** (default) and **eager all-400 background fill** vs purely lazy.

---

## 5. Self-throttle (target ≤ ~80% CPU, machine-wide)

`throttle.py` provides a **cross-process** cap so many agents don't collectively saturate:
- Primary: every session is single-threaded (§3b), so 1 inference ≈ 1 core.
- Cross-process gate: a system-wide semaphore / lockfile sized to `floor(0.8 * cpu_count)` concurrent *inference workers* across ALL agents (named-mutex or a counter file in the cache dir with stale-PID reclaim). An agent acquires a slot before its inference loop, releases after.
- Optional adaptive backoff: sample `psutil.cpu_percent`; if total > 80%, sleep-spin before acquiring. Keeps headroom for the user's own work.
- Sane N defaults (`--n 2000` dev, `--full`/`200k` opt-in) + early-exit on first A/B divergence (byte-equal mode only needs *whether* it diverges).

---

## 6. `models/` decision
Currently task018-only (55 `task018_structural_v*.onnx`, zero other references). Repurpose as the
parent registry: `models/taskNNN.onnx` = current parent for `ab` mode (maintained by
`build_submission.py` / submit-captain). Move task018's version-spam into `task018/prototypes/`.
Caveat: `.onnx` gitignored → parents are machine-local (fine for single-machine multi-agent loop).

---

## 7. Rollout (phased, low-risk)
1. Build `tools/ng_gate.py` + `ng_harness/` + corpus cache; keep all existing scripts working.
2. Update agent playbook (`.ai_instructions/playbooks/`) to **mandate `ng_gate.py`**, **ban** hand-rolled harnesses, 100k+ N defaults, and parallel seed-window processes.
3. (Optional) sweep-delete the ~50 redundant per-task scripts after adoption.

---

## 8. Open decisions before build
1. Corpus **N cap** per task: 200k (default) or lower?
2. Corpus fill: **eager all-400 background** (true "full corpus") or lazy-on-demand only?
3. Throttle mechanism: named-mutex/semaphore vs lockfile-counter (both cross-process; pick by dependency tolerance — `psutil` available?).
4. `models/` as parent registry: confirmed reuse (per user). Migrate task018 spam out?

---

## 9. Integration with the existing `tests/` suite + repo conventions

There are TWO distinct test layers — keep them separate:
- **`tests/` (Alpha-owned unittest CI suite)** — 30 `test_*.py` covering the *tooling/orchestration* layer (submit-gate, board policy, path resolution, `verify_against_live`, spread patterns, detector/builder parity). Run: `python -m unittest discover -s tests -p "test_*.py"` (README claims 584/584). `test_detector_builder_parity.py` is the gate to promote a task to SOLVED (detector + Kaggle-faithful `convert_from_numpy` ONNX), keyed off `tools/inspect_tasks.py:SOLVED_TASKS`. **This is the build_submission layer, not the prototype layer.**
- **per-task prototype gates** — the ad-hoc agent scripts `ng_gate.py` replaces. These feed `.hermes/board/staged_submits/` → submit-captain. `ng_gate.py` is a *prototype-stage* gate; it does not replace the SOLVED parity gate.

**Requirements this imposes on the harness build:**
1. **Add CI coverage**: `tests/test_ng_harness.py` + `tests/test_ng_gate.py`, written in the existing style (resolve `NG_ROOT`/`DATA_DIR` via `os.environ.get(VAR, default)`, import `neurogolf_utils` + the module under test, plain `def test_*()` asserts). Must pass under `python -m unittest discover -s tests`. Cover: session factory caps threads; encode/decode round-trips vs `nu.convert_to_numpy`; registry resolves task→uuid/parent/pairs; corpus determinism (same seeds → identical pairs) + cache hit; throttle slot acquire/release + stale-PID reclaim.
2. **Don't break the tools the harness wraps** — `test_verify_against_live.py`, `test_submit_gate*.py` must stay green (so wrap, don't mutate, those tools).
3. **Follow env-var conventions** (`WHERE_DATA_LIVES.md` §"Env vars"): reuse `NEUROGOLF_DATA_DIR`, `NEUROGOLF_PYTHON`, `NEUROGOLF_MAIN_ROOT`, `NEUROGOLF_NOGIT_ROOT`; add `NEUROGOLF_ARGEN_CACHE`. Always `os.environ.get(VAR, default_under_root)`.
4. **Run agents via `${NEUROGOLF_PYTHON}`** (the junctioned `.venv` Python 3.14), not a bare `py`, for dep consistency across worktrees.
5. **Session-start docs**: `WHERE_DATA_LIVES.md` (filesystem map) and `CURRENT_STATE.md` (score/queue snapshot) are the two ~80-line files agents read first. The corpus + harness must be discoverable from `WHERE_DATA_LIVES.md` (see §0 ⚠️). Cross-link this PLAN from there too.

---

## 10. Visible window + kill-on-close (orphan safety) — REQUIRED

**Problem this fixes:** the exact task018 incident — an agent "finished long ago" but its `argen_verify` sweeps kept running detached, eating CPU until manually hunted down. Long sweeps and the corpus-fill job MUST NOT be able to outlive their owner.

**Requirement:** every `ng_gate.py` run (especially `truth --full`, `battery`, and `fill-corpus`) runs in a **visible console window**, and **closing that window — or any abrupt death of the launcher (taskkill, parent shell exit, crash) — kills the entire worker process tree.** No orphans, ever.

**Mechanism (port from the reference, don't reinvent):** `Q:\AIz\Intel_Arc_Stuff\Iarc_Recorder\start_iarc_recorded_run.py`:
- Windows **JobObject** created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK` (`_make_kill_job()`); the handle is held in a module global for the launcher's whole lifetime, so the kernel triggers KILL_ON_JOB_CLOSE on all bound children the moment the launcher exits *however it exits*.
- Every spawned worker/ORT process attached via `AssignProcessToJobObject` (`_attach_pid_to_kill_job(pid)`), with per-PID diagnostics + a fallback explicit `TerminateProcess` on shutdown for PIDs that couldn't be attached (AppX container / perms).
- `BREAKAWAY_OK` lets a small kill-watchdog spawn with `CREATE_BREAKAWAY_FROM_JOB` so it survives to clean up.
- Same pattern the user's TIMMY (llama-server) and benchmark harness already use → established, reuse it.

**Build tasks:**
- Factor the JobObject helper into **`ng_harness/killjob.py`** (lifted from the reference) so `ng_gate.py` and other tools share one implementation. `ng_gate.py` creates the kill-job on startup and attaches every child (worker procs in the throttle pool, the corpus generator, ORT sessions if sub-processed).
- **Visible window**: launch the work in a `CREATE_NEW_CONSOLE` window titled with `task/mode/N/progress` so the user can see it and close it to reclaim CPU instantly. (If invoked already inside a console, run in-place but still create+bind the kill-job.)
- **Port the reference's contract tests** into `tests/`: `test_orphan_kill` (kill the launcher → assert zero surviving children) and `test_clean_exit` (normal finish → zero leftovers), adapted to `ng_gate.py`.

**Interaction with §5 throttle:** the kill-job makes the cross-process throttle robust — when a window closes, its workers die AND its semaphore slots free. Stale-PID reclaim in the throttle counter is then only a backstop, not the primary cleanup.
