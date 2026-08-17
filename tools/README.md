# tools/

Analysis utilities + build pipeline helpers. Each `*.py` here is a standalone CLI script.
Archived/retired scripts live in `tools/_archive/` — if a doc references a tool that isn't
here, check there before assuming it's gone. (Refreshed 2026-06-06; lane-era labels dropped.)

## Bootstrap for new agents

```bash
python tools/agent_helpers/context_pack.py --task N  # ~14 KB curated pack: state, rule, don't-dos, prompt, staged, cited §s
python tools/agent_helpers/show_task.py --task N     # just the one task_state row
python tools/state.py --terse                        # one-line repo snapshot
```

**Per-task helper catalog:** `Onnx_Advanced/diagnostics_recipes.md` ("when you want X, run Y") —
read it before writing a new diagnostic script; it probably exists in `tools/agent_helpers/`.

## Maintenance / invariants

- `repo_lint.py` — standing invariants from the audit cycles (dead tool refs, retired-system refs, hardcoded paths, dup §-anchors, token-bomb banners, oversize task journals, prompt contract, staging hygiene). On-demand; writes `Logs/lint_last.md`; exit 1 on errors. Workers: `--only task-md-size --task N` before session close.

## The core loop (what actually ships)

- `ng_gate.py` — THE verification harness (visible / truth / ab / fill-corpus). Self-throttles; caches corpus on Q: drive. See `WHERE_DATA_LIVES.md`.
- `_score_v2.py` — canonical local scorer: `score_network_v2(file) -> (macs, memory, params)`. Static graphs price exactly; dynamic graphs are fiction (§39).
- `swap_and_submit.py` — captain's ship tool: swaps one task ONNX into submission.zip (validates 401-entry invariant) + submits.
- `build_dispatcher_task.py NNN` — builds one task's dispatcher-picked payload from `build_submission.py` (grep `_task<NNN>_` there; NEVER read it whole — 965 KB).
- `agent_helpers/` — per-task diagnostics (extract_task, inspect_graph, largest_intermediate, diff_graphs, find_op, dump_intermediates, adversarial_against_live, show_task, ...). Catalog: `Onnx_Advanced/diagnostics_recipes.md`.

## Adversarial / hidden-risk checks (which one is canonical?)

- `agent_helpers/adversarial_against_live.py` — **canonical per-candidate check**: synthetic adversarial inputs vs the LIVE extract, agreement % + verdict.
- `silent_fail_check.py` — arc-gen pass-rate gate across the whole submission (`--min-arcgen-N`); use for package-level sweeps, not per-candidate.

## Analysis / triage (alphabetical, live tools only)

- `_scan_new_patterns.py` / `_spread_audit.py` — spread-knowledge scans across all 401 tasks.
- `arc_gen_enrich.py` — pulls fresh ARC-GEN ground-truth pairs (independent oracle).
- `batch_stage_polish.py` / `spread_polish_sweep.py` — batch math-equivalent polish passes.
- `cascade_targets.py` — ranks rewrite targets by cascade-pool membership + cost headroom.
- `catalog_organic_prototypes.py` — inventories per-task prototype dirs.
- `compact_mask_analyzer.py` — static analysis of channel-redundant initializers.
- `cost_edge_probe.py` — validator/onnx-tool cost-formula edge-case prober.
- `cost_predictor.py` — fast cost model used during dispatcher selection.
- `dispatcher_audit.py` — verifies the cost-aware dispatcher picks the cheapest matching pattern.
- `inspect_tasks.py` — visualize task pairs; `--summary`; `SOLVED_TASKS` literal lives here (advisory — see SCHEMA.md note).
- `lut_density_check.py` — LUT-pattern file-size sanity check.
- `mine_new_angles.py` / `mine_rule_based_detectors.py` / `mining_lib.py` / `task_inspect.py` — rule-mining sweeps + shared helpers.
- `onnx_repair.py` — triages external ONNXs (READY / BANNED-OP / DYNAMIC-SHAPE / ...).
- `promote_task_work.py` — copies prototype work from scratch dirs into git-tracked per-task folders (preserves work).
- `rotate_runs_log.py` — rotates `Logs/runs.jsonl` when above threshold.
- `scan_hash_mlp_signatures.py` — probes canonical ONNXes for the §28 hash-MLP fingerprint.
- `scan_unsolved.py` — sweeps the unsolved set for tractable targets.
- `uncashed_prototypes.py` — finds prototype work on disk that never shipped.
- `verify_submission.py` — full-zip sanity: every entry loads + scores; cost/points tally.

Retired to `_archive/` (2026-06-05/06): external_blend, blend_diff, karnak_xref, mine_karnak_targeted,
audit_external_cascade, all_local_external_sweep, split_alpha_lanes, harvest_alpha_b, hermes-era tools,
task014/153_synthetic_guard, `_score_zip_to_json` (superseded by targeted `_score_v2` loops), and more.

## Cross-worktree usage

`data/` is junctioned (see `WHERE_DATA_LIVES.md`). From an external checkout, set:

```bash
export NEUROGOLF_MAIN_ROOT="/path/to/K-NeuroGolf"
export NEUROGOLF_DATA_DIR="${NEUROGOLF_MAIN_ROOT}/data/competitions/neurogolf-2026"
python tools/<tool>.py [args]
```
