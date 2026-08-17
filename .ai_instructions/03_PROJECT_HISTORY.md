# 03_PROJECT_HISTORY.md — closed-item audit trail

> Read on demand only. This is the durable record of completed work moved out of `03_PROJECT.md` to keep that file ~120 lines. The session log at `Logs/sessions/<date>_<file>.md` is the per-session narrative; this is the per-item summary ledger.
>
> *Established: 2026-05-10 reorg. Last entries: 2026-05-05.*

---

## Closed-item ledger

| Date | Lane | Type | Item | Outcome | Source |
|---|---|---|---|---|---|
| 2026-05-01 | Both | TODO | Phase 0 — competition spec ground truth | Done; verified facts in `Research/00_COMPETITION_SPEC.md` (license / disallowed ops / shape / scoring formula sourced from `neurogolf_utils.py`) | `Logs/sessions/2026-05-01_0343_phase-0-kaggle-research.md` |
| 2026-05-01 | Both | TODO | Phase 1 — acquire materials (Kaggle CLI, data, starter notebook, pinned deps) | Done; 5.59 MB → `data/competitions/neurogolf-2026/` | same |
| 2026-05-01 | Alpha | TODO | Phase 2 — minimum-viable submission (`build_submission.py` validator-pass) | Done; identity-weight build at 3,772 bytes | same |
| 2026-05-01 | Alpha | TODO | Phase 3 — submit apple #1 | Done; PR #3 merged; baseline shipped | same |
| 2026-05-01 | Alpha | TODO | Submit banana #2 (50.00) — 2 transpose tasks | Done | session 2026-05-01 |
| 2026-05-01 | Beta | TODO | First research pass — A1–A4 + B1–B6 + 20 J topics | Done; 4 briefs filed; per-topic notes in `Research_Archive/notes/` | session 2026-05-01 |
| 2026-05-01 | Beta | Fixed-bug | Color-perm partial mappings produced non-permutation Gather indices | Fixed in PR #6 (`70d217b`); detector validates `len(set(indices)) == 10` | `Logs/lessons.md` 2026-05-01 entry |
| 2026-05-01 | Beta | TODO | Cost-model verification (Where=0, OneHot=0, params=element count, memory=value_info bytes) | Done; documented in `04_DATA.md` Cost-model section | sessions 2026-05-01 + lessons |
| 2026-05-02 | Alpha | TODO | PR #16 — alpha pattern set (+392.40 pts; 11 new patterns; 13/400 → 40/400) | Done | session 2026-05-02 |
| 2026-05-02 | Alpha | TODO | PR #19 — post-merge `runs.jsonl` verification (+400 entries) | Done | session 2026-05-02 |
| 2026-05-02 | Alpha | TODO | PR #20 — audit fixes B1/B2/R1/R3/R4/E1/E2/S1/S3/S5 (+10.62 pts) | Done | `Logs/sessions/2026-05-02_2101_audit-fixes.md` |
| 2026-05-02 | Alpha | TODO | PR #21 — deep-audit D1–D12 (+1.78 pts; cost-aware dispatcher; 4 new tools) | Done | `Logs/sessions/2026-05-02_2203_deep-audit-fixes.md` |
| 2026-05-02 | Beta | TODO | Build `tools/silent_fail_check.py` | Done; pre-submit verifier with `--min-arcgen-N N` flag | `Logs/sessions/2026-05-02_2330_beta-silent-fail-check.md` |
| 2026-05-02 | Beta | TODO | Build `tools/cost_predictor.py` | Done; ~1 ms/task; 0.00% delta vs `score_network` | session 2026-05-02 |
| 2026-05-02 | Beta | TODO | Deep-read `multi-source-onnx-solver.ipynb` | Done; 28 candidates → `05_BACKLOG.md` | `Research_Archive/notes/deep-read-multi-source-onnx-solver.md` |
| 2026-05-02 | Beta | TODO | Deep-read `locally-solved.ipynb` (160 hand-coded) | Done; 5 confirmed `[B→A]` handoffs added | `Research_Archive/notes/deep-read-locally-solved.md` |
| 2026-05-02 | Beta | TODO | Deep-read `logic-driven-ensembe.ipynb` | Done; ~50 lines reusable repair logic catalogued | `Research_Archive/notes/deep-read-logic-driven-ensembe.md` |
| 2026-05-02 | Beta | TODO | Mine new fingerprint slices (a / b / c) | Done; 5 new `[B→A]` items in `05_BACKLOG.md` Tier 1 | `Research_Archive/notes/mine-fingerprint-slices-2026-05-02.md` |
| 2026-05-02 | Beta | Dismissed | `Compress` exploit bypass-of-banned-op-check | Dismissed; 3 hypotheses tested, all rejected. Catalogued but not exploited per ethical-play stance | `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits` |
| 2026-05-02 | Both | TODO | Switch to shared-collaborator (blacklist) `.gitignore` (PR #13) | Done; all instructions / research / logs now tracked | session 2026-05-02 |
| 2026-05-02 | Both | TODO | Submit grape #7 — first blend (3,261.15 LB, +3,045.82) | Done; PR #23 | `Logs/sessions/2026-05-03_0125_grape-blend-translation.md` |
| 2026-05-02 | Alpha | TODO | Translation pattern (`[B→A]` from deep-read) | Done; **0 net new hits** but kept as primitive (12 unit tests; cost-aware dispatcher preserves cheaper match) | same |
| 2026-05-02 | Owner | TODO | Decide file-sharing model for `.ai_instructions/03_PROJECT.md` | Resolved by PR #13 — shared-collaborator pattern bulk-add | session 2026-05-02 |
| 2026-05-03 | Both | Cleanup | Prune merged local + remote branches (PR #22, PR #24) | Done; 2 local + 1 remote orphan deleted | `Logs/sessions/2026-05-03_0200_prune-merged-branches.md` |
| 2026-05-03 | Both | TODO | PR #25 — AI-instructions overhaul (00/01/02/playbooks drop-in + new 04-07 siblings + 03 restructure 503→179 lines) | Done; merged at `22dd858` | `Logs/sessions/2026-05-03_0220_ai-instructions-overhaul.md` |
| 2026-05-03 | Alpha | TODO | PR #26 — color-replace + pixel-perm + shape-changing-pixel-perm (+8 SOLVED, +128 pts) | Done; merged at `ff4fdcb` | session 2026-05-03 |
| 2026-05-03 | Beta | Closed | Color-swap (2-color) `[B→A]` from deep-read | Demoted; pre-flight mining showed 0 unsolved hits before implementation | `05_BACKLOG.md` Demoted; session 2026-05-03 |
| 2026-05-03 | Both | TODO | Submit honeydew #8 — blend (organic 48 incl color-replace/pixel-perm/scpp + Apache/CC0 externals) | Done; **scored 3,275.37 LB** (+14.22 vs grape) | session 2026-05-03 |
| 2026-05-03 | Both | TODO | Tier-3 R&D pre-flight mining + enclosed-region-fill pattern (PR #27) | Done; pre-flight mined 6 candidates, 4 dropped at 0 hits, J.22 enclosed-region-fill landed (+2 SOLVED: tasks 002 + 251, +22.97 organic pts) | (this session log) |
| 2026-05-03 | Beta | TODO | Beta-track session 3: new mining angles + blend_diff tool + 2 deep-reads (PR #28) | Done; merged at `d8532eb`. `tools/blend_diff.py` shipped | `Logs/sessions/2026-05-03_1630_beta-mining-blend-diff-deepreads.md` |
| 2026-05-03 | Both | TODO | Handoff polish (post-PR-#28-merge) — README + 03/04 staleness fixes + branch cleanup | Done | `Logs/sessions/2026-05-03_1713_handoff-polish-post-pr28.md` |
| 2026-05-03 | Both | TODO | Beta-independence package (cross-lane prep): backlog transcription + `tools/mining_lib.py` shared helpers + SOLVED_TASKS drift-fix + `tools/task_inspect.py` | Done | (this session log) |
| 2026-05-03 | Beta | TODO | Beta 20-hr session: 5 new tools + 5 task recipes + 4 deep-reads + bbox-fill characterization | Done | `Logs/sessions/2026-05-03_1830_beta-20hr-session.md` |
| 2026-05-03 | Alpha | TODO | Alpha pattern set #2 — 7 new patterns (+10 SOLVED, +142.42 organic pts) | Done | session 2026-05-03 |
| 2026-05-03 | Alpha | TODO | Submit imbe #9 — organic 941.44 (after pattern set #2) | Done; **scored 889.03 LB** | same |
| 2026-05-04 | Both | TODO | PR #35 — 3-pass deep audit | Done; +18.21 organic LB | `Logs/sessions/2026-05-03_2200_audit-deep-pass.md` |
| 2026-05-04 | Both | TODO | Submit jackfruit #10 — blend (60 organic + Apache/CC0 externals) | Done; **scored 3,284.81 LB** (+9.44 vs honeydew) | session 2026-05-04 |
| 2026-05-04 | Alpha | TODO | Pattern set #3 (PR #36) — 5 new patterns + 1 picker variant | Done; +7 SOLVED, +93.26 organic pts | same |
| 2026-05-04 | Both | TODO | Submit kiwi #11 — blend | Done; **scored 3,353.19 LB NEW HIGH** | same |
| 2026-05-04 | Both | TODO | 40-hr autonomous push: 3 patterns + external pool refresh + 4 fruits | Done. +3 organic SOLVED (kron-block-up / const-largest-bbox / const-framed-cell). 4 submissions reaching 4043.11 | `Logs/sessions/2026-05-04_1057_both-40hr-push.md` |
| 2026-05-04 | Both | TODO | 40-hr push #2: 3 complex organic patterns | Done. +3 organic SOLVED, +37.27 LB pts | `Logs/sessions/2026-05-04_1700_both-40hr-complex-targets.md` |
| 2026-05-04 | Both | TODO | 20-hr Beta-prep push: 2 patterns + external pool refresh + 4 Beta recipes | Done. +2 organic SOLVED, +24.82 LB pts | `Logs/sessions/2026-05-04_1830_both-20hr-beta-backlog-fill.md` |
| 2026-05-05 | Beta | TODO | 40-hr Beta session (PR #40): 4 recipes cracked + task071 progress | Done. task389/task206/task263/task174 each at 266-267/267 | `Logs/sessions/2026-05-05_0008_beta-40hr-session.md` |
| 2026-05-05 | Both | TODO | focused-dhawan rule-rewrite recovery | Done. Diagnosed Kaggle rule rewrite (2026-05-04 18:27 UTC: MACs out, profiler-trace memory). +2 SOLVED. **All 14 prior submissions ERRORed under new rules** | `Logs/sessions/2026-05-05_0030_focused-dhawan-rule-rewrite-recovery.md` |
| 2026-05-05 | Both | TODO | 20-hr Beta-backlog-fill: 8 NEW Beta recipes filed + cushion-fix re-eval | Done. 8 truly-unsolved tasks have detailed recipes | `Logs/sessions/2026-05-05_1606_both-20hr-beta-backlog-fill.md` |
| 2026-05-05 | Both | TODO | Beta-60hr implementation push (PR open): task174 detector fix + 4 new ONNX patterns + CASCADE_FLOOR dead code removal | Done. SOLVED +4 → 105 | `Logs/sessions/2026-05-05_1841_git-cleanup-and-task174-tie-fix.md` |
| 2026-05-10 | Both | TODO | PR #60 — data layout reorg via Windows junctions | Done; 1.6 GB moved from repo tree to `.notongit/NeuroGolf_nogit/` with junctions. Live zips tracked. `WHERE_DATA_LIVES.md` added. | `Logs/sessions/2026-05-10_1530_git-cleanup-and-data-layout-reorg.md` |

> **Closed mining attempts** (~17 patterns confirmed unproductive across 2026-05-01 → 2026-05-02 + 2 in 2026-05-03 PR #26 mining + 8 more in 2026-05-03 pattern-set #2 mining + 35+ in 2026-05-04 pre-flight rounds A-HH; only smallest_bbox picker yielded a hit on task036): see `05_BACKLOG.md` "Demoted / dropped from backlog" table.

> **Cascade-risk task list (historical stub set, pre-fig11 ladder):** `{1, 2, 16, 20, 45, 53, 65, 73, 98, 129, 130, 143, 171, 172, 179, 210, 211, 220, 239, 240, 241, 242, 243, 251, 256, 257, 258, 259, 260, 261, 262, 276, 282, 283, 294, 309, 331, 337, 346}` (39 total). The current fig11 ladder has supplanted this stubbing strategy; current don't-touch list lives in `CURRENT_STATE.md`.
