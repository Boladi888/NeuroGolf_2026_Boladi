# K-Arc2 cross-reference — what ARC research already exists in the sister project

> **DISCOVERY 2026-05-26**: The user has a sibling project at `<WORKSPACE_ROOT>\K-Arc2` for the **ARC Prize 2026 / ARC-AGI-2 Kaggle competition** (different from NeuroGolf 2026; same ARC puzzle family but different scoring rules). Its `Research/` folder contains 40+ numbered notes covering 2020-2026 ARC research, including substantial post-Jan-2026 work that is outside Opus 4.7's training cutoff. Its source code includes Python implementations of techniques directly applicable to NeuroGolf, especially **rule disambiguation / discriminating-input synthesis**.
>
> **Implication for NeuroGolf research**: Cards G1, G2, G3 (added 2026-05-26 this session) are REDUNDANT — K-Arc2 already answers them. Cards moved to `Todo/Declined/` with this cross-reference as the disposition.

## K-Arc2 in one paragraph

K-Arc2 is the user's workspace for the official ARC Prize 2026 Kaggle competition (`arcprize.org`). That competition is a $725K-guaranteed event using the ARC-AGI-2 dataset (released May 2025 by Chollet et al., `arXiv:2505.11831`). The K-Arc2 stack uses Qwen3-4B + LoRA TTT + AIRV (Augment-Infer-Revert-Vote) ensembles, scoring ~5.83% on the Kaggle public LB as of 2026-04-30 (Boston build). It has its own Research/, code, eval datasets, and submission pipeline. NeuroGolf 2026 is a DIFFERENT competition with a DIFFERENT scoring rule (`max(1, 25 - log_e(memory + params))` static ONNX, no LLM at inference) but the underlying ARC puzzle data is shared family.

## Why ARC-AGI-2 matters for NeuroGolf specifically

Per K-Arc2 note 02, ARC-AGI-2 introduces FOUR new difficulty axes vs v1:

1. **Symbolic interpretation** — objects in the grid stand for *something other than themselves*; meaning defined within the task
2. **Compositional / multi-step reasoning** — apply rule R, then rule R' to the output, possibly multiple times
3. **Contextual rule application** — same rule but *which way* it applies depends on context
4. **In-context symbol definition** — symbols whose meaning is established within THIS specific task

These axes likely characterize our remaining trap-active tasks:
- **task255** (corridor-fill): suspected axis 2 (compositional) or axis 4 (in-context symbol if color 3 is "established by demos")
- **task366** (two-half scene/library object copy): suspected axis 1 (symbolic — anchor colors stand for "match by count") + axis 2 (compositional — anchor cardinality drives ordering)
- **task157** (canonical hash-MLP, untouched): need axis classification before rule-decode attempt
- **task319** (canonical, untouched): need axis classification

## Direct mappings: NeuroGolf question → K-Arc2 answer

### Self-consistency trap / rule disambiguation (G2-equivalent)

| Question | K-Arc2 source |
|---|---|
| How to generate synth inputs that distinguish candidate rule hypotheses? | `rule_disambiguation.py` — LITERAL implementation. Takes a candidate rule + alternative rule hypotheses, verifies each against demos, returns whether candidate is unambiguous (no alternatives also fit). Threshold K configurable. |
| How to filter candidates by consistency? | `consistency_filter.py` |
| How to filter via synthetic perturbations? | `synthetic_perturbation_filter.py` |
| How to filter by confidence after AIRV? | `confidence_filter.py` — three-tier (z-score dominance / single-pred min_votes / ambiguous-pair separation) |
| Symbolic regression techniques for rule search | `Research/notes/35_Symbolic_Regression.md` — Pareto-front parsimony, age-fitness diversity, ε-lexicase per-case selection, e-graph dedup, semantic backprop. Code in `pareto_front.py`, `lexicase_select.py`, `age_fitness_select.py`, `semantic_backprop.py`. **None of these are in NeuroGolf's current pipeline.** |
| Stability-under-augmentation as candidate selector | `Research/notes/06_Algorithmic_Toolkit.md` — ARChitects 2024 winning insight. D4 × S_10 augment-infer-invert-vote. |
| Adversarial perturbation filtering (ARC-TGI / NVARC SDG) | `Research/notes/14_RE_ARC_v2_Synthetic_Data.md` |

### ARC-AGI-2 rule family taxonomy (G3-equivalent)

| Question | K-Arc2 source |
|---|---|
| What's new in v2 vs v1? | `Research/notes/02_ARC_AGI_2_vs_ARC_AGI_1.md` — full delta with citations to Chollet paper |
| Chollet's 4 core knowledge priors | `Research/notes/03_Core_Knowledge_Priors.md` |
| Empirical v2 stats (grid sizes, color use, train pairs per task) | `Research/notes/04_Local_Eval_Profile.md` — 67.5% same-shape, 49/120 multi-test, median 16×16 |
| Per-task axis classification for 120 eval tasks | `Research/notes/17_Eval_Task_Axis_Tagging.md` + `derived/eval_axis_tags.csv` |
| Cross-task similarity clustering | `Research/notes/18_Task_Family_Clustering.md` + `derived/eval_task_clusters.csv` |
| Per-task human time-to-solve (hardness signal) | `Research/notes/19_Human_Time_To_Solve.md` — HF dataset `arcprize/arc_agi_2_human_testing` |
| Failure-mode taxonomy per architecture | `Research/notes/21_Per_Architecture_Failure_Modes.md` |

### Post-Jan-2026 ARC research (G1-equivalent)

| Question | K-Arc2 source |
|---|---|
| April 2026 competitive intelligence | `Research/notes/26_Mid_Contest_Approaches_20260426.md` — TRM-family operator race, HRM mechanistic critique, CoreThink 30.8% on v2 |
| May 2026 frontier-LLM scores on v2 | `Research/notes/36_Frontier_Models_v2_Refresh.md` — GPT-5.5 xHigh 85.0%, Gemini 3 DT v2 84.6%, Claude Opus 4.7 Max 75.8% |
| MindsAI 2025 source code (CodeT5-660M, MIT) | `Research/notes/37_MindsAI_2025_Source_Review.md` + `primary_sources/arc_2025_mindsai/` (12 MB clone) |
| CoreThink 30.8% v2 paper (April 2026, arXiv 2604.02434) | `Research/notes/38_CoreThink_Compositional_NeuroSymbolic_Study.md` |
| Boston recipe + NVARC playbook | `Research/notes/39_Boston_Recipe_Archive.md` + `Research/notes/41_NVARC_L4x4_Source_Review.md` |
| Logic profiling techniques | `Research/notes/44_logic_profiler_review.md` |
| CompressARC paradigm | `Research/notes/46_compressarc_paradigm.md` — likely interesting for NeuroGolf's compression focus |
| Turbo-DFS with augmentation | `Research/notes/48_turbo_dfs_aug32.md` |
| EDA features survey | `Research/notes/49_eda_features_survey.md` |
| 2024-2025 winner code-level distillation | `Research/notes/12_Code_Level_Reading_2024_2025_Winners.md` |
| Top approaches 2020-2025 history | `Research/notes/05_Top_Approaches_2020_to_2025.md` |
| Postmortems 4th-10th place 2024 | `Research/notes/13_Postmortems_4th_to_10th.md` |

## Actionable items for NeuroGolf right now

**Highest-EV transfer from K-Arc2** for our remaining unsolved tasks at 6197.00 LB:

1. **Port `rule_disambiguation.py` semantics into our pre-ship gate**. Currently our synth gate is "candidate matches X random inputs with ≥99% accuracy against own numpy reference". Add: "candidate is the ONLY rule from a set of K alternatives that fits 265/265 visible". This directly addresses the self-consistency trap. K-Arc2 default K=1 (any-alternative-fits → reject), with optional K=5 lenient mode.

2. **Classify each unsolved task by v2 axis** (notes 02 + 17). Manual axis tagging for task157/task255/task319/task366 before next rule-decode attempt. Hypothesis: tasks failing the self-consistency trap (task255, task366) likely have axis-1 or axis-2 patterns that need decomposition-aware decoding.

3. **Apply semantic backpropagation** (`semantic_backprop.py` + note 35) to task366 specifically. Given a partial program with one unknown subtree, invert the executed program from the desired output back through the known operators. Could surface what the "anchor cardinality" rule actually computes.

4. **CoreThink Compositional Neuro-Symbolic** (note 38, arXiv 2604.02434, April 2026) hit 30.8% on v2 public WITHOUT TTT or LLM. The paper is post-Jan-2026 and may have ARC-style compositional reasoning techniques applicable to fixed-graph ONNX. Worth a focused read.

5. **CompressARC paradigm** (note 46) — name suggests compression-oriented ARC solving, directly aligned with NeuroGolf's scoring. Worth a focused read.

## What does NOT transfer

K-Arc2 is designed for a runtime that allows LLMs (Qwen3-4B, 4×L4 GPUs, 12 hours). NeuroGolf has ZERO inference budget for LLMs — only static ONNX graphs scored on memory+params. Therefore the following from K-Arc2 do NOT apply to NeuroGolf:

- TTT (test-time training) — needs runtime fine-tuning, banned in NeuroGolf
- AIRV (Augment-Infer-Revert-Vote) — needs multiple LLM inference passes
- Any Qwen/CodeT5/Mistral runtime — needs LLM at inference
- Per-task LoRA — same
- Most of the TRM/HRM operator race — needs runtime
- Frontier-LLM scores — not relevant to our zero-LLM constraint
- Pass@2 hedging — NeuroGolf is single-attempt

What DOES transfer: rule-decoding methodology, synth gate design, v2 family taxonomy, post-2024 algorithmic insights (Pareto front, semantic backprop, e-graph dedup, etc.), augmentation toolkit (D4 × S_10), connected components patterns, stability selection.

## Status of newly-created G-cards

| Card | Status | Reason |
|---|---|---|
| G1 | DECLINED 2026-05-26 | Notes 26, 36, 37, 38, 41, 12, 25, 13 in K-Arc2 cover post-Jan-2026 ARC research comprehensively. |
| G2 | DECLINED 2026-05-26 | `rule_disambiguation.py` + `consistency_filter.py` + `synthetic_perturbation_filter.py` + note 35 (Symbolic Regression) cover discriminating-input synthesis from multiple angles. |
| G3 | DECLINED 2026-05-26 | Note 02 has the exact v2-vs-v1 taxonomy with the 4 novel axes. Note 17 has per-task axis tagging methodology. |

The actionable work is now IMPLEMENTATION (port rule_disambiguation semantics, manually axis-classify our 4 remaining tasks) rather than research.
