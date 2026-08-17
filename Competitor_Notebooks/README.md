# Competitor Notebook Survey

This is the retained analytical survey from the May 2026 review of public Kaggle NeuroGolf notebooks. The raw third-party notebooks and downloaded zip payloads were removed during archive cleanup because their Kaggle metadata had no explicit license. Keeping this survey preserves the useful competitive intelligence without implying Apache-2.0 coverage for other people's notebook files.

The survey is historical. It names public Kaggle account slugs and notebook slugs so future agents can re-find sources if needed, but the current repository does not vendor those notebooks.

## Attribution and Scope

Notebook authors and source accounts observed in the survey included vyankteshdwivedi, afr1ste, jonathanchan, konbu17, artemnazemtsev, ashok205, sigmaborov, aliafzal9323, needless090, beicicc, karnakbaevarthur, svanikkolli, and magmacot.

One pulled notebook, the-epic-sax-ensembler, was from a different competition and was not relevant to NeuroGolf.

## 2026-05-08 NAS/local Artifact Update

The local NAS scrape showed May 6/7 kernels and datasets beyond the original notebook catalog. High-value current-rule sources visible locally included vyankteshdwivedi/neurogolf-multi-source-onnx-solver, afr1ste/neurogolf-5480-41-current-rules-score, jonathanchan/ngc26-constraint-smart-logic-mix-blending, konbu17/neurogolf-2026-may-7-updated, and the official starter refresh.

Local competitor artifacts were already represented under the machine-local data area, including kernel-output-vyanktesh-multi-source, kernel-output-jonathanchan-ngc26, kernel-output-konbu17-may7, many afr1ste and beicicc CC0 artifacts, and magmacot_5700. See `Research_Archive/notes/beta-backlog-intake-2026-05-08.md` for the Beta intake and task145 all-local sweep result.

## Catalog Summary

Sorted by public score where known, then by content density.

| Kaggle notebook slug | Public LB | Content kind | Follow-up artifact/source | Survey value |
|---|---:|---|---|---|
| neurogolf-6645-39-public-score-open-solution | 6,645.39 | Thin wrapper; unzips attached artifact | neurogolf-6645-39-open-submission-artifact | Pull dataset; notebook itself was glue |
| neurogolf-2026-rule-based-onnx-solver | about 6,285 | Master Ensemble Solver v7; multi-source blend with validation throughout | In-notebook code | Densest methodology source |
| neurogolf-2026-blended-401-tasks-4-27 | 6,254.65 | One-click blend submission; notes task363 hidden-test silent fail | Chained upstream notebooks | Critical silent-fail finding |
| neurogolf-6233-36-public-score-open-solution | 6,233.36 | Thin wrapper; unzips attached artifact | neurogolf-6233-36-open-submission-artifact | Pull dataset; notebook itself was glue |
| neurogolf-2026-4-27-lb-5126 | 5,126.37 | Bug-Free Blend; deliberately avoids profiler quirks | Chained upstream notebooks | Exploit catalogue plus April 21 reference |
| neurogolf-5044-56-current-rules-score | 5,044.56 | Post-current-rules public refresh | Attached dataset | Skim only |
| neurogolf-4808-21-post-apr-28-update | 4,808.21 | Post-April-28 rescore artifact; describes parallel per-task search | Attached dataset | Confirms metric update impact |
| neurogolf-multi-source-onnx-solver | n/a | Code-only multi-source solver pipeline | In-notebook code | Likely engine behind the rule-based solver blend |
| neurogolf-all-task-logic-complexity-map | n/a | ARC logic decoder and task categorization | In-notebook code | High-ROI planning/mapping source |
| neurogolf-acking-multiple-tasks-part-3 | n/a | Per-task hand-coded ONNX program ideas | In-notebook code | Pattern-recipe ideas |
| neurogolf-locally-solved | n/a | 160 shipped tasks, roughly 197 solved locally | ashok205/neuro-golf-2026 | Maybe useful for ONNX comparison |
| neurogolf-logic-driven-ensembe | n/a | Conservative I/O repair and blend wrapper | Chained upstream notebooks | Skim only |
| neurogolf-2026-public-data | n/a | Repackages dataset ONNX files | sigmaborov/golf-solve-agent | Pull dataset to inspect |
| the-epic-sax-ensembler | n/a | Tabular CSV ensembler from another competition | n/a | Not relevant to NeuroGolf |

## Carried-Forward Findings

These were the useful points from the survey pass, retained for future similar projects.

### Cost-Model and Exploit Recon

1. Compress-with-fake-zero-batch exploit was referenced by name in the Bug-Free Blend notebook as a profiler quirk that the author deliberately avoided. It was likely related to using a disallowed or normally-counted operator in a way the validator missed. The project stance was to understand this defensively, not use it.
2. April 21 update was mentioned separately from the April 28 metric update and may explain earlier public-score discontinuities.
3. Constant-not-counted and dynamic-shape under-counting appeared to match already-cataloged accounting gaps.
4. Positive-cost wrappers were a meta-technique class where adding nominal graph structure could reduce the profiler's total through an accounting gap.

### Methodology and Planning

5. A top-player workflow described broad parallel per-task candidate search first, followed by per-pattern-family cost reduction. That complemented, rather than replaced, the project's own pattern-mining loop.
6. Defensive I/O naming verification was completed on 2026-05-02: all 400 ONNX files in `submissions/` correctly emitted tensors named `input` and `output`.

### Resolved by Deep Reads on 2026-05-02

- Positive-cost wrappers were decoded as three layered exploits in the tackling-multiple-tasks part 3 source: static-to-symbolic output/value_info dims, initializer-to-Constant conversion, and an output identity wrapper through Shape, ConstantOfShape, and Add. The project stayed clean by construction: no Constant nodes, no symbolic dims, and no ConstantOfShape.
- The top-blender architecture loaded candidates from up to 11 source notebooks, then picked the highest-priority per-task candidate that passed check_network, score_network, and onnxruntime correctness.
- High-value Kaggle slugs to re-pull, if network access is ever needed again: afr1ste/neurogolf-6285-95-public-score-open-solution, aliafzal9323/neurogolf-2026-tiny-onnx-solver, svanikkolli/arc-nano-engine, and karnakbaevarthur/neurogolf-2026-task-transformation-library.
- Kaggle API endpoint access was unreliable from the owner's machine during the survey period: competitions list worked, while datasets download and kernels output timed out.

### Silent-Fail Risk

The Blended 401-Task notebook reported that task363 passed train, test, and local ARC-GEN checks but scored zero on Kaggle's hidden test. The implication for future work is clear: visible-pair validation is necessary but not sufficient, and private hidden-test risk grows as solutions become more complex.

### Pattern and Artifact Follow-Ups

The highest-ROI historical deep reads were the all-task logic complexity map, the rule-based ONNX solver, the multi-source solver, and the tackling-multiple-tasks part 3 source. Artifact pulls worth comparing, if ever refreshed from Kaggle, were the 6645.39 open submission artifact, the 6233.36 open submission artifact, ashok205/neuro-golf-2026, and sigmaborov/golf-solve-agent.
