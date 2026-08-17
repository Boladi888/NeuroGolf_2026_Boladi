# 01_whole_graph_rewrites — worked case studies

Whole-graph rewrites replace the existing op chain entirely with a different algorithm achieving the same output. Highest leverage when polish techniques don't apply.

When polish (techniques.md §1-§17) can't get you to +0.1 LB, this is the next step.

Read every exemplar in this category. Cross-references to other categories follow.

## Files

- `task376_input_to_bool.md` — ★ input → BOOL whole-graph propagation. The highest-leverage technique. Cross-ref to `02_bool_first/` (rule 8).
- `task049_least_color_bbox.md` — bounding-box extraction algorithm. Object_Based family canonical.
- `task382_cumsum_gather.md` — geometric task encoded as CumSum + Gather. Spatial_and_Geometric canonical.
- `task119_diag_conv.md` — Conv reformulation with custom kernel.
- `task115_axis_factor.md` — runtime axis-factorization for separable rules.

Also cross-references:
- task110 (periodicity-detection rebuild) — lives in `09_periodicity_detection/`.
- task071 (ring-mask dynamic) — lives in `05_static_to_dynamic/`.
