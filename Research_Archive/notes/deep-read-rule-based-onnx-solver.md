# Deep-read: `neurogolf-2026-rule-based-onnx-solver.ipynb`

> Beta-track deep-read, 2026-05-03 20-hr session. Source: `Research/primary_sources/neurogolf-2026-rule-based-onnx-solver.ipynb` (16 cells, ~120 KB).

## Summary

Author Imaad Mahmood. Claims **6285+ LB / 401 tasks solved**. Strategy: rule-based detector library (40+ detectors) → ONNX builder per detector → multi-source blend → artifact-floor lock.

The notebook is a TREASURE TROVE of rule-based detector patterns that aren't yet in our build pipeline. Many overlap with our existing patterns or our backlog, but several are NEW and immediately usable as proxy detectors.

## Detector inventory (40+)

Ported a subset (19) into `tools/mine_rule_based_detectors.py` and ran against truly-unsolved 150:

| Detector | Truly-unsolved hits | Sample tasks |
|---|---|---|
| **bbox-fill** | **8** | 66, 102, 105, 243, 255, 265, 363, 367 |
| period-row-tiling | 3 | 102, 112, 124 |
| period-col-tiling | 3 | 102, 112, 178 |
| (all others) | 0 | (h/v/diag/both-symmetry-complete; outline-extract; keep-largest/smallest; nonzero/bg-recolor; translation; largest-cc-bbox-crop; concat-rot180-w; color-filter-remove; frame-border) |

### bbox-fill semantics — actually a LOOSE detector

My ported `bbox-fill` detector matches when:
- Outside the bbox of non-zero input cells, input == output.
- Inside the bbox, only originally-zero cells can change.
- All changed cells change to the SAME fixed color C (consistent across all pairs of the task).

This is broader than "fill the entire bbox interior" — it matches **any task where the change set is `{0-cells} → fixed color C`**. The 8 hits include:
- Tasks where 3 fills enclosed-region-style interior (66, 367 likely)
- Tasks where lines/connectors are drawn between markers (363 likely — this matches the task363 LB-5126 silent-fail recipe in `Research/notes/deep-read-lb-5126-bug-free-blend.md`!)
- Tasks where bbox is fully filled (105 candidate)
- Tasks where the bg gets filled inside object (255, 265 candidates)

These 8 hits are **CANDIDATES** for further per-task characterization, not a single shippable pattern.

### Notable detectors I haven't yet ported / tested

From the notebook (cells 5-12) but not in my mining sweep:
- **detect_period_row_tile_onnx / detect_period_col_tile** with non-trivial periods (the 3 hits I got are trivial period=1 or period=ih cases).
- **detect_crop_then_scale** (factor 2/3/4/5) — fixed-position crop + nearest-neighbor upscale.
- **detect_fixed_row_select** — output = a specific row of input, optionally tiled.
- **detect_crop_plus_color** — fixed-position crop + per-cell color map.
- **detect_concat_rot180_w** — input concatenated with its 180-rotation horizontally.
- **detect_border_add** — output = input padded with fixed-color border (configurable thickness).

These are worth a follow-up mining sweep in a future session.

### Builder/network insights (cells 5, 6, 7, 10)

The notebook implements ONNX builders for each detector:
- `make_period_row_tile_onnx(period, in_h, in_w, out_h)` — Slice + Tile pipeline. Confirmed cheap.
- `_slice_channel_node` — extract a single color channel via Slice (clean idiom we already use).
- `build_rotation_gather` — direct Gather-based rotation (we have `fixed_rot180` etc.).

### Multi-source blending strategy (cell 15)

Author uses 5 source ZIPs:
- afr1ste op37 artifact (LB 6285+ floor)
- artem cluster (rule-based blend)
- 3 other sources

Key strategy: **artifact-floor lock** — start with the highest-LB external as the baseline, then ONLY override per-task if a cheaper / verified option exists. This is structurally identical to our `external_blend.py` pick-cheapest-passing logic, but with explicit floor protection.

### EXCLUDED_TASKS list

Cell 15 references `EXCLUDED_TASKS` (not shown in my read) — likely tasks the author manually excludes from the blend. Worth a follow-up read of cell 15 in a future session.

## Cross-reference vs our backlog

| Their detector | Our state |
|---|---|
| bbox-fill (8 hits) | NEW; immediately useful as candidate-task identifier. Add to backlog. |
| hole-fill | We have enclosed-region-fill; functionally similar, different boundary semantics. Different detector might catch additional tasks. |
| outline-extract | In backlog Tier 3 (`05_BACKLOG.md`). 0 hits in our mining. Demote to Tier 4? |
| keep-largest-color / keep-smallest-color | In backlog Tier 3. 0 hits in our mining. Demote. |
| period-row-tiling / period-col-tiling | In backlog Tier 2 (multi-source-onnx-solver Tier A #11). 3 trivial hits each. Worth full implementation pass. |
| frame-border / border-add | NEW. 0 hits in mining. Add to Tier 4. |
| nonzero-recolor / bg-recolor | NEW. 0 hits in mining. Demote / drop. |
| diag-symmetry-complete | NEW. 0 hits. Drop. |
| translation (with shifts) | NEW for shape-PRESERVING. 0 hits in mining. The `multi-source-onnx-solver` "Translation Tier S" finding (0 hits in backlog Demoted) is corroborated. |
| largest-cc-bbox-crop | NEW. 0 hits. Add to Tier 4 — may need 4-conn vs 8-conn variant. |
| concat-rot180-w | NEW. 0 hits. Drop. |
| color-filter-remove | NEW. 0 hits. Drop. |
| crop-then-scale | NOT mined yet. Add to Tier 3. |
| fixed-row-select | NOT mined yet. Add to Tier 3. |
| crop-plus-color | In backlog Tier 2. NOT mined yet here. |

## Action items / backlog additions

1. **[B→A] bbox-fill cluster (8 candidate tasks: 66, 102, 105, 243, 255, 265, 363, 367)**: per-task characterization needed (each may be a different specific rule). Cross-reference: task363 matches the LB-5126 silent-fail recipe; task243 was already in our backlog from karnak xref. Add cluster note to Tier 2.
2. **[B→A] Per-task dispatch for the bbox-fill cluster**: when each task's specific rule is characterized (Phase 4 follow-up), file individual `[B→A]` items.
3. **[B→A] crop-then-scale + fixed-row-select**: not yet mined. Add `tools/mine_rule_based_detectors.py` extensions in next session.
4. **No new tools needed beyond `mine_rule_based_detectors.py`.**

## Closed sub-questions

- **"Does rule-based-onnx-solver have novel patterns we lack?"** — Yes (bbox-fill, period-tiling-with-non-trivial-period, crop-then-scale, fixed-row-select). Most others are in our backlog.
- **"Should we adopt their multi-source blend strategy?"** — Already done via `external_blend.py`. Their "artifact-floor lock" idea is worth porting if we ever risk regression on a previously-good submission.

---

*Author: Beta-track 2026-05-03 deep-analysis session. Tool: `tools/mine_rule_based_detectors.py`. Cross-reference: `Logs/sessions/2026-05-03_1830_beta-20hr-session.md`.*
