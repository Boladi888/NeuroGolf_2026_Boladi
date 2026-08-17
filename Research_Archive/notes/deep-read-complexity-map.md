# Deep-read: `neurogolf-all-task-logic-complexity-map.ipynb` (ARC Logic Decoder)

**Date read:** 2026-05-03

## Headline

The notebook is a **visualization tool for an external dataset** (karnak's `neurogolf-2026-task-transformation-library`). It classifies each of the 400 tasks by:
- Transformation types used (Mirror, Shift, etc.)
- Complexity score (1-10)
- Grid-size change flag

It does NOT define new patterns or contribute its own ONNXs — it's a **taxonomy / triage tool** for picking which tasks to attempt.

## Structure

Three sections:
1. **Task Decoder** — labels each task with logic types + transformation types.
2. **Output Preview** — task list with transformation counts, difficulty scores, grid size changes.
3. **Logic Visualizer** — side-by-side ARC grid comparison with the produced logic labels.

## What this implies for NG

The actual data lives at `https://www.kaggle.com/datasets/karnakbaevarthur/neurogolf-2026-task-transformation-library`. We already have karnak's ONNX dataset locally at `data/competitor_artifacts/karnak-task-transformation/` (Apache-2.0). **Whether the same dataset includes the complexity-map labels needs verification** — if yes, we have it for free; if no, the dataset is just the ONNXs.

If we can extract the complexity scores + transformation labels:
- Filter to **complexity ≤ 4 + truly-unsolved** = ranked candidate list of "easy unsolved tasks"
- Cross-reference with our `tools/blend_diff.py` unsolved-150 list to get a prioritized backlog
- Saves ~30-60 min vs visualizing each unsolved task by hand

If only the ONNXs are bundled (no labels), this notebook is mostly a UI wrapper with no actionable data for us.

## Limitations

- Notebook code is mostly UI scaffolding; no novel ONNX patterns to lift.
- "Complexity 1-10" is the author's subjective scale; meaning of each level isn't defined in the notebook.
- Labels classify INPUT/OUTPUT structure, not the SOLUTION pattern needed.

## Sources

- `Research/primary_sources/neurogolf-all-task-logic-complexity-map.ipynb`
- `data/competitor_artifacts/karnak-task-transformation/` (local copy of source dataset)

## GAPS

- **Verify whether the local karnak dataset includes the complexity-label CSV / JSON** alongside the ONNXs. If yes, lift it into a Beta-track query tool. If no, lower priority.
- Other unread competitor notebooks (9 of 14 still unread post-PR-#28). Priority order TBD based on titles + LB scores.
