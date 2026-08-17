# Deep-read: `neurogolf-4808-21-post-apr-28-update.ipynb`

> Beta-track 20-hr session, Phase 9c. Source: `Research/primary_sources/neurogolf-4808-21-post-apr-28-update.ipynb` (9 cells, ~5 KB).

## Summary

Pure release-artifact notebook. Shares author's submission pack scoring **4808.21 LB after the April 28 metric update**. No novel patterns or recipes — the notebook just verifies + loads + displays a pre-built submission.zip + task_by_task_release_table.csv.

**Strategic value to NG**: minimal. Confirms April 28 update happened (already known per `Research/notes/deep-read-lb-5126-bug-free-blend.md`). The author's method:
> "I subscribed to ChatGPT Pro and used many parallel coding sessions to search task-specific ONNX candidates, then spent most of the remaining effort reducing model cost family by family."

So: per-task ChatGPT-generated ONNXs + cost reduction. NOT a generalizable pattern library.

## Cited source datasets (potential mining targets)

The author credits 3 sources used in their pack:

1. **`aliafzal9323/neurogolf-2026-tiny-onnx-solver`** — Kaggle code dataset. Worth pulling and cross-referencing as another external blend candidate.
2. **`karnakbaevarthur/neurogolf-all-task-logic-complexity-map`** — already deep-read 2026-05-03 (per `Research/notes/deep-read-complexity-map.md`).
3. **`needless090/neurogolf-onnx-v31`** — Kaggle dataset. Another external pack candidate.

## task_by_task_release_table.csv (referenced but not pulled)

The notebook references a per-task release table with columns: `task_id`, `cost`, `score_points`, `zip_member`, `included`. If pulled, this would let us cross-reference our 50 SOLVED costs against the author's 4808-LB pack to identify cost-displacement opportunities.

**Action item:** if Kaggle CLI access is convenient, pull this dataset:
```
kaggle datasets download -d <author>/neurogolf-4808-21-post-apr-28-metric-update
```
Then cross-reference with `tools/blend_diff.py` to identify per-task-cost wins/losses.

## Findings vs existing knowledge

- April 28 update existence: already confirmed by LB-5126 deep-read.
- 4808.21 LB ceiling without exploit: useful data point — sits between our 3275 (current) and the 5000+ exploit-driven scores.
- Our gap to 4808 is ~1500 LB pts. Bridging that requires either (a) more SOLVED tasks (we're at 50; 4808-LB pack covers 395 with task-specific ONNXs), or (b) external blending with `aliafzal9323` / `needless090` packs.

## Action items

1. **Pull `aliafzal9323/neurogolf-2026-tiny-onnx-solver` + `needless090/neurogolf-onnx-v31`** as external blend candidates (next session's `tools/external_blend.py` candidate-pool refresh).
2. **Pull `task_by_task_release_table.csv`** for cost-cross-reference opportunities.
3. **No backlog additions** from this notebook — pure artifact release.

## Closed sub-questions

- **"Does this notebook surface novel patterns?"** — No. Pure pack release.
- **"What's the post-April-28 ceiling without exploits?"** — At least 4808.21, achievable via ChatGPT-driven per-task ONNX generation + cost reduction over months.

---

*Author: Beta-track 2026-05-03 deep-analysis session. Cross-reference: `Logs/sessions/2026-05-03_1830_beta-20hr-session.md`.*
