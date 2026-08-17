# Archived 2026-06-21 — dominated staged candidates

157 `staged_submits/taskNNN_*` folders moved here (not deleted) because they are **dominated**:
their folder-name `cost` is **>= the current team-best cost** in
`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`. The team already has an
equal-or-better candidate for each of these tasks, so they were superseded dead weight in the
staging area.

**Safety checks done at move time:**
- Re-verified "dominated" against the *current* manifest (not a stale snapshot).
- Skipped any folder modified in the prior 3h (none were) — avoids touching a worker's in-flight build.
- Per-folder move with lock protection (a move fails rather than partially-moves if a file is open) — 0 failed.
- Spot-checked that no `cost < team-best` winner was swept (task104/094/321/005 etc. remain in `staged_submits/`).

**Not touched:** the 106 remaining `staged_submits/` folders — the missed-wins (`cost < team-best`,
incl. the 33 shipped to LB 7311.74) plus any recently-active builds.

**To restore a folder:** `mv _dominated_archive_2026-06-21/<folder> ../`
