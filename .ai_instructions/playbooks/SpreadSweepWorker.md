# Spread-Sweep Worker Prompt — drop-in for a Sonnet session

Use this as a one-shot prompt to a fresh Sonnet (or any Claude) session.
Change the single line `MY_GROUP = <pattern_name>` to the group you want
worked. Everything else is parametric.

---

```
MY_GROUP = onnxsim   # CHANGE THIS LINE ONLY. Valid: onnxsim | safe_opt | scalar_broadcast_ones | slice_axis_collapse | crop_to_size_advisory

You are a NeuroGolf spread-sweep worker. Your only job is to apply the
`$MY_GROUP` shrink pattern to a list of tasks, verify each one against the
official live ONNX, stage the candidate, and commit the staged package +
task notes back to git so the submit-captain can submit.

Working directory: C:\path\to\K-NeuroGolf
Current baseline: submissions/submission.zip @ sha EE90DD57 (task117 worker-7
crop15+F16 swapped in, score 5759.16).

Your task list (per-task expected byte savings) is in
`Logs/spread_scan_groups_for_sonnet.json` under the key `$MY_GROUP`. Open it
and iterate.

Per-task workflow:

1.  Activate the venv. All scripts assume `.venv/Scripts/python`.

2.  For AUTOMATIC groups (onnxsim, safe_opt, scalar_broadcast_ones,
    slice_axis_collapse):

      .venv/Scripts/python tools/polish_candidate.py \
        --task <NNN> --pattern $MY_GROUP \
        --base submissions/submission.zip \
        --out-dir submissions/_spread_candidates/<NNN>_$MY_GROUP/

    The polish tool reads taskNNN.onnx from the base zip, applies the named
    pattern, writes the new taskNNN.onnx into out-dir, and emits a report.

3.  For the MANUAL crop_to_size_advisory group: read the existing precedent
    `The_400_Tasks/Task_by_Task/task117/prototypes/task117_crop15_f16out_t_f7689940/build_task117_crop15_f16out.py`
    — it shows the exact pattern (prepend Slice(input, ends=[1,10,N,N]) +
    rewrite range30/c29i/c30i initializers + append Pad back to [1,10,30,30]
    + ClearField('value_info') before ORT load). Write an analogous
    `build_taskNNN_cropK.py` per task, where K is the task's max active
    grid (from the scan report's `extras.max_active_grid` field).

4.  Verify with the OFFICIAL gate (this is the only safety check that matters):

      .venv/Scripts/python tools/verify_against_live.py \
        --proto <path_to_new_task_NNN.onnx> --tid <NNN> \
        --live-zip submissions/submission.zip

    Required: `Prototype vs LIVE: M/M` where M is the full pair count. If
    any pair fails, DROP the candidate. Do not ship.

5.  Score with `_score_v2.py`:

      .venv/Scripts/python tools/_score_v2.py --onnx <path>

    The new cost must be strictly less than the original from the base zip.
    If equal or higher, drop the candidate.

6.  Adversarial perturbation (informational; OFFICIAL gate at step 4 is the
    real safety check):

      .venv/Scripts/python tools/_score_v2.py --onnx <path> --adversarial

    Threshold: **≥30% aggregate**. Lower bound reflects empirical reality —
    task162 shipped at 54.8% adversarial and scored exactly its expected
    +0.22 LB on Kaggle, with no degradation across the perturbation
    families that ARE on-contract. The perturbations test off-contract
    inputs (shift/rotate move ARC content off the (0,0) origin which
    NEVER occurs in the real eval pipeline), so failures there don't
    predict hidden-bad on official data. Below 30% suggests on-contract
    behavior is also at risk; widen the crop window or drop. 30-95% is
    the safe-to-ship band.

7.  If all gates pass, stage the candidate. Path convention:

      staged_submits/taskNNN_$MY_GROUP_cost<NNN>/
        task<NNN>.onnx              # the new ONNX
        manifest.json               # see template below
        # optional: submission.zip (a copy of the rebased package; not required if a later step builds it)

    manifest.json template:

      {
        "schema": "neurogolf.spread_sweep.v1",
        "candidate_id": "taskNNN_$MY_GROUP_t_<8hex>_on_5759_16",
        "task_id": "taskNNN",
        "worker_owner": "neurogolf-sweep-$MY_GROUP",
        "pattern": "$MY_GROUP",
        "package_sha256": "<sha of stacked submission.zip if built; else empty>",
        "lineage_base_candidate_id": "task117_crop15_f16out_t_f7689940_20260518T2212Z_on_5758_04",
        "changed_package_entries": ["taskNNN.onnx"],
        "scores": {
          "base_cost": <from score_v2>,
          "candidate_cost": <from score_v2>,
          "cost_delta": <candidate - base>,
          "point_delta": <log(base/candidate)>
        },
        "expected_point_gain": <same as point_delta>,
        "tests_run": [
          "verify_against_live --tid NNN: M/M proto-vs-LIVE",
          "_score_v2: cost X -> Y",
          "<adversarial result if run>"
        ],
        "risk_notes": [],
        "rejected_attempts": []
      }

8.  Append a row to the task's per-task notes file
    `The_400_Tasks/Task_by_Task/taskNNN.md` under the "Attempts" table.
    Use exactly the existing date/Candidate/Result/Cost/Notes format. Link
    to the staged path and the build script.

9.  Move to the next task in the group. Continue until the list is exhausted
    OR you hit a hard error budget (10 verify-failures in a row).

10. Final commit per session:

      git add -A The_400_Tasks/Task_by_Task/ submissions/_spread_candidates/
      git commit -m "spread-sweep $MY_GROUP: <count> staged candidates, total <bytes>B saved"

      # NOT to be done by Sonnet (the auto-sync cron handles canonical):
      # NEVER directly edit submissions/submission.zip or current_best_manifest.json
      # NEVER call kaggle submit
      # NEVER run neurogolf_sync_current_best.py

Rules:
- One pattern per session. Do not touch other groups' tasks.
- One task at a time. Do not batch-process — each task gets independent
  verify + score gates.
- Do NOT submit to Kaggle. Submit-captain has exclusive Kaggle authority.
- Do NOT delete files. If a candidate fails its gates, log the failure in
  taskNNN.md and move on; leave the file artifacts on disk for the next
  worker.
- If the polish_candidate.py output for a task is no-op (no savings),
  skip without staging — the scan was against an older canonical and the
  win was likely already absorbed by a prior submit.
- Per-task .md notes always survive even if no candidate ships. Capture
  the analysis: what was tried, what was the verdict, why.

Stop conditions:
- Group list exhausted (success)
- 10 consecutive verify-failures (something systemic; report and stop)
- Operator explicitly asks you to stop

When done, your final reply must include a single-line summary:
  "spread-sweep $MY_GROUP complete: <staged_count> staged candidates,
   <total_bytes> B saved, expected combined gain <pts> LB"
```

---

## Operator launch checklist (for the operator, not the Sonnet)

For each group, in a fresh Claude Code window:

1. cd C:\path\to\K-NeuroGolf
2. Paste the prompt above. Change line 1 (`MY_GROUP = ...`) for that session.
3. Send. Then `/loop` (dynamic, no interval — the worker self-paces task by task).

Suggested order (lowest-risk first):
1. `slice_axis_collapse` — 53 tasks × ~4 B; trivial, almost certainly clean.
2. `scalar_broadcast_ones` — 34 tasks × ~100 B avg; usually clean.
3. `safe_opt` — 22 tasks; moderate.
4. `onnxsim` — 26 tasks; some already absorbed in current canonical (de-dupe).
5. `crop_to_size_advisory` — 375 tasks, manual reasoning. Run a heavy model (Opus) on this if available; potential ~+11 LB across the group.

Total realistic upside before submit-captain rate limits and hidden-bad
attrition: order of +5 to +20 LB cumulative if everything lands.
