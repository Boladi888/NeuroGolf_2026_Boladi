# DONE 2026-05-23T21:20Z — session consolidation: solve-from-scratch unlock pattern CONFIRMED + 4 ships +37 LB

## Session ships (5949.01 → 5986.13)

| Time | Task | Pattern | LIVE → new | Predicted | Actual | LB |
|---|---|---|---|---|---|---|
| 18:33Z | task066 wallpath_rule | solve-from-scratch unlock #1 (lookup-LIVE) | 21,408 → 11,769 | +0.60 raw | **+15.62** (26×) | 5949.01 → 5964.63 |
| 19:57Z | task092 moment_score_einsum | structural-LIVE raw log_e ×1.0× | 198,818 → 4,352 | +3.82 | **+3.83** | 5964.63 → 5968.46 |
| 20:50Z | task277 bbox_area_min_edges | solve-from-scratch unlock #2 (NN-LIVE) | 46,539 → 16,748 | +15.28 max | **+15.27** | 5968.46 → 5983.73 |
| 21:18Z | task005 v40_compact_rayboxes | structural-LIVE raw log_e ×1.0× | 183,854 → 16,738 | +2.40 | **+2.40** | 5983.73 → 5986.13 |

## Two calibration patterns established

### Pattern A: solve-from-scratch unlock (lookup-LIVE / NN-LIVE on cherry13b=0 task)
- Required: LIVE-contribution = 0 confirmed via cherry13b bulk
- Required: structural rewrite (no [N≈266,K] table, no trained MLP, no ScatterElements from memorized positions)
- Required: 266/266 vs expected (NOT vs LIVE — LIVE is wrong on visible) + task-shaped synth > 50%
- Result: lands at exact theoretical max `25 - log_e(new_cost)` — both task066 and task277 hit it within ±0.01
- 2 confirms / 2 attempts = 100% rate so far

### Pattern B: structural-LIVE compression
- LIVE genuinely contributes positive LB (not in cherry13b bulk)
- Cost compression yields exactly raw log_e × 1.0×
- 2 confirms (task092 +3.83 predicted +3.82, task005 +2.40 predicted +2.40)
- Predictions accurate within ±0.05 LB — quote them directly to workers

## Bulk cherry13b 11-task triage (19:45Z, 1 submit credit)

Replaced 11 lookup-LIVE tasks (76, 96, 101, 118, 158, 219, 233, 277, 285, 363, 367) with cost-0 cherry13b stubs in ONE Kaggle submit. Result: 5964.63 → 5964.63 = **0.00 LB delta**. All 11 confirmed LIVE-contribution = 0.

**Refined interpretation** (overturns prior "zero-weight family" diagnostic):
- Cherry13b 0.00 delta = LIVE wrong on hidden = LIVE-contribution is 0
- Does NOT mean task has 0 latent public weight (task066 proved this; was bulk-tested at 0 then unlocked +15.62)
- Cost compression of LIVE-equivalent on these 11 = dead lever
- Structural rewrite = the only unlock path

## task145 -15.30 LB regression (19:01Z, reverted via re-submit)

Calibration ship at 0/50 task-shaped synth-live regressed by -15.30 LB. Refined rule:
- **Task-shaped synth-live > 50% required before staging any candidate** (hard gate)
- 0/X generic random adversarial can be OK if rule-shaped (task170 precedent at 0/100 generic shipped clean)
- 0/X task-shaped synth-live IS predictive of hidden-bad regression
- What matters is what the synth diverges ON — task-relevant axes vs off-contract noise

## Family ceiling forecast

10 remaining tasks at LIVE-contribution = 0 (76, 96, 101, 118, 158, 219, 233, 285, 363, 367), each carries +12-17 LB ceiling at Pattern A. Combined family ceiling: **+120-150 LB** if pattern holds across all.

Standing structural-LIVE prompts (002, 161, 017, 243, 025) each carry +1.2-1.4 LB at Pattern B. Combined: ~+6-7 LB.

## Files updated this session
- `techniques.md` ships table — 4 new rows
- `traps.md §14e` — 4 new entries (18:33Z, 19:01Z, 19:45Z, 20:50Z)
- `by_symptom.md` — new section "By LIVE architecture signature" with 3 entries
- `CHANGELOG.md` — 12th-pass entry
- `The_400_Tasks/Task_by_Task/{task066,092,277,005}.md` — SHIPPED entries
- `The_400_Tasks/task_state.csv` — task005/task277 rows updated; 11 bulk-cherry13b rows tagged
- `Logs/submissions_ledger.csv` — 4 manual-ship entries (kanban off this session)
- 4 lookup-LIVE prompts (task367/219/233/096) — rewritten with pattern-confirmed framing
- Memory: `feedback_lookup_family_detection_refined.md` — second-confirm entry

## Open items for next session
- 7 active workers iterating on lookup-LIVE / structural-LIVE tasks
- 4 lookup-LIVE prompts ready for fresh assignment (task367/219/233/096)
- 5 structural-LIVE prompts ready (task002/161/017/243/025)
- task092 + task005 (structural-LIVE prompt has shipped; remove from queue if it's still listed)
