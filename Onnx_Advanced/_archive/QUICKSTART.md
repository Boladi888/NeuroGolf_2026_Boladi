# QUICKSTART — the 5-minute path for time-pressed workers

If you have time to read everything, follow `README.md`. If you don't, follow this.

## The minimum path before touching ONNX (10 minutes total)

1. **Read your task's graph diagnostic (2 min)**
   Open `The_400_Tasks/Task_by_Task/taskNNN.md`. Scroll to the `## Graph Diagnostic` section at the bottom. Note:
   - Cost / memory / params
   - Max active grid (from the JSON)
   - Quick-look flags (has_Conv? first_node_is_Slice? ends_with_Cast_to_F32?)
   - Suggested techniques

2. **Extract the canonical ONNX (10 sec)**
   ```
   .venv/Scripts/python tools/agent_helpers/extract_task.py NNN
   ```
   Drops `taskNNN.onnx` into your CWD.

3. **Read your task's entry in the router (2 min)**
   Open `task_techniques_index.md`. Find your task's family section. Find your task ID. Read the linked exemplar file in full.

4. **Read the trap catalog (3 min)**
   `traps.md`. Five minutes of trap-scan prevents hours of debugging. Especially:
   - ORT does NOT support `Where(bool, bool, bool)`.
   - OneHot: INT32 and INT64 indices BOTH work — but `depth` dtype must MATCH indices (the old "INT64 only" rule was a task124 misdiagnosis; traps §2).
   - `g.ClearField("value_info")` after any shape-changing surgery.
   - 1-ch0 trap for foreground detection.

5. **Read `core.md` (3 min)**
   NeuroGolf-specific facts. Scoring formula, encoder details, validator rejection criteria.

## After that, if you have more time

| Read next | Time | When useful |
|---|---|---|
| `cost_floor_reference.md` | 5 min | Before designing — price-shop your op chain |
| `Ort_Advanced/ort_quirks.md` | 5 min | If your model fails ORT load |
| `Ort_Advanced/ort_errors.md` | 2 min lookup | When you hit a specific error message |
| `techniques.md` | 15 min | The full rewrite toolkit (18 techniques) |
| `Algorithm_Library/algorithm_patterns.md` | 10 min | When you don't know what your task computes |
| `Algorithm_Library/rule_discovery_methodology.md` | 5 min | If your task needs a from-scratch rebuild |
| Other exemplars in your category | 10 min each | When polish techniques aren't enough |

## Decision tree

```
Cost > 200k?
├── max active grid << 30?  → crop-to-active-grid + F16-output (techniques.md §2 + §3, exemplar task117)
└── Conv-heavy?              → consider input→BOOL whole-graph (exemplar task376)

Cost 20k-200k?
├── many F32 [1,10,30,30] intermediates? → F16 placement (techniques.md §4, exemplar task055)
├── multiple Equal/Less chains?           → Equal-fold variants (techniques.md §5)
└── Mul of two bool casts?                 → bool_andfold (techniques.md rule 1)

Cost < 20k (near-floor)?
├── Concat with bool-eligible inputs?     → bool_concat keeping (rule 7, exemplar task299 — 60% reduction)
├── Already shape-only?                    → check for opset bump (rule 6) or scalar broadcast (§17)
└── Otherwise                              → DO NOT DISMISS. Inspect graph for bool-Concat / Pad chains.
```

## Symptom-driven navigation

If your model has a specific structural issue, jump straight to `by_symptom.md` — it maps graph patterns to exemplars without going through the technique taxonomy first.

## Diagnostics

Don't write your own. See `diagnostics_recipes.md` for the lookup of "I want to know X, run script Y". Half the scripts agents write already exist in `tools/agent_helpers/`.

## Hard rules (no exceptions)

These won't change. Pin them to working memory:

- **verify_against_live must return 100% M/M** before staging. Exception only for tasks with <10 visible pairs.
- **Adversarial-against-live**: run + report, **advisory only** (per `core.md` 2026-05-22 update). Low % does NOT mean broken — tests off-contract inputs Kaggle never sees. Empirical: task064 candidate at 9.7% shipped +1.20 LB clean. Exception: documented hidden-correctness-sensitive tasks (task008-style 3/86 history) keep the ≥30% gate per their own don't-do log.
- DO NOT submit to Kaggle (submit-captain only).
- DO NOT edit `submissions/submission.zip` directly.
- DO NOT skip the visible verify gate.
- DO NOT use `Sub(1, ch0)` for foreground detection (the 1-ch0 trap).

## When you're done

If you discovered a new technique / ORT quirk / trap not in the library — drop a file in `.update/` (see `.update/README.md`). A sorter agent integrates contributions later.
