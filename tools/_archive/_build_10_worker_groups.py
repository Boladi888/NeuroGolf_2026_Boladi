"""Build 10 worker groups for fresh task rebuilds on the 5835.30
canonical. Groups by (family, cost tier) so similar tasks stay together."""
import json
from collections import defaultdict
from pathlib import Path

import openpyxl

wb = openpyxl.load_workbook('The_400_Tasks/NeuroGolf_Task_Status.xlsx')
ws = wb['Tasks']
headers = [c.value for c in ws[1]]


def ci(name):
    return headers.index(name) + 1


tasks = []
for r in range(2, ws.max_row + 1):
    tid = ws.cell(row=r, column=ci('task')).value
    fam = ws.cell(row=r, column=ci('family')).value
    transforms = ws.cell(row=r, column=ci('transformations')).value
    cost = ws.cell(row=r, column=ci('current_cost')).value or 0
    complexity = ws.cell(row=r, column=ci('complexity_1_10')).value
    if not tid:
        continue
    tasks.append({
        'tid': tid,
        'family': fam or 'Unclassified',
        'transformations': transforms or '',
        'cost': cost,
        'complexity': complexity or 0,
    })

# Filter to actionable (cost >= 1000 — skip near-floor tasks)
actionable = [t for t in tasks if t['cost'] >= 1000]
print(f'Actionable (cost >= 1000): {len(actionable)} of {len(tasks)}')


def tier(cost):
    if cost >= 50000:
        return 'HUGE'
    if cost >= 20000:
        return 'BIG'
    if cost >= 5000:
        return 'MID'
    return 'SMALL'


# Group definitions: 10 groups with rough cost-tier × family axes
GROUPS = [
    # (group_name, filter_fn, family_label, tier_label, description)
    ('PR_HUGE', lambda t: t['family'] == 'Pattern_Recognition' and t['cost'] >= 50000,
     'Pattern_Recognition', 'HUGE',
     'Tile/mirror/wave repetition patterns. Tend toward Conv-heavy or per-cell-rule encodings.'),
    ('PR_BIG', lambda t: t['family'] == 'Pattern_Recognition' and 20000 <= t['cost'] < 50000,
     'Pattern_Recognition', 'BIG', 'Same family, mid-cost rewrites.'),
    ('PR_MID', lambda t: t['family'] == 'Pattern_Recognition' and 5000 <= t['cost'] < 20000,
     'Pattern_Recognition', 'MID', 'Same family, smaller wins available.'),
    ('PR_SMALL', lambda t: t['family'] == 'Pattern_Recognition' and t['cost'] < 5000,
     'Pattern_Recognition', 'SMALL',
     'Near-floor pattern-recognition tasks. Wins rare; only chase if a single Constant or scalar collapse is obvious.'),
    ('OB_HUGE', lambda t: t['family'] == 'Object_Based' and t['cost'] >= 50000,
     'Object_Based', 'HUGE',
     'Object segmentation / counting / coloring. Look for ReduceSum/ReduceMax bounding-box detectors and ScatterND fills.'),
    ('OB_BIG', lambda t: t['family'] == 'Object_Based' and 20000 <= t['cost'] < 50000,
     'Object_Based', 'BIG', 'Same family, mid-cost rewrites.'),
    ('OB_MID_SMALL', lambda t: t['family'] == 'Object_Based' and t['cost'] < 20000,
     'Object_Based', 'MID/SMALL', 'Smaller object-based wins.'),
    ('CL_HUGE', lambda t: t['family'] == 'Color_and_Logical' and t['cost'] >= 30000,
     'Color_and_Logical', 'HUGE',
     'Color-mapping / boolean-logic-driven tasks. Frequent Where/Equal chains; bool_andfold and or_cast_collapse often apply.'),
    ('CL_MID_SMALL', lambda t: t['family'] == 'Color_and_Logical' and t['cost'] < 30000,
     'Color_and_Logical', 'MID/SMALL', 'Smaller color/logic wins.'),
    ('SG_ALL', lambda t: t['family'] == 'Spatial_and_Geometric',
     'Spatial_and_Geometric', 'ALL',
     'Spatial transforms (rotation, reflection, tiling). Often very low cost; only big wins come from algorithm rebuilds.'),
]

assigned = set()
groups_data = {}
for name, fltr, fam, tier_lbl, desc in GROUPS:
    members = [t for t in actionable if fltr(t) and t['tid'] not in assigned]
    members.sort(key=lambda x: -x['cost'])
    for t in members:
        assigned.add(t['tid'])
    groups_data[name] = {
        'family': fam,
        'tier': tier_lbl,
        'description': desc,
        'members': members,
        'sum_cost': sum(t['cost'] for t in members),
    }

# Sanity: anything actionable not assigned?
unassigned = [t for t in actionable if t['tid'] not in assigned]
print(f'Unassigned (Unclassified or floor): {len(unassigned)}')

FAMILY_TRICKS = {
    'Pattern_Recognition': {
        'inspiration': 'task117 (crop15+F16 -100525), task374 (conv_run_combined -11728), task381 (crop10), task233 (Equal-fold -32000)',
        'tricks': '1) Crop-to-active-grid if max_active_grid<30. 2) F16-output trick on Concat->Cast tails. 3) bool_andfold on mask Mul chains. 4) shared_cast_equal_fold on N-way color comparisons.',
        'rebuild_hint': 'Many pattern tasks compute the same output via convolution OR predicate-mask OR ReduceSum-based encoding. Try the smallest-rank formulation: if the output is binary, use bool ops (1 byte/elem) throughout and Cast to F32 only at output.',
    },
    'Object_Based': {
        'inspiration': 'task266 (scalar_broadcast), task167/task103 (opset9_attr_shrink), task266 (Conv-based bounding box)',
        'tricks': '1) ReduceMax(axes=[2]) + ReduceMax(axes=[3]) for bounding-box. 2) ScatterND for filling. 3) separable_mask_collapse on row/col masks. 4) nonzero_shortcut on Mul(input, gate)+ReduceSum chains.',
        'rebuild_hint': 'Object-based tasks often build a wide [1,C,H,W] intermediate before reducing. Try projection-cache: compute the reductions directly from input (see Logs/lessons.md task071 ring-mask insight) and skip the wide intermediate.',
    },
    'Color_and_Logical': {
        'inspiration': 'task233 (Equal-fold -32000), task219 (Equal-fold -1056), task293 (And-fold), task196 (Or+Cast)',
        'tricks': '1) Equal-fold on Sub->Mul(self)->Less<1. 2) bool_andfold on Cast(bool)*Cast(bool) Muls. 3) or_cast_collapse on Add(Cast(gt_hi),Cast(lt_lo)). 4) flip_where on Mul(b, Sub(1, Where(c,b,0))).',
        'rebuild_hint': 'Stay bool through every And/Or/Not chain. Cast to F32 only at the boundary where F32 arithmetic against non-mask values is required. See Logs/lessons.md "bool-first compute" rule.',
    },
    'Spatial_and_Geometric': {
        'inspiration': 'task129 (tile_to_edge_pad), task140 (opset10_pad_attr), task005 (onnxsim folded)',
        'tricks': '1) Pad+Slice collapse via identity. 2) opset upgrade for pad-attr -> pad-input. 3) Constant-fold padding values. 4) Tile -> Reshape+Expand sometimes wins.',
        'rebuild_hint': 'Spatial tasks often have a redundant Pad before Slice or vice versa. Look for opportunities to drop one. If the model uses Tile with a small repeat factor, try Expand instead (no element duplication).',
    },
}

# Generic prompt template
def make_prompt(group_name, group_data):
    members = group_data['members']
    family = group_data['family']
    tier_lbl = group_data['tier']
    desc = group_data['description']
    tricks = FAMILY_TRICKS.get(family, FAMILY_TRICKS['Pattern_Recognition'])
    task_list = ', '.join(t['tid'] for t in members)
    return f'''```
You are a NeuroGolf 2026 Kaggle worker. Your assigned group is {group_name}.

Working directory: C:\\Users\\Qx\\Documents\\Py_cx\\K-NeuroGolf
Current canonical: submissions/submission.zip @ 5835.30 LB (sha 5D9AABE0)
.venv at .venv/Scripts/python

Family: {family} ({tier_lbl} cost tier)
{desc}

Your tasks ({len(members)}, by cost desc): {task_list}

Inspiration prototypes (look at their build_*.py for technique):
  {tricks['inspiration']}

Tricks to try first, in order:
  {tricks['tricks']}

Family-specific rebuild hint:
  {tricks['rebuild_hint']}

Per task:
1. Extract canonical taskNNN.onnx from submissions/submission.zip.
2. Inspect: node count, op-type histogram, value_info shapes. Look
   for 30x30 intermediates (crop opportunity) and Flatten->MatMul/Gather
   (crop blocker).
3. Look at the inspiration prototypes under
   The_400_Tasks/Task_by_Task/<inspirational_task>/prototypes/build_*.py
   - they are shipped exemplars in similar shape; mirror their technique.
4. Apply the tricks above in order.
5. If all polish tricks fail to lower cost AND the task is still worth
   points (cost >= 5000), REBUILD from scratch in a different algorithm
   family (see rebuild hint).
6. Verify EVERY candidate before staging:
     .venv/Scripts/python tools/verify_against_live.py \\
       --proto <new.onnx> --tid <NNN> --live-zip submissions/submission.zip
   Required: "OK Prototype matches LIVE" AND "Prototype vs LIVE: N/N"
   with no failures. HARD GATE. Sub-100% verify drops the candidate
   (see Logs/lessons.md wildcard rule: 7/8 verify-fail candidates
   regressed -13 to -17 LB on Kaggle).
   EXCEPTION: tasks with very few visible pairs (<10) can have verify
   failure but still pass hidden — task000 confirmed this 2026-05-20.
7. Score: .venv/Scripts/python tools/spread_score_isolated.py <new.onnx>
   Must be strictly lower cost than canonical taskNNN.onnx.
8. Stage under .hermes/board/staged_submits/taskNNN_<technique>_t_<8hex>_on_5835_30/
   with taskNNN.onnx + manifest.json (template:
   .ai_instructions/playbooks/SpreadSweepWorker.md step 7).
9. Append to The_400_Tasks/Task_by_Task/taskNNN.md under "Attempts".
10. Final commit:
     git add -A The_400_Tasks/Task_by_Task/ .hermes/board/staged_submits/
     git commit -m "fresh worker {group_name}: N staged candidates, B bytes saved"

Rules:
- ONE task at a time. Independent verify + score gates per task.
- DO NOT submit to Kaggle. Submit-captain has exclusive Kaggle authority.
- DO NOT edit submissions/submission.zip directly.
- DO NOT skip the verify gate (except for tasks with <10 visible pairs).
- If 10 consecutive tasks produce no candidate, STOP and report.

When done, report:
  {group_name} complete: N staged candidates, B bytes saved, expected ~X LB
```'''


# Generate md
lines = []
lines.append('# Fresh worker groups on 5835.30 canonical (2026-05-20)\n')
lines.append(f'Canonical: 5835.30 LB (sha 5D9AABE0). 10 worker groups by (family, cost tier).\n')
lines.append(f'**Total actionable tasks**: {len(actionable)} (cost >= 1000)\n')
lines.append(f'**Total assigned**: {len(assigned)}\n')

lines.append('\n## Group index (sorted by sum cost)\n')
lines.append('| # | Group | Family | Tier | N | Sum cost |')
lines.append('|---:|---|---|---|---:|---:|')
sorted_keys = sorted(groups_data.keys(), key=lambda k: -groups_data[k]['sum_cost'])
for i, k in enumerate(sorted_keys, 1):
    g = groups_data[k]
    lines.append(f'| {i} | `{k}` | {g["family"]} | {g["tier"]} | {len(g["members"])} | {g["sum_cost"]:,} |')

lines.append('\n---\n')

for i, k in enumerate(sorted_keys, 1):
    g = groups_data[k]
    lines.append(f'## {i}. `{k}` ({len(g["members"])} tasks, sum cost {g["sum_cost"]:,})\n')
    lines.append(f'**Family**: {g["family"]} ({g["tier"]} cost tier)\n')
    lines.append(f'**Description**: {g["description"]}\n')
    lines.append('**Tasks** (by cost desc):\n')
    lines.append('| task | cost | transformations | complexity |')
    lines.append('|---|---:|---|---:|')
    for t in g['members'][:50]:  # cap display at 50; full list in prompt
        cmplx = t['complexity'] if t['complexity'] else '?'
        trans = (t['transformations'] or '')[:50]
        lines.append(f'| {t["tid"]} | {t["cost"]:,} | {trans} | {cmplx} |')
    if len(g['members']) > 50:
        lines.append(f'| ... ({len(g["members"]) - 50} more) | | | |')
    lines.append('')
    lines.append('**Copy-paste prompt for this group**:\n')
    lines.append(make_prompt(k, g))
    lines.append('')

Path('Logs/worker_groups_10_on_5835_30.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote Logs/worker_groups_10_on_5835_30.md  ({len("".join(lines)):,} chars)')

print(f'\nGroup sizes:')
for k in sorted_keys:
    g = groups_data[k]
    print(f'  {k:<15} N={len(g["members"]):>3}  sum_cost={g["sum_cost"]:>10,}  ({g["family"]} / {g["tier"]})')
