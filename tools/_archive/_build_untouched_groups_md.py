"""One-off script: cluster the 234 untouched tasks into groups of 3-5
and write Logs/untouched_worker_groups_2026-05-19.md with a generic
worker prompt at the end."""
import json
from collections import defaultdict
from pathlib import Path

with open('Logs/untouched_235_features.json') as f:
    feats = json.load(f)

def family(top_ops, n_nodes):
    if n_nodes <= 5: return 'TRIVIAL'
    ops = set(top_ops.keys())
    if 'Conv' in ops: return 'CONV'
    if 'Gather' in ops or 'GatherND' in ops or 'ScatterND' in ops: return 'GATHER'
    if 'MatMul' in ops or 'Gemm' in ops: return 'MATMUL'
    if 'ReduceMax' in ops or 'ArgMax' in ops or 'TopK' in ops: return 'RMAX'
    if 'ReduceSum' in ops or 'ReduceMean' in ops: return 'RSUM'
    if any(o in ops for o in ('Where', 'Less', 'Greater', 'Equal')): return 'PRED'
    if 'Slice' in ops or 'Pad' in ops: return 'CROPPAD'
    if any(o in ops for o in ('Mul', 'Add', 'Sub', 'Cast')): return 'MATH'
    return 'OTHER'

def cost_tier(cost):
    if cost <= 0: return 'NIL'
    if cost >= 50000: return 'HUGE'
    if cost >= 20000: return 'BIG'
    if cost >= 5000:  return 'MID'
    return 'SMALL'

tagged = []
for f in feats:
    if 'error' in f: continue
    tagged.append({
        'tid': f['tid'], 'cost': f['cost'], 'n_nodes': f['n_nodes'],
        'family': family(f['top_ops'], f['n_nodes']),
        'tier': cost_tier(f['cost']),
    })

buckets = defaultdict(list)
for t in tagged:
    buckets[(t['tier'], t['family'])].append(t)
for k in buckets:
    buckets[k].sort(key=lambda x: -x['cost'])

def chunk(tasks, ideal=3, max_size=5):
    out = []
    i, n = 0, len(tasks)
    while i < n:
        remaining = n - i
        if remaining <= max_size:
            out.append(tasks[i:])
            break
        # If taking 'ideal' leaves a sub-3 straggler, absorb stragglers
        # by taking a larger chunk now (up to max_size).
        if remaining - ideal < 3:
            take = min(remaining, max_size)
            if remaining - take in (1, 2):
                take = remaining
        else:
            take = ideal
        out.append(tasks[i:i+take])
        i += take
    return out

groups = {}
for (tier, fam), tasks in sorted(buckets.items()):
    if len(tasks) <= 5:
        groups[f'{tier}_{fam}'] = tasks
    else:
        for idx, c in enumerate(chunk(tasks, ideal=3, max_size=5), 1):
            label = chr(ord('a') + idx - 1)
            groups[f'{tier}_{fam}_{label}'] = c

sizes = sorted(len(v) for v in groups.values())
total = sum(len(v) for v in groups.values())
print(f'Total groups: {len(groups)}')
print(f'Size dist: min={sizes[0]}, p50={sizes[len(sizes)//2]}, max={sizes[-1]}')
print(f'Total tasks: {total}')

with open('Logs/untouched_groups_finetuned.json','w') as f:
    json.dump({k: [{'tid': t['tid'], 'cost': t['cost'], 'n_nodes': t['n_nodes']} for t in v]
               for k, v in groups.items()}, f, indent=2)

sorted_keys = sorted(groups.keys(), key=lambda k: -sum(t['cost'] for t in groups[k]))

FAMILY_GUIDE = {
    'RSUM': {
        'desc': 'Sum/mean reduction-heavy (ReduceSum, ReduceMean - counting / area / voting tasks)',
        'inspiration': 'task030, task200, task163, task277 (add_chain_to_sum); task233, task219 (equal_fold_distance_threshold)',
        'tricks': '1) Sub->Mul(self,self)->Less<1 -> Equal-fold. 2) Add-chain -> single Sum. 3) Mul(x, all-ones-init) -> x.',
    },
    'RMAX': {
        'desc': 'Max-reduction / argmax / TopK (selection / "find the brightest" / "pick winner")',
        'inspiration': 'task354 (mul_by_all_ones_drop), task248, task234 (scalar_broadcast_ones)',
        'tricks': '1) Crop-to-active-grid if max grid < 30. 2) Replace 1-channel-broadcast init with scalar. 3) Slice-axis-collapse for no-op Slices.',
    },
    'CONV': {
        'desc': 'Convolutional (Conv layer - spatial filters, edge detection, neighborhood ops)',
        'inspiration': 'task117 (crop15+F16-output trick, -100525 cost), task374 (conv_run_combined, -11728), task381 (crop10), task097 (crop20)',
        'tricks': '1) Crop-to-active-grid: prepend Slice + rewrite range30/c29i/c30i + append Pad. 2) F16-output trick: drop final Cast(F16->F32) when output is binary. 3) Conv kernel pruning.',
    },
    'GATHER': {
        'desc': 'Gather / lookup-table (Gather/GatherND/ScatterND - color-mapping, KNN-style indexing)',
        'inspiration': 'task266 (scalar_broadcast), task167/task103 (opset9_attr_shrink), task129 (tile_to_edge_pad)',
        'tricks': '1) Compress lookup-table init (often all-zeros tail). 2) Replace gather with arithmetic if table is structured. 3) WARNING: do NOT pre-crop if Flatten->MatMul/Gather is downstream (KNN blocker).',
    },
    'PRED': {
        'desc': 'Predicate / boolean-mask-driven (Where/Less/Greater/Equal - conditional logic)',
        'inspiration': 'task233 (Equal-fold), task219 (Equal-fold), task218',
        'tricks': '1) Sub->Mul(self,self)->Less<1 -> Equal. 2) Fold redundant Where(cond,x,x) -> x. 3) Combine consecutive predicates into single boolean expression.',
    },
    'MATMUL': {
        'desc': 'Dense linear (MatMul/Gemm - typically Flatten->MatMul classification tails)',
        'inspiration': 'task062 (MatMul). Note: usually NOT crop-friendly (Flatten->MatMul blocker)',
        'tricks': '1) Cast MatMul weights F32->F16 if accuracy permits. 2) Prune zero rows/columns in weight matrix. 3) Replace small MatMul with explicit Conv1x1 + broadcast.',
    },
    'CROPPAD': {
        'desc': 'Slice/Pad-heavy (already cropping/padding - may be over-elaborate)',
        'inspiration': 'task140/task087 (opset10_pad_attr), task129 (tile_to_edge_pad), task005 (onnxsim folded)',
        'tricks': '1) Pad+Slice -> identity collapse. 2) opset upgrade for pad-attr -> pad-input. 3) Constant-fold padding values into static init.',
    },
    'MATH': {
        'desc': 'Pure arithmetic (Mul/Add/Sub/Cast - algebraic chains without higher-level structure)',
        'inspiration': 'task030 (add_chain), task200, task163',
        'tricks': '1) Add-chain -> Sum. 2) Mul(x, all-ones) -> x. 3) Constant-fold runs of arithmetic.',
    },
    'TRIVIAL': {
        'desc': 'Trivial graphs (<=5 nodes - often already near-minimal; payoff rare)',
        'inspiration': 'task056 (opset9_attr_shrink), task129',
        'tricks': '1) Check if a single Constant can replace the whole graph. 2) opset downgrade for attr-vs-input encoding. 3) LOW-priority - skip unless an obvious win.',
    },
    'OTHER': {'desc': 'Unclassified', 'inspiration': 'see other groups', 'tricks': 'try Equal-fold, add-chain-to-sum, scalar_broadcast first.'},
}

L = []
L.append('# Untouched-234 worker groups (post-megastack162 baseline)\n')
L.append('Generated 2026-05-19 against canonical 5806.87. Each group is a self-contained worker assignment.\n')
L.append('Target group size: 3-5 tasks. Smaller groups exist when a structural family has fewer tasks at that cost tier.\n')
L.append(f'**Total groups**: {len(groups)} | **Total tasks**: {total}\n')
L.append('\n## Group index (sorted by cost opportunity)\n')
L.append('| # | Group | N | Sum cost | Family |')
L.append('|---:|---|---:|---:|---|')
for i, k in enumerate(sorted_keys, 1):
    g = groups[k]
    parts = k.split('_')
    fam = parts[1] if len(parts) >= 2 else ''
    L.append(f'| {i} | `{k}` | {len(g)} | {sum(t["cost"] for t in g):,} | {fam} |')

L.append('\n---\n')

def make_prompt(group_name, family_guide, tasks):
    """Build the per-group prompt with GROUP_NAME, family, tricks, inspiration,
    and task list all substituted in. Worker can paste directly."""
    task_list = ', '.join(t['tid'] for t in tasks)
    return f'''```
You are a NeuroGolf 2026 Kaggle worker. Your assigned group is {group_name}.

Working directory: C:\\Users\\Qx\\Documents\\Py_cx\\K-NeuroGolf
Current canonical: submissions/submission.zip @ 5806.87 LB
.venv at .venv/Scripts/python

Your tasks ({len(tasks)}): {task_list}

Family: {family_guide["desc"]}
Inspiration prototypes (look at their build_*.py for technique):
  {family_guide["inspiration"]}
Tricks to try first, in order:
  {family_guide["tricks"]}

Per task:
1. Extract canonical taskNNN.onnx from submissions/submission.zip.
2. Inspect it: node count, op-type histogram, value_info shapes. Look
   for 30x30 intermediates (crop opportunity) and Flatten->MatMul/Gather
   (crop blocker).
3. Look at the inspiration prototypes under
   The_400_Tasks/Task_by_Task/<inspirational_task>/prototypes/build_*.py
   - they are shipped exemplars in the same family; mirror their technique.
4. Apply the tricks above in order.
5. If all polish-style tricks fail to lower cost AND the task is still
   worth points (cost >= 5000), REBUILD the ONNX from scratch in a
   different algorithm family:
     - Conv-heavy   -> try predicate-mask reformulation
     - Gather-heavy -> try ReduceMax-based one-hot lookup
     - RSUM         -> try Conv1x1 with constant weights
     - RMAX         -> try ArgMax + ScatterND
     - PRED         -> try direct arithmetic on integer-encoded grids
   The rebuild MUST verify_against_live 100% M/M before staging.
6. Verify EVERY candidate before staging:
     .venv/Scripts/python tools/verify_against_live.py \\
       --proto <new.onnx> --tid <NNN> --live-zip submissions/submission.zip
   Required: "OK Prototype matches LIVE" AND "Prototype vs LIVE: N/N"
   with no failures. HARD GATE - no exceptions. Sub-100% verify drops
   the candidate (see Logs/lessons.md 2026-05-19 wildcard regression:
   3 sub-100% candidates regressed -13 to -46 LB).
7. Score the candidate:
     .venv/Scripts/python tools/spread_score_isolated.py <new.onnx>
   Must be strictly lower cost than canonical taskNNN.onnx. Drop if equal
   or higher.
8. Stage winners under:
   .hermes/board/staged_submits/taskNNN_<technique>_t_<8hex>_on_5806_87/
   with taskNNN.onnx + manifest.json (template in
   .ai_instructions/playbooks/SpreadSweepWorker.md step 7).
9. Append a row to The_400_Tasks/Task_by_Task/taskNNN.md under "Attempts"
   regardless of outcome - capture what was tried and the verdict.
10. Final commit:
     git add -A The_400_Tasks/Task_by_Task/ .hermes/board/staged_submits/
     git commit -m "untouched {group_name}: N staged candidates, B bytes saved"

Rules:
- ONE task at a time. Independent verify + score gates per task.
- DO NOT submit to Kaggle. Submit-captain has exclusive Kaggle authority.
- DO NOT edit submissions/submission.zip directly.
- DO NOT call neurogolf_sync_current_best.py.
- DO NOT skip the verify gate.
- If all {len(tasks)} tasks produce no staged candidate, STOP and report.

When done, report:
  {group_name} complete: N staged candidates, B bytes saved, expected ~X LB
```'''

for i, k in enumerate(sorted_keys, 1):
    g = groups[k]
    parts = k.split('_')
    fam = parts[1] if len(parts) >= 2 else ''
    guide = FAMILY_GUIDE.get(fam, FAMILY_GUIDE['OTHER'])
    L.append(f'## {i}. `{k}` ({len(g)} tasks, sum cost {sum(t["cost"] for t in g):,})\n')
    L.append(f'**Family**: {guide["desc"]}\n')
    L.append(f'**Inspiration prototypes**: {guide["inspiration"]}\n')
    L.append(f'**Tricks to try first**: {guide["tricks"]}\n')
    L.append('**Tasks** (by cost desc):\n')
    L.append('| task | cost | n_nodes |')
    L.append('|---|---:|---:|')
    for t in g:
        L.append(f'| {t["tid"]} | {t["cost"]:,} | {t["n_nodes"]} |')
    L.append('')
    L.append('**Copy-paste prompt for this group**:\n')
    L.append(make_prompt(k, guide, g))
    L.append('')

Path('Logs/untouched_worker_groups_2026-05-19.md').write_text('\n'.join(L), encoding='utf-8')
print('Wrote Logs/untouched_worker_groups_2026-05-19.md')
