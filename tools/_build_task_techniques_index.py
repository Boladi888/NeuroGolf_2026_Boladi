"""Build Onnx_Advanced/task_techniques_index.md.

Sources:
  1. xlsx Tasks sheet — family, current_cost, transformations
  2. The_400_Tasks/Task_by_Task/task*/staged_candidates/**/manifest.json — pattern/method/method-name per task
  3. The_400_Tasks/Task_by_Task/taskNNN/prototypes/build_*.py — file listing per task

Group by family. Within family, sub-group by technique class. Each line is a one-line entry pointing to exemplar (if it exists) or to the build script (if it doesn't).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import openpyxl

# 1) Load family + cost from xlsx
wb = openpyxl.load_workbook('The_400_Tasks/NeuroGolf_Task_Status.xlsx')
ws = wb['Tasks']
headers = [c.value for c in ws[1]]


def ci(name):
    return headers.index(name) + 1


task_info = {}
for r in range(2, ws.max_row + 1):
    tid = ws.cell(row=r, column=ci('task')).value
    if not tid:
        continue
    task_info[tid] = {
        'family': ws.cell(row=r, column=ci('family')).value or 'Unclassified',
        'transformations': ws.cell(row=r, column=ci('transformations')).value or '',
        'current_cost': ws.cell(row=r, column=ci('current_cost')).value or 0,
    }

# 2) Mine task-local staged candidate manifests for technique attributions
manifest_tech = defaultdict(list)  # tid -> list of {pattern, method_snippet, dir}
staged_root = Path('The_400_Tasks/Task_by_Task')
for mf in staged_root.glob('task*/staged_candidates/**/manifest.json'):
    d = mf.parent
    try:
        m = json.loads(mf.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        continue
    tid = m.get('task_id') or m.get('task')
    if isinstance(tid, int):
        tid = f'task{tid:03d}'
    if not tid:
        mm = re.search(r'(task\d{3})', d.as_posix())
        tid = mm.group(1) if mm else None
    if not tid:
        continue
    pattern = m.get('pattern') or m.get('attempt_kind') or ''
    method = m.get('method') or ''
    cost_delta = (m.get('scores') or {}).get('cost_delta')
    manifest_tech[tid].append({
        'pattern': pattern,
        'method': method[:150],
        'cost_delta': cost_delta,
        'dir': d.as_posix(),
    })

# 3) Mine prototypes/ for build script names
proto_scripts = defaultdict(list)  # tid -> list of build_*.py
proto_root = Path('The_400_Tasks/Task_by_Task')
for tid in task_info:
    pdir = proto_root / tid / 'prototypes'
    if pdir.exists():
        for f in pdir.rglob('build_*.py'):
            proto_scripts[tid].append(f.name)

# 4) Map task -> exemplar (if we have one — to be filled as exemplars are written)
EXEMPLAR_MAP = {
    'task376': 'exemplars/01_whole_graph_rewrites/task376_input_to_bool.md',
    'task049': 'exemplars/01_whole_graph_rewrites/task049_least_color_bbox.md',
    'task382': 'exemplars/01_whole_graph_rewrites/task382_cumsum_gather.md',
    'task119': 'exemplars/01_whole_graph_rewrites/task119_diag_conv.md',
    'task115': 'exemplars/01_whole_graph_rewrites/task115_axis_factor.md',
    'task293': 'exemplars/02_bool_first/task293_andfold.md',
    'task021': 'exemplars/02_bool_first/task021_mul_mask_to_where.md',
    'task196': 'exemplars/02_bool_first/task196_or_cast_flip_where.md',
    'task273': 'exemplars/02_bool_first/task273_drop_pad_cast_bool.md',
    'task342': 'exemplars/02_bool_first/task273_drop_pad_cast_bool.md',
    'task319': 'exemplars/02_bool_first/task319_opset_bump_pad_uint8.md',
    'task299': 'exemplars/02_bool_first/task299_bool_concat.md',
    'task233': 'exemplars/02_bool_first/task233_equal_fold.md',
    'task368': 'exemplars/02_bool_first/task368_shared_cast_equal_fold.md',
    'task359': 'exemplars/02_bool_first/task359_bool_first_where.md',
    'task117': 'exemplars/03_crop_and_f16/task117_crop15_f16.md',
    'task055': 'exemplars/03_crop_and_f16/task055_per_channel.md',
    'task128': 'exemplars/03_crop_and_f16/task128_crop15.md',
    'task281': 'exemplars/03_crop_and_f16/task281_crop13.md',
    'task364': 'exemplars/03_crop_and_f16/task364_crop22.md',
    'task384': 'exemplars/03_crop_and_f16/task384_f16_drop.md',
    'task133': 'exemplars/04_algebraic_identities/task133_equal_fold_reducesum.md',
    'task327': 'exemplars/04_algebraic_identities/task327_add_chain_to_sum.md',
    'task349': 'exemplars/04_algebraic_identities/task349_10_site_equal_fold.md',
    'task163': 'exemplars/04_algebraic_identities/task163_add_chain.md',
    'task071': 'exemplars/05_static_to_dynamic/task071_ring_mask.md',
    'task201': 'exemplars/05_static_to_dynamic/task201_reduce_reorder.md',
    'task306': 'exemplars/06_cse_dead_op/task306_slice_cse.md',
    'task093': 'exemplars/06_cse_dead_op/task093_dead_relu_post_maxpool.md',
    'task302': 'exemplars/06_cse_dead_op/task302_drop_relu_shift.md',
    'task240': 'exemplars/06_cse_dead_op/task240_dead_end.md',
    'task374': 'exemplars/07_conv_reformulation/task374_conv_run_combined.md',
    'task323': 'exemplars/07_conv_reformulation/task323_conv_bool.md',
    'task266': 'exemplars/08_gather_lookup/task266_scalar_broadcast.md',
    'task167': 'exemplars/08_gather_lookup/task167_opset9_attr_shrink.md',
    'task129': 'exemplars/08_gather_lookup/task129_tile_to_edge_pad.md',
    'task118': 'exemplars/08_gather_lookup/task118_bool_scatter.md',
}


# 5) Bucket tasks within each family by technique class. Heuristic: use the most
#    informative source of attribution — manifest 'pattern' field first, then
#    'method' keywords, then file-name keywords.
TECH_CLASSES = [
    # (label, predicate)
    ('Whole-graph rewrites', lambda techs, fs: any(
        'input_to_bool' in t or 'bool_input' in t or 'fresh' in t or 'rebuild' in t or
        'diag_conv' in t or 'cumsum' in t or 'axisfactor' in t or 'bbox' in t
        for t in techs)),
    ('Bool-first family (input/Concat/Cast/Pad bool keeping)', lambda techs, fs: any(
        'bool' in t or 'andfold' in t or 'mul_mask' in t or 'or_cast' in t or
        'flip_where' in t or 'concat_bool' in t or 'drop_cast' in t or
        'opset_bump' in t or 'pad_bool' in t or 'uint8' in t
        for t in techs)),
    ('Crop-to-active-grid + F16 placement', lambda techs, fs: any(
        re.match(r'crop\d+', t) or 'crop' in t.lower() or 'f16' in t.lower() or
        'cast_after' in t or 'f16drop' in t or 'f16out' in t
        for t in techs)),
    ('Algebraic identities (Equal-fold, Add-chain, scalar)', lambda techs, fs: any(
        'equal_fold' in t or 'equalfold' in t or 'add_chain' in t or
        'scalar_broadcast' in t or 'shared_cast' in t or 'cast_fold' in t
        for t in techs)),
    ('Static-to-dynamic / rank reduction / projection-cache', lambda techs, fs: any(
        'ring' in t or 'dyn' in t or 'reduce_reorder' in t or 'reduce_sum_reorder' in t
        for t in techs)),
    ('CSE / dead-op elimination', lambda techs, fs: any(
        'cse' in t or 'noop' in t or 'dce' in t or 'drop_relu' in t or 'shift16' in t
        for t in techs)),
    ('Conv reformulation', lambda techs, fs: any(
        'conv_bool' in t or 'conv_run' in t or 'diag_conv' in t
        for t in techs)),
    ('Gather/lookup compression', lambda techs, fs: any(
        'gather' in t or 'lookup' in t or 'scatter' in t or 'opset9' in t or
        'tile_to' in t or 'attr_shrink' in t
        for t in techs)),
]


def classify_techniques(techs, fs):
    """Return list of class labels that apply to the task."""
    labels = []
    for label, pred in TECH_CLASSES:
        if pred(techs, fs):
            labels.append(label)
    return labels or ['Uncategorized (no clear technique tag yet)']


# 6) Compose the markdown
families = ['Pattern_Recognition', 'Object_Based', 'Color_and_Logical', 'Spatial_and_Geometric']

lines = []
lines.append('# task_techniques_index.md — by family, by technique class\n')
lines.append('Grouped task index. For each family, sub-grouped by technique class. Each entry:')
lines.append('- `taskNNN` (cost) → exemplar path if available, else `prototypes/build_*.py` file, else "no exemplar — see staged manifest".')
lines.append('')
lines.append('**Usage**: locate your task ID below. Read the listed exemplar(s) AND any in the same technique class for variant detail.')
lines.append('')
lines.append('---\n')

for family in families:
    family_tasks = [(tid, info) for tid, info in task_info.items() if info['family'] == family]
    family_tasks.sort(key=lambda x: -x[1]['current_cost'])

    lines.append(f'## {family} ({len(family_tasks)} tasks)\n')

    # Build per-class buckets within the family
    class_buckets = defaultdict(list)
    untagged = []
    for tid, info in family_tasks:
        techs = []
        for m in manifest_tech.get(tid, []):
            techs.append(m['pattern'])
            techs.append(m['dir'])
            techs.append(m['method'])
        fs = proto_scripts.get(tid, [])
        techs.extend(fs)
        techs_str = ' '.join(t.lower() if t else '' for t in techs)
        labels = classify_techniques([techs_str], fs)
        if labels == ['Uncategorized (no clear technique tag yet)']:
            untagged.append((tid, info, fs))
            continue
        for label in labels:
            class_buckets[label].append((tid, info, fs))

    # Emit each class bucket
    for label, pred in TECH_CLASSES:
        bucket = class_buckets.get(label, [])
        if not bucket:
            continue
        lines.append(f'### {label}\n')
        for tid, info, fs in bucket:
            exemplar = EXEMPLAR_MAP.get(tid)
            best_manifest = sorted(
                manifest_tech.get(tid, []),
                key=lambda x: x.get('cost_delta', 0) or 0,
            )
            best = best_manifest[0] if best_manifest else None
            cost = info['current_cost']
            delta_str = ''
            if best and best.get('cost_delta'):
                delta_str = f' (staged Δ={best["cost_delta"]:+,})'
            if exemplar:
                lines.append(f'- `{tid}` ({cost:,}){delta_str} → [{exemplar}]({exemplar})')
            elif fs:
                fs_short = ', '.join(fs[:2])
                lines.append(f'- `{tid}` ({cost:,}){delta_str} → `The_400_Tasks/Task_by_Task/{tid}/prototypes/{fs_short}`')
            elif best:
                lines.append(f'- `{tid}` ({cost:,}){delta_str} → see staged `{best["dir"]}/manifest.json`')
            else:
                lines.append(f'- `{tid}` ({cost:,})')
        lines.append('')

    if untagged:
        lines.append(f'### Untagged ({len(untagged)} tasks — no transferable technique attribution yet)\n')
        lines.append('These tasks have no entries in task-local `staged_candidates/` manifests and no build_*.py in their `prototypes/` dir. They are candidates for future worker attention. Listed by cost descending:')
        lines.append('')
        for tid, info, _ in untagged[:30]:  # cap untagged listing
            lines.append(f'- `{tid}` ({info["current_cost"]:,}) — {info["transformations"][:50]}')
        if len(untagged) > 30:
            lines.append(f'- ... ({len(untagged) - 30} more untagged tasks)')
        lines.append('')

    lines.append('---\n')


# 7) Save
out = Path('Onnx_Advanced/task_techniques_index.md')
out.write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote {out} ({sum(len(l) for l in lines):,} chars across {len(lines)} lines)')
