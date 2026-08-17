"""Quick characterization scan of all unsolved tasks.

Prints summary statistics per task to help identify promising pattern clusters.
"""
import os, sys, types, pathlib, collections
for n in ('matplotlib','matplotlib.pyplot','IPython','IPython.display'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['IPython.display'].display = lambda *a,**kw: None
sys.modules['IPython.display'].FileLink = lambda *a,**kw: None
sys.modules['IPython'].display = sys.modules['IPython.display']
sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / 'data' / 'competitions' / 'neurogolf-2026')))
sys.path.insert(0, str(DATA_DIR / 'neurogolf_utils'))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / 'tools'))
import neurogolf_utils as nu
nu._NEUROGOLF_DIR = DATA_DIR.as_posix() + '/'
from inspect_tasks import SOLVED_TASKS

def shapes(pairs):
    return [(len(p['input']), len(p['input'][0]) if p['input'] else 0,
             len(p['output']), len(p['output'][0]) if p['output'] else 0)
            for p in pairs]

def n_colors_in(pairs):
    return [len({v for row in p['input'] for v in row}) for p in pairs]

def n_colors_out(pairs):
    return [len({v for row in p['output'] for v in row}) for p in pairs]

def size_consistent(pairs):
    sh = shapes(pairs)
    return len(set(sh)) == 1

def same_size(pairs):
    sh = shapes(pairs)
    return all(s[0]==s[2] and s[1]==s[3] for s in sh)

def colors_in(pairs):
    return sorted({v for p in pairs for row in p['input'] for v in row if v != 0})

def colors_out(pairs):
    return sorted({v for p in pairs for row in p['output'] for v in row if v != 0})

unsolved = [t for t in range(1, 401) if t not in SOLVED_TASKS]

# Quick scan: consistent shapes, same in/out size, color counts
rows = []
for tn in unsolved:
    ex = nu.load_examples(tn)
    pairs = ex.get('train',[]) + ex.get('test',[]) + ex.get('arc-gen',[])
    if not pairs:
        continue
    sh = shapes(pairs)
    ish = set((s[0],s[1]) for s in sh)
    osh = set((s[2],s[3]) for s in sh)
    same = same_size(pairs)
    ci = colors_in(pairs)
    co = colors_out(pairs)
    n_in = n_colors_in(pairs)
    n_out = n_colors_out(pairs)
    in_shape = list(ish)
    out_shape = list(osh)
    rows.append({
        'tn': tn,
        'npairs': len(pairs),
        'n_in_shapes': len(ish),
        'n_out_shapes': len(osh),
        'same': same,
        'in_shape': in_shape[0] if len(ish)==1 else None,
        'out_shape': out_shape[0] if len(osh)==1 else None,
        'ci': ci,
        'co': co,
        'n_in_min': min(n_in),
        'n_in_max': max(n_in),
        'n_out_min': min(n_out),
        'n_out_max': max(n_out),
    })

# Group by interesting properties
print("=== SAME SIZE, SINGLE IN-SHAPE, SINGLE OUT-SHAPE ===")
subset = [r for r in rows if r['same'] and r['n_in_shapes']==1 and r['n_out_shapes']==1]
print(f"Count: {len(subset)}")
for r in subset:
    print(f"  t{r['tn']:03d}: {r['in_shape']} npairs={r['npairs']} ci={r['ci']} co={r['co']}")

print()
print("=== SHAPE CHANGING, CONSISTENT SHAPES ===")
subset2 = [r for r in rows if not r['same'] and r['n_in_shapes']==1 and r['n_out_shapes']==1]
print(f"Count: {len(subset2)}")
for r in subset2:
    print(f"  t{r['tn']:03d}: {r['in_shape']}->{r['out_shape']} npairs={r['npairs']} ci={r['ci']} co={r['co']}")

print()
print("=== VARIABLE SHAPE TASKS ===")
subset3 = [r for r in rows if r['n_in_shapes']>1 or r['n_out_shapes']>1]
print(f"Count: {len(subset3)}")
for r in subset3:
    print(f"  t{r['tn']:03d}: npairs={r['npairs']} in_shapes={r['n_in_shapes']} out_shapes={r['n_out_shapes']} ci={r['ci']} co={r['co']}")
