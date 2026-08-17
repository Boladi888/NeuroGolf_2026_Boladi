import os, sys, types, pathlib, numpy as np
for n in ('matplotlib', 'matplotlib.pyplot', 'IPython', 'IPython.display'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['IPython.display'].display = lambda *a, **kw: None
sys.modules['IPython.display'].FileLink = lambda *a, **kw: None
sys.modules['IPython'].display = sys.modules['IPython.display']
sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']
NG_ROOT = pathlib.Path('.').resolve()
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / 'data' / 'competitions' / 'neurogolf-2026')))
sys.path.insert(0, str(DATA_DIR / 'neurogolf_utils'))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / 'tools'))
import neurogolf_utils as nu
nu._NEUROGOLF_DIR = str(DATA_DIR).replace('\\', '/') + '/'

def load(tn):
    ex = nu.load_examples(tn)
    return ex.get('train', []) + ex.get('test', []) + ex.get('arc-gen', [])

# task395: vertical NOR (top 3 rows vs bottom 3 rows)
pairs = load(395)
ok = sum(1 for p in pairs if np.array_equal(
    ((np.array(p['input'])[:3] == 0) & (np.array(p['input'])[3:] == 0)).astype(int) * 2,
    np.array(p['output'])))
print(f"task395 vertical-NOR: {ok}/{len(pairs)}")

# task347: horizontal OR (no separator, W_in=6=2*3)
pairs = load(347)
ok = 0
for p in pairs:
    a = np.array(p['input']); o = np.array(p['output'])
    Wo = a.shape[1] // 2
    left = a[:, :Wo]; right = a[:, Wo:]
    pred = ((left != 0) | (right != 0)).astype(int) * 6
    if np.array_equal(pred, o): ok += 1
print(f"task347 horizontal-OR: {ok}/{len(pairs)}")

# task072: vertical OR with separator (row of all 4s)
pairs = load(72)
ok = 0
for p in pairs:
    a = np.array(p['input']); o = np.array(p['output'])
    sep_row = None
    for r in range(a.shape[0]):
        if np.all(a[r] == 4): sep_row = r; break
    top = a[:sep_row]; bot = a[sep_row+1:]
    pred = ((top == 2) | (bot == 2)).astype(int) * 3
    if np.array_equal(pred, o): ok += 1
print(f"task072 vertical-OR-sep: {ok}/{len(pairs)}")

# task386: horizontal NOR (separator col=1 at col 3)
pairs = load(386)
ok = 0
for p in pairs:
    a = np.array(p['input']); o = np.array(p['output'])
    Wo = 3; sep_col = 3
    left = a[:, :sep_col]; right = a[:, sep_col+1:sep_col+1+Wo]
    pred = ((left == 0) & (right == 0)).astype(int) * 3
    if np.array_equal(pred, o): ok += 1
print(f"task386 horizontal-NOR: {ok}/{len(pairs)}")

# task043: cross-product row0 x col9 marking
pairs = load(43)
ok = 0
for p in pairs:
    a = np.array(p['input']); o = np.array(p['output'])
    row0 = a[0]; col9 = a[:, -1]
    expected = a.copy()
    for r in range(a.shape[0]):
        if col9[r] == 5:
            for c in range(a.shape[1]):
                if row0[c] == 5 and a[r, c] == 0:
                    expected[r, c] = 2
    if np.array_equal(expected, o): ok += 1
print(f"task043 cross-product: {ok}/{len(pairs)}")
