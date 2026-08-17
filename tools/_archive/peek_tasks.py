"""Show first few pairs for a list of tasks to spot patterns visually."""
import os, sys, types, pathlib
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

def show(tn, nshow=3):
    ex = nu.load_examples(tn)
    pairs = ex.get('train',[]) + ex.get('test',[]) + ex.get('arc-gen',[])
    print(f'\n=== task{tn:03d} ({len(pairs)} pairs) ===')
    for i, p in enumerate(pairs[:nshow]):
        print(f'  pair {i}: in={p["input"]}')
        print(f'          out={p["output"]}')

tasks = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else []
for t in tasks:
    show(t)
