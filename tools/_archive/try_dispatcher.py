"""Try the dispatcher on a list of task numbers and report matches."""
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
import build_submission as bs

def load(tn):
    ex = nu.load_examples(tn)
    return ex.get('train', []) + ex.get('test', []) + ex.get('arc-gen', [])

tasks = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else []
for tn in tasks:
    pairs = load(tn)
    try:
        matches = bs._collect_matches(pairs)
        if matches:
            print(f't{tn:03d}: {[m[0] for m in matches]}')
        else:
            print(f't{tn:03d}: no match')
    except Exception as e:
        print(f't{tn:03d}: ERROR {e}')
