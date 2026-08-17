import os, sys, types, pathlib, numpy as np, math
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
import build_submission as bs
import onnx, onnxruntime as ort

m = bs.rank_vertical_bars_network(batch=1)
print(f'Built: {len(m.graph.node)} nodes, {len(m.graph.initializer)} inits')
onnx.save(m, 'submissions/_task010.onnx')
ok = nu.check_network('submissions/_task010.onnx')
print(f'check_network: {ok}')
ex = nu.load_examples(10)
pairs = ex.get('train', []) + ex.get('test', []) + ex.get('arc-gen', [])
sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
right, wrong, _ = nu.verify_subset(sess, pairs)
print(f'verify: {right}/{len(pairs)}, wrong={wrong}')
from _score_v2 import score_network_v2
r = score_network_v2('submissions/_task010.onnx')
cost = r[1] + r[2]
pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
print(f'cost={cost}, pts={pts:.4f}')
