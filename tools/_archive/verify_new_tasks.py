import os, sys, types, pathlib, numpy as np, warnings
warnings.filterwarnings('ignore')
for n in ('matplotlib','matplotlib.pyplot','IPython','IPython.display'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['IPython.display'].display = lambda *a,**kw: None
sys.modules['IPython.display'].FileLink = lambda *a,**kw: None
sys.modules['IPython'].display = sys.modules['IPython.display']
sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']
NG_ROOT = pathlib.Path('.').resolve()
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / 'data' / 'competitions' / 'neurogolf-2026')))
sys.path.insert(0, str(DATA_DIR / 'neurogolf_utils'))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / 'tools'))
import neurogolf_utils as nu
nu._NEUROGOLF_DIR = str(DATA_DIR).replace('\\', '/') + '/'
from build_submission import (
    _detect_vertical_halves, vertical_halves_network,
    _detect_horizontal_or, horizontal_or_network,
    _detect_boolean_halves, boolean_halves_network,
    _detect_cross_product, cross_product_network,
)
import onnxruntime as ort

def load(tn):
    ex = nu.load_examples(tn)
    return ex.get('train', []) + ex.get('test', []) + ex.get('arc-gen', [])

def run_onnx(model, pairs):
    sess = ort.InferenceSession(model.SerializeToString())
    ok = 0
    for p in pairs:
        a = np.array(p['input'], dtype=np.float32)
        H, W = a.shape
        oh = np.zeros((10, 30, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                oh[int(a[r, c]), r, c] = 1.0
        out = sess.run(None, {'input': oh[np.newaxis]})[0][0]
        Hout, Wout = np.array(p['output']).shape
        pred = np.argmax(out[:, :Hout, :Wout], axis=0)
        if np.array_equal(pred, np.array(p['output'])):
            ok += 1
    return ok

tasks = [
    (72,  _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (144, _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (227, _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (236, _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (318, _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (395, _detect_vertical_halves,  lambda m: vertical_halves_network(m['H_in'],m['W'],m['H_half'],m['mode'],m['out_color'],sep_row=m['sep_row'])),
    (347, _detect_horizontal_or,    lambda m: horizontal_or_network(m['H'],m['W_half'],m['out_color'])),
    (386, _detect_boolean_halves,   lambda m: boolean_halves_network(m['H'],m['W_in'],m['W_out'],m['check_channel'],m['out_color'])),
    (43,  _detect_cross_product,    lambda m: cross_product_network(m['N'],m['col_idx'],m['template_color'],m['out_color'])),
]

for tn, detector, builder in tasks:
    pairs = load(tn)
    m = detector(pairs)
    if m is None:
        print(f'task{tn:03d}: NOT DETECTED')
        continue
    net = builder(m)
    score = run_onnx(net, pairs)
    status = 'OK' if score == len(pairs) else 'FAIL'
    print(f'task{tn:03d}: {score}/{len(pairs)} [{status}]')
