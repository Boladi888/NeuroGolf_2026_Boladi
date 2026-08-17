import onnx, json, numpy as np, onnxruntime as ort
from onnx import helper

SRC = r"<REPO_ROOT>\Team_Best_Onnx\task017.onnx"
m = onnx.load(SRC)
g = m.graph
# expose every intermediate as a graph output
existing = {o.name for o in g.output}
extra = []
for n in g.node:
    for o in n.output:
        if o and o not in existing:
            g.output.append(helper.make_empty_tensor_value_info(o))
            extra.append(o)
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

def encode(grid):
    x = np.zeros((1,10,30,30), dtype=np.float32)
    for r,row in enumerate(grid):
        for c,val in enumerate(row):
            x[0,val,r,c] = 1.0
    return x

def decode(out):
    # out [1,10,30,30] -> argmax over channel for top-left 21x21
    a = out[0]  # [10,30,30]
    idx = a.argmax(0)
    return idx[:21,:21]

d = json.load(open(r"<REPO_ROOT>\data\competitions\neurogolf-2026\task017.json"))
for split in ['train']:
    for ex_i, ex in enumerate(d[split]):
        gi = np.array(ex['input']); go = np.array(ex['output'])
        x = encode(gi.tolist())
        outs = sess.run(None, {'input': x})
        names = [o.name for o in g.output]
        res = dict(zip(names, outs))
        pred = decode(res['output'])
        ok = np.array_equal(pred, go)
        print(f"=== {split}[{ex_i}] correct={ok} ===")
        # period detection tensors
        G = res['G'].flatten(); GG=res['GG'].flatten(); dd=res['dd'].flatten()
        t = res['t'].flatten(); V=res['V'].flatten(); pi=res['pi'].flatten()
        print(" G  :", np.round(G,1).tolist())
        print(" dd :", np.round(dd,1).tolist())
        print(" t=GG-dd:", np.round(t,1).tolist())
        print(" Kc :", [-999,420,399,378,357,336,315,294,273,252])
        print(" V  :", V.astype(int).tolist(), " -> pi(argmax)=", pi.tolist())
        print(" D shape:", res['D'].shape, " D[0,:3,:6]=\n", np.round(res['D'][0,:3,:6],1))
        if not ok:
            print(" MISMATCH cells:", int((pred!=go).sum()))
        break
    break
print("\nintermediates:", extra)
