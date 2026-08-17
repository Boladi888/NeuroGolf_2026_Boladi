import onnx, json, numpy as np, onnxruntime as ort
from onnx import helper

SRC = r"<REPO_ROOT>\Team_Best_Onnx\task017.onnx"
m = onnx.load(SRC); g = m.graph
existing = {o.name for o in g.output}
for n in g.node:
    for o in n.output:
        if o and o not in existing:
            g.output.append(helper.make_empty_tensor_value_info(o)); existing.add(o)
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
names = [o.name for o in g.output]

def encode(grid):
    x = np.zeros((1,10,30,30), dtype=np.float32)
    for r,row in enumerate(grid):
        for c,val in enumerate(row):
            x[0,val,r,c] = 1.0
    return x

def hself(x):  # horizontal all-channel self-conv, shifts 0..9  -> like Sho's G
    # G[p] = sum over c,h,w of x[c,h,w]*x[c,h,w+p]
    out=[]
    for p in range(10):
        out.append(float((x[0,:,:, :30-p]*x[0,:,:, p:]).sum()))
    return np.array(out)

def vself(x):  # vertical all-channel self-conv, shifts 0..9
    out=[]
    for p in range(10):
        out.append(float((x[0,:, :30-p, :]*x[0,:, p:, :]).sum()))
    return np.array(out)

d = json.load(open(r"<REPO_ROOT>\data\competitions\neurogolf-2026\task017.json"))
allex = d['train']+d['test']+d['arc-gen']
Kc = np.array([-999,420,399,378,357,336,315,294,273,252], dtype=float)

mism_dd=0; mism_period=0; n=0
worst=[]
for ex in allex:
    gi=np.array(ex['input']); x=encode(gi.tolist())
    outs = sess.run(None, {'input': x}); res=dict(zip(names,outs))
    G = res['G'].flatten(); dd_true = res['dd'].flatten(); pi=int(res['pi'].flatten()[0])
    Gh = hself(x); Gv = vself(x)
    # check Sho's G == my horizontal self-conv
    if not np.allclose(G, Gh):
        pass
    # hypothesis: dd == raw vertical self-conv Gv ?
    if not np.allclose(dd_true, Gv, atol=1e-3):
        mism_dd += 1
        if len(worst)<5:
            worst.append(('dd', pi, np.round(dd_true,1).tolist(), np.round(Gv,1).tolist()))
    # hypothesis: 2*Gh - Gv == Kc uniquely at true period?
    t_marg = 2*Gh - Gv
    sel = [p for p in range(10) if abs(t_marg[p]-Kc[p])<0.5]
    if sel != [pi]:
        mism_period += 1
        if len([w for w in worst if w[0]=='per'])<5:
            worst.append(('per', pi, sel, np.round(t_marg,1).tolist()))
    n+=1

print(f"examples: {n}")
print(f"dd != raw vertical self-conv Gv : {mism_dd}/{n}")
print(f"2*Gh - Gv fails to uniquely select true period pi: {mism_period}/{n}")
print("\nsamples:")
for w in worst[:8]:
    print(" ", w[0], "pi=",w[1], w[2], w[3] if len(w)>3 else '')
