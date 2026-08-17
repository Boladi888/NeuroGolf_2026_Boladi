import onnx, numpy as np
from onnx import shape_inference, TensorProto
p = r"<REPO_ROOT>\Team_Best_Onnx\task017.onnx"
m = onnx.load(p)
m2 = shape_inference.infer_shapes(m, strict_mode=True)
g = m2.graph
itemsize = {TensorProto.FLOAT:4, TensorProto.FLOAT16:2, TensorProto.UINT8:1, TensorProto.INT8:1,
            TensorProto.INT32:4, TensorProto.INT64:8, TensorProto.BOOL:1, TensorProto.DOUBLE:8}
tmap = {t.name: t for t in list(g.input)+list(g.value_info)+list(g.output)}
node_out = {}
for n in g.node:
    for o in n.output:
        if o: node_out[o]=n.op_type
print(f"{'tensor':10} {'op':8} {'dtype':8} {'shape':16} {'bytes':>7}")
total=0
rows=[]
for name,t in tmap.items():
    if name in ('input','output'): continue
    tt=t.type.tensor_type
    dims=[d.dim_value for d in tt.shape.dim]
    ne=1
    for d in dims: ne*=d
    b=ne*itemsize.get(tt.elem_type,4)
    total+=b
    rows.append((b,name,node_out.get(name,'?'),TensorProto.DataType.Name(tt.elem_type),dims))
for b,name,op,dt,dims in sorted(rows,reverse=True):
    print(f"{name:10} {op:8} {dt:8} {str(dims):16} {b:>7}")
print(f"{'TOTAL MEMORY':10} {'':8} {'':8} {'':16} {total:>7}")
# params
params=sum(int(np.prod(i.dims)) for i in g.initializer)
print("PARAMS (init elem count):", params)
print("COST = memory + params =", total, "+", params, "=", total+params)
