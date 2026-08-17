"""task221 floor builder v3 — ConvTranspose-stride-3 kron with UNIFIED-threshold macro mask.
Rule: input 3x3 one-hot in [1,10,30,30]. k=#nonzero, z=9-k=macro M. output=kron(macro,input):
MxM macro of 3x3 tiles, first k row-major = input stamp, rest bg-tile, outside zeros.

Both conv-input channels are one broadcasting compare:  c < min(C[ch] - r*z, z)  with C=[z*z, k].
  ch0 (active) = (r<z)&(c<z)         [bg-fill region]   <=>  c < min(z*z - r*z, z)
  ch1 (stamp)  = active & (r*z+c<k)  [first k cells]     <=>  c < min(k   - r*z, z)
-> no per-cell idx tensor, no separate A/M bool masks. Conv weight [2,10,3,3] = [W_active | [-fg,colors]].
"""
import numpy as np, onnx, sys
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

OPSET = 13

def f32(n,a): return nh.from_array(np.asarray(a,np.float32), n)
def i32(n,a): return nh.from_array(np.asarray(a,np.int32), n)
def i64(n,a): return nh.from_array(np.asarray(a,np.int64), n)

inits, nodes = [], []
# constants
inits.append(i64('s_starts',[1,0,0]))            # input[:,1:10,0:3,0:3]
inits.append(i64('s_ends',[10,3,3]))
inits.append(i64('s_axes',[1,2,3]))
inits.append(i32('row', np.arange(10,dtype=np.int32).reshape(1,1,10,1)))
inits.append(i32('col', np.arange(10,dtype=np.int32).reshape(1,1,1,10)))
inits.append(i32('nine', np.array(9,np.int32).reshape(1,1,1,1)))
inits.append(f32('neg_one', np.array(-1,np.float32).reshape(1,1,1,1)))
inits.append(i64('axis1',[1]))                    # ReduceSum axes (opset13)
wa = np.zeros((1,10,3,3),np.float32); wa[0,0]=1.0 # W_active: ch0=ones, rest 0
inits.append(f32('W_active', wa))

# body
nodes.append(h.make_node('Slice',['input','s_starts','s_ends','s_axes'],['grid']))     # [1,9,3,3]
nodes.append(h.make_node('ReduceSum',['grid'],['nz'],keepdims=1))                       # k (float)
nodes.append(h.make_node('Cast',['nz'],['k'],to=TP.INT32))                              # [1,1,1,1]
nodes.append(h.make_node('Sub',['nine','k'],['z']))                                     # z=9-k
nodes.append(h.make_node('Mul',['z','z'],['zsq']))                                      # z*z
nodes.append(h.make_node('Concat',['zsq','k'],['C'],axis=1))                            # [1,2,1,1]=[z^2,k]
nodes.append(h.make_node('Mul',['row','z'],['row_z']))                                  # r*z [1,1,10,1]
nodes.append(h.make_node('Sub',['C','row_z'],['Tpre']))                                 # C[ch]-r*z [1,2,10,1]
nodes.append(h.make_node('Min',['Tpre','z'],['T']))                                     # min(.,z) [1,2,10,1]
nodes.append(h.make_node('Less',['col','T'],['Xb']))                                    # c<T -> [1,2,10,10] bool
nodes.append(h.make_node('Cast',['Xb'],['X'],to=TP.FLOAT))                              # conv image [1,2,10,10]
# conv weight: stamp row = [-fg, colors]; stacked under W_active
nodes.append(h.make_node('ReduceSum',['grid','axis1'],['Pfg'],keepdims=1))              # fg per cell [1,1,3,3]
nodes.append(h.make_node('Mul',['Pfg','neg_one'],['negPfg']))
nodes.append(h.make_node('Concat',['negPfg','grid'],['W_stamp'],axis=1))                # [1,10,3,3]
nodes.append(h.make_node('Concat',['W_active','W_stamp'],['W'],axis=0))                 # [2,10,3,3]
nodes.append(h.make_node('ConvTranspose',['X','W'],['output'],strides=[3,3]))           # [1,10,30,30]

g = h.make_graph(nodes,'task221_floor',
    [h.make_tensor_value_info('input',TP.FLOAT,[1,10,30,30])],
    [h.make_tensor_value_info('output',TP.FLOAT,[1,10,30,30])], inits)
m = h.make_model(g, opset_imports=[h.make_opsetid('',OPSET)]); m.ir_version=8
onnx.checker.check_model(m, full_check=True)
out = sys.argv[1] if len(sys.argv)>1 else 'task221_v3.onnx'
onnx.save(m, out); print('saved', out)
