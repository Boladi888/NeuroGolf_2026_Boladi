# task315 active-mask row elimination certificate

Reviewer lock: start from live SHA `013dbc694b331f0991b4e4f9a637a01c318e96a0f89289052e15bf526280ebb7` and try only a one-node terminal `Einsum` rewrite that removes the active-mask row from the live coordinate basis `B[3,30]`.

## Bar

Authoritative refreshed bar: 146 from `python tools/ng_current_bar.py --task 315 --json`, manifest owner Vamshi, updated `2026-07-02T02:35:15Z`. Live profile: static/profile 146, memory 0 + params 146.

## Live final equation and charged shapes

Live node:

`Einsum('nehw,eq,ek,ajl,jm,mh,lo,oh,dst,su,uw,tv,vw,nfHW,fr,bzA,zB,BH,AC,CH,gDE,DF,FW,EG,GW,aIJ,IK,Kx,JL,Lx,bMN,MO,Ox,NP,Px,dQR,QS,Sy,RT,Ty,gUV,UX,Xy,VY,Yy,cp,ci,pqr,ik->ncxy')`

Initializers:

- `B`: `[3,30]` = 90 params; rows are active mask, `x % 3`, `floor(x / 3)` on active positions and zero outside.
- `E`: `[10,2]` = 20.
- `P`: `[3,2,2]` = 12.
- `T`: `[2,2,2]` = 8.
- `WL`: `[2,3]` = 6.
- `WB`: `[2,3]` = 6.
- `Q`: `[2,2]` = 4.

## Locked rewrite shape

The locked row-elimination rewrite would replace `B[3,30]` with `B2[2,30]` containing only the remaining live coordinate rows:

- `B2[0,x] = x % 3` for active `x < 9`, else 0.
- `B2[1,x] = floor(x / 3)` for active `x < 9`, else 0.

`WL/WB` can be shrunk or folded, and `P/T/Q/E` can be retuned, but a one-node terminal `Einsum` still receives output coordinate `x` only through products of `B2[:,x]` and coordinate-independent initializers.

## Infeasibility proof

For the row-eliminated live basis, `B2[:,0] == B2[:,9] == [0,0]`. Therefore any one-node `Einsum` using only this coordinate basis has identical coordinate factors at output row `x=0` and padding row `x=9`. Since all other operands are independent of output row except through `B2[:,x]`, the graph must satisfy:

`output[n,c,0,y] == output[n,c,9,y]` for every input, channel `c`, and output col `y` whenever the same `y` basis is used.

The task requires the opposite. Example train pair 3 has input `(0,0)=2`, so expected active output cell `(0,0)` has channel 2 lit, while row 9 is outside the 9x9 active grid and must be all-zero padding. Thus `output[0,2,0,0] = 1` but `output[0,2,9,0] = 0`.

This contradiction is independent of the tuned values in `P/E/T/Q/WL/WB`: after deleting the active-mask row, the final `Einsum` has no remaining coordinate signal that distinguishes active origin `(0,0)` from padded coordinates whose mod/floor rows are also zero.

## Verdict

No ONNX build was produced for this rework pass. The locked one-node row-0 deletion is dead before build by coordinate collision. The only escape would use a different two-row basis that re-encodes active status into one or both coordinate rows, which is not the requested live-basis row deletion and would no longer be a pure elimination of row 0 from `B=[active, mod, floor]`.
