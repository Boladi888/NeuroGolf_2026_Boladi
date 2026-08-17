# task055 — compressed history (moved from task055.md 2026-06-06; task233 convention)

## Architecture 1: staircase-code compare-form (v1-v7, superseded by bitand v8-v12)

Full pattern write-up: `Onnx_Advanced/techniques.md` #58. v7 = the compare-arch floor
(2,386), fully certified + staged at `staged_submits/task055_staircase_v7_u8_cost2386`.

### Algorithm sketch (v7)

```
r8[h] = sum_w 1[inp==8]   via Einsum('bchw,c->bh', input, onehot8)   # [1,30] F32, free input
c8[w] = sum_h ...         via Einsum('bchw,c->bw', ...)
act = r8 != 0 (Cast->BOOL);  line = r8 > 2          # counts {0,2,W}: >2 <=> divider
pre = CumSum(line as F16)                            # 0=above, 1=between, 2=below
top = (pre<1)&act; mid = (pre==1)&!line&act; bot = (pre>1)&!line&act   # act-gated = LIVE Where semantics
X[1,10,30,1] U8 = stack[top+bot, bot, top, mid, mid, 0, mid, 0, act+line, 0]   # row codes
Y[10,1,30]  U8 = stack[min(!l,!r), !mc, !mc, !r, !l, y8, !mc, y8, (!act)+(!line), y8]
output = Greater(X, Y)    # broadcast [1,10,30,30] BOOL = the FREE graph output
```
Decoder's `>0.0` IS the comparison. ch8 = 3-level staircase: x8=act+line vs
y8=2-(act+line); x8>y8 <=> (line_r&act_c)|(act_r&line_c). ch0 = corners = tb_r x lr_c.

### Cost analysis (v7 = 2,340 mem + 46 params; construction-floor table)

| stage | bytes | notes |
|---|---:|---|
| 2x Einsum [1,30] F32 | 240 | ORT Einsum is F32-only on CPU; 2-operand form trusted (S42) |
| act/line cmp + unsqueeze (BOOL) | 240 | compare on [1,30], unsqueeze the 1-byte bools (-120 vs F32 unsq) |
| 2x (Cast F16 + CumSum F16) | 240 | CumSum dtype floor = F16 (no U8/I8 kernel; F16 needs opset>=14) |
| band bools p0,p1,p2,nl,na,b0,b1,b2 x2 | 480 | p0=Less(pre,1), p2=Greater(pre,1) — only k1 scalar survives |
| row codes (3 casts + tb-Add + act/line casts + m8r) | 240 | tb=Add(u_top,u_bot) beats Or+Cast by 30 |
| col codes (3 Not-bools + 4 casts + Min + nact chain + y8) | 300 | y_lr=Min(u8) beats And+Cast by 30; De Morgan reuses !l,!r |
| X concat [1,10,30,1] U8 + Y concat [10,1,30] U8 | 600 | the assembly floor under compare-arch: 2 stacks |
| output Greater | 0 | named 'output' -> skipped by scorer |
| params | 46 | oh8[10], f2, k1, ax2 (shared via Y rank-3), uxr[2], uxc[1], ZR8[30] |

Compare-arch floor ~2,300-2,400: a fundamentally different assembly must beat the 600B
stack pair with <3 operands feeding a free output op — that's what the bitand
architecture (techniques #58b) did: `BitwiseAnd(BitwiseAnd(x, K[1,10,1,1]), y)` puts the
channel expansion at 300 B and dropped the floor to 1,141 (v12).

### v1 (3,801, F16 codes + Sub output)

Same staircase, F16 codes, `output = Sub(X, Y)` (decoder >0 compares). Fully certified
(vis 263/263, truth 200k, ab 200k byte-equal); staged at
`staged_submits/task055_staircase_v1_cost3801` as the most conservative rung
(F16 output dtype identical to LIVE, opset 17, no U8/BOOL novelty).
