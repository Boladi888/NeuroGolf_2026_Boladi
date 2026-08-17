# task349 — archived history (moved 2026-06-24 to keep task349.md under budget)

This is the legacy **LIVE-byte-compression** ladder (V1–V11, 2026-05). These all
predate the decoded-generator rewrite (the worker3 accumulator-fold, current best
15710). They started from the old 201200 F16 LIVE graph and shrank it by op
substitution; they are superseded by the from-scratch accumulator architecture.
Kept for provenance only. See `task349.md` for the current rule, attempts, and floor.

## Attempts (legacy)
- 2026-05-19 | F16 output trick (drop Cast(f16->f32)->output, expose f16 intermediate as output) | Verified 100% adversarial. cost 201200->183200 (-18000, +0.0937 pts). Build: `work/build_f16_trick_sweep.py`. Staged in `f16trick_9tasks_on_5764_4228_20260519T1546Z` [A771767D]. | cost 183200 | worker-7 session 2026-05-19

| Date | Candidate | Result | Cost | Notes |
|------|-----------|--------|------|-------|
| 2026-05-18 | worker-7 / card `t_9a70100c` | **10-site Equal-fold**: collapse 5 `Less(Mul(Sub(ConvOut, k_scalar), self), 0.0001)` chains into `Equal(ConvOut, k_scalar)` AND 5 `Less(ConvOut, 0.5)` into `Equal(ConvOut, v7=0.0)`. Valid: all 10 ConvOut are sums-of-one-hot = exact-int F16. | **-18001 cost** (mem 201060->183060; params 140->139). 30/30 byte-exact; verify 267/267; adversarial 267/267 ×7. | 183199 | +0.0937 pts. |
| 2026-05-21 | worker-7 / radical | **V3a — remove Pad-Slice shift from col fill**: feed Mul directly into MaxPool; frame mask v137 already handles overlaps. | 267/267 verify, 4806/4806 adv. | 105473 | not staged (V4 superseded). |
| 2026-05-21 | worker-7 / radical | **V4 — merged Conv**: 5×(eq1+eq2+And) → 5×(merged_Conv+Equal+Cast). Kernel `[-1,1×W,-1]`, pads=[0,1,0,W] = W iff isolated W-run. | 267/267 verify, 4806/4806 adv. | 84699 | not staged (V5 superseded). |
| 2026-05-21 | worker-7 / radical | **V5 — BOOL v144 shortcut**: NOT/AND/Where(BOOL) replaces F16 Where chain. | 267/267 verify, 4806/4806 adv. | **82898** | **SHIPPED** +0.79 LB (CHANGELOG 2026-05-21). Path A only. |
| 2026-05-29 | claude / V6 | UINT8 narrowing of 24 F16 [1,1,30,30] masks + CumSum→MaxPool(31×1) running-OR. Sum→Add chain (Sum rejects U8). | 267/267 + 5000/5000 bit-equal ×3. | 63097 | not staged. |
| 2026-05-29 | claude / V7 | Max replaces Add chain (for 0/1, Max==Sum>0). | 267/267 + 8000/8000 ×4. | 57697 | not staged. |
| 2026-05-29 | claude / V8 | QLinearConv INT8 weights `[-1,+1×W,-1]`, U8 out, y_zp=2. Equal threshold `W+2`. Drops v52 F16. | 267/267 + 10000/10000 ×5. | 51401 | not staged. |
| 2026-05-29 | claude / V9 | UINT8 shift-NOT chain (Slice+Pad+Sub) replaces BOOL Greater+Pad+Slice+Not+Cast. Cast(F32→U8) direct (one-hot guarantee). | 267/267 + 10000/10000 ×5. | 49536 | superseded by V10. |
| 2026-05-29 | claude / V10 ALG REWRITE | Drops per-W col-fill chain → single `MaxPool(ch9_u8, k=(31,1), pad=(30,0,0,0))` "any 9 above in column". Task-shaped-clean (diverges on non-task-shaped, §27d-supplement task285). | 267/267 + 15000/15000 task-shaped + 46.2% generic (expected). | 36955 | superseded by V11. |
| 2026-05-29 | claude / V11 QLINEARCONV BIAS | per-W bias `B=-W+1` makes QLinearConv = 1 iff sum=W else saturate 0. Drops 5×_meq BOOL + 5×Cast U8 = -9000B. | 267/267 + **45000/45000 task-shaped**. | **27955** | **STAGED** `task349_v11_qlinearconvbias_on_current/`. +1.087 LB. (This is the stale local 27955 — beaten by Aditya's 17509 and worker3's 15710.) |

## Cost analysis (legacy)

Live cost 201200 (memory 201060 + params 140). Memory dominates (99.93%). The 5
Sub+Mul squared-distance chains (k=2,4,6,8,10) were ~15000 bytes; Equal-folding
removed nearly all. The 5 Less<0.5→Equal substitutions save no memory (same shape)
but drop a threshold init. Base ConvOut values are sums of one-hot cells through
all-ones / end-cap kernels → integer-valued, so squared-distance / small-positive
thresholds equal integer equality (all-ones width-k: ConvOut ∈ {0..k}; end-cap
[1,0..,1]: ConvOut ∈ {0,1,2}).
