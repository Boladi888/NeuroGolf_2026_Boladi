> ⚠️ **SUPERSEDED 2026-07-12 — the "forced floor ≈3963" claim below is REFUTED.** A raw signed-mask
> identity `2·G − Conv(S,S) == 21·(21−p)` (S=1−2·bg, 0 transform params) replaces Sho's rank-5 D, and
> H-only detection (H-period==V-period 266/266) + a small partial-sum correction lands a BYTE-EQUAL
> candidate at **3857 (−106)**. The error here was treating Sho's 510 rank-5 D-factor *params* as forced;
> only the D-plane (~1764) and P (1080) are near-forced. See `task017.md` 2026-07-12 WIN +
> `staged_candidates/task017_rawmask_honly_cost3857/`. The Sho-graph decode + lever tests below remain accurate.

# task017 — FALSE-FLOOR WALL @ 3963 (Sho)  —  DROP  (2026-07-10, bo_ng_cld_worker2)

## Live bar (refreshed `ng_current_bar --task 17`)
- **live_cost = 3963**, owner **Sho**, sha256 `353c425531f79ab7c0ca703042926531dab8e898c0aa833a9ca184cac8265dfc`
- authoritative source = manifest (2026-07-09T01:09Z, points 16.7152); manifest + workbook + Team_Best_Onnx all agree, none stale.
- profile: `static=3963 profile=3963` (memory 3364 + params 599; fully static, no grader-gap).
- **Supersedes** all local narratives: 11582/10153/9892/7802 and the staged **4992** periodic-hole-fill (now ABOVE the bar).

## Prompt target vs reality
- Prompt: `3963 -> ~720`, class SWEEP, gap 5.50×; Direction = "color-index collapse: delete charged carrier/canvas/template"; Fast probe = "replace per-color path with ArgMax/index-plane".
- **The prompt's crack is already realized.** Sho's 3963 graph is already color-collapsed and already uses ArgMax/index-plane. There is no per-color carrier to delete. `720 = team_best/5.5` is an EV heuristic, not a construction.

## Decoded live graph (13 nodes, opset 17, ir 8)
```
G   = Conv(input, input, pads=[0,0,0,9])      # [1,1,1,10] horiz all-channel autocorrelation (match counts per shift 0..9)
GG  = Add(G, G)                                # 2*G
D   = Einsum(input, m10, <rank-5 separable factors> )  # [1,21,21] F32  2D hole-aware structure
dd  = Conv(D, D, pads=[0,9])                   # [1,1,10]  column autocorrelation of D
t   = Sub(GG, dd)                              # [1,1,1,10]  t[p] = 2*G[p] - dd[p]
V   = Equal(t, Kc)                             # Kc = 21*(21-p) = [-999,420,399,378,357,336,315,294,273,252]
pi  = ArgMax(Cast(V,u8), axis=3)               # selected period p (the shift where t==Kc)
q   = Mod(a30[0..29], pf=float(pi))            # residue index per position
Pb  = Equal(q, ar[0..8])  ; P = Cast(Pb,f32)   # [1,9,30] residue one-hot
out = Einsum(P,P,input,P,P, m10, mu,mu,mu,mu)  # [1,10,30,30] residue-class fill (FREE output)
```
Key trick: **`t[p] = 2*G[p] - dd[p]` equals the hole-independent constant `21*(21-p)` exactly at the true period** (verified: period-6 example gives t[6]=512-197=315=21*15). `m10=[-1,1,1,...]` weights the hole channel (ch0) −1. Color is contracted inside both Einsums; the only 10-channel tensors are the graph input and output, both **excluded from cost** (`calculate_memory` line 234).

## Cost breakdown (memory 3364 + params 599 = 3963)
| tensor | op | dtype | shape | bytes |
|---|---|---|---|---:|
| D | Einsum | F32 | [1,21,21] | **1764** |
| P | Cast | F32 | [1,9,30] | **1080** |
| Pb | Equal | BOOL | [1,9,30] | 270 |
| q | Mod | F16 | [1,1,30] | 60 |
| t / dd / GG / G | — | F32 | small | 40 each |
| Vf / V / pi / pf | — | — | small | ≤10 each |

params 599 = rank-5 separable factors `rxHI30[30,5] S_vh[21,5] rxLO30__p2/p3 S_vl__p2/p3` (510) + essential `a30/mu 60 + m10/Kc 20 + ar 9` (89).

## Levers attacked — ALL DEAD
1. **Collapse D → marginal vertical self-conv** (would drop 1764→40 B + kill ~510 params → ~1650 cost).
   **FALSIFIED 266/266.** `dd != raw vertical all-channel self-conv Gv` on every example; and `2*Gh - Gv == Kc` selects the true period in **0/266** (empty set every time). The 2G−dd hole-cancellation is NOT separable into marginal H/V autocorrelations — D encodes a genuine 2D hole-aware reconstruction. Repro: `scratchpad/test_marginal.py` (copied below).
2. **F16 on D or P** (would halve 1764→882 and 1080→540).
   **Blocked.** Both D and P are produced by Einsums that read the official **F32 `input`**; Einsum needs uniform dtype, so F16 requires `Cast(input→f16)` = full [1,10,30,30] F16 = **18000 B** (the standing task017 don't-do trap; see Don't-do log "Cast input→f16 before Slice").
3. **Single index-plane detect + fill** (argmax→i8 plane, Equal-based period test).
   **Proven WORSE.** This is exactly the staged `task017_periodicholefill_cost4992` route (4992 > 3963).
4. **Shave the rank-5 separable factors (510 params).**
   Not attempted — this is a `rung-epsilon` shave, explicitly out of scope for a false-floor attack, and risks correctness. Even a full elimination (impossible; they define D) would land ~3453, still not a floor route to 720.

## Verdict
**DROP.** Charged per-color component already gone (Sho realized the color-index collapse); the one structural collapse lever (D→marginal) is falsified 266/266; F16 is trap-blocked; the index-plane alternative is proven worse. Forced floor for a *correct, hole-independent, 2D-conflict-checked* periodic hole-fill ≈ **3963** (D 1764 F32 and P 1080 F32 are near-forced). The ~720 target is a false floor.

## One remaining hypothesis (low confidence, for a future batch only)
A fundamentally different period *encoding* that materializes the 2D conflict structure at < F32 without an input cast — e.g. building D directly in a packed integer/bool lane whose Conv-autocorrelation still reproduces the exact `2G−dd==21·(21−p)` cancellation. No known ORT path does this (Conv forces F32; bit-lane Conv autocorrelation is unproven for this cancellation). Treat as research, not a shave.

## Reproduce
```
& $env:NEUROGOLF_PYTHON tools/ng_current_bar.py --task 17 --json
& $env:NEUROGOLF_PYTHON tools/agent_helpers/profile_real_cost.py --task 17 --cand Team_Best_Onnx/task017.onnx --arcgen 266
# decode + falsification scripts archived beside this file in repro_2026-07-10/:
#   repro_2026-07-10/decode.py         (dumps G/dd/t/V/pi on real inputs)
#   repro_2026-07-10/test_marginal.py  (266/266 falsification of the D->marginal collapse)
#   repro_2026-07-10/breakdown.py      (per-tensor byte breakdown of the 3963 graph)
```
