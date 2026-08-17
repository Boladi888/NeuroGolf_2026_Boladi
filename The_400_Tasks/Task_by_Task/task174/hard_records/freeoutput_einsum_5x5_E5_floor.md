# DEAD ROUTE — task174 TemplatePadFreeOutput false-floor attack → WALL

**Recorded:** 2026-07-11 by `bo_ng_cld_worker1` (Claude lane).
**Verdict:** DROP / WALL. No candidate strictly below the live bar exists in this
formulation, and every independent from-scratch reformulation prices **≥ 867**.

## Live bar (refreshed this session — supersedes ALL stale docs)
- **live_cost = 867** (`ng_current_bar.py --task 174 --json`; authoritative source = manifest.csv, owner **Sho**, updated 2026-07-10T09:55Z).
- live SHA-256 = `64745efcf4d40c5adbe5ad6801204f6466cc835c535593eaa375f5e68834c6fa`
  (= `Team_Best_Onnx/task174.onnx`, graph name `t174_w36b`, 2643 bytes, opset 17).
- **profile = 867** over 266 real inputs (`memory 584 + params 283`); `static=None`
  (dynamic Slice ⇒ shape-inference declines; baked value_info supplies the static baseline).
- NOTE: `task174.md` manual content and `context_pack`/`task_state.csv` are ARCHAEOLOGICAL
  (they show local best 2225 / team-best 4786 / LIVE 5763). The real bar is **867**. Do not
  chase "beat 2225" — that is a ~+1358 loss vs live.

## What the prompt's Direction targeted is ALREADY GONE
Prompt Direction: "Pad/Concat/template sweep: avoid stored/materialized canvas, generate
predicate/index and let final output op materialize. Delete charged carrier/canvas/template."
Forced-floor claim: "output canvas bytes and a small predicate/index are forced; repeated
Pad/Concat planes are not."

Sho's 2026-07-10 `t174_w36b` **is** the collapsed free-output construction the prompt anticipated.
There is **no stored 30×30 canvas** and there are **no repeated Pad/Concat planes** left to dissolve:
- ONE Pad (`C5h[1,1,5,5]→Hstk[1,2,5,5]`, adds a single ones-plane for the background channel, 100 B).
- TWO tiny Concats (`stf[3]`=12 B slice-index build; `F2[2,10]`=40 B color-map).
- The whole `[1,10,30,30]` output is materialized **only as the free graph output** by the final Einsum.

Per the prompt's own workflow rule ("if it is already gone … return DROP with the refreshed
floor note. This is a false-floor attack, not a rung−epsilon shave"), this is a DROP.

## The task (decoded from ARC-GEN visuals)
Select the **left–right-symmetric** colored object; output its bbox crop, color-mapped, at the
**top-left** of the 30×30 canvas. Background cells *inside* the crop bbox must set channel-0=1;
cells *outside* the bbox are all-zero.

## Exact charged breakdown of the live 867 graph (every tensor forced)
Cost model (verified reproduces 867 exactly): `cost = memory + params`.
`params` = Σ initializer **element counts** (dtype-independent). `memory` = Σ intermediate
**bytes** = max(static value_info, runtime-observed), dtype matters.

params = 283:  E5[5,30]=150 · V[2,30]=60 · w8[30]=30 · thr[1,10]=10 · e0h[1,10]=10 · pads8f[8]=8 · eps[2,2]=4 · k155[3]=3 · axesT[3]=3 · oh0v[2]=2 · zero/padv/cneg=1·3
memory = 584: C5 F32[1,1,5,5]=100 · Hstk F16[1,2,5,5]=100 · C5h F16[1,1,5,5]=50 · D/Ns/sf F32[1,10]=40·3 · F2 F16[2,10]=40 · starts/ends INT64[3]=24·2 · sfh/Fsub F16[1,10]=20·2 · stf F32[3]=12 · eqz/gt/sel BOOL[1,10]=10·3 · ci INT64[1]=8 · sr/lsr/nr/r0f/sc/lsc/nc/c0f/cif F32[·]=4·9

### Forced-floor argument (exact margins, not "tried hard")
1. **Crop 5×5 is measured-minimal.** Max output bbox over all 266 (train+test+arc-gen) = **5×5**
   (sizes include 1×5, 2×5, 5×2, 3×4, 4×3). A crop smaller than 5×5 drops correct pixels ⇒ the
   crop path (`C5` 100 + `C5h` 50 + `Hstk` 100 = 250) cannot shrink.
2. **`E5[5,30]`=150 is the forced in-Einsum placement.** Verified E5 = `[I₅ | 0₅ₓ₂₅]` (fixed
   identity, top-left, NOT data-dependent). To emit a 30-wide **free** output from a 5-wide crop
   *inside* the final Einsum, the embedding's output dim must be 30 ⇒ [5,30]. The 25 zero columns
   are the price of a directly-30-wide free output.
3. **`V[2,30]`=60, `w8[30]`=30 are the free-input soft-slice.** They zero-mask the input's 30-wide
   spatial axis to the active 10×10; a real Slice to 10-wide charges thousands of bytes.
4. **All F32 `[1,10]`/scalar tensors are Einsum/math outputs on the F32 input ⇒ dtype-forced.**
   Narrowing to F16 requires a Cast that adds a tensor ≥ the saving (the F32 Einsum output still exists).
5. **INT64 `starts`/`ends`[3]=48 forced by Slice** (index dtype), minimal shape [3]=(chan,row,col).

## Independent from-scratch formulations, all priced ≥ 867
| # | Formulation | Price | Why it loses |
|---|---|---:|---|
| 1 | Free-output Einsum + 5×5 crop + E5 placement (LIVE) | **867** | the floor |
| 2 | Einsum→`[1,10,5,5]` then Pad→`[1,10,30,30]` | ~1217 | drops E5 (−150 params) but materializes a **500 B** `[1,10,5,5]` F16 intermediate (Pad input, not free) ⇒ +350 |
| 3 | F32 end-to-end crop path (skip `C5h` cast) | ~957 | Hstk/F2 double to F32 (100→200, 40→80) ⇒ +90; F16 operands are optimal |
| 4 | Per-channel Slice+Pad 30×30 canvas (old local line) | 2225 | stored canvas; documented superseded rung |

## One remaining hypothesis (low confidence)
Fold the presence count `Ns` (40 B) into the geometric predicate `D` so a single Einsum both
tests LR-symmetry AND excludes absent/background colors. Blocker: an absent color has `D=0`
trivially (empty is symmetric), so an independent presence signal is required; making `D` nonzero
for absent colors needs a pixel-independent additive term that would also perturb present colors.
Max upside if solved ≈ 40 B → 827, still far above the 720 EV estimate. Not pursued — below the
SWEEP "one-crack / stop on priced wall" bar.

## Do-not-re-explore
- Do NOT rebuild the 2225 per-channel/canvas line "to beat 2225" — the live bar is 867.
- Do NOT try to shrink the crop below 5×5 (max output is exactly 5×5).
- Do NOT replace E5 with a post-Einsum Pad (materializes ≥500 B).
- Do NOT slice the F32 input to 10-wide to shrink V/w8 (slice ≫ the 90 params saved).
