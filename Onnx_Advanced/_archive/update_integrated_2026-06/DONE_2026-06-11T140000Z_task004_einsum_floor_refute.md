# task004 — refuted a canvas-family floor; the real floor is Aditya's row-bit shifted-basis Einsum (2,662)

**Where:** task004 (horizontal shear: parallelogram color cells shift right by 1 except bottom-row + the cell above the box's bottom-right corner). worker-3 session 2026-06-11. Build/decode scripts: `The_400_Tasks/Task_by_Task/task004/prototypes/{verify_rule,design_min_einsum,decode_einsum,ablate}.py`. Reference net: `Team/submission.zip::task004.onnx`.

## What I discovered

1. **A "PROVEN FLOOR ~7.5–8.2k, <1k infeasible" claim in `task004.md` was FORMULATION-RELATIVE and wrong as a global floor.** It only priced the canvas+Equal family (collapse 10ch one-hot → F32 color plane 1,024 + 900 U8 canvas + ~14 BOOL/U8 logic planes @256 = ~6k mem). Aditya already holds task004 at **2,662** (mem 360 + params 2,302), **3× cheaper**, via a free-output single Einsum. This is a concrete repeat of traps §68 ("floors are formulation-relative — enumerate final-op families FIRST"). The stale floor was the result of never pricing the free-Einsum family.

2. **The winning architecture is a 3-node "row-bit + shifted-basis free-output Einsum"** (composes #119 deep-separable output, #113 place-value packing, #118 basis-matrix axis remap):
   - `Bcur = Conv(input, W_bitrow[1,10,1,30]) → [1,1,30,1]`: per-row **place-value color-presence code** `Σ_c cm[r,c]·2^c` (one F32 number encodes the whole row's colored-cell pattern; exact since active ≤16 cols → ≤2^16).
   - `Baug = Conv(Bcur, W_shift_next[2,1,2,1], pad top=1) → [1,2,30,1]`: augments each row with next-row context (for the bottom-row / corner stay logic).
   - `output = Einsum('birc,bjrq,bmrw,tqc,kq,ftmk,if,oi->borq', input, input, Baug, S_col[2,30,30], P_basis[2,30], C_coef[2,2,2,2], H_feature[10,2], D_channel[10,10])` — THE graph output (free). `input` appears twice: operand1 (`birc`) = colored source at col c; operand2 (`bjrq`, j summed) = the **active-mask at the OUTPUT position**, which replaces the canvas sentinel/`inside` plane entirely. `S_col[0]`=identity δ(q,c) (stay), `S_col[1]`=shift δ(q,c+1) (shift); `C_coef`/`Baug`/`P_basis` decode per-cell stay-vs-shift from the packed row-bits; `D_channel`/`H_feature` relabel channels and rebuild bg(ch0).

3. **Forced-component floor proof (why 2,662 can't be beaten — not even by 1 byte):**
   - **Only sub-3k final-op family is the free single-Einsum.** Canvas+Equal ~6k; GridSample shear ≥7,200 (forced `[1,30,30,2]` F32 grid); broadcast-Add / Where-from-input / ScatterND / OneHot all need a materialized `[1,10,30,30]` (≥36,000 F32 or ≥2,560 BOOL pre-pad).
   - **`S_col[2,30,30]=1,800` is irreducible inside that family.** The rule maps input col `c` → output col `q∈{c,c+1}`. In a single free-output Einsum the contracted `c`-axis is pinned to 30 by the input operand and the output `q`-axis is pinned to 30 by the output shape, so the always-zero cols 17..29 (active ≤16×16, corpus-verified 200k) **cannot be cropped — they share labels with the [1,10,30,30] input/output.** Two offset options ⇒ `[2,30,30]`. Both branches proven load-bearing (zeroing either → 0/265). Splitting stay/shift into 2 Einsums+Add re-materializes a `[1,10,30,30]` plane (≥36k) — strictly worse. The shift cannot become a scalar (place-value ×2) because unpacking the packed code back to per-column one-hot needs the same `[30,*]` basis.
   - mem 360 (`Bcur`120+`Baug`240, Conv outputs over 30 rows that pass through to the output → cannot crop free) and the 502 non-`S_col` params are each at natural shape.

## Why it matters

- **Transferable trap correction:** any task .md whose floor proof only enumerates the canvas/plane family is suspect — price the free single-Einsum (Aditya class) before believing it. Several of our "5–8k floors" on placement/shift/shear tasks may be 2–3× beatable the same way.
- **Reusable idiom:** "per-row/col place-value code via a `[1,C,1,W]` Conv → feed a deep free-output Einsum whose basis operands do the geometric placement" is the general recipe behind his cheapness. For a conditional **horizontal shift/shear** specifically, the placement is a `[2,W,W]` identity+shift tensor and is the hard floor.
- **Team-score consequence:** for task004, a build anywhere in (2,662, 8,229] adds ZERO points (Aditya books 2,662 on the cheap side). No sub-2,662 build exists. Don't spend more worker time compressing our 8,229 canvas build.

## What's missing from the library

- `techniques_catalog.md` / `by_symptom.md` have no "conditional horizontal shift / shear → `[2,W,W]` identity+shift basis in a free-output Einsum" route. The closest is #121 (GridSample shear) which is ~3× more expensive here. Suggest a by_symptom row: "output = input with a per-cell ±1 column remap (shear/shift) → free Einsum with a `[k,W,W]` offset-basis operand; floor = k·W² when both col axes are pinned to the [.,.,H,W] input/output." Reference this task004 proof.
