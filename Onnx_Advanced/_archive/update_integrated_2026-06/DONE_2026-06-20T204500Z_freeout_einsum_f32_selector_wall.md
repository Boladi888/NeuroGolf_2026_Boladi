# Free-output mirror Einsum walls when F32 input forces F32 selectors that exceed the crop

**Where:** task351 (16×16 mirror-crop recovery, F32 `[1,10,30,30]` input → `[1,10,30,30]` output). Builders `The_400_Tasks/Task_by_Task/task351/prototypes/build_task351_v10.py` (v10, cost 2197) and `build_task351_v11.py` (v11, cost 1788). Team-best stays v8=1498 / v9=1358 (GridSample crop + free-output Pad).

## What I discovered (quantified crossover for #109/#118/#119)

The free-output separable Einsum (`output = Einsum(input, Pr, Sr, Pc, Sc)`, eq `bkrc,Rt,tr,Cs,sc->bkRC`, output named `output` ⇒ uncharged) genuinely dissolves BOTH the materialized 5×5 crop AND the final Pad — and it is decode-CORRECT (visible 265/265, A-B byte-equal 20,265/20,265 vs LIVE). But it can LOSE on cost, because the crop is replaced by **data-dependent selector matrices that are locked to the input dtype**:

- Selectors `Sr,Sc[5,30]` must be **F32** because the graph `input` is F32 and **ONNX Einsum rejects a mixed F32-input + F16-operand contraction** (`ShapeInferenceError: Inputs has inconsistent type tensor(float16)`, re-verified onnx 1.21 / ort 1.24 on 2026-06-20). So selectors = `2·5·30·4 = 1,200 B`, which is *larger* than the `10·5·5·4 = 1,000 B` F32 crop they replace. Net +200 B core wall before locator overhead.
- The selector axis **cannot shrink to `[5,16]`** (skipping always-zero input cols): Einsum contracts the selector's source axis against the input's full 30-length axis, so the selector source axis is forced to 30. Slicing the input to 16 first is a charged `[1,10,16,16]` intermediate (≥5,120 B).
- The "GridSample straight to `[1,10,30,30]` output" variant (deletes crop AND Pad with NO selectors) needs a `[1,30,30,2]` grid = 3,600 B (F16) — also a wall. The compact 100 B grid only works WITH a materialized 1,000 B crop.

## The crossover law (transferable)

> Free-output separable Einsum beats a GridSample/Gather **crop + free-output Pad** only when the selector bytes `Σ 2·K·N·itemsize` are **below** the crop bytes `C·k·k·itemsize`. When the graph input is F32 (NeuroGolf task inputs are F32 one-hot) the selectors are F32-locked (no mixed-dtype Einsum), so for a `k×k` crop over `C` channels with source extent `N`, the selectors win **only if `2·k·N < C·k·k`, i.e. `2N < C·k`**. task351: `2·30=60` vs `10·5=50` → selectors lose. For wider channels or larger crops (`C·k > 2N`) the free-output basis wins; price `2·k·N·4` against `C·k·k·4` FIRST.

This is the F32-input dual of the existing #109 "carrier-dtype tax (task091)" note: there the `[30,30]` F32 selectors (7,200 B) lost to a U8 Gather tail; here the `[5,30]` F32 selectors (1,200 B) lose to the 1,000 B F32 crop. Same root cause — **F32-input forces F32 selection structures; the free-output exploit only pays off when the deleted output canvas/crop is bigger than the forced selectors.**

## What's missing from the library

`techniques.md §109/§119` document the free-output Einsum win but don't give the explicit selector-vs-crop byte inequality (`2N < C·k`) for the F32-input case, and don't state that a GridSample-crop + free-output-Pad is often the cheaper free-output variant (compact grid 100 B + crop, vs full `[K,N]` selectors). The §17 mixed-dtype Einsum trap (F32 input + F16 operand → shape-inference error) is in the task351 don't-do log but worth a one-line entry in `ort_reality.md` / `traps.md` as a general Einsum dtype rule.
