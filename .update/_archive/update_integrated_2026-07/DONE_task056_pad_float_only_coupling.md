# Transferable insight — "free attribute-pads" ⟺ "float-only Pad"; you can't buy a non-float Pad cheaply

**Source**: task056 batch 2026-07-07 (`bo_ng_cld_worker2`). Audited LIVE 33 (Vamshi, 2026-07-02);
this is a **floor confirmation** (33 exhaustively proven optimal), not a new win. Kills a tempting
"drop the Cast" micro-shave that looks free but is blocked by an ONNX opset/type coupling.

**The trap it kills**: On a one-hot-canvas task, the final `Pad`-to-`[1,N,30,30]` is fed by a small
float intermediate that exists ONLY to satisfy Pad's dtype (e.g. `active_channels [1,K,1,1] F16` = 2K B).
The gate compares outputs with `np.array_equal`, which is **value-based, not dtype-based** — so a BOOL
or UINT8 output *would* match the float32 one-hot reference. Tempting conclusion: "Pad a BOOL directly,
delete the Cast, save 2K B." **This is false at the floor opset.**

**The coupling (why it's blocked)**:
- Attribute `pads` / `starts` / `ends` / `axes` (= **0 counted params**) exist ONLY in the opset-≤10
  op versions: `Pad-2`, `Slice-1`.
- `Pad-2` type constraint T is **float-only** (`f16 / f32 / f64`). ORT opset-9 Pad on bool/uint8/int8
  → `INVALID_GRAPH: Type 'tensor(bool)' of input ... of operator (Pad) ... is invalid`.
- Bool/int Pad exists ONLY in `Pad-13+` (opset ≥11), where `pads` becomes an **input initializer**
  (+2·rank params = +8 for 4-D) AND `Slice-10+` takes starts/ends/axes as **input initializers**
  (+~15 params for two single-cell slices).

So: **free pads ⟺ float Pad ⟺ the small F16 pre-Pad intermediate is forced.** There is no opset that
gives you both attribute (0-param) pads and a non-float Pad. F16 (2 B/elem) is the smallest legal
pre-Pad float.

**Measured (task056, K=6 channels)**:
- opset-9 F16 Pad (live): **33** = memory 32 + params 1.
- opset-13 bool Pad, Cast deleted: **44** = memory 20 + params 24. The −12 B memory is swamped by
  +23 params from initializer-based slices/pads.

**Rule of thumb**: on any `Concat→Cast(F16)→Pad`-to-canvas graph, do NOT try to delete the F16 Cast by
switching Pad to bool/int — the opset bump to get a non-float Pad costs more params (pads + slice
inits) than the memory it saves, unless the graph has NO slices and few output dims. Keep opset-9
attribute pads/slices and eat the smallest-float (F16) pre-Pad tensor.

**Related**: encode gate uses value-equality (`np.array_equal(out, encode_grid(output))`), so output
dtype is free to differ from LIVE's F16 *for correctness* — but that freedom is unusable here because
the op that produces the output (Pad) constrains the dtype, not the grader.
