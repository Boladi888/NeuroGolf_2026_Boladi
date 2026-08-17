# ConvInteger x_zero_point is scalar-only in the pinned ORT (per-channel 1-D rejected at runtime)

## What I discovered
ONNX spec says `ConvInteger`'s `x_zero_point` may be a 1-D **per-input-channel** tensor, but the
pinned onnxruntime CPU kernel **rejects it at runtime**:

```
RUNTIME_EXCEPTION : ... conv_integer.cc:47 ConvInteger::Compute
IsScalarOr1ElementVector(X_Zero_Point) was false. Must be a scalar or 1D tensor or size 1.
```

So the input zero-point is effectively a single scalar. (`w_zero_point` per-output-channel is fine;
only the INPUT zero-point is scalar-locked.) `onnx.checker` passes — the failure only appears when
you build a session and run it, so a graph can look valid and still be dead.

## Why it matters (a real decode-golf trap)
When you decode a per-cell K-state plane to a one-hot output with `BitwiseAnd(v,[m0,m1]) → ConvInteger`,
the "background" state is value 0, and `0 & anymask = 0` puts background at the conv **origin (0,0)**.
With a **scalar** input zero-point `z`, the background output is `−z·(w0+w1)` and the sign constraints
force `z<1`, which contradicts integer `z≥1` — so a scalar-z conv **cannot** make background's channel
`>0` while the other states stay `≤0`. That is exactly why such graphs need an extra `Add(+const)`
plane before the `BitwiseAnd` to move background off-origin (task219's forced `+3` plane, 150 B on a
15×10 grid). A **per-channel** `x_zero_point=[0,1]` removes the origin problem (clean weights exist:
ch_bg=(-1,-1), and the two active channels =(1,0)/(0,1) on masks `[1,2]`) — but ORT won't run it, so
the `+const` plane is unavoidable, not a golf oversight.

## Where I hit it
- task219 (2444 F16-CumSum graph), trying to drop the `+3` plane before `ConvInteger`.
- Repro prototype: `The_400_Tasks/Task_by_Task/task219/prototypes/c1_drop_plus3.onnx` (checker-valid, ORT-fails).
- Full write-up: `The_400_Tasks/Task_by_Task/task219/hard_records/forced_floor_2444.md`.

## What's missing from the library
`ort_reality.md` documents ConvInteger *usage* (weighted channel-sum, U8 stamps, GRADER-VALIDATED)
but nowhere notes the scalar-only `x_zero_point` limit or the "off-origin `+const` is forced for a
background/K-state decode" corollary. Add it so future decode-golf doesn't re-derive the proof.

## Sorter note: Onnx_Advanced/ort_reality.md — extend the ConvInteger dtype/quirk rows with: "x_zero_point is scalar-only in the pinned ORT (1-D per-channel rejected at runtime: IsScalarOr1ElementVector); w_zero_point per-output-channel is OK. Corollary for K-state→one-hot decode: background=value-0 sits at the conv origin, and scalar-z can't sign-separate it (algebra: needs z<1 vs integer z≥1), so an Add(+const) plane before BitwiseAnd is FORCED to move background off-origin — not removable (task219 +3 plane)."
