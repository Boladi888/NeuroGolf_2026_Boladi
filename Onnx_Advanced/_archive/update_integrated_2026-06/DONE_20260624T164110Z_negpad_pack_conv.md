# DONE — negative-pad crop applies to PACK/extraction Convs, not just label Convs

**Topic:** §41 / #88 negative-Conv-pad crop generalizes to every Conv whose
output is sliced down a spatial axis — including bit-pack extraction Convs.

**Source:** task286 v6 (worker, 2026-06-24). Team-best v5 (53,464) had already
applied the §41 negpad trick to the *label* Conv (1x1 kernel, pads `[0,0,-5,-5]`)
but left the four `pack` Convs (the row-bit extraction Convs, kernel `[1,25]`
stride `[1,25]`) cropping their 30-row output to 25 with a trailing `Slice`.

**Lever:** replace `Conv(...) -> Slice(height 30->25)` with
`Conv(..., pads=[0,0,-(30-25),0])`. ORT opset 18 accepts negative Conv pads on
one axis only; output `[1,1,30,1] -> [1,1,25,1]` directly. Byte-identical
(the `[1,25]` kernel already reads only cols 0..24, so only the row crop's
mechanism changes).

**Saving (task286):** removed 4× F32 `[1,1,30,1]` pre-Slice intermediates
(480 B runtime) + 4 Slice nodes + the `s0/s25/sax` slice initializers (12
params). 53,464 -> 52,972 (-492), full A/B byte-equal 200265/200265 vs LIVE.

**General rule:** after building any extraction/pack Conv that feeds a spatial
`Slice` crop, fold the crop into the Conv via negative end pads. The pre-Slice
intermediate (full uncropped spatial extent) is a real charged tensor; folding
it away is free and exact. Check every Conv->Slice pair, not only the obvious
label Conv.

**Cross-ref:** techniques.md #88 (negpad lane shifts), §41 negative-pad Conv
crops. Suggested home: a one-line addendum on cost_floor_reference.md /
op_cost_patterns.md noting that the negpad-crop fold is per-Conv-output, so
multi-Conv graphs should audit every Conv->Slice pair.
