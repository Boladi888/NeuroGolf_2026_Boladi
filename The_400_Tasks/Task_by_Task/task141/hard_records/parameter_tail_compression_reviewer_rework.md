# parameter-tail compression reviewer rework

- bar: 106 bytes, Team_Best/manifest SHA-256 `9b03cb7d6b60847aa58dd706ea6254e16144c1ee80d8f652828e087548c5ab81`, owner Vamshi, rechecked with `python tools/ng_current_bar.py --task 141 --json`.
- scope: reviewer-locked parameter-only tail compression of the 106-byte final-Einsum graph. Kept both F32 `[2]` moment vectors (`row_feat`, `col_feat`) materialized. Did not revive v13/v14 raw-input moment fusion.
- routed techniques cited: `INDEX.md` routes `techniques.md` by section; relevant sections are #67 free-output output-node pricing, #82 singleton-object scalar reads, #89 ORT left-deep Einsum runtime rules, and #119 deep separable output/scalar coefficient solve.

## Attempts

1. `task141_v16_no_final_poly_sumA.onnx`: removed `final_poly` and let the first shared `affine_sub` sum over its leading axis. Static score `(0, 16, 88)` / cost 104, SHA-256 `89ad2897d5d54927090f52d2ea7e345f2e3e6e8fee4a85c7cd4aaa10b54bd2a0`. Failed `ng_gate visible`: `[visible] 0/265 (0.00%)  FAIL`.
2. `task141_v17_share_nonbg_as_channel_sign.onnx`: removed `channel_sign` and reused `nonbg` as the final channel sign factor. Static score `(0, 16, 80)` / cost 96, SHA-256 `3ee933d4e8229589b734fbacd4c3a6086a9dad0c745dd837a5801c79a79ac467`. Failed `ng_gate visible`: `[visible] 0/265 (0.00%)  FAIL`.

## Forced tensors under the locked hypothesis

- `nonbg[10] = [0,1,...,1]` is forced for the #82-style singleton marker coordinate reads. Replacing it with `channel_sign[-1,1,...,1]` contaminates the row/column moments with the active-grid background mass; correcting that requires extra active-grid algebra or folding marker selection into the final raw-input contraction, which is the v13/v14 pathological family.
- `channel_sign[10] = [-1,1,...,1]` is forced for the final signed output: marker channel and background channel need opposite signs over the same diagonal polynomial, while absent colors must remain zero. Reusing `nonbg` removes the negative ch0 branch and visible-fails 0/265. Factoring `channel_sign = 2*nonbg - 1` needs a constant-one channel basis or additive output branch, which costs back the saved vector or materializes a charged full-output term.
- `final_poly[2]` is forced as an independent two-coefficient threshold polynomial. Summing the shared `affine_sub` axis is the zero-param alternative, and it visible-fails 0/265. Precontracting it into a separate first-use tensor would replace 2 params with a dense `[2,2]` coefficient tensor (4 params), so it cannot beat 106.

Conclusion: within the non-pathological 3-node formulation, the 90 params are forced as `coord_basis60 + nonbg10 + channel_sign10 + affine_sub8 + final_poly2`, plus the two F32 `[2]` moment vectors for 16 memory bytes. Routes #67/#89/#119 explain why the output field itself is free, but all small operands remain charged. Any exact sub-106 route found so far either removes a forced factor and visible-fails, or deletes a moment vector by adding raw input copies to the final high-arity Einsum; that latter route has real static savings but exceeded the default gate wall budget and is outside this reviewer lock.
