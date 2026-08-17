# task217 int8 signed ConvInteger route dead

- Live bar used: 366, sha256 `60156f39344ce78300e20f571abb526e298406a2babfc2acecd22ac8145c999b`, source `python tools/ng_current_bar.py --task 217 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- Attempted route: recenter the final writer around signed {-1,+1} masks and signed {-1,0,+1} dynamic ConvInteger weights so the final `ConvInteger` can omit zero-point machinery and save roughly one scalar/byte versus the live U8 0/2 centered formulation.
- Exact failed gate: in-memory ORT kernel smoke before model build. `Where(BOOL, INT8, INT8)` failed with `NOT_IMPLEMENTED : Could not find an implementation for Where(16)`, and `DepthToSpace(INT8)` failed with `NOT_IMPLEMENTED : Could not find an implementation for DepthToSpace(13)`. `ConvInteger(INT8)` itself loaded, but the required INT8 carrier construction does not.
- Why dead: the route needs an INT8 tensor through the mask outer-product and D2S interleave. ORT cannot execute the needed INT8 `Where`/`DepthToSpace` kernels, so the apparent signed-zero-point saving is not a legal scorer path.
- One remaining hypothesis: only a route that keeps the carrier UINT8 until the final free output, or avoids D2S entirely with a different legal sparse writer, can beat the 366 bar.
