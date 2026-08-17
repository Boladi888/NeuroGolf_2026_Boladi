# task324 — two durable findings (opset-12 U8 ReduceMax/Max dtype unlock + diagonal-spread is a conv floor, not an einsum)

Source: worker-324 session 2026-06-19. Builder `The_400_Tasks/Task_by_Task/task324/prototypes/build_task324_u8mc_v1.py`; staged `staged_submits/task324_gencorner_u8mc_cost13621/` (sha256 7dbba6bb...). Beat team-best `dilated_bridge_v4` (15,667) -> 13,621 (-2,046, -13.1%), byte-equal vs parent across full 200k AB and 200k truth = 100.000%, profile==static==13621 (266 arc-gen).

## Finding 1 (ort_reality / dtype x opset matrix): ReduceMax and Max ACCEPT UINT8 at opset 12+, REJECT it at opset 11

Probed 2026-06-19 on the pinned ORT:
- opset 11: `ReduceMax(U8)` and `Max(U8)` -> `INVALID_GRAPH Type Error` (rejected). `ReduceSum(U8)` rejected at every opset (use F16 or I32).
- opset 12 and 13: `ReduceMax(U8)`, `ReduceMax(I8)`, `Max(U8)`, `Where(..., U8, U8)` all OK.

The old task324 note "Mul/ReduceMax/Max/Sub reject UINT8" is **opset-11-specific** and was the sole reason an F16 `canvas_h` copy + F16 mc masked planes existed. Bumping the model opset 11->12 (a one-line change; Unsqueeze/ReduceSum keep their axes-attribute form at 12, so no other node churn) let the mc-detection (`Where(mask, canvas_u8, 0u8) -> ReduceMax(U8) -> Max(U8,1u8)`) and the plaid "row/col-has-B" (`Cast(B_pos->U8) -> ReduceMax(U8, axis)` = the OR) run entirely in U8. Removed one 800B F16 plane and halved three more (800B F16 -> 400B U8). -2,046 B, exact/byte-equal.

The lever: **when a graph carries an F16 plane only because a U8 reduce/clamp was "unsupported", re-probe at opset 12-13 before accepting it.** U8 planes are half the bytes of F16. ReduceSum still needs I32/F16, but ReduceMax/Min/Max/Where/Equal/Greater are all U8-clean at 12+. (Suggested touch: add a row to the ort_reality dtype x opset matrix: ReduceMax/Max U8 = opset>=12.)

## Finding 2 (confirms #133 / the task158 DONE: diagonal spread is a conv floor, not a free-output einsum)

task324's prompt proposed dissolving the diagonal X-Conv (`x_kernel [1,1,39,39]` = 1,521 params, both diagonals in one kernel) into a "free-output separable Einsum over per-dot r+c / r-c index bases (#118)". Assessed NOT-VIABLE (independent analysis agreed), same root cause as task158's stamp finding:

- A single diagonal line `{(r,c): r+c==s}` on HxW is a **rank-min(H,W)** partial-permutation matrix, NOT a sum of a few `row⊗col` outer products -> #109/#119 free-output separable einsum explicitly does not apply.
- A literal #118 incidence basis `A[r,c,s]=[r+c==s]` is `[H,W,H+W-1]` (20x20x39 = 15,600 elems, ~10x the kernel); both diagonals ~31,200. CP/Tucker at rank-20 ~= 20*(20+39) per side ~= 2,400-3,160, still ~2x the conv.
- #8e's `[H,H]` superdiagonal shift is for a FIXED finite stencil (single +1 step); a diagonal is a full ray (union of all offsets), not a stencil.
- ORT Einsum can't consume BOOL -> the marker plane would have to be F32, doubling it.

Root cause (transferable): **Conv exploits translation-invariance — it stores the diagonal-generating line ONCE and reuses it by translation (Toeplitz). Einsum has no translation-invariance primitive, so its basis factors pay rank x dims and cannot compress a high-rank shift-invariant operator.** For diagonal spread / ray-cast / translation-invariant detect-and-place, the conv IS the floor; the einsum is the wrong tool. (Pairs with the task158 DONE Finding 2 — same family.) The achievable wins on these graphs are the surrounding plane dtypes (Finding 1), not the conv kernel.
