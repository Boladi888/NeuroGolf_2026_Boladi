# task040 reviewer locked hypothesis: implicit all-ones half row

- Live bar: cost 80, SHA256 `01ba84e2bca7236e35b1b674ebe44e86b809ffb65de9c8e29e2f3c6d39f44127`, source `python tools/ng_current_bar.py --task 40 --json` / manifest owner Vamshi.
- Parent graph: one terminal free-output `Einsum` with equation `ncuv,pu,qv,nars,a,c,pr,qs->ncrs` and params `half[2,30]=60`, `gate[10]`, `sign[10]`.
- Routed docs used: `Onnx_Advanced/INDEX.md` routes cited sections; techniques #67 says the graph output node is not charged; techniques #89 says in-kernel `Einsum` temporaries are not charged but operands still are; `cost_floor_reference.md` says params are element-count and dtype does not save params, only shape compression does.

## Algebra

Let `A(x)=1` for active positions `0..4`, `B(x)=1` for active positions `5..9`, and zero outside the fixed 10x10 grid. The live same-half projector on one axis is:

`M(u,r) = A(u)A(r) + B(u)B(r)`.

On the active `10x10` submatrix this is block diagonal with two `5x5` all-ones blocks, so `rank(M)=2`. With `O=A+B` and `D=A-B`, the equivalent signed form is:

`M(u,r) = (O(u)O(r) + D(u)D(r)) / 2`.

The scalar factor is irrelevant under the final `>0` decode, but the additive two-term structure is not. A single terminal `Einsum` can hide products inside the kernel (#89), but to add `O⊗O` and `D⊗D` inside the same node it needs a contracted term axis with two independent rows, e.g. `basis[p,x]=[O(x),D(x)]`. That is still a rank-2 spatial basis and still costs `2*30` initializer elements. If `O` is omitted, the remaining `D⊗D` term is rank 1 and is positive for both same-half and opposite-half pairs; across two axes, `D(u)D(r)D(v)D(s)` also lights the opposite-opposite quadrant, so it is not the same projector.

## ONNX variants

- Control: `The_400_Tasks/Task_by_Task/task040/prototypes/task040_half_signed_2row_equiv.onnx`
  - SHA256 `770f6cdad9e5a78c59006610fd800633f551bf0f2d178053bc0c5ef447a557d5`
  - Static score `(0, 0, 80)`, profile `static=80 profile=80`.
  - Gate: visible `266/266 PASS`.
  - Meaning: `[O,D]` is algebraically equivalent to the live `[A,B]`, but it does not reduce charged params.
- Sub-80 attempt: `The_400_Tasks/Task_by_Task/task040/prototypes/task040_half_signed_product_only.onnx`
  - SHA256 `e875a48961b175b3a816db99076f41a248bae69faa5ded65a350c3ca5e3af711`
  - Static score `(0, 0, 50)`.
  - Gate: visible `0/266 FAIL`.
  - Meaning: deleting the all-ones row leaves only the rank-1 signed product; it is not the same-half projector.

## Verdict

The locked hypothesis does not produce a valid sub-80 candidate. Under the reviewer constraints (same terminal free-output `Einsum`, no added node, no charged intermediate, no sparse initializer, no output merge), the exact same-half projector still needs a rank-2 spatial basis or an added charged/additive term. The free-output/Einsum-final trick makes the contraction temporary free, not the missing independent operand row.
