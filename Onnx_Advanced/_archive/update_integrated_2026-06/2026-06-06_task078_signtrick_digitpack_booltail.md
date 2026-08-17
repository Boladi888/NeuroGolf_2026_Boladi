# Pending library update — task078 (col-sort 15,756 → 842, 3-rung ladder staged)

Source: task078 worker session 2026-06-06. Full detail: `The_400_Tasks/Task_by_Task/task078.md`
+ `Logs/lessons.md` 2026-06-06 sign-trick entry. All three rungs ab byte-equal 200,266/200,266.

## Promote to techniques.md

1. **Sign-trick output planes** (counting/threshold rule families): the `>0.0` decode means
   a channel plane can be the raw signed margin `count − (r+.5)` — no clamp-to-one-hot.
   Half-integer thresholds make every comparison tie-free and F16-exact. A between-two-
   thresholds band = `ReduceProd` of the two margins when both-positive is impossible
   (lo ≤ hi guarantees it): product > 0 ⟺ in-band. Replaces 2 Relu + 3 Sub per plane.
2. **Digit-packed multi-count einsum**: K per-column counts can ride ONE einsum output via
   place-value sel coefficients (e.g. {c1: 101, c2: 100} → `n1 + 100·n12`). Comparisons at
   the high place-value work directly on the packed stream when the low digits are bounded
   below half the radix (margin argument); unpack low digits with `Mod(packed, radix, fmod=1)`
   (const divisor — no SIGFPE). Halves the count-extraction memory.
3. **Einsum size-1 free subscript**: `bchw,kjc->bkjw` with sel `[k,1,10]` emits rank-4
   `[1,k,1,30]` directly — no Unsqueeze after a rank-3 einsum. 0 extra params.
4. **Comparison/BOOL tail** for stack-band outputs: `Greater(n_lo, r+.5)`, `Greater(r+.5, n_hi)`,
   band = `Equal(b_lo, b_hi)` (XNOR; both-true impossible). BOOL planes are 1 B/elem and
   BOOL Pad/Concat/output all work at opset 13. traps §12 escape clause: these bools are
   DEFINED by tie-free exact comparisons, not reinterpreted float data; gate per §NOVEL-OP
   full-corpus ab.

## Already applied to ort_reality.md §7 (this session)

- `ReduceProd` row added and FULLY probed (2026-06-06): BOOL/INT8/UINT8/INT16 ✗ (load-reject),
  INT32/INT64/F16/F32 ✓ — identical profile to ReduceSum. No follow-up needed.

## Mod note (candidate for ort_reality op rows)

`Mod(F32, fmod=1)` with a constant divisor runs exact on integer-valued floats (probed
opset 13, values ≤ 1010). The traps Div/Mod-by-INPUT→SIGFPE warning is about runtime-zero
divisors; const divisors are safe.
