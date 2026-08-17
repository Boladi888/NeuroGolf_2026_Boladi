# task357 (11,082 → 375 staged, byte-equal, 29.6×): Conv-as-placer output + ConvInteger lane (2026-06-07)

**Night addendum — contraction-final (312/304 class) kill template for equality-coupled
outputs:** when the output couples (r,c) through an INPUT-SELECTED permutation/equality
(`[c == p_W(r)]`), the params-only einsum-final class is priceable and dies: (a) CP lower
bound via matricization PLUS the per-column combinatorial-rectangle test (R=10 disproven
in one line: column-0 incidence is r∈odd for W=2 vs r≡1 mod 4 for W=3 — not a rectangle)
⇒ ≥(dims-sum)·R params; (b) the single-product ±1 trick (`u[o]·act0[r]·s[c]·D[r,c]`,
free input reads for act0/s) is killed by einsum axis-matching — D inherits the FULL
output extent ([30,30] F16 = 1,800 B) or pays identity-lifts (600 params); (c) sum-axis
`Σ_τ T·U·V` needs every term rank-1 in (r,c); (d) Toeplitz/mod-congruence reformulation
(`[t∓c ≡ 0 mod 2W−2]`) re-imports a [9,39] dense congruence bank (351 params, rank≈9).
Full worked numbers: task357.md §Contraction-final. Use these four sub-checks as the #68
kill-number template whenever the final-op enumeration reaches the contraction class on
an equality/permutation-structured task. (312 folded because its structure is separable
under contraction; the discriminator is SEPARABILITY of the (r,c)-coupling, not op choice.)

Ladder: 1,312 (Equal-tail) → 514 (Conv-F16 tail) → 404 (ConvInteger) → 387 (nibble bank).
All rungs byte-equal vs LIVE on the full corpus. The big transferables:

1. **#67 UPGRADE — Conv with asymmetric END-pads as the output op = Pad + channel-expand
   FUSED, both free.** `Conv(X[1,1,10,10], W[10,1,1,1], pads=[0,0,20,20])` emits
   [1,10,30,30] directly: `pads` is an ATTRIBUTE (zero params — beats Pad's 8-param init),
   placement is top-left corner, and the 1×1 weights do the one-hot/channel expansion that
   Equal-tails pay a [30,30] u8 canvas (900 B) for. The pre-output tensor drops from 900 B
   (Equal-tail canvas) / 1,000 B (Pad-tail one-hot) to the bare coupling grid (200 B F16 /
   100 B u8). With #69 sign-arithmetic (X=+1/−1/0, w=±1) two channels cost one plane.
   **Direct upgrade candidate for the 97-task `Pad→output` spread map (#67 SPREAD TARGET):
   any task whose pre-pad is channel-decodable can fuse the decode INTO the placer Conv.**
   task357's first floor proof (1,312 "Equal-tail is the cheapest final family") fell to
   exactly this — #68 in action: the Conv family was mis-killed as "translation-invariant"
   without checking the pads-as-attr placement trick.

2. **ConvInteger lane (probe-verified local ORT, STAGED as grader probe — not yet
   grader-validated):** `ConvInteger(x u8, w i8, x_zero_point)` runs on CPU, is NOT in the
   banned list, outputs INT32 (decodes fine via >0), and **pads with the ZERO POINT** —
   so x_zp=1 gives signed {+1,−1,0} semantics in a 1-byte-per-cell plane with correct
   all-zero padding. Halves the coupling-plane cost vs F16 Conv. task357 v6/v7 are the
   validation probes; if they score, add ConvInteger + Div(u8)/Mod(u8) to the validated list.

3. **Kernel probes settled (local ORT 1.24 CPU):** `Where(bool,i8,i8)` NOT_IMPLEMENTED
   (Where-u8 ✓ #74, Where-f16 ✓ new); `Div(u8)`, `Mod(u8, fmod=0)` ✓ exist; ConvInteger
   u8×i8 ✓; ConvInteger pad-fill = x_zero_point ✓.

4. **Nibble-pack pattern for tiny banks — use the HALF-SPLIT layout:** values ≤15 pack
   2-per-byte ([9,10]→[9,5], 45 params). Pack `p[k] + 16·p[k+5]` (first half in lo-nibble,
   second half in hi), NOT adjacent pairs: then `Concat(Mod(.,16), Div(.,16), axis=0)`
   restores order DIRECTLY — no Reshape node, no [2]-shape init (v7 387 → v8 375). Net win
   whenever bank_params/2 − 2 consts > ~15 B of unpack intermediates. Pairs with the
   parameter-LUT pattern from the earlier task357 note.
