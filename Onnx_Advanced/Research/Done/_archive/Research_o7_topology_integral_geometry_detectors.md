# Topology and Integral-Geometry Detectors as Free-Input Local-Pattern Reductions

**Intended save path:** `Research/Done/o7_topology_integral_geometry_detectors.md` `[PROMPT]`

## Scope and assumptions

[PROMPT] I assume the project setting exactly as stated: a static ONNX graph over input `[1,10,30,30]` F32 one-hot, with all padding cells all-zero across channels, no `Loop/Scan/If/Sequence/NonZero/Unique/Compress`, static shapes, and a cost model `cost = memory + params`, where initializer and `Constant` element counts contribute to `params`, and only intermediate tensors contribute to charged `memory`. `[PROMPT]`

[DERIVED] Everything below is stated for a **binary mask** `M ∈ {0,1}^{30×30}` obtained from one chosen input color channel by `Slice`/`Gather`, or from a fixed union of colors by a 1×1 linear selector over the one-hot channels; because the input is one-hot, any fixed sum of selected channels remains binary per cell. `[DERIVED]`

[PRIMARY] Stephen Gray’s original paper makes the key classical point for this card: on the square lattice, **area, perimeter, and Euler number are locally countable**, i.e. computable from bounded neighborhoods and then summed globally; Horn later summarizes the same trio as the “locally computable” discrete binary-image properties. `[PRIMARY]` citeturn11view0turn12view0turn21view0

[PRIMARY] Gray also assumes a **background border** around the image when scanning local patterns, and the later Euler-number literature keeps the same convention. `[PRIMARY]` [DERIVED] For a `30×30` mask, the safest exact lowering is therefore to add a one-cell zero border with `Pad`, giving a `32×32` padded mask and exactly `31×31 = 961` valid `2×2` windows. `[DERIVED]` citeturn4view1turn18search0

[PRIMARY] The topology caveat is essential: on the square grid, foreground/background connectivity must be taken as a **dual pair**—foreground 4-connected with background 8-connected, or foreground 8-connected with background 4-connected. Kong and Rosenfeld showed that if one uses the *same* 4- or 8-connectedness for both foreground and background, the Euler characteristic is **not** locally computable. `[PRIMARY]` citeturn8search0turn8search2turn8search8

## Euler characteristic via local patterns

[PRIMARY] Gray’s square-lattice derivation is the cleanest exact statement for this card. Let a `2×2` window be written in row-major order as `[[a,b],[c,d]]`, and let `n(Q1), n(Q2), n(QD), n(Q3), n(Q4)` be the counts of quads with, respectively, one foreground cell, two edge-adjacent foreground cells, two diagonally opposite foreground cells, three foreground cells, and four foreground cells. Then Gray’s two exact formulas are:
`4 χ_fg=8 = n(Q1) - n(Q3) - 2 n(QD)` and
`4 χ_fg=4 = n(Q1) - n(Q3) + 2 n(QD)`.
Gray’s notation is `W` for “ones joined” (diagonal foreground cells count as connected, i.e. foreground 8-connectivity) and `Z` for “zeros joined” (diagonal foreground cells count as disconnected, i.e. foreground 4-connectivity / background 8-connectivity). `[PRIMARY]` citeturn12view2turn12view3turn7view0

[PRIMARY] The proof is local and exact: Gray subdivides the digitized set into a cell complex and tabulates, for each quad type, the local contribution to `V - E + F`; because those contributions depend only on the quad type, summing them over all `2×2` windows gives the global Euler characteristic exactly. `[PRIMARY]` [DERIVED] This is the cubical-complex / vertex-edge-face proof requested by the prompt. `[DERIVED]` citeturn12view1turn12view2

### Exact `2×2` Euler weight table

[PRIMARY] The exact weights implied by Gray’s formula are the following. This table gives the **complete** local weight assignment for all 16 row-major `2×2` patterns. `[PRIMARY]` citeturn7view0turn12view2

| code | pattern `[[a,b],[c,d]]` | weight for `χ` with foreground 4 / background 8 | weight for `χ` with foreground 8 / background 4 |
|---|---|---:|---:|
| `0000` | `[[0,0],[0,0]]` | `0` | `0` |
| `0001` | `[[0,0],[0,1]]` | `+1/4` | `+1/4` |
| `0010` | `[[0,0],[1,0]]` | `+1/4` | `+1/4` |
| `0011` | `[[0,0],[1,1]]` | `0` | `0` |
| `0100` | `[[0,1],[0,0]]` | `+1/4` | `+1/4` |
| `0101` | `[[0,1],[0,1]]` | `0` | `0` |
| `0110` | `[[0,1],[1,0]]` | `+1/2` | `-1/2` |
| `0111` | `[[0,1],[1,1]]` | `-1/4` | `-1/4` |
| `1000` | `[[1,0],[0,0]]` | `+1/4` | `+1/4` |
| `1001` | `[[1,0],[0,1]]` | `+1/2` | `-1/2` |
| `1010` | `[[1,0],[1,0]]` | `0` | `0` |
| `1011` | `[[1,0],[1,1]]` | `-1/4` | `-1/4` |
| `1100` | `[[1,1],[0,0]]` | `0` | `0` |
| `1101` | `[[1,1],[0,1]]` | `-1/4` | `-1/4` |
| `1110` | `[[1,1],[1,0]]` | `-1/4` | `-1/4` |
| `1111` | `[[1,1],[1,1]]` | `0` | `0` |

[DERIVED] The weight table is exactly equivalent to the class formulas above: singletons contribute `+1/4`, three-ones quads contribute `-1/4`, edge-adjacent two-ones quads contribute `0`, and the only sign flip between foreground-4 and foreground-8 occurs on the diagonal pair patterns `0110` and `1001`. `[DERIVED]`

### ONNX lowering

[DERIVED] The cheapest exact ONNX lowering I found is an **encoded-quad Conv** rather than one Conv channel per pattern. Use a single fixed `2×2` stencil
`Kcode = [[8,4],[2,1]]`, applied to the zero-padded mask. Every `2×2` binary pattern then maps exactly to its integer code in `0..15`, and exact pattern indicators are obtained by `Equal(code_map, const_code)`. `[DERIVED]` [PRIMARY] `Conv`, `Pad`, `Equal`, `Cast`, and `ReduceSum` are standard ONNX operators; `Einsum` is also available, but the encoded-Conv route is smaller here. `[PRIMARY]` citeturn17search0turn17search1turn17search2turn17search3turn18search0turn17search13

```text
M   = Slice(input, color_channel)                     # [1,1,30,30]
Mp  = Pad(M, pads=[0,0,1,1,0,0,1,1], value=0)        # [1,1,32,32]
C   = Conv(Mp, W=[[8,4],[2,1]])                      # [1,1,31,31], exact codes 0..15
Ik  = Cast(Equal(C, k), to=INT32 or F16/F32)         # one indicator map per needed code
S4  = Sum(I1,I2,I4,I8) - Sum(I7,I11,I13,I14) + 2*Sum(I6,I9)
S8  = Sum(I1,I2,I4,I8) - Sum(I7,I11,I13,I14) - 2*Sum(I6,I9)
χ4  = S4 / 4
χ8  = S8 / 4
```

[SECONDARY] There is also a literature-proved **three-pattern compression** under a fixed raster orientation:
foreground 4-connectivity: `χ = #1000 - #1110 + #1001`,
foreground 8-connectivity: `χ = #1000 - #1110 - #0110`.
Arce et al. summarize these exact formulas from earlier work by Sossa et al. and related bit-quad papers. `[SECONDARY]` [DERIVED] For ONNX this means the encoded-quad map only needs three `Equal` tests instead of ten. `[DERIVED]` citeturn6view0turn6view1turn6view3turn6view4turn22search16

### Cost and dtype verdict

[DERIVED] For the **full Gray symmetric formula** using the encoded-quad Conv, the core parameter count is:
- `4` for the `[[8,4],[2,1]]` stencil,
- `8` for the `Pad` vector,
- `10` scalar pattern-code constants (`1,2,4,6,7,8,9,11,13,14`),
- `3` combination coefficients (`+1,-1,±2`),
- optionally `1` scalar for divide-by-4 if you need the exact scalar `χ`,
for a total of **26** constant elements, plus **2** more only if your chosen `ReduceSum` version requires axes as an initializer. `[DERIVED]`

[DERIVED] For the **three-pattern compressed formula**, the core parameter count is:
- `4` stencil,
- `8` pad vector,
- `3` scalar pattern codes,
- `3` combination coefficients,
for **18** constant elements, again plus optional `ReduceSum`-axes constants. `[DERIVED]`

[DERIVED] Under the project’s established “free-input reduction” convention, both lowerings are effectively **~0 charged memory** if the reduction chain is not surfaced as named intermediates. `[PROMPT][DERIVED]` [DERIVED] A conservative explicit-materialization bound is still tiny: the spatial code map has `31×31 = 961` elements; with `K=3` indicators the dominant casted indicator volume is `3×31×31 = 2883` elements, i.e. `5766` bytes in F16 or `11532` bytes in F32, while `K=10` gives `9610` elements, i.e. `19220` bytes in F16 or `38440` bytes in F32. `[DERIVED]`

[DERIVED] The dtype verdict is favorable. The encoded quad map only holds integers `0..15`, so `Equal` comparisons are exact in F16/F32. The full Gray numerator has worst-case absolute value at most `2 × 961 = 1922`, which is still below the F16 exact-integer limit of `2048`. The three-pattern compressed numerator is bounded by `961`. Therefore **Euler characteristic is F16-exact on a single `30×30` binary mask**. `[DERIVED]`

### Section table

| feature | exact local formula (2×2 / vertex-edge-face weights) | connectivity (4/8) caveat | ONNX lowering (Conv stencil + reduce / einsum) | params | charged memory | F16-exact or needs INT32/F32 | ARC predicate served | proof-or-counterexample given? | source tags |
|---|---|---|---|---:|---|---|---|---|---|
| Euler characteristic `χ`, full symmetric Gray form | `4χ_fg=4 = n(Q1)-n(Q3)+2n(QD)`; `4χ_fg=8 = n(Q1)-n(Q3)-2n(QD)`; full pattern weights listed above | Must use dual connectivity pair: fg-4/bg-8 or fg-8/bg-4; same-for-both is the classic silent bug | Encoded `2×2` `Conv` with `[[8,4],[2,1]]` → `Equal` against needed codes → `ReduceSum` → linear combine | `26` core, `28` if axes tensors counted | `≈0` by project convention; conservative explicit map `31×31×10` casted indicators | **F16-exact** on one binary mask | exact `χ`, and thresholded predicates derived from `χ` | **Yes**: Gray’s local `V-E+F` table | `[PROMPT][PRIMARY][DERIVED]` |
| Euler characteristic `χ`, exact three-pattern compression | fg-4: `χ = #1000 - #1110 + #1001`; fg-8: `χ = #1000 - #1110 - #0110` | Orientation-sensitive raster convention; still needs the same fg/bg duality | Same encoded `2×2` `Conv`, but only three `Equal` tests then one reduction/combine | `18` core, `20` if axes tensors counted | `≈0`; conservative explicit map `31×31×3` casted indicators | **F16-exact** | exact `χ` with lower param cost | **Yes, sketch**: cited exact compression of Gray under fixed scan orientation | `[SECONDARY][DERIVED]` |

[PRIMARY] The formulas and the connectivity caveat in the table come directly from Gray’s square-lattice derivation and the later review literature; the ONNX lowering and the exact param/memory arithmetic are my deductions from those formulas and the project cost model. `[PRIMARY][DERIVED]` citeturn12view2turn12view3turn8search0turn8search2turn6view0turn6view3turn6view4turn17search0turn17search1turn17search3turn18search0

## Hole count without labeling

[PRIMARY] For a binary image, the Euler number is the number of foreground objects minus the number of holes in those objects. That immediately gives the exact identity
`holes = components - χ`. `[PRIMARY]` [DERIVED] In Betti-number language for a planar binary mask, this is `β1 = β0 - χ`. `[DERIVED]` citeturn1view3turn1view0turn19search0

[PRIMARY] Therefore, under the common ARC-side assumption that the mask being measured is a **single connected object**, hole count is just
`holes = 1 - χ`,
with no labeling, no flood fill, and no iteration. `[PRIMARY]` [DERIVED] If the object class is also foreground-4-connected with no isolated diagonal-contact ambiguity, then `QD=0`, so the very cheap corner version `holes = 1 - (n(Q1)-n(Q3))/4` is exact. `[DERIVED]` citeturn1view3turn12view2turn1view0

[DERIVED] For the project’s sign-decoded output convention, the most useful Boolean lowering is
`has_hole = (1 - χ) > 0`.
With the full Gray numerator `S = 4χ`, this can be emitted as `4 - S`; the final graph output is positive exactly when the single connected object has at least one hole. `[DERIVED]`

[DERIVED] For **component count** `β0` itself, the boundary is sharp in practice but not as beautifully theorem-backed as Euler. On useful restricted classes where **each component contributes exactly one bounded-radius anchor**, `β0` *is* a local count. The cleanest exact case is pairwise-disjoint filled axis-aligned rectangles: every rectangle contributes exactly one occurrence of any fixed outer-corner singleton code (for example code `1000` under one raster convention), so rectangle count is an exact single local reduction. `[DERIVED]`

[DERIVED] Once the class is widened to arbitrary blobs, that anchor property disappears: a single component can contribute many singleton-corner quads, and a one-pixel bridge can merge two components without giving any bounded-radius “one per component” witness. `[DERIVED]` [SECONDARY] The practical literature therefore treats unrestricted connected-component counting by labeling / connectivity algorithms rather than by one-shot local valuations. `[SECONDARY]` citeturn25search1turn25search8turn25search10

### Section table

| feature | exact local formula (2×2 / vertex-edge-face weights) | connectivity (4/8) caveat | ONNX lowering (Conv stencil + reduce / einsum) | params | charged memory | F16-exact or needs INT32/F32 | ARC predicate served | proof-or-counterexample given? | source tags |
|---|---|---|---|---:|---|---|---|---|---|
| hole count for a **single connected object** | `holes = 1 - χ`; with fg-4 and no `QD`, `holes = 1 - (n(Q1)-n(Q3))/4` | Same connectivity choice as the Euler detector you reuse; single-component assumption is mandatory | Reuse Euler detector, then add scalar `1` and negate/subtract | Euler-detector params `+1` scalar if explicit | Same as Euler detector | **F16-exact** | `is_hollow`, `has_a_hole`, “donut vs solid” | **Yes**: by definition `χ = components - holes` | `[PRIMARY][DERIVED]` |
| component count for disjoint filled axis-aligned rectangles | `β0 = #1000` under one fixed raster convention, or any equivalent unique outer-corner singleton code | Exact only for pairwise-disjoint **filled rectangles** with zero border convention | Encoded `2×2` `Conv` → `Equal(code, singleton_code)` → `ReduceSum` | `13` core (`4` stencil + `8` pad + `1` code), plus axes if counted | `≈0`; conservative explicit `31×31` indicator map | **F16-exact** | number of rectangles / tiles | **Yes**: one designated outer corner per rectangle, exactly one per component | `[DERIVED]` |
| unrestricted component count `β0` on arbitrary masks | No confirmed exact finite-local one-shot formula from the sources reviewed; treat as non-local for this card | General arbitrary blobs / merges / bridges break any fixed local anchor rule | Use labeling / flood-style route, not a local reduction | — | — | — | generic component counting | **Boundary**: restricted-class yes, unrestricted no for this card | `[SECONDARY][DERIVED]` |

[PRIMARY] The hole formula is exact because it is just the digital definition of Euler number rewritten; the rectangle-count row is an exact restricted-class construction; the unrestricted-`β0` row is the practical boundary suggested by the connected-components literature, which uses labeling/connectedness algorithms rather than a Gray-style valuation. `[PRIMARY][SECONDARY][DERIVED]` citeturn1view3turn25search1turn25search8turn25search10

## Other intrinsic volumes and exact local detectors

[SECONDARY] In integral-geometry language, the classical 2D intrinsic volumes / Minkowski functionals are **area, perimeter, and Euler characteristic**. Horn’s note and standard intrinsic-volume references align with that triad. `[SECONDARY]` [PROMPT] Area is already covered elsewhere in your board, so I only restate it briefly for completeness. `[PROMPT]` citeturn21view0turn20search2turn20search3

[PRIMARY] Gray’s exact square-lattice **digital perimeter** formula is
`P = n(Q1) + n(Q2) + 2 n(QD) + n(Q3)`.
This is the exact exposed-edge boundary length of the union of foreground unit squares, not a Euclidean-contour approximation. `[PRIMARY]` [DERIVED] It is connectivity-agnostic in the sense that it measures exposed pixel edges; what changes with 4/8 choices is topology, not this edge count. `[DERIVED]` citeturn12view0

[DERIVED] There is also a cheaper exact formula using adjacency pairs:
`P = 4F - 2H - 2V`,
where `F` is area (foreground-cell count), `H` is the number of horizontal `11` pairs, and `V` is the number of vertical `11` pairs. The proof is one line: every foreground cell contributes four edges, and every shared horizontal or vertical edge cancels two boundary edges. `[DERIVED]` [PRIMARY] Horn’s summary is consistent with this pairwise-local view of perimeter. `[PRIMARY]` citeturn21view0turn12view0

[PRIMARY] Gray’s quad formulas also give exact **corner counts**. `n(Q1)` is the count of convex outer corners; `n(Q3)` is the count of concave inner corners. In the 4-connected no-diagonal-ambiguity case, the corner-codification literature writes the same Euler identity as `χ = (N1 - N3)/4`, where `N1` and `N3` count vertex corners touching one and three foreground cells, respectively. `[PRIMARY]` [DERIVED] For many ARC objects that are orthogonal and free of isolated diagonal contacts, this means corner counting is an especially cheap exact detector for “rectangle-like”, “hollow-box”, and corner-threshold predicates. `[DERIVED]` citeturn1view0turn12view2

[SECONDARY] By contrast, chain-code / Freeman / traced / Crofton perimeters are **different definitions**, often chosen to approximate continuous Euclidean boundary length, and the sources explicitly describe them as approximations or as definition-dependent. `[SECONDARY]` [DERIVED] They are therefore **out of scope** for this card’s “exact local reduction” criterion. `[DERIVED]` citeturn15view0turn15view1

### Section table

| feature | exact local formula (2×2 / vertex-edge-face weights) | connectivity (4/8) caveat | ONNX lowering (Conv stencil + reduce / einsum) | params | charged memory | F16-exact or needs INT32/F32 | ARC predicate served | proof-or-counterexample given? | source tags |
|---|---|---|---|---:|---|---|---|---|---|
| area `F` | `F = ReduceSum(M)` | None | Direct `ReduceSum` on mask | `0` or small axes constant only | `≈0` | **F16-exact** (`F ≤ 900`) | area thresholds, size tests | **Yes**: direct count | `[PROMPT][DERIVED]` |
| perimeter `P`, exact Gray quad form | `P = n(Q1)+n(Q2)+2n(QD)+n(Q3)` | No topology caveat for exposed-edge count; this is not a Euclidean approximation | Encoded `2×2` `Conv` → `Equal` on 14 nontrivial codes → `ReduceSum` with diagonal weight `2` | about `27` core (`4` stencil + `8` pad + `14` code constants + `1` diagonal extra-weight scalar), plus axes if counted | `≈0`; conservative explicit `31×31×14` casted indicators | **F16-exact** because weighted sum is bounded by `1922` | boundary-length thresholds, contour complexity | **Yes**: Gray’s exact local perimeter formula | `[PRIMARY][DERIVED]` |
| perimeter `P`, exact pair-count form | `P = 4F - 2H - 2V`, with `H=#horizontal 11`, `V=#vertical 11` | Same exposed-edge measure as above | `ReduceSum(M)` plus exact `1×2` and `2×1` pair detectors (`Conv == 2`) | about `9–12` core constants depending combine style | `≈0`; conservative explicit pair maps `30×29` and `29×30` | **Needs INT32/F32 if implemented literally**, because `4F` can reach `3600` even though final `P ≤ 1800` | boundary thresholds at very low param count | **Yes**: four-edges-minus-shared-edges proof | `[DERIVED]` |
| convex / concave corner counts | `C+ = n(Q1)`, `C- = n(Q3)`; with no `QD`, `χ = (C+ - C-)/4`; rectangle iff `C+=4`, `C-=0`, `χ=1` | Exact as corner counts; the simplified Euler relation needs no isolated diagonal-contact ambiguity | Encoded `2×2` `Conv` → `Equal` on singleton and three-ones codes → `ReduceSum` | about `20` core (`4` stencil + `8` pad + `8` code constants), plus axes if counted | `≈0` | **F16-exact** (`C+ , C- ≤ 961`) | number-of-corners, rectangle / hollow-box tests, concavity tests | **Yes**: each local vertex type corresponds to one corner incidence | `[PRIMARY][DERIVED]` |

[PRIMARY] The perimeter and corner identities are exact in Gray’s digital sense; Horn’s “pairs for perimeter, three cells at a vertex for Euler” summary matches the same structural decomposition; and the approximation caveat for traced / Crofton-style perimeters is stated explicitly in modern measurement docs. `[PRIMARY][SECONDARY]` citeturn12view0turn21view0turn15view0turn15view1

## Provable boundary of non-local predicates

[DERIVED] Let `Φ_r(I) = Σ_x φ(neighborhood_r(I,x))` be any bounded-radius local-sum functional. If two images `A` and `B` are placed more than `2r` apart, then every local neighborhood lies entirely in one copy or the other, so
`Φ_r(A ⊔ B) = Φ_r(A) + Φ_r(B)`.
This *separated-union additivity* is the key reason Euler works: it is a valuation/local sum. `[DERIVED]`

[PRIMARY] Gray explicitly highlights additivity for the locally computable quantities, and Horn restates the same locality principle in discrete form. `[PRIMARY]` [DERIVED] This yields a short impossibility proof for several tempting predicates. `[DERIVED]` citeturn11view0turn21view0

[DERIVED] **Connectedness** is not a finite local sum: take any nonempty connected image `C`. If connectedness were `Φ_r`, then for two far-separated copies we would get
`Φ_r(C ⊔ C) = Φ_r(C) + Φ_r(C)`,
but the left-hand side should encode “disconnected” while the right-hand side doubles the value on one connected component. No scalar additively accumulated local sum can serve as the exact connectedness predicate on arbitrary images. `[DERIVED]`

[DERIVED] The same locality barrier rules out, for arbitrary masks, exact one-shot local-reduction implementations of: **“are pixels A and B in the same component?”**, **reachability**, **shortest-path length**, **largest-component area**, and any **per-component aggregation** such as “does some component contain color `c` and have area > t?” Those tasks require global component identity, which bounded local neighborhoods do not carry. `[DERIVED]`

[DERIVED] A related warning applies to alluring global shape predicates such as **general convexity** or **global orthogonal convexity**: a finite local-sum reduction can count local concavities (`n(Q3)`), but the Boolean statement “this arbitrary mask is convex” is not itself an additive local valuation. `[DERIVED]`

[SECONDARY] The practical exact route the wider literature uses for these nonlocal tasks is precisely **connected-component labeling / connectivity propagation**, not a Gray-style valuation. `[PROMPT] In your own toolkit, the existing non-algebraic route is the iterated `MaxPool` flood. `[PROMPT]` [SECONDARY] MathWorks’ connected-components documentation is representative of the standard exact route on binary images. `[SECONDARY]` citeturn25search1turn25search10

## Overall verdict

[DERIVED] The highest-value exact additions are bluntly these. **First, add Euler characteristic immediately**: use the encoded-quad `Conv` with `[[8,4],[2,1]]`; either the full Gray weights (`+1/4` on singletons, `-1/4` on three-ones, `±1/2` on diagonals) or, better for params, the exact three-pattern compression `χ_fg=4 = #1000 - #1110 + #1001` and `χ_fg=8 = #1000 - #1110 - #0110`; this costs about **18 core params** in the compressed version, has **≈0 charged memory** under your convention, and is **F16-exact** because all accumulators stay below `2048`; it directly serves ARC families like **single-object hole detection**, **ring vs solid**, and **topology class tests**. **Second, hole count for a single connected object is a free corollary**: `holes = 1 - χ`, so a Boolean `has_hole` detector is just a sign test on `1 - χ` with no labeling at all. **Third, add exact exposed-edge perimeter**: if you want F16-only safety, use Gray’s exact quad formula `P = n(Q1)+n(Q2)+2n(QD)+n(Q3)`; if you are happy to accumulate in INT32/F32, the cheapest graph is `P = 4F - 2H - 2V` from area plus horizontal/vertical `11` pairs. **Fourth, add convex/concave corner counts** `C+=n(Q1)` and `C-=n(Q3)`, which are exact, tiny, and ARC-useful for **rectangle**, **box**, **hollow frame**, and **corner-threshold** predicates; in the no-diagonal-ambiguity 4-connected setting, they also give the simplified exact identity `χ=(C+-C-)/4`. **Fifth, keep unrestricted connectedness / same-component / path tasks on the nonlocal side** and stop hunting for a one-shot local trick there: the algebraic shortcut buys you **Euler, holes, perimeter, and corner structure**; it does **not** buy you generic labeling, reachability, or per-component aggregation, for which your existing **iterated MaxPool flood** remains the right route. `[DERIVED][PROMPT]` citeturn12view0turn12view2turn12view3turn6view0turn6view3turn6view4turn21view0turn25search1
