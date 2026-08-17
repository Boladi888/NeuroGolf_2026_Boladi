live bar: 360, sha256 dc7be94781b9f121753f5873a123425927af1b776e722e6092ceb6e2b228839a

Attempted route: dense tensor-network shave of Sho's single-Einsum graph by factoring the charged initializer bases (`VH`, `VL`, `G`, `H`, `M`) without adding charged intermediates.

Failed gate: pre-gate rank/cost check, before visible. `M` is full rank 10, so a two-factor color map needs at least 200 params vs current 100. `VH` rank 4 and `VL` rank 3 are already cheaper as direct dense matrices than matrix factors. `G` has CP lower bound rank 4 (`(t,a)|p` rank 4) so CP cost is at least 40 vs current 32. `H` has CP lower bound rank 3 so CP cost is at least 24 vs current 18. The combined mixed-radix comparator tensor `T[a,l,p,q] = sum_t G[t,a,p]*H[t,l,q]` has current dense split cost 50; exact TT checks over all axis orders gave best cost 75, and the useful unfolding rank is exactly the current 2-term split.

Why dead: the live graph is already a one-op free-output formulation with zero memory. The 360 cost is pure dense params (`VH 120 + VL 90 + G 32 + H 18 + M 100`), and the checked exact dense decompositions do not drop any component below its current representation.

One remaining hypothesis: a non-linear sign-threshold polynomial or a different comparator basis could beat 360 only if it represents the 10-column triangular interval relation with fewer than the current 260 comparator/basis params while preserving one-hot background suppression; this was not built.
