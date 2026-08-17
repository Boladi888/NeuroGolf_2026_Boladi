# Free-output Einsum: when operand factoring CANNOT save params (task011 floor proof)

**Where:** task011 (missing-8 key-block stamp). v7 = team-best = 901 (mem 153 + params 748),
a single free-output separable Einsum `buhw,rih,sjw,mrs,iH,jW,ijm,muc->bcHW`. Prompt asked to
factor the two largest operands (`chan_map [2,10,10]`=200, `src [3,4,30]`=360). Verdict: 901 is
the hard floor; all four factoring directions measured dead. Builder probes in `/tmp`; two-node
attempt cost 72,649.

## The transferable rule (belongs in techniques.md #118 basis-as-operand AND the FREE-OUTPUT doctrine)

A free-output Einsum charges ONLY its operands' element counts (the output tensor is uncharged).
Two hard laws govern whether you can shrink an operand:

1. **An identity / diagonal coupling operand `δ(u,c)` is IRREDUCIBLE within a single node.**
   `matrix_rank(identity)=N` (full). Any low-rank factor `[N,k]+[k,N]` needs k=N → `2N²` > `N²`.
   In task011, `chan_map[0]` is identity (the stamp-content channel passthrough, c=u) → its 100
   elements are floored. You CAN eliminate it by **renaming the contracted index to the output
   index** (`c=u`, dropping the operand entirely) — BUT only if that term can stand as its own
   node. The moment a second term (here the separator, `chan_map[1]` = rank-1 `u→ch5`) also needs
   the output index free, you must KEEP both planes stacked under a mode axis (`muc`), and the
   identity plane drags its full `N²` along. Rank-1-ness of the OTHER plane buys nothing because
   you cannot peel it into a cheaper operand without splitting the node.

2. **Splitting a free-output Einsum into ≥2 nodes + Add re-charges the full output-shape
   intermediate, EACH.** Measured on task011: the 2-node factoring dropped params 748→568 (chan
   200 → tm0/tm1/e_all/e5 = 52) but birthed two `[1,10,30,30]` fp32 intermediates (`content`,
   `sep`) = 72,000 B memory before the Add → total cost **72,649**, i.e. **80× worse**. The
   free-output property (output never charged) survives ONLY while the output is produced by a
   SINGLE terminal node. Any `Add`/`Concat`/`Where` merging two output-shaped tensors charges both.
   **This is the general trap behind "factor the operand into the free-output Einsum" advice: a
   factor that stays as an extra OPERAND on the one node is free; a factor that becomes a separate
   NODE feeding a merge is catastrophic.**

3. **Trailing-zero operand columns are only "free" slack if the corresponding axis is an
   INTERMEDIATE you control — never when it's pinned to the graph I/O shape.** task011's `src`
   `[3,4,30]` and `out [4,30]` use only cols 0-10 (19 zero cols). But the 30-axis is the
   fixed 30×30 competition canvas: `src`'s maps to the graph INPUT spatial dim, `out`'s to the
   graph OUTPUT spatial dim. Shrinking either requires a Slice of the I/O, which materializes a
   ≥2,420 B (fp16) charged intermediate — far more than the ~228 params saved. (Confirmed earlier
   by the v_w8a/crop ladder: the crop intermediate was always the dominant cost.)

## Scoring fact worth pinning (verified in `neurogolf_utils.calculate_params`)

`params = Σ math.prod(init.dims)` — **counts ELEMENTS, dtype-independent.** So converting a big
initializer (`chan_map`, `src`) to fp16 saves **0 params**. Only `calculate_memory` is
dtype-scaled (`num_elements * itemsize`), and it applies to value_info'd intermediates, not inits.
Corollary: in a graph whose cost is params-dominated with a tiny memory footprint (task011: mem
153), fp16 tricks are useless; the only lever is reducing initializer ELEMENT counts, which the
two laws above usually forbid.

## Why it matters

Saves future workers from re-walking the "factor chan_map/src" rabbit hole on task011 (and any
free-output separable-Einsum task). The pattern "operand is rank-1 / has zero columns, therefore
factorable" is FALSE unless the factor can ride as an extra operand on the single output node
without changing which index is free. The clean diagnostic: *does my factoring add a NODE or just
an OPERAND?* Operand = free; node = 18 KB-class memory charge.
