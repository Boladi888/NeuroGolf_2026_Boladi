# Strided-Slice anchor for 3-spaced lattice detection — beats F32-marginal anchor (task195)

## What I discovered

**A cheaper anchor primitive for "first-active row/col on a strided lattice" than the two-F32-marginal + clamp + ArgMax pattern.** When the rule samples a `k`-spaced lattice anchored at the first active row/col (here `kron(P,P)` from a 3-spaced gray lattice), you can locate AND read the lattice in one cheap head using a **strided Slice**, instead of full-width `[1,30]` F32 Einsum marginals:

```
Slice(input, axes=[0,1,2,3], starts=[0,5,k,k], ends=[1,6,END,END], steps=[1,1,k,k])  # [1,1,S,S], S small
Cast -> U8                      # 1-byte plane, e.g. [1,1,5,5] = 25 B
ReduceMax(axes drop H) -> [S]   # row presence (5 B)
ReduceMax(axes drop W) -> [S]   # col presence
ArgMax -> top, left             # FIRST strided lattice position with the marker
Add [0,1,2]                     # the 3 consecutive strided picks
Gather (axis=W) then Gather (axis=H)  -> P [1,1,3,3]   # the sampled pattern, U8
```

Why it works: the marker only sits on the `k`-spaced lattice, so a strided window
`{k, 2k, 3k, ...}` always contains the whole lattice, and `ArgMax`-of-first-strided-hit
gives the lattice's first occupied strided index. `+[0,1,2]` reads the 3×3 pattern.
**Plain `Gather` with INT32 indices is fine here** — this is NOT GatherND, so the
INT64-index restriction (and its 216 B index grid) does not apply.

Cost on task195: this anchor head charges ~150 B total. The alternative head we had
(freeoutkron) used two `[1,30]` F32 Einsum marginals (240 B) + a binary-clamp→ArgMax
chain (120 B) + an INT64 GatherND index `[1,3,3,3]` (216 B) = **576 B** for the same
"locate + read 9 cells" job. **The strided Slice is ~426 B cheaper.**

## Where I encountered it

- task195. Builder: `The_400_Tasks/Task_by_Task/task195/prototypes/build_task195_hybrid_slicekron.py`
- Staged: `staged_submits/task195_hybrid_slicekron_cost525/` (cost 525, profile==static==525).
- I found the strided-Slice primitive by **reading the team-best onnx** (Aditya, 921) directly — node-for-node. It was sitting in plain sight in `Team_Best_Onnx/task195.onnx`.

## Why it matters (two compounding lessons)

1. **The cheaper-anchor lesson:** strided Slice + ReduceMax + plain Gather beats the
   F32-marginal + clamp + GatherND anchor for strided-lattice "first-active" detection.

2. **The process lesson (the bigger one):** task195 had a documented "WALL: cannot
   cross 921" from a prior beat-or-wall audit (2026-06-20). That audit did an
   exhaustive per-tensor floor proof — but **only of OUR formulation's head**. It
   never read the team-best graph it was trying to beat. The team-best used a
   completely different, cheaper anchor primitive. **Before declaring a WALL against
   a known team-best cost, READ THE TEAM-BEST ONNX** (`extract`/`onnx.load` it, dump
   nodes+initializers). The construction you "don't have documented" is often
   literally inside the file you're trying to beat, and is directly transplantable.
   This reinforces #68 (floor proofs are formulation-relative).

3. **Hybrid bonus:** the final 525 also keeps OUR free-output separable Einsum
   (#109/#119), which the team-best graph LACKS — Aditya still materialized a
   `[1,6,9,9]` U8 = 486 B label canvas + kron/out5 + Where/Pad tail. So the win was
   "their cheaper anchor + our cheaper output", each side missing what the other had.
   General move: when both your graph and the team-best beat each other on DIFFERENT
   sub-blocks, splice the cheaper sub-block from each.

## What's missing from the library

I did not find a "strided-Slice first-active-on-lattice anchor" primitive in
`techniques_catalog.md` / `by_symptom.md` (I searched anchor/first-active/lattice
routes). The closest entries are about free-output Einsum placement (#109/#118/#119)
and ArgMax-of-first-active (which assumes a full-width marginal + binary clamp). The
strided-Slice-as-anchor idea — and especially "ReduceMax on a small strided U8 window
+ plain INT32 Gather" as a sub-1-byte-plane alternative to F32 marginals — deserves a
short technique entry, routed from by_symptom "locate the first active row/col of a
k-spaced lattice". Also worth a process note in the beat-or-wall doctrine: read the
team-best graph before writing a WALL verdict.
