# task182 hard record: crop the 16x16 shape-hash map to 14x14

Live bar: **4460** (`ng_current_bar.py --task 182`), owner Boladi, sha256
`b2c1867a3daee66808dbf4ad64acc64a617b140393acfbaefbac40d2dc3652e8`, updated 2026-07-11T00:40:30Z.
Paradigm: conv shape-hash (see task182.md CURRENT decode).

Attempted route: the hash Conv (`v_w12 [1,10,5,5]` powers-of-random-weights on channel-1) emits
`v_t17` FLOAT **[1,1,16,16] = 1024 B**, the single biggest tensor (23% of cost). The random
generator places sprites at `rows/cols = randint(1, size-6) = randint(1,14)`, so blue anchors
fire only at placement `(row,col) in [1,14]^2` (verified by instrumenting `v_t19`: seed anchors
(8,1),(12,14),(3,14),(13,4) — all in [1,14]). Cropped the hash Conv pads `[0,0,-10,-10]` ->
`[-1,-1,-11,-11]` => `v_t17 [1,1,14,14]=784`, adjusted the stamp to 18x18 and the final Pad to
`[1,1,11,11]`. Est. total saving ~512 B (hash 1024->784, anchors 256->196 x2, stamp 400->324 x2).

Failed gate: `ng_gate visible --task 182` -> **265/267 FAIL**.

Why dead: the hand-authored **train/test** examples in `task182.json` use EDGE placements outside
the random [1,14] range. `validate()` train[2] `rows=[1,0,9,10,14,15]` has a matching idx-5 blue
sprite at **row 15** (recolor at rows 16,17); test `rows=[14,0,1,3,8,8,8,15]` uses rows 0 and 15
and col 0. Diff vs parent: exactly 2 pairs (train idx2: 6 px @ rows16-17; test idx4: 5 px @
rows9-11) — both anchored at row 15 / col edge. 300/300 pure arc-gen pass, but VISIBLE-FAIL AXIOM
kills it. The full 16x16 (placement range [0,15]^2) is FORCED for visible safety; F32 forced by
Conv output dtype. **v_t17 = 1024 B is irreducible.** Do not re-attempt any spatial crop of the
hash/anchor/stamp planes. (Cols never exceed 14 across all visible+random, but cropping cols to
[0,14] on the strength of 5 authored examples is an unjustified visible-fail gamble for ~64 B.)

Remaining hypothesis: none for cropping. The only dtype trick (F16 hash) needs an extra Cast
(map born F32 by Conv) -> net +512, dead. Blue-plane extraction for a UINT8/QLinearConv match
costs a 3600 B F32 Einsum plane >> 1024, dead.
