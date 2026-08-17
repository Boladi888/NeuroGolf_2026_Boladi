# task182 hard record: replace the row paired-add frame detector with direct ArgMax

Live bar: **4460** (Boladi, sha `b2c1867a...`, 2026-07-11). Conv-hash paradigm.

Context: frame localization is asymmetric. The COLUMN axis uses a direct `ArgMax(v_t5)` (0 extra
tensors). The ROW axis uses a paired-add: `v_t1=v_t0[0:14]`, `v_t2=v_t0[6:20]`, `v_t3=v_t1+v_t2`,
`ArgMax(v_t3)` — three FLOAT [1,14] tensors = **168 B**. The paired-add detects a COMPLETE 7x7
frame (gray rows r and r+6 both present, 6 apart), robust against the generator's incomplete-gray
decoy fragments.

Attempted route: drop `v_t1/v_t2/v_t3` and point `ArgMax(v_t4)` straight at the row profile `v_t0`
(save 168 B -> ~4292).

Failed gate: `ng_gate visible --task 182` -> **266/267 FAIL** (1 miss).

Why dead: at least one visible pair has an incomplete-gray decoy (or a taller single gray row) that
beats the true frame's top border under a raw row-ArgMax; the paired-add's "two borders 6 apart"
constraint is what rejects it. The 168 B is LOAD-BEARING. (Column direct-ArgMax is safe empirically
— its decoys don't create false single-column peaks — so the asymmetry is real, not an oversight.)

Remaining hypothesis: computing the paired sum via one Conv `[1,0,0,0,0,0,1]` on `v_t0` needs a
Reshape `[1,30]->[1,1,30]` (+120 B copy) + conv out (96 B) = 216 > 168, so it LOSES. The Einsum
can't emit 4D directly (`nchw,c->nh` is 2D), so no reshape-free conv route. Row detection is at its
floor. Do not re-attempt row direct-ArgMax or a conv-based row detector.
