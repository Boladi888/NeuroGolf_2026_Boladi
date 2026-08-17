live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route: replace final `Equal(color30_u8, color_bank)` with OneHot.

Failed gate: ORT load for the bool-valued probe failed with `NotImplemented: Could not find an implementation for OneHot(11) node with name 'onehot_tail'`.

Why dead: the current index canvas is `[1,1,30,30]`; OneHot inserts a depth axis, so a legal `[1,10,30,30]` output first needs a charged squeeze/rank-change path. Even if a supported F32-valued OneHot variant loaded, that extra 900 B squeeze canvas makes it byte-negative versus the existing Equal tail before correctness.

Remaining hypothesis: OneHot only belongs here if the color index is already rank `[1,30,30]` or the output active area is below the #81 small-grid crossover.
