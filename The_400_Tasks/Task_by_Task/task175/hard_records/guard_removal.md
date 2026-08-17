live bar: manifest 1231, sha b2b7db986c3d9fb9b2743deb4589985f7e4c418ad7823b3e2b411baa7034a725; local parent predictor 1115.

attempted route: delete or reuse the invalid-channel guard false sentinel in the current bitset architecture.

failed gates: `no_where_guard` visible 0/266; `zero_false_guard` visible 0/266; `false_period` visible 0/266; `false_raw` visible 222/266. `false_limit` visible 266/266 but graph-cost tie because it removes only a scalar uncharged by cost_predictor.

why dead: invalid output channels collide with real mask bits unless the false shift is safely beyond bit 10; reusing `v_t9` is safe but not charged, so it is not a cost win.

one remaining hypothesis: only revisit if another rewrite already computes a high safe sentinel as a charged non-scalar that can replace a charged guard tensor.
