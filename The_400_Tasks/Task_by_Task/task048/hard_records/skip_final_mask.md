live bar: 480, sha f1835f58bce71c8dcede1ad7cb892e39c3e19ac4540070ed614e4a3e05182569.
attempted route: remove final `reach_bits4 = r3_expanded & passable_bits` and hit-test `r3_expanded` directly.
failed gate: visible. `skip_final_mask_probe` = 215/270 FAIL at profile cost 472.
why dead: the unmasked last dilation crosses zero/background gaps into the target dilation on official cases.
one remaining hypothesis: a different target test might absorb one final-mask error, but the current target dilation already uses the cheapest visible-clean byte-equal scalar shave.
