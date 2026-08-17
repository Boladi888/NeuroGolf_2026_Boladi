live bar: 480, sha f1835f58bce71c8dcede1ad7cb892e39c3e19ac4540070ed614e4a3e05182569.
attempted route: keep only one horizontal side of the target-column dilation by dropping either `other_cols_l` or `other_cols_r`.
failed gate: visible. `target_drop_l` = 267/270 FAIL; `target_drop_r` = 267/270 FAIL, both at profile cost 478.
why dead: officials require target dilation on both horizontal sides. The only safe target-column shave found was different: drop the original-center scalar from the intermediate OR while keeping both side shifts.
one remaining hypothesis: no local one-side target reduction remains; further target savings need a different flood/target framing.
