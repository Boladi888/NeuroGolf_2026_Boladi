live bar: 480, sha f1835f58bce71c8dcede1ad7cb892e39c3e19ac4540070ed614e4a3e05182569.
attempted route: replace lowbit seed extraction with direct gathered `seed_row_bits` as the seed columns.
failed gate: visible. `seedrow_direct_probe` = 247/270 FAIL at profile cost 475.
why dead: official cases can have both boxes on the selected row; direct row bits seed both boxes instead of one 2x2 source box.
one remaining hypothesis: a cheaper leftmost-pair extractor would help, but known alternatives tie the current `x & -x` cost once scalar params and intermediates are counted.
