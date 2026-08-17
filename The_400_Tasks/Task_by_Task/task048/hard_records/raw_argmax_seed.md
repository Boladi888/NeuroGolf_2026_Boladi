live bar: 480, sha f1835f58bce71c8dcede1ad7cb892e39c3e19ac4540070ed614e4a3e05182569.
attempted route: delete `row_has_twos_u8` and let `ArgMax` read raw `twos_bits`; also tried reversing the first horizontal shift to match the rightmost-box bias.
failed gate: visible. `raw_argmax_probe` = 263/270 FAIL; `raw_argmax_r0_right_probe` = 266/270 FAIL.
why dead: raw row-byte priority chooses a different seed box on official cases, and the current asymmetric first expansion/target dilation no longer matches.
one remaining hypothesis: a fully symmetric first expansion can repair raw seed choice, but it adds at least the 8-byte lane saved here and does not beat the 480 bar.
