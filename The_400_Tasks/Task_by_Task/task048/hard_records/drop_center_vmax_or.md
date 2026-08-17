live bar: 480, sha f1835f58bce71c8dcede1ad7cb892e39c3e19ac4540070ed614e4a3e05182569.
attempted route: delete `r2_expanded = Or(r2_vmax,r2_horiz)` and/or `r3_expanded = Or(r3_vmax,r3_horiz)`.
failed gate: visible. drop-both = 269/270 FAIL; drop-r2 = 269/270 FAIL; drop-r3 = 269/270 FAIL.
why dead: `MaxPool` over packed row bytes is numeric max, not bitwise vertical OR; the center-row OR is load-bearing in later rounds.
one remaining hypothesis: exact vertical OR needs row shifts/pads and costs more than the 8-byte center term in this graph.
