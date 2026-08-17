# task032 compact_vertical_qlinearconv hard record

- Live bar used: cost 403, sha256 `e3ca4a62f12fe88ff9852a6159c24108f8438d030eff62ecf6a74267146754ce` from `python tools/ng_current_bar.py --task 32 --json` / team manifest.
- Attempted route: replace the 6x6 `bottom -> code -> square -> Concat -> QLinearConv` tail with a final 6-row vertical-kernel `QLinearConv` over compact per-column features, so the output op would place rows directly.
- Exact falsification: LP/separability probes over legal generator states showed `[packed_y, packed_y^2 mod 256, n_col]` infeasible for color channels 1..9 with shared QLinearConv bias; adding active-column as a fourth no-bias feature remained infeasible for color channels (`ch1 row2` first failure) for every legal base `D=46..64`.
- Why dead: the UINT8 square aliases the packed color/count classes, and the 3-4 feature vertical kernel cannot linearly separate color singleton outputs while also enforcing row placement and active-grid padding. A 5-feature version would need at least `10*5*6 = 300` final weights, which prices above the 403 live bar before safety gates.
- One remaining hypothesis: a compact final kernel only has a chance if it gets a cheap true non-aliased nonlinear color feature, not `UINT8 x*x mod 256`; otherwise Sho's full 6x6 code/square classifier is the lower-cost representation.
