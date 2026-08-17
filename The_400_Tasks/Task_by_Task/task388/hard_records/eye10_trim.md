# task388 eye10_trim dead route

- live bar: 1087 from `python tools/ng_current_bar.py --task 388 --json`
- live sha: `3faca8164fe79aba25cae6527e5b72f268cf0180f2b5eabae5af5982e135483b`
- attempted route: shrink live initializer `EYE` from `[10,11]` to `[10,10]` to remove the apparent all-zero sentinel column.
- failed gate: pre-gate source-read kill; no visible/truth/ab run.
- why dead: the graph stores shifted color ids. `EYE[:,0]` is the no-color sentinel, while `EYE[:,10]` maps shifted id 10 to real output channel 9. ARC-GEN allows the unique nonzero color `X=9`, so column 10 is hidden-live even if a small visible slice missed it.
- one remaining hypothesis: replace `Gather(EYE, fi)` with a different shifted-id-to-one-hot materialization only if it saves more than the 110-param table without creating an extra `[10,1,6,6]` intermediate.
