# task372 hard record: separable moment/rank route

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: prompt-suggested row/column moment or low-term separable output formulation.
- Pre-build kill check: fresh ARC-GEN numpy probe over 64 generated examples found shapes `(11,11)->(5,11)` for all samples and verified the exact rule `overlay rows 0..4 with rows 6..10; lower non-background/non-5 wins, else upper`.
- Failed evidence: per-color output masks commonly have rank 3-5 over the 5x11 active output (`color0 rank5 on 57/64`; nonzero colors also reach rank 5). A low-term free-output separable Einsum would need one row-selector/column-selector term per mask rank, while a terminal row-shift basis would require a charged `30x30` row matrix or a dense spatial selector.
- Why dead under current bar: even the marginals-only skeleton would materialize row/column statistics (`[1,10,30,1]` and/or `[1,10,1,30]`, about 1,200 bytes each in F32) before rendering, already above the 200-byte one-node Conv bar. The current Conv encodes the 6-row offset as an attribute and pays only `10*10*2*1 = 200` params.
- One remaining hypothesis: a future win is more likely a terminal-op encoding or Conv ABI exploit than a moment rewrite; search for legal one-node ops with both spatial offset and cross-channel fallback semantics.
