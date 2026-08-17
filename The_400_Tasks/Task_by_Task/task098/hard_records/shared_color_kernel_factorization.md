route: share the repeated color-channel 3x3 kernel

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: factor the nine identical nonzero-color 3x3 boundary kernels into one shared detector, then route that detector back across color channels with `Mul`/`Where`/`Equal`/broadcast logic.

Exact failed gate: pre-build price gate. The duplicated color kernels cost only `9 channels * 9 taps = 81` params; keeping one shared copy would save at most 72 params. The first required shared detector or route mask is at least `[1,1,30,30]` BOOL/U8 = 900 bytes, before any routing params or final op.

Why dead: the current graph is params-only and pays no memory. Moving repeated color logic into a reusable node creates a charged full-canvas plane that is more expensive than the params it deletes.

Remaining hypothesis: a legal weight-sharing primitive inside the final output node, or a scorer-supported structured/sparse Conv weight, would be needed to exploit the repeated 3x3 kernels.
