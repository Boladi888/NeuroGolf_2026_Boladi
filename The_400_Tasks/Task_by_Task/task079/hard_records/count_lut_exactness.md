# task079 count-LUT exactness dead route

- Live bar: cost 758, sha256 `834fe149b126376ade77a246b1ebcebf8e6abc31472d84d4881c07eaa630ad31`, source `ng_current_bar.py` authoritative manifest.
- Attempted route: keep Vamshi's count-index selector, swap the final BOOL patch decoder to `UINT8 Mul -> Equal`, and retune/extend the `lut` initializer.
- Failed gates:
  - parent-LUT tail probe: cost 660, visible 266/266, but generated A/B hit `Gather` out-of-bounds `idx=21` on the 19-entry LUT.
  - simple `lut22` extension: visible 266/266, profile 663, truth 1965/2000 (98.250%) => low truth.
  - unconstrained tuned `lut22`: truth 1984/2000 (99.200%) but visible 264/266 => visible fail, dead by §VISIBLE-FAIL.
- Why dead: count-only ordering has contradictory generated-corpus constraints, including both `11 > 12` and `12 > 11`; no exact LUT ordering exists without another feature.
- One remaining hypothesis: add one compact sprite-density/row-shape tie-break feature while preserving the 663-byte tail shave; the current charged floor is dominated by `band` 168B, `opatch` 90B, and selector tensors at 40B each.
