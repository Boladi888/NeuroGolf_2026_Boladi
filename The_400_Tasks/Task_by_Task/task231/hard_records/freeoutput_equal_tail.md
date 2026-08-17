# freeoutput_equal_tail

- live bar: cost 168, sha256 `6f13ae96ff65787e0aea9794968ca32c3913dfe9ccbcf55b422ebce75131862d` (`ng_current_bar`, manifest source)
- attempted route: price the prompt's color-index / Equal / Pad output-tail replacement against the current graph
- exact failed gate: prebuild cost floor check; padded one-channel label canvas is 900 B, and small-grid one-hot before Pad is `10 * 5 * 20 = 1000` B at this generator bound
- why dead: both Equal-tail variants exceed the prompt target 67 and the current 168-cost Team_Best; current graph has no charged `[1,10,H,W]` prepad left, only the 120 B INT32 Gather index plus scalar width selectors
- one remaining hypothesis: a single terminal op that copies input columns and gates output width in-kernel without a materialized index vector; naive free-output selectors are F32-locked and price above 168
