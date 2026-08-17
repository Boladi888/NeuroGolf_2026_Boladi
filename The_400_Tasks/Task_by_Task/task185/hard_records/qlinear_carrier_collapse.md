# qlinear_carrier_collapse

- Live bar used: cost 1096, sha256 `855b6183dbfe556915d22b23d48a6c58e7d00e64d22c2d77fb71645228d1edf5` (`ng_current_bar.py --task 185 --json`, manifest source).
- Attempted route: prompt-requested QLinearConv/U8 carrier collapse over the current v3 GridSample/tree graph.
- Failed gate: pre-build pricing/recon, before visible. No candidate was built because the current graph has no U8/INT8 feature tensor feeding the sampler; QLinearConv requires INT8/UINT8 input, so this route must first materialize a full F32 input cast (`[1,10,30,30]` U8 = 9000B) or build a color-index plane, both larger than the whole 1096-cost live graph.
- Why dead: task notes already priced the related rare-color-index plane as `+~1584B`, and library/ORT facts confirm QLinearConv rejects direct F32 graph input. Adding casts would erase any dtype win.
- Remaining hypothesis: only scalar-byte polish or a fundamentally new sampler that avoids both F32 selector planes and full-frame input casting is plausible; v5 U8 step-chain found a 3B scalar polish win.
