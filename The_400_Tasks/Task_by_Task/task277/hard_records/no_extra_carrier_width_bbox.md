live bar: 3481 bytes, sha f727187c16c9d3caad99e7cbb71827701f4d9573e2378adc325f354951148c36 (ng_current_bar manifest, 2026-07-02)

attempted route: no-extra-flood-carrier rewrite from the ARC-GEN generator. The generator places two copies of a width-W sprite and one width-(W-1) column-deleted sprite, so the natural carrier-free idea is a bbox/width detector: color foreground inside width-W components as duplicate and the smaller component(s) as unique.

failed gate: price gate before full build. Even the optimistic skeleton is above the prompt's 1200-byte target before exact edge/validity logic: forced ch8 F32 read 400B + mask_bool 100B + 3-channel pre-Pad tail 300B + at least one narrow detector/input carrier 100B, leaving <300B for all width/height/boundary maps. A scoreable detector needs multiple per-dimension anchor maps for heights 2..4 and widths 2..4; those maps alone exceed the remaining budget and the older bbox/area prototypes already sit far above the current 3481 family.

why dead: removing the five flood rounds is not enough unless component assignment is also solved with a single very compact detector. The straightforward bbox-width formulation needs separate validity maps to avoid false larger boxes and adjacent-component leakage, so it misses the forced-floor target before truth/AB gates.

one remaining hypothesis: a single signed QLinearConv/Conv score that encodes width class, boundary-clear, and edge-nonempty tests in one output map might revive this route; it must be priced as one fused detector first and then stress-tested on fresh ARC-GEN split-column cases.
