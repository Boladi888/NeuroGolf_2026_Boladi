# task193 depthwise sub-210 footprint infeasibility

- Live/team bar used: 260, sha256 `2d169ad9818e988befc624641f554289f002913b9a149531df4829563ff6a406`, source `python tools/ng_current_bar.py --task 193 --json` / team manifest.
- Attempted route: single-op zero-memory depthwise Conv with footprint area below 20 cells while preserving the 3x3 cross predicate for channels 1-9.
- Exact failed gate: pre-build visible-label feasibility in `prototypes/probe_depthwise_footprints.py`; channel-0 patches for 3x3, 3x4, 3x5, 3x6, 3x7, and 3x8 footprints have direct label conflicts on the visible corpus, so any such linear Conv would have a visible miss.
- Why dead: same local ch0 patch must emit both background and non-background under the official visible pairs; no linear weights can separate those conflicts, independent of LP margin.
- One remaining hypothesis: byte-equal compression may still exist by fitting to the current 260-byte parent rather than ground truth, but it must keep a single zero-memory Conv or it will exceed the 260-byte bar.
