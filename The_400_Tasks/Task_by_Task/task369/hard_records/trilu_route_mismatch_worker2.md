# task369 - Trilu triangular mask route is mismatched

- Live bar: cost **982**, sha256 `06b03bcda0ed2309c93c2dee084bb9032a057483332ace744d1ff58b10695b01`, source `python tools/ng_current_bar.py --task 369 --json` / manifest.
- Attempted route: prompt-named `Trilu(all-ones)` triangular/ray mask rewrite.
- Failed gate: pre-build rule/recon kill check. `context_pack.py --task 369`, BoBrain context, and the task-local rule all decode task369 as fixed 10x10 zero-adjacency labeling: preserve color 5; recolor zero cells by 4-neighbor zero degree and the lone-neighbor heavy-degree test.
- Why dead: the required mask is not triangular/ray-like and does not need an upper/lower-triangular relation. Building Trilu variants would be a visible-zero representation-family burn, not a plausible floor route.
- One remaining hypothesis: only revisit Trilu if a new task-local sidecar proves a different current-best parent whose actual output rule is triangular/ray-like; otherwise stay on the QLinearConv two-hop zero-adjacency formulation.

