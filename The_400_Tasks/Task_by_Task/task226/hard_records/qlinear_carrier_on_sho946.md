# task226 QLinear carrier route on Sho 946

- Live bar: 946, SHA-256 `43803c4f30de75afee846f09b939e0ea8b98369cc20975392d7c0316b12a60a0`, source `python tools/ng_current_bar.py --task 226 --json` / team manifest, owner Sho.
- Attempted route: prompt-directed QLinearConv/U8 carrier collapse on the current-best graph.
- Failed gate: pre-build feasibility, before visible. The current best is already a single F32 `Einsum` output with `memory=0 params=946`; there is no charged F32 carrier to narrow, and ORT requires QLinearConv input to be U8/INT8. A full-frame `Cast(input -> U8)` would add about 9000 B and erase any possible 24-byte rank-prune gain.
- Why dead: QLinearConv can only win here if a cheap U8/INT8 feed already exists or if it replaces the one-node output entirely; neither is present in the Sho 946 parent.
- One remaining hypothesis: a new one-node or output-node factorization with fewer than 922 params, especially recompressing the remaining `m` rank rows, could still beat the staged rank prune.
