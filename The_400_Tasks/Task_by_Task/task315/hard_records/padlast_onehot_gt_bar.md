# task315 Pad-last one-hot lower-bound record

- Live bar used: 146, SHA-256 `013dbc694b331f0991b4e4f9a637a01c318e96a0f89289052e15bf526280ebb7`, source `python tools/ng_current_bar.py --task 315 --json` manifest result, owner Vamshi.
- Attempted route: prompt-suggested small-grid decoded `[1,10,9,9]` one-hot followed by free Pad to the fixed `[1,10,30,30]` output.
- Exact failed gate: pre-build lower-bound price. The one-hot tensor alone is `10*9*9 = 810` charged elements/bytes before any source read, mask, or params.
- Why dead: 810 is below the old 900-byte full-canvas Equal-tail, but it is strictly above the refreshed 146-byte live bar. Building/gating variants in this representation cannot produce a reviewer-ready win.
- Remaining hypothesis: a lower-rank terminal `Einsum` that preserves zero charged intermediates and reduces dense params below 146.
