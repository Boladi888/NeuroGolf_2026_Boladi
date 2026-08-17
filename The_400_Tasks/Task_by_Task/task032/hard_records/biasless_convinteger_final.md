# task032 biasless_convinteger_final hard record

- Live bar used: cost 403, sha256 `e3ca4a62f12fe88ff9852a6159c24108f8438d030eff62ecf6a74267146754ce` from `python tools/ng_current_bar.py --task 32 --json` / team manifest.
- Attempted route: replace Sho's final two-feature `QLinearConv` code classifier with a biasless `ConvInteger`-style final margin, deleting QLinearConv scales/bias while still decoding by `output > 0`.
- Exact failed gate: pre-build separability gate in `The_400_Tasks/Task_by_Task/task032/prototypes/probe_compact_final_routes.py`.
  - Command: `python The_400_Tasks/Task_by_Task/task032/prototypes/probe_compact_final_routes.py`
  - Result line: `ConvInteger zero-point feasible: NO for z1,z2 in 0..255`
- Why dead: without per-output bias, the final classifier would be `A*(x-z1)+B*(x^2-z2)` for one shared UINT8 zero-point pair. No shared zero point can make channel 0 fire for codes `1..6` and each color channel fire for exactly one of `7..15` while keeping code `0` off.
- One remaining hypothesis: a biasless integer final only revives if another already-needed feature supplies a free per-channel/per-row bias; adding such a feature as a normal carrier costs more than the deleted 10-element bias.
