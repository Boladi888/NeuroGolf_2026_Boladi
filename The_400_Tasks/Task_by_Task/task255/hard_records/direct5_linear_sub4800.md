# task255 direct5 linear sub-4800 hard record

- Live bar used: 13,983, SHA-256 `09708d7ac2de84fb8ec07e06f75561edabef1f99a219e9f191450301e8bb8cdf`, from `python tools/ng_current_bar.py --task 255 --json` authoritative source `manifest`.
- Attempted route: one-F32-plane direct 5x5 raw-input linear mask, `Conv(input)->Greater->Where`, built at `../prototypes/task255_direct5_linear_probe.onnx`.
- Probe price: static cost 4,762 = memory 4,500 + params 262. Profile was not run because visible failed.
- Exact failed gate: `python tools/ng_gate.py visible --task 255 --cand The_400_Tasks/Task_by_Task/task255/prototypes/task255_direct5_linear_probe.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the only static local Conv-threshold skeleton under 4,800 is at most 5x5. Official visible context audit found 797 conflicting 5x5 raw-input keys requiring both fill and no-fill labels; 7x7 still has 30 conflicts and already exceeds 4,800 by params; 9x9 is the first conflict-free raw local radius but is over the prompt ceiling before any extra guard.
- One remaining hypothesis: abandon single local mask classification and look for a nonlocal compressed carrier that keeps row/column interval state without full 30x30 masks, or accept a higher floor above 4,800.
