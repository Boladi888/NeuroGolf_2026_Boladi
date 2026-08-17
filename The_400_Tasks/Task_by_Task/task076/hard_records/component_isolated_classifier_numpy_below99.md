# task076 component-isolated transform classifier

- Live bar used: 19,811, SHA `A23CDED50F406EC61DD27B094876FF28199C0EA578424FB11B48B366C0A71E02` from `python tools/ng_current_bar.py --task 76 --json`.
- Reviewer-locked route: start from decoded generator/reference and build a component-isolated transform classifier to remove the `3x7x7` candidate-board/body-probe ladder.
- Probe source: `The_400_Tasks/Task_by_Task/task076/prototypes/probe_task076_component_classifier.py`, using the task076 ARC-GEN corpus cache (`seed_offset=2026`, `cached=200000`, `n_raw=200349`) so the cases are generated ARC-GEN cases, not bundled visible pairs.
- Failed gate 1: `python The_400_Tasks/Task_by_Task/task076/prototypes/probe_task076_component_classifier.py --n 3000 --source cache --solver c2` -> `2964/3000 (98.800%)`; first failures `(16, 7)`, `(42, 2)`, `(168, 4)`, `(289, 2)`, `(399, 17)`.
- Failed gate 2: `python The_400_Tasks/Task_by_Task/task076/prototypes/probe_task076_component_classifier.py --n 3000 --source cache --solver object` -> `2967/3000 (98.900%)`; first failures `(16, 4)`, `(42, 2)`, `(168, 4)`, `(289, 2)`, `(399, 12)`.
- Why dead for this pass: the reviewer kill rule was explicit: stop immediately if the numpy classifier is `<99%` on 3000 fresh ARC-GEN cases. Both the c2-anchor simplification and the full marker-compatibility component reference are below 99%, so no ONNX skeleton was built or priced.
- One remaining hypothesis: inspect failure 16/42 to learn the missing generator quirk, likely involving transform ambiguity or source choice under symmetric body/marker layouts; only after a corrected numpy classifier clears >=99% should an ONNX skeleton be attempted.
