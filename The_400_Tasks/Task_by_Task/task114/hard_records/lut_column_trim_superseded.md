# task114 lut-column-trim superseded

- Live bar used: 467, sha256 `50a6cc0667655779cdd52d453c635d5a77eee5a7a0d5689b349cf72ddec7825f`, source `python tools/ng_current_bar.py --task 114 --json` / manifest.
- Attempted route: prompt-specified `[4,25]` LUT column trim.
- Failed gate/check: pre-gate representation check, not a visible/truth failure. Refreshed `Team_Best_Onnx/task114.onnx` is `task114_factorized_edgepad_einsum_v5` with three `Einsum` nodes and no LUT initializer; task-local `task114_mode2_cost460` is also the factorized Einsum family. The `[4,25]` LUT route only describes stale older artifacts.
- Why dead for this bar: trimming LUT columns cannot improve the current effective graph because there are no LUT columns in the authoritative 467 live/team-best or pending 460 staged candidate.
- One remaining hypothesis: optimize or full-gate the inlined factorized-Einsum probes (`task114_inline_einsum` cost/profile 443) rather than reopening the old LUT formulation.
