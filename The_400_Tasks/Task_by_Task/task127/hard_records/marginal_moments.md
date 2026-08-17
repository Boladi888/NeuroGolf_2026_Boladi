# task127 marginal moments route hard record

- live bar used: cost 330, sha256 `19c7f0baaa6b30f77aafe3cb70dc7bc96792b20ac19fc0f5841ba7bf1467fb0d`, source `python tools/ng_current_bar.py --task 127 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- attempted route: row/column/color marginal skeleton (`prototypes/probe_task127_marginals.py`) before building an output tail.
- failed gate: fast numpy rule probe, before visible/truth. Collision found on legal ARC-GEN color lists:
  - `colors_a=(1, 1, 2, 1, 2, 1)`
  - `colors_b=(1, 2, 1, 1, 1, 2)`
  - shared row/column/color marginal key `(2, 1, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)`
  - outputs differ, so marginals-only state cannot determine the panel colors.
- priced skeleton: `prototypes/task127_marginal_skeleton.onnx`, score `(macs=0, memory=80, params=190)`, cost 270, sha256 `d097ea71893c69e2dd80b43eb7960358d8231adfed852fa3eca70081c585f64f`.
- why dead: the 60-byte nominal gap from skeleton cost 270 to live 330 is not usable; any correct output tail has to restore the joint `(panel_row, panel_col, color)` state lost by the marginals. The current live 330 graph carries that joint state directly as `[1,5,2,3]` descriptors and uses the minimum 4x4 grouped ConvTranspose painter.
- one remaining hypothesis: only a new final-op family that keeps joint panel state inside the terminal output op, without 30-length row/column selector banks or a charged compact tensor, could beat 330. Direct final-Einsum selector banks appear to exceed the bar.
