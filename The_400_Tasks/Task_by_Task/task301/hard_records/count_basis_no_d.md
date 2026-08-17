# task301 count-basis / no-d refactor

- Live/team-best bar: cost 430, SHA-256 `b01f14e6667d853308d75e797b8f3b702bb204524a57468e06e03aa72537555c`, source `python tools/ng_current_bar.py --task 301 --json` / manifest.
- Effective local staged rung to beat: cost/profile cost 428, SHA-256 `cf6a3738253302d9118cb8959c7b2fedfab9d703c541f2d5b465ba2d0e2a11d4`, path `staged_candidates/task301_seek64bool_cost428/task301.onnx`.
- Attempted route: replace the current channel features `[nz, t, t^2, black]`, where `t = count + h - w`, with count-derived features such as `[count*nz, (count*nz)^2, black]`, or delete the scalar `d = h - w` by refitting around `s = count + h`.
- Failed gate: pre-visible constrained support check over generator support `W=3..9`, `H=W..min(W+3,12)` found no positive-margin separator for the count-derived basis. The `s = count + h` no-d rewrite also fails the existing global basis: expanding the current logits needs a color-channel `W^2` constant term, but `[1, W, H, H*W]` cannot express `W^2` across all `H=W+d`, `d=0..3`.
- Why dead: adding an explicit `W^2` scalar/glob feature cancels the only 2-byte memory saving from deleting `d` and increases `u` params, so this route cannot beat the staged 428 rung in the current polynomial/Einsum formulation.
- One remaining hypothesis: a genuinely different spatial/global basis could still reduce `u` or `chan_feat`, but it must change the final basis, not only swap `t` for count-derived features.
