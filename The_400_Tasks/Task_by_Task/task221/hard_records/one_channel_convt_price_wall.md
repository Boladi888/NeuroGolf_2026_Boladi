# task221 one-channel ConvTranspose price wall

- Live bar used: 635 from `python tools/ng_current_bar.py --task 221 --json`; manifest SHA `f8029feb04334d79723b2e61c000e0a95c829941f0d956532c3ded628fea7fb2`, owner Vamshi, updated 2026-07-02T02:35:15Z.
- Attempted route: the prompt's 1-channel ConvTranspose probe, replacing the old two-channel macro-resolution ConvT input with `X[1,1,10,10]`.
- Failed gate: pre-visible price gate. Even the optimistic operands are `X[1,1,10,10]` F32 = 400 bytes plus dynamic motif weight `W[1,10,3,3]` F32 = 360 bytes, for 760 bytes before the mask/count/color scaffolding.
- Why dead: 760 is already worse than the refreshed live/team-best bar 635, so no representation in this family can be reviewer-ready for this batch unless it avoids either the dynamic 3x3 motif weight or the 10x10 macro image. A fixed `W` cannot encode arbitrary input color/motif from one scalar macro channel under a final linear ConvTranspose and NeuroGolf `>0` decode.
- One remaining hypothesis: keep the current free-output Einsum family and reduce dense selector/coefficient initializers; `prototypes/task221_g_term_coeff.onnx` is the active local shave candidate at profile cost 617 pending full gates.
