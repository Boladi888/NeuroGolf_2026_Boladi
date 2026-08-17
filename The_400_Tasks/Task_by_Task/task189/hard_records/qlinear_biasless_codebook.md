live bar: 521 bytes, sha256 fe2ea6bc155e42c00f364540ebb1100e650d45b2a2663dbc8dd97bc2a18937bd (ng_current_bar/team manifest, 2026-07-09T01:09:17Z).

attempted route: delete the 10-element final QLinearConv bias by finding a fixed-threshold 2-feature UINT8 codebook; first tested Sho's existing `[code & 15, code & 240]` points, then bounded alternate mask probes.

exact failed gate: feasibility probe, before ONNX build. Existing code points had no int8 weight solution for thresholds {1,2,4,8,16,32,64,128} over x_zero_point 0..255. Fixed alternate masks `{55/aa, aa/55, 33/cc, cc/33, 5a/a5, a5/5a, 3c/c3, c3/3c}` also found no 10-class separable set in the bounded radial pick.

why dead: the current 2-feature classifier needs per-channel affine offsets. Removing `q_b` requires every color code to be isolated by a shared-threshold line under a scalar zero point; the checked codebooks do not satisfy that, and adding normalized/extra feature planes costs more memory than the 10 params saved.

one remaining hypothesis: a solver-guided search over three or more feature masks could find a biasless classifier, but it must beat the added feature-plane cost; a 3-channel input already adds 36+ bytes, so the likely floor remains worse than the live 521.
