live bar: 1454 bytes, Team_Best_Onnx/task222.onnx sha256 796a53ac92654a261b32e46945196f32ba83d8e5a352768abb0f98762add76c2.

route attempted: exact-rank factor the fused graph's c_th_vl F32 [2,16,4] threshold tensor. Its row-mode flattening rank is 5, so a factorization `[16,5] x [2,4,5]` would price at 120 params versus 128, a theoretical -8 if every use stayed inside the terminal Einsum.

exact failed gate: construction gate failed before a candidate ONNX. The current fused equation already uses 50/52 legal ASCII Einsum labels; digit and unicode-label parser probes (`prototypes/_einsum_digit_label_probe.onnx`, `prototypes/_einsum_unicode_label_probe.onnx`) both failed strict shape inference with `Rank of input 0 (1) does not match the equation indices (0)`. Safe size-1/forced-label merges free only six labels, but the factorization needs eight independent rank labels for the eight c_th_vl occurrences.

why dead: sharing a rank label between independent c_th_vl occurrences couples the sums and changes the polynomial. Splitting the factor into a preparatory node materializes a [2,16,4] tensor (512 bytes) to save only 8 params, so it loses the 1454 bar.

one remaining hypothesis: a different fused formulation with fewer independent labels, not a local factor splice of Sho's equation, could reopen a sub-1454 route.
