# task074 termfold deletion probes

Live/team bar used: 262, profile 262, SHA-256 `4c312226397c7e96c15da8ca6dd6f03c5a58d76815f8ef7401e49221b0bd2773` from the workbook Tasks row and team manifest.

Attempted route: delete/fold apparently small operands from Vamshi's one-Einsum term-fold graph.

Exact failed gates:
- `no_channel_mask_cost252_probe`: score `(0, 0, 252)` cost 252; visible `[visible] 0/267 (0.00%) FAIL`.
- `no_switch_all_terms_cost258_probe`: score `(0, 0, 258)` cost 258; visible `[visible] 0/267 (0.00%) FAIL`.
- `same_term_no_switch_cost258_probe`: score `(0, 0, 258)` cost 258; visible `[visible] 0/267 (0.00%) FAIL`.
- `drop_high_bit_cost202_probe`: score `(0, 0, 202)` cost 202; visible `[visible] 0/267 (0.00%) FAIL`.

Why dead: color 9 must be killed explicitly, the off-diagonal term switch is load-bearing for the normal/transpose orbit union, and all four coordinate bits are needed to distinguish visible orbit classes. A visible miss is a certain hidden zero.

One remaining hypothesis: only a non-naive factorization can go lower; the current staged win is the shared-coordinate-basis rewrite at 212, and further progress would need to compress the channel mask/switch/H2 basis without creating a charged pre-output tensor.
