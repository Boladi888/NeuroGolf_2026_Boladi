# task095 d=5 single-Einsum re-solve reviewer return

- live bar used: 208, manifest/team-best sha256 `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`, owner Vamshi, updated 2026-07-12T15:15:54Z.
- locked hypothesis tested: exact dense rank-5 re-solve of Vamshi's one-node free-output F32 Einsum class, equation `nkhw,ha,ra,wb,cb,ks,nprc,pq,oe,qes->norc`, with `V[30,5]`, `F[10,2]`, `C[2,2,2]`, no sparse initializer, and no Pad/Equal/Where/crop tail.
- solver/probe pass: `python The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_resolve.py`
- fresh rule check: decoded numpy rule matched ARC-GEN `3000/3000`.
- attempt cap honored: exactly 3 d=5 initializations/formulations, 1200 Adam steps each.
- best result: `eigh5_parent_fc` initialization at step 160 reached official `265/265` and fresh ARC-GEN `2999/3000`.
- other results: `eigh5_parent_fc_noisy` best official `265/265`, fresh `2998/3000`; `random_dense_vfc` best official `0/265`, fresh `0/3000`.
- stop reason: reviewer prefilter required official `265/265` and fresh `3000/3000`; best dense d=5 sign solution missed fresh by one row, so no ONNX was built and no `ng_gate` candidate gate was run.
- summary artifact: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_rank5_resolve_summary.json`.
- remaining hypothesis: a d=5 solution, if it exists, likely needs a more global constrained solve than this bounded three-initialization pass; this pass did not produce a reviewer-ready sub-208 candidate.
