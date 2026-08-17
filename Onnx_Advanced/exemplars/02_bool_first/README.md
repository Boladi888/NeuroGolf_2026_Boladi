# 02_bool_first — worked case studies

The 8-rule bool-first family. Each rule keeps mask values in BOOL (1 byte/elem) instead of F32 (4 bytes/elem) at some point in the graph.

Read every exemplar. The rules compose — a single task can benefit from multiple rules simultaneously.

## Files (by rule number, ascending)

- Rule 1 `task293_andfold.md` — Mul of two bool casts → And + single Cast.
- Rule 2 `task021_mul_mask_to_where.md` — Mul of bool cast × value → Where.
- Rules 3 + 6 `task196_or_cast_flip_where.md` — Add of two bool casts (mutex) → Cast(Or); binary-mask flip_where.
- Rule 5 `task273_drop_pad_cast_bool.md` — Drop Cast before Pad when downstream is bool-tolerant.
- Rule 6 alt `task319_opset_bump_pad_uint8.md` — opset bump + UINT8 Pad instead of BOOL (when downstream needs numeric dtype).
- Rule 7 `task299_bool_concat.md` — Concat of bool-eligible inputs stays BOOL. **Especially valuable for near-floor tasks.**
- Rule 8 `task376_input_to_bool.md` — input → BOOL whole-graph propagation. Cross-ref to `01_whole_graph_rewrites/`.
- Equal-fold (algebraic, bool-adjacent) `task233_equal_fold.md`.
- N-way Equal-fold opset-10 quirk `task368_shared_cast_equal_fold.md`.
- ORT `Where(bool,bool,bool)` workaround `task359_bool_first_where.md`.

## When applying multiple rules

Apply in this order for safety:
1. First, apply rule 8 (input → BOOL) if eligible. This propagates BOOL through the whole graph and many of the other rules become trivial.
2. Then apply local rules (1, 2, 3, 4) for remaining F32-mask operations.
3. Apply Concat-keeping (rule 7) for any Concat that's now bool-eligible.
4. Apply Pad-related rules (5, 6) at the output boundary.

If rule 8 doesn't apply (downstream needs Conv/MatMul/F32 arithmetic on input), still apply rules 1-7 as local optimizations.
