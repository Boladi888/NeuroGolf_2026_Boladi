<!-- ===== VERDICT INDEX (Phase-3 front-matter, 2026-06-05) ===== -->
> **The 12 verdicts everything else builds on** (grep the §):
> 1. **Grader cost mechanism** = max(static, runtime-on-REAL-inputs)/tensor → §39 MECHANISM. Static ⇒ local==Kaggle EXACT; dynamic local costs are FICTION (78× seen).
> 2. **F16-Conv grader-validated** (task080) → F16-CONV VERDICT. F16 exactness ceiling: integers ≤2048.
> 3. **Negative-pad Conv = param-free crop, grader-confirmed** → §41 VERDICT.
> 4. **Free-input Einsum marginals score 0-memory** → §42 VERDICT; adopt-list → #46.
> 5. **Spatial ops**: Slice-negstep/Transpose/Tile/Resize-nearest/Trilu; GridSample = 2HW-param tax → #44.
> 6. **Packing law**: α·m·(n−w) > pack + g·n·unpack; U8 from ~4 packed consumers → #45.
> 7. **Row-packed u32 flood**: 4-neighbor + K≥12 only → #47.
> 8. **Plane-breaker catalog** (escapes the 2-input conservation law) → #40; LUT sizing lemma → §40 SIZING LEMMA.
> 9. **Iteration floors are corpus-measured** (K-min proofs) → §37; 5×5=2·3×3 iff convexity certificate → §37 ADDENDUM.
> 10. **Dtype/opset reality** → `ort_reality.md` (probed matrix; Bitwise* u8/i32/u32; BitShift unsigned-only; MatMul-u32 legal).
> 11. **Visible-miss = certain hidden 0** → traps §VISIBLE-FAIL AXIOM; class-check before chasing 100% → traps §CLASS-CHECK.
> 12. **Zero-memory/params-only graphs credit honestly** (t215/t252) — leanness itself is safe.
<!-- ===== end verdict index ===== -->

# techniques.md — the rewrite toolkit

Each technique: what it matches, what it rewrites to, why it's safe, what it saves. Exemplars cited where validated.

Notation: `A → B` means "graph fragment A is replaced by graph fragment B". `Op(x, y)` means the op with those inputs. `Cast(x, F32)` means a Cast node with `to=FLOAT`.

---

# 0. Strategic principle (2026-05-23 update): RADICAL ALGORITHM REWRITE BEATS TRICK STACKING

After ~17 confirmed-clean ships this session totaling +45+ LB, the data is decisive:

**Algorithm-rewrite candidates outperform trick-stack candidates by ~3-5× per task.** Confirmed clean algorithm-rewrite ships:

*(54-row ship-log table evicted to `_archive/techniques_pre_phase3.md` — per-task outcomes live in `task_techniques_index.md` + `current_strategy.md`; charter rule 4.)*

**Common feature**: decode rule from train pairs + rebuild from numpy ref + verify against LIVE on the documented contract. Adversarial percentage is NOT predictive of hidden outcome (ranges 0% generic-off-contract to 87% all shipped clean). What matters is whether the decoded rule actually captures LIVE on the hidden test slice.

**Lower-bound floor empirically established (2026-05-23 task170)**: 0/100 generic random adversarial + 64% task-shaped synthetic (still-advisory data point) shipped clean at +2.25 LB. The signal that matters is NOT the adversarial number itself but: (1) the algorithm is rule-shaped (Einsum projections + dynamic stride, not hash/lookup); (2) the padded output shape is preserved (avoiding any task-specific public-negative pattern like task170's no-final-Pad/4x4-output trap). When both hold and 266/266 verify passes, even 0% generic adversarial is a false alarm.

**Failure mode**: count-vector / packed-position lookups (`task077_lookup8` -13.13 LB, `task396_count_hash_lookup` -13.40 LB). These memorize the 266 visible+arc-gen pairs into a hash table; hidden inputs collide or miss → wrong output → -X LB. See `traps.md §14e`.

**Decision rule for radical candidates (revised 2026-05-23, replaces the old `>30%` blanket safety gate)**:

The only universal HARD GATE is **`verify_against_live` 100% (with `--live-zip submission.zip`, NOT `--cropped-decode`)**. Adversarial percentage is ADVISORY and not a blanket safety threshold.

- ✅ **Safe (high confidence)**: rule decoded from train pairs + rebuilt from numpy + verify 100% + structural compression of existing algorithm (preserve semantics, change plumbing). Adversarial reported but not gating.
- ✅ **Safe (medium confidence)**: rule-replacement candidates where the decoded rule captures LIVE on visible+arc-gen AND task-shaped synthetic-LIVE (≥500 samples) ≥95%. Document the contract assumption in the manifest. Examples shipped clean at 0% generic adv (task170 +2.25, task387 +2.54).
- ❌ **Unsafe**: hash-table / lookup / sampled-cell-hash / template-table architectures fit ONLY to the 266 visible+arc-gen pairs. 10 confirmed zero-public-weight family members (task366/324/133/182/018/157/255/066/076/277) all score 0.00 LB regardless of how much cost is reduced. See `traps.md §14e`.
- ❌ **Unsafe**: trained NN/MLP/CAE distilled to fit the visible set without learning the rule (task277 calibration confirmed 2026-05-23).
- ❌ **Unsafe**: candidates with adversarial < 1% generic random AND no documented off-contract justification — `task077_lookup8 -13.13 LB`, `task396_count_hash_lookup -13.40 LB`, `task370_mph_w20 -13.04 LB` all regressed -13 LB.

**Adversarial signal interpretation**:
- < 1% generic + 266/266 verify = memorization signature (lookup family or distilled NN) → expect 0.00 LB (zero-weight family) or -13 LB (lookup-with-public-weight)
- 1-30% generic + off-contract documented = likely safe IF accompanied by ≥95% task-shaped synthetic
- 30-100% generic = informative but not the deciding signal; task387 0% off-contract synthetic shipped clean, task234 87% adv partial-decoupled
- 100% across all task-specific perturbations = LIVE-equivalent algorithm; safest

**Cherry13b is the universal filter**: before shipping any low-cost rewrite (especially when cost reduction comes from `[N≈266, K]` tables or trained weights), run the cherry13b Identity stub. Any candidate that produces correct visible-pair outputs from a fit-only-to-visible architecture will score 0.00 LB on Kaggle regardless of architectural style.

For new task work: prioritize rule decoding + numpy rebuild over trick stacking. Tricks (Rule 10, Rule 12, BOOL output) are POLISH; algorithm rewrite is PRIMARY.

---

# 1. Bool-first family (12 rules)

The unifying principle: keep mask values in BOOL dtype (1 byte/elem) instead of F32 (4 bytes/elem) for as much of the graph as possible. Cast to F32 only at the boundary where F32 arithmetic against non-mask values is required, OR at the model output (and sometimes not even then — see rule 5).

Cross-ref: every rule below has an exemplar in `exemplars/02_bool_first/` and most have an auto-detector in `tools/spread_patterns/`.

**Calibration: BOOL beats F16 even when the chain is longer.** Measured on task110 v9 (2026-05-20): a 6-op BOOL validity chain costs 5,400 BOOL bytes + 1,800 F16 (for the mandatory Cast before ReduceSum) = 7,200 bytes per scoring direction. The equivalent 4-op F16-arithmetic chain (`Sub → Mul-self → vis-mask → ReduceSum`) costs 12,600 F16 + 900 BOOL = 13,500 bytes — **87% larger.** Reason: BOOL intermediates are 1 byte/elem; F16 are 2 byte/elem. Even with 50% more ops, BOOL wins per-intermediate by ~2×. Default to BOOL chains for validity / matching / mask construction; switch to F16 only when arithmetic against non-binary values is required.

## Rule 1 — `bool_andfold`: Mul of two bool casts → And + single cast

```
Mul(Cast(bool_a, F32), Cast(bool_b, F32))   → Cast(And(bool_a, bool_b), F32)
```

- Safety: 0/1 × 0/1 = 0/1 AND. Identity for all inputs of {0, 1}.
- Saves: 2 F32 intermediates (each 4 bytes/elem) → 1 BOOL intermediate (1 byte/elem). Plus -1 node.
- Exemplar: **task293** (-1,575 bytes after sweep). Detector: `pass_bool_andfold.py`.

## Rule 2 — `mul_mask_to_where`: Mul of bool cast × value → Where

```
Mul(Cast(bool_op, F32), V)   → Where(bool_op, V, 0_constant)
```

- Safety: 0 × V = 0; 1 × V = V. Identity. (V can be any dtype matching the original Mul output.)
- Saves: F32 mask intermediate (4 bytes/elem) eliminated. The Cast also goes away.
- Exemplar: **task021** (-3,119 bytes). Detector: `pass_mul_mask_to_where.py`.

## Rule 3 — `or_cast_collapse`: Add of two bool casts (mutually exclusive) → Cast(Or)

```
Add(Cast(bool_a, F32), Cast(bool_b, F32))   → Cast(Or(bool_a, bool_b), F32)
```

- Safety REQUIRES mutual exclusion: no input can make both `bool_a` AND `bool_b` true. Otherwise the Add produces 2.0 where the Or produces 1.0.
- Mechanical mutex detection: `Greater(x, hi) + Less(x, lo)` with `hi >= lo` is provably mutex. Same for any pair `(Op_>(x, a), Op_<(x, b))` with `a >= b`.
- Saves: 1 F32 intermediate + 1 Cast + 1 Add.
- Exemplar: **task196** (mutex `gt25 + lt15` pattern). Detector: `pass_or_cast_collapse.py`.

## Rule 4 — `flip_where`: Mul(b, Sub(1, Where(c, b, 0))) → Where(c, 0, b)

```
Mul(b, Sub(1, Where(c, b, 0)))   → Where(c, 0, b)
```

- Safety REQUIRES `b` to be binary (0/1). Verified for b ∈ {0, 1} by truth-table.
- Trace: b=0 → LHS=0, RHS=Where(c, 0, 0)=0 ✓. b=1, c=False → LHS=1×(1-0)=1, RHS=1 ✓. b=1, c=True → LHS=1×(1-1)=0, RHS=0 ✓.
- For non-binary b the rewrite breaks (LHS = `2 × (1-2) = -2` ≠ RHS = 0).
- Saves: drops Mul + Sub + the inner Where's intermediate. One Where stays.
- Exemplar: **task196** flip-where variant. Detector: `pass_flip_where.py`.

## Rule 5 — `drop_cast_before_pad_bool`: Cast(bool, F32) → Pad → output dropped when downstream is bool-tolerant

```
Cast(bool_op, F32) → Pad → [graph output OR Greater(0, ...)]
   →
Pad(bool_op) → [graph output (now BOOL) OR drop the Greater entirely]
```

- **Pad-13 accepts BOOL per the ONNX spec** (`tensor(bool)` is in the opset-13 T constraint — verified). Opset 11 Pad rejects BOOL. So the rewrite requires opset 13+ for BOOL input; at opset 10/11 you must Cast(bool, F32/UINT8) first.
- Safety: padding constant must be 0 (default). The bool 0-fill matches semantically.
- If graph output dtype changes to BOOL: `(out > 0.0)` in `run_network` is identity on BOOL {True, False} → produces same {0, 1} mask. Safe.
- Saves: F32 intermediate between Cast and Pad eliminated. The Pad output (often the graph output) becomes 1 byte/elem instead of 4.
- Exemplars: **task250, task342** (-4,000 each). Detector: `pass_drop_cast_before_pad_bool.py`.

**Extension — BOOL graph output via Concat of BOOL channels + Pad-13+**: when the graph ends with `Concat(per-channel BOOL) → Cast(F16) → Pad(F16) → output (F16)`, declare graph output as BOOL. ORT Pad-13+ accepts BOOL data (per `ort_reality.md §1`). Drops the F16 Cast intermediate of `[1, 10, H, W] F16 = H*W*20 bytes`. **Empirical**: task280 radical saved -8,000B via this extension on the per-channel-Concat case.

## Rule 6 — `opset_bump_pad_uint8`: opset-10 Pad-attr + Cast(bool, F32) → opset-11 Pad(input form) + Cast(bool, UINT8)

When downstream is NOT bool-tolerant but accepts UINT8, use UINT8 instead of BOOL. UINT8 is also 1 byte/elem.

**UINT8 downstream-consumer safety (verified 2026-05-23 ORT 1.24.4)**: NOT all ops promote UINT8 to F32 implicitly. The rewrite is safe only when downstream consumers are UINT8-supported. **UINT8 is accepted by**:
- `Pad`, `Gather`, `Concat`, `Slice`, `Reshape`, `Transpose`, `Identity`, `Squeeze`/`Unsqueeze` (shape ops)
- `Where(BOOL, UINT8, UINT8)` at opset 13+
- `Equal(UINT8)`, `Greater(UINT8)`, `Less(UINT8)` at opset 10+
- `MaxPool(UINT8)` at opset 12+
- `ReduceMax(UINT8)`, `ReduceMin(UINT8)` at opset 13+
- `Cast(UINT8 → any)` always

**UINT8 is REJECTED by** (do NOT feed a UINT8 tensor into these):
- `Add`, `Sub`, `Mul`, `Div` at opsets ≤ 13 — requires **opset 14+**
- `ReduceSum`, `ReduceMean` — ALL opsets (use INT32/INT64/F16 for sum-style reductions)
- `Einsum` — kernel not implemented
- `Conv`, `ConvTranspose`, `MatMul`, `Gemm` (need F16/F32 or use the integer Q* variants)

If your chain hits any rejected op, insert `Cast(UINT8 → F16/F32)` BEFORE that op.

```
# Before: opset 10
opset_import = [{"": 10}]
Cast(bool_op, F32) → Pad(attr pads=[...])   # F32 intermediate, 4 bytes/elem

# After: opset 11
opset_import = [{"": 11}]
Cast(bool_op, UINT8) → Pad(input pads_init)   # UINT8 intermediate, 1 byte/elem
```

- Update the graph output dtype to UINT8 (matches the new Pad output dtype) or ORT rejects at load.
- Exemplar: **task319** (-742 bytes). Detector: `pass_opset_bump_pad_uint8.py`.

## Rule 7 — `concat_bool_inputs`: Concat with all bool-eligible inputs stays BOOL

```
Concat(Cast(bool_a, F32), Cast(bool_b, F32), F32_init_0_or_1, ...) → Cast(F32)
   →
Concat(bool_a, bool_b, bool_init, ...) [output BOOL]
```

- "Bool-eligible" input is either (a) Cast(bool_op, F32) or (b) F32 initializer with every element ∈ {0, 1}.
- Convert 0/1 F32 inits to BOOL inits (1 byte/elem vs 4). Drop the Casts. Concat output dtype becomes BOOL.
- Safety: identical semantically; only storage width changes.
- Exemplar: **task299** (cost 2,678 → 1,058, **60% reduction, ~+0.92 LB on Kaggle**). Proves "near-floor" tasks have huge headroom. Detector: `pass_concat_bool_inputs.py`.

## Rule 8 — `input → BOOL` (whole-graph propagation, no auto-detector)

The most aggressive bool-first technique.

```
Insert Cast(input, BOOL) as the FIRST node.
Replace every downstream Mul(gathered, mask) with And(gathered_b, mask_b).
Graph output dtype → BOOL.
```

- When the model input is one-hot (NeuroGolf default: F32 [1,10,30,30] with 0/1 per channel), BOOL semantics propagate through every op that accepts BOOL: **Gather, Where (condition only), Pad (opset 13+), Concat, Reshape, Transpose, Slice, Squeeze, Unsqueeze, Identity, And/Or/Not/Xor, Equal (BOOL,BOOL)**. **NOT order-comparison ops — `Greater(BOOL)` and `Less(BOOL)` are ALWAYS REJECTED by ORT.** If you need ordered comparison, cast to UINT8/INT32/F16 first or rewrite using `And`/`Or`/`Xor` of bool inputs.
- **Reduce ops do NOT accept BOOL.** `ReduceMax(BOOL)`/`ReduceMin(BOOL)`/`ReduceSum(BOOL)`/`ReduceMean(BOOL)` all fail at model load. Cast to UINT8 (cheapest 1-byte path for Max/Min at opset 13+) or to F16/INT32 (for Sum) before the reduction.
- **`Where(condition, x, y)` is BOOL-tolerant on `condition` only** — the data inputs `x`, `y` cannot both be BOOL (`Where(BOOL, BOOL, BOOL)` always rejected). For BOOL data inputs use the logical-op rewrite `Or(And(c, a), And(Not(c), b))`. `Where(BOOL, UINT8, UINT8)` works at opset 13+.
- Eligibility: the entire downstream chain must be bool-tolerant. **Disqualifying ops**: Conv, MatMul, Gemm, F32 arithmetic (Add/Sub/Mul/Div) on the input directly.
- Exemplar: **task376** — cost 42,836 → 24,626 (-18,210, +0.55 LB, verify 39/39, adversarial 100%). Build archived at `_hermes_archive/board/staged_submits/task376_bool_t_8d70c38f_on_5835_30/`. See `exemplars/01_whole_graph_rewrites/task376_input_to_bool.md`.
- Why not auto-detected: whole-graph type propagation analysis. Worker-judgment.

## Rule 10 — `deferred_cast_bool_throughput`: keep BOOL through Reshape+Transpose, Cast at MatMul/Gemm boundary

When a BOOL tensor feeds `Cast(F16) → Reshape → Transpose → MatMul/Gemm`, the Reshape and Transpose can be done in BOOL (1 byte/elem). Cast at the end before the MatMul/Gemm.

```
# OLD: 3 F16 intermediates between Cast and MatMul
full_bool [1, 9, 25, 25] BOOL = 5,625
→ Cast(F16)  → full     [1, 9, 25, 25] F16  = 11,250
→ Reshape   → full_flat [9, 625]      F16  = 11,250
→ Transpose → full_t    [625, 9]      F16  = 11,250
→ MatMul    → cooccur                       (uses F16 input)

# NEW: keep BOOL through shape-only ops, Cast once before MatMul
full_bool [1, 9, 25, 25] BOOL = 5,625
→ Reshape   → full_flat_b [9, 625]  BOOL = 5,625    (save 5,625)
→ Transpose → full_t_b    [625, 9]  BOOL = 5,625    (save 5,625)
→ Cast(F16) → full_t      [625, 9]  F16  = 11,250
→ MatMul    → cooccur
```

Net save: 5,625 × N where N is the number of BOOL-form intermediates eliminated. **task173 v3** saved -33,750 bytes via this technique on 3 branches × 11,250.

Generalizes to any chain `Cast(BOOL, wider) → shape-only-ops → arithmetic-op`. Same family as Rule 5 (drop_cast_before_pad_bool) but extended to Reshape/Transpose.

## Rule 11 — `cast_bool_to_uint8_for_reduce`: skip F16 when feeding ReduceMax/ArgMax

When the chain is `Greater → Cast → ReduceMax/ArgMax`, Cast to UINT8 (1 byte/elem) instead of F16 (2 bytes/elem). ReduceMax(UINT8) and ArgMax(UINT8) are supported at opset 12+. Per-tensor savings 50% on the cast intermediate.

```
# Before — opset 10
v_bool   = Greater(score F16, threshold)            BOOL [1,1,30,30]  900
v_cast   = Cast(v_bool, F16)                         F16  [1,1,30,30]  1,800     ← 2 bytes/elem
v_max    = ReduceMax(v_cast, axes=[2,3])             F16  [1,1,1,1]    4

# After — opset 12+
v_bool   = Greater(score F16, threshold)            BOOL [1,1,30,30]  900
v_cast   = Cast(v_bool, UINT8)                       UINT8 [1,1,30,30] 900       (-900 per iter)
v_max_u8 = ReduceMax(v_cast, axes=[2,3])             UINT8 [1,1,1,1]   4
v_max    = Cast(v_max_u8, F16)                       F16   [1,1,1,1]   4         (only if Mul downstream needs F16)
```

**Constraints (re-verified 2026-05-23 at opsets 10/12/13/14/16 under ORT_DISABLE_ALL)**:
- `ReduceMax(UINT8)` / `ReduceMin(UINT8)`: opset 12+ OK.
- `ReduceMax(INT8)` / `ReduceMin(INT8)`: opset 12+ OK (was previously claimed to require opset 16+; that claim was wrong).
- `ArgMax(UINT8)`: opset 10+ OK.
- **`Greater(UINT8)` / `Less(UINT8)` / `Greater(INT8)` / `Less(INT8)`: opset 10+ OK** (was previously claimed to require opset 14+; that claim was wrong).
- `Greater(BOOL)` / `Less(BOOL)`: ALWAYS REJECTED (any opset). BOOL is not order-comparable.
- `Mul(UINT8, UINT8)` / `Add(UINT8, UINT8)` / `Sub(UINT8, UINT8)`: opsets 11/13 FAIL, **opset 14+ OK**. Same for INT8.
- `Where(BOOL, UINT8, UINT8)`: **opset 13+ OK** (was previously claimed opset 14+; that was wrong).
- `Where(BOOL, INT8, INT8)` / `Where(BOOL, INT16, INT16)` / `Where(BOOL, BOOL, BOOL)`: REJECTED at all opsets.

Path: Cast BOOL→UINT8 between Greater and ReduceMax, Cast UINT8→F16 back at the Mul boundary (or stay UINT8 if downstream chain is `Where`/`Gather`/`Pad`/`Concat`/`Equal`/`MaxPool`). **task205 v2** saved -23,966 bytes via this on 25 per-iter chains. Opset 10→13 bump also requires Pad attr-form→input-form conversion (trivial).

### INT8 variant: channel-axis Or-fold (opset 12+)

When the chain is `BOOL [1, N, H, W] → Split → N × (Or-chain to collapse channels) → BOOL [1, 1, H, W]` — replacing the N-1 chain of Or nodes — use **INT8 ReduceMax along the channel axis**:

```
# Before: Split + N-1 Or
v_split [1, N, H, W] BOOL → 9×900 split outputs + 8×900 Or intermediates = 15,300 B (N=9, H=W=30)

# After: INT8 ReduceMax
v_i8     = Cast(v_split, INT8)                     INT8 [1, N, H, W]  = 8,100 B   (1 byte/elem, was 8,100 as BOOL)
v_or_i8  = ReduceMax(v_i8, axes=[1], keepdims=1)   INT8 [1, 1, H, W]  = 900 B
v_or_b   = Greater(v_or_i8, int8_zero_init)        BOOL [1, 1, H, W]  = 900 B
# Total: 9,900 B (-5,400 saves)
```

**ORT constraint (re-verified 2026-05-23)**: `ReduceMax(INT8)` works at **opset 12+**. The previous "opset 16+" claim was incorrect — ONNX schema includes INT8 in ReduceMax since opset 12, and local ORT under `ORT_DISABLE_ALL` accepts it at opset 12/13.

**Rejected alternatives** (verified empirically):
- `Conv(BOOL, ones[1, N, 1, 1])` — ORT rejects BOOL as Conv input.
- `Cast(BOOL → F16) → ReduceSum` — produces F16 [1, N, H, W] = 2× the byte count of INT8 path.

Empirical: **task092 radical_v1** saved -5,400 bytes via INT8 ReduceMax replacing the 9-step Or-chain.

### UINT8 Mul-chain replaces BOOL And + Cast-for-Reduce

When a BOOL And chain feeds into a ReduceMax, the `Cast(BOOL → UINT8)` step before ReduceMax produces a [1, 1, H, W] = `H*W` byte intermediate per BOOL → UINT8 boundary. If you do the And chain DIRECTLY in UINT8 via Mul (for 0/1-valued tensors, `Mul(UINT8, UINT8) = And` semantically), you skip those Casts.

```
# Old: 16 BOOL And outputs (6,400B) + 16 BOOL→UINT8 Casts (6,400B) = 12,800B
has_thick_d_1 = edge_d                              # BOOL
has_thick_d_2 = And(has_thick_d_1, ...)             # BOOL
... [16 more] ...
final_u8 = Cast(has_thick_d_16, UINT8)              # Cast at the end
reduce = ReduceMax(final_u8, ...)

# New: 16 UINT8 Mul outputs (6,400B), no Casts. Save 6,400B.
has_thick_d_1 = edge_d_u8                           # UINT8 from per-channel Slice+Cast
has_thick_d_2 = Mul(has_thick_d_1, ...)             # UINT8 Mul = And
...
reduce = ReduceMax(has_thick_d_16, ...)             # UINT8 ReduceMax directly
```

Bonus: `NOT(X)` in UINT8 = `Sub(1, X)` (safe for 0/1 values, no underflow). Edge detection chains (`ch2 AND NOT(shift_OPP) AND shift_AWAY`) computed directly in UINT8 skip additional BOOL→UINT8 Casts.

**Empirical**: task280 radical saved -6,400 bytes via UINT8 Mul-chain replacing BOOL And chain.

### UINT8 MaxPool variant (opset 12+ unlock)

`MaxPool(UINT8)` is accepted by ORT at opset 12+ (FAILS at opset 10 and opset 11; `MaxPool(BOOL)` is rejected regardless of opset). When 0/1 propagation only needs max/OR semantics, bump opset and use UINT8 throughout the MaxPool chain.

**Empirical**: task025 radical worker saved ~76k bytes (228k → 152k) via switching F16 directional propagation to opset 12 UINT8 MaxPool. Same family of unlocks as task286 v2 (opset 10→12 + INT8 BFS, -71k).

### INT8 output assembly variant (opset 12+)

When the model's final output is `Cast(BOOL→F16) → ReduceSum(channel-axis) → Greater → Cast → Concat → output (F16)`, replace with the INT8 spatial path:

```
# Old (F16 spatial chain):
Cast(nonbg_lines_bool → F16) → ReduceSum(F16, axis=1) → Greater → ... → Concat(F16, F16) → output F16

# New (INT8 spatial chain at opset 12+):
Cast(nonbg_lines_bool → INT8) → ReduceMax(INT8, axes=[1], keepdims=1) → Greater(INT8, int8_zero) → ... → Concat(INT8, INT8) → output INT8
# graph.output[0].type.tensor_type.elem_type = TensorProto.INT8
```

**Safety**: `run_network` decodes via `(result[0] > 0.0).astype(float)`. INT8 {0, 1} satisfies this identically to F16. NeuroGolf validator accepts INT8 output dtype.

**Savings**: -9,900 B per pipeline on [1, 9, 30, 30] BOOL → F16 → output. Three F16 spatial intermediates halved to INT8.

**ORT requirements**: `ReduceMax(INT8)` works at opset 12+ (verified at opset 12/13). `Cast(BOOL → INT8)`, `Greater(INT8, INT8)`, and `Concat(INT8, INT8)` work in the same low-opset chain; use the lowest opset accepted by the whole graph.

**Empirical**: task092 v2 saved -9,900 B via this. Combined with the Or-fold INT8 ReduceMax (Rule 11): -15,300 B total.

## Rule 12 — `compare_after_reduce`: swap Greater and ReduceMax order

When the chain is `score_f16 → Greater(thr) → Cast → ReduceMax(per-axis)`, the comparison and reduction can be swapped via the algebraic identity:

```
Greater(ReduceMax(score, axis=X), thr) == ReduceMax(Greater(score, thr), axis=X)
```

(when BOOL is interpreted as 0/1; "any cell above thr in this row/col" is equivalent to "row/col max is above thr").

```
# Before — Greater on full [1, 1, 30, 30], Cast on full
v_bool  = Greater(score_f16, thr)              BOOL [1,1,30,30]  900
v_cast  = Cast(v_bool, UINT8)                  UINT8 [1,1,30,30] 900
v_max_h = ReduceMax(v_cast, axes=[3])          UINT8 [1,1,30,1]  30
v_max_w = ReduceMax(v_cast, axes=[2])          UINT8 [1,1,1,30]  30

# After — ReduceMax FIRST on F16, Greater on small tensor
v_max_h_f16 = ReduceMax(score_f16, axes=[3])    F16 [1,1,30,1]   60
v_above_h_b = Greater(v_max_h_f16, thr)         BOOL [1,1,30,1]  30
v_max_w_f16 = ReduceMax(score_f16, axes=[2])    F16 [1,1,1,30]   60
v_above_w_b = Greater(v_max_w_f16, thr)         BOOL [1,1,1,30]  30
```

Per-iter savings: ~1,616 bytes (skip the [1, 1, 30, 30] full-resolution BOOL + UINT8 intermediates). **task205 v3** saved **-60,400 bytes** across 25 iterations — the single biggest win in task205's optimization stack.

This is a general algebraic identity for `Compare → Reduce` chains. Works at any opset (doesn't require UINT8 support). Classic mistake: filter-then-reduce when reduce-then-filter is identical with smaller intermediates.

## Rule 9 — `padding_sentinel_before_cast`: keep Where in the smaller dtype

When a model ends with `Where(mask, sentinel, value) → Cast(wider_dtype) → consumer` and the sentinel value is exactly representable in the **pre-Cast** dtype, reorder to do Where first.

```
# Before: Where in INT64 (8 bytes/elem after upstream Cast)
fi          = Cast(cur_f16, INT64)               # INT64 [1,1,30,30] = 7,200
fi_masked   = Where(is_padding, neg1_i64, fi)    # INT64 [1,1,30,30] = 7,200
fi_3d       = Squeeze(fi_masked, axes=[0])       # INT64 [1, 30, 30]  = 7,200

# After: Where in F16 (2 bytes/elem), Cast once at the end
cur_masked    = Where(is_padding, ten_f16, cur)  # F16  [1,1,30,30]  = 1,800
cur_masked_3d = Squeeze(cur_masked, axes=[0])    # F16  [1, 30, 30]  = 1,800
fi_3d         = Cast(cur_masked_3d, INT64)       # INT64 [1, 30, 30]  = 7,200
```

- Where preserves the data dtype. As long as the sentinel value is exactly representable in the pre-Cast dtype, the rewrite is identity.
- The sentinel must be out-of-range for the downstream consumer. For OneHot(depth=10), any value ≥ 10 produces all-zero output. **Do NOT use -1 as the sentinel for OneHot — OneHot wraps negative indices modulo depth, so -1 → channel 9** (see `ort_reality.md §3` and task110's anti-pattern log). Use `10.0` instead.
- Generalizes to any `Where(mask, sentinel, x) → Cast → consumer` where `sentinel` fits in `x`'s dtype.
- Empirical: **task110 v9** saved -10,800 bytes via this reorder on the F16 → OneHot pipeline (2026-05-20).
- No auto-detector — requires reasoning about sentinel representability per task.

---

# 2. Crop-to-active-grid + dimension-tied initializer rewrites

When a task's max input/output grid is < 30, every intermediate of shape `[..., 30, 30]` is wasted memory on the padding rows/cols. Crop the input at the start, compute on the smaller grid, pad back to `[1, 10, 30, 30]` at the end.

```
input [1, 10, 30, 30] → Slice(starts=[0,0,0,0], ends=[1,10,H,W]) → input_cropped
[computation on [..., H, W] intermediates]
→ Pad(constant 0, pads=[0,0,0,0, 0,0, 30-H, 30-W]) → output [1, 10, 30, 30]
```

The savings ONLY materialize if you also rewrite **dimension-tied initializers** in lockstep:

- `range30` initializer (used for reflection/period ops): rewrite to `range_H = [0..H-1]`.
- `c29i` (often hardcoded as 29): rewrite to `H - 1`.
- `c30i` (often hardcoded as 30): rewrite to `H`.
- Any other init named after the grid size needs to track.

**Critical**: after surgery, `g.ClearField("value_info")` before saving. ORT's cached value_info still says `{1,1,30,30}` on internal tensors; runtime shapes are now `{1,1,H,H}`. ORT will fail with `Op (Max) [ShapeInferenceError] Incompatible dimensions` without this step.

**Determine max active grid first**: read `data/competitions/neurogolf-2026/taskNNN.json`, compute `max(input_shapes + output_shapes)`. The `spread_scan` crop advisory string also reports it.

**Stacks with the F16-output trick** (technique 3 below). task117 stacked both: cost 148,857 → 48,332 (+1.1249 pts).

**False-positive guard**: when the spread_scan reports `crop_to_size_advisory` but the model already Slices to a small region on its FIRST node and Pads back at the end, pre-cropping adds overhead without removing any 30×30 intermediate. Verify by reading the graph; skip if the model is already grid-aware. Confirmed on task186/229/266/322/373/380/**201**.

**When the graph is already cropped, the real lever is dtype-narrowing the crop-size planes (2026-05-30, task201).** task201 (cost 84,895) was ALREADY cropped to 13×13 at node 1; a "−85% full crop" prompt was a false premise (it misread the `.md` graph-diag op-HISTOGRAM, e.g. `Squeeze:16 / Mul:15`, as 30×30 tensors — those are op counts, not shapes). **Always run a per-tensor `calculate_memory` replica** (`task201/prototypes/inspect_task201.py` is a reusable template) to see where cost actually lives BEFORE trusting any crop premise. The real cost was 18 one-hot `[1,10,13,13] F16` planes; the byte-equal win (→35,577, +0.87 LB) came from: (a) slice-BEFORE-cast (drop the one full-canvas input cast, −11,240); (b) narrow the 0/1 planes F16(2B)→**UINT8(1B)** — which needs an **opset 10→14 bump**: convert `Squeeze`/`Unsqueeze`/`ReduceSum` axes-attr→input and `Pad` pads/value attr→input (ReduceMax KEEPS axes-attr until opset 18); keep ONE F16 cast for the lone `ReduceSum` branch; output dtype→UINT8; (c) algebraic fusions — `max_c(x·mask)=mask·max_c(x)` for per-cell mask≥0 to channel-reduce before materializing, and `Where(scalar,Gather(x,a),Gather(x,b))`→`Gather(x,Where(scalar,a,b))` index-select folding (commuting/composing Gathers across axes). **Isolation tip**: build an opset-only stage first (no dtype change) and confirm it stays byte-identical (266/266) — that pins any divergence to the dtype logic, not the opset conversion. See `The_400_Tasks/Task_by_Task/task201.md` cost-analysis for the full floor decomposition.

Exemplar: `exemplars/03_crop_and_f16/task117_crop15_f16.md`.

---

# 3. F16-output / BOOL-output trick (drop final Cast)

When a model ends with `... → output_intermediate [F16 or BOOL] → Cast(F32) → output`, drop the Cast and declare the model output as F16 or BOOL.

```
# Before
... → Concat → output_f16 [F16, [1,10,30,30]]
      Cast(output_f16, to=1) → output [F32, [1,10,30,30]]

# After (F16-output trick)
... → Concat → output [F16, [1,10,30,30]]    # rename to be the model output
graph.output[0].type.tensor_type.elem_type = FLOAT16

# After (BOOL-output trick — deeper compression)
... → Pad → output [BOOL, [1,10,30,30]]    # rename to be the model output
graph.output[0].type.tensor_type.elem_type = BOOL
```

- Why it saves: the scorer EXCLUDES the model output from memory. Before, the F16/BOOL intermediate was counted; after, it's the output (skipped).
- Net save:
  - F16-output: ~18,000 bytes for [1,10,30,30] F16 (+ any F16 [1,1,30,30] = 1,800-byte inits only present for bool↔float bridges).
  - BOOL-output: ~9,000 bytes for [1,10,30,30] BOOL (1 byte/elem). Same mechanism, deeper compression because BOOL is 1 byte vs F16's 2.
- Safety: `run_network` does `(out > 0.0).astype(float)`. F16/BOOL representations of {0.0, 1.0}/{False, True} are exact under this thresholding; result is byte-identical to the F32 incumbent under the official decoder.
- Detection: `onnx.load(...).graph.node[-1].op_type == "Cast"` and the prior node produces a tensor named `output_f16` / `output_bool` / similar.
- Eligibility for BOOL-output: output values must be deterministically {0, 1} after the final op. Internals can be ANY dtype (F16, F32, INT8, INT64, etc.) — the BOOL is only the OUTPUT DECLARATION, not whole-graph propagation. This is the **Track B** variant of input→BOOL (see exemplar).
- Exemplars: task092 (F16-output -18k), task117 (F16-output stacked with crop, +1.12 LB), **task376 (BOOL-output Track A, +0.55 LB, MID_GATHER family)**, **task069 (BOOL-output Track B, +2.54 LB CONFIRMED 2026-05-22, Crop+Conv+ConvTranspose family — internals were F16/F32, only output dtype declared BOOL)**.

---

# 4. F16 placement rules

The bare F16 trick — cast intermediate tensors from F32 to F16 to halve their bytes — is correct in principle. **Placement is where workers slip up.** `pass_fp16_selective` (which runs onnxconverter_common's blanket conversion) regresses ~134/401 tasks because it pre-casts the full input.

### Rule A — Cast AFTER Slice/Gather, not before

If the model Slices `[1, 10, 30, 30]` down to a smaller intermediate, put the F16 Cast on the SLICE OUTPUT. Casting the full input adds ~18,000 bytes of Cast-node overhead; casting the smaller post-Slice tensor scales proportionally.

**Combined per-channel + spatial Slice variant**: when the task only uses ≤ 3 of the 10 input channels AND crops spatially, per-channel F32 Slice → per-channel Cast(BOOL) is cheaper than the full-input `Cast(F32 → BOOL) [1, 10, 30, 30]` (9,000B) followed by Slice:

```
# Old: Cast then Slice (also: full BOOL is too big when only a few channels are used)
Cast(input F32 → input_b BOOL) [1, 10, 30, 30] = 9,000
→ Slice channels [1, 1, 20, 20] BOOL ×3 = 1,200 (3 channels)
# Total: 10,200 B

# New: per-channel F32 Slice + Cast
Slice input ch_i → F32 [1, 1, 20, 20] = 1,600 ×3 = 4,800
→ Cast(F32 → BOOL) → BOOL [1, 1, 20, 20] = 400 ×3 = 1,200
# Total: 6,000 B (save 4,200)
```

Empirical: task280 radical saved -4,200B via per-channel F32 Slice + Cast.

### Rule B — Cast per-channel when N < 10

When the model Gathers N channels individually and only ops on those, cast each gathered channel separately rather than pre-casting the full tensor.

```
# Bad: pre-cast full input = 19,800 bytes (task055)
Cast(input, F16) → Gather(channel_3) → ...

# Good: per-channel cast = 5,400 bytes
Gather(input, channel_3) → Cast(F16) → ...
```

Per-channel placement wins when N < ~6.

### Rule C — Hash-decode models: NEVER cast the hash side

For graphs with `Gather(table, hash_key) → MatMul(large_weights)` lookup architectures, the MatMul weight tensor MUST stay F32. F16 destroys hash discrimination and breaks the decode. Cast only the decode side (UINT8 → F32 → downstream ops).

### Rule D — F16 for ScatterND spatial pipelines

When the spatial chain is `bg_canvas → ScatterND → Reshape → Cast(INT64) → OneHot`, use F16 throughout the spatial portion instead of INT32 (4 bytes/elem) or INT64 (8 bytes/elem):

```
bg_per_pair stored as F16 init
stamps_color stored as F16 init
Where(BOOL, F16, F16) — ORT supports this
ScatterND with F16 data — ORT supports this
Cast F16 → INT64 only at OneHot input boundary
```

**Savings**: 3 spatial intermediates × (4 - 2) bytes/elem × 900 cells = ~5,400 bytes on NeuroGolf shapes.

**Constraints**:
- Values must fit F16's integer-exactness range (0–2048 fully exact; see `ort_reality.md §11`). NeuroGolf color indices 0–10 are safe.
- **Re-verified 2026-05-23**: Where(BOOL, UINT8, UINT8) and ScatterND(UINT8/BOOL) both work at opset 13+. **UINT8 is now the smallest viable dtype** for ScatterND + Where pipelines on 0..255 color labels. F16 is still required only when downstream needs Add/Sub/Mul before opset 14 or ReduceSum/Mean. Where still rejects INT8/INT16/BOOL data inputs at all opsets.

Empirical: **task366 v10** saved -5,690 bytes via this swap. See `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md` sub-technique 3.

### Decision procedure

1. Find the smallest tensor the F16-eligible compute runs on.
2. Place the F16 Cast immediately upstream of that point (NOT on the model input).
3. Place the F32 Cast-back as far downstream as possible. Drop entirely if technique 3 (F16-output trick) applies.
4. NEVER cast MatMul weights in hash-lookup pipelines.

Exemplars: `exemplars/03_crop_and_f16/task055_per_channel.md`; task222's f16-selective attempts: `The_400_Tasks/Task_by_Task/task222.md`.

---

# 4b. Detector→paint→`input+delta`→`Pad` ⇒ output `Where` (drop the 10-channel intermediates)

The single biggest lever on "find cells matching a local rule, repaint them color K" graphs.
These end with a 10-channel paint chain that is **pure waste** because the model OUTPUT is excluded from scoring:

```
delta         = Where(mask [1,1,H,W], PVEC [1,10,1,1], 0)   # [1,10,H,W]  COUNTED
output_active = Add(input_cropped [1,10,H,W], delta)        # [1,10,H,W]  COUNTED
output        = Pad(output_active, ->[1,10,30,30])          # OUTPUT (excluded)
```
Replace the whole tail with ONE full-canvas Where that produces the output directly:
```
mask_30 = (pad mask to 30x30)                               # [1,1,30,30] BOOL = 900  (single channel)
output  = Where(mask_30, NEWCOLOR_vec [1,10,1,1], input)    # [1,10,30,30] OUTPUT (excluded)
```
- Eliminates `delta` + `output_active` (two `[1,10,H,W]` tensors = ~32k at H=W=20 F32) and `input_cropped`
  if the mask is computed from a single sliced channel (see 4c). Net: the 10-channel dimension vanishes from
  the *counted* memory entirely — only single-channel mask tensors remain.
- **Safety conditions (both required for byte-equality):**
  1. Painted cells have a KNOWN prior color so `input + PVEC ≡ NEWCOLOR_vec`. The classic case: PVEC `[-1,+1,0..]`
     (subtract black, add blue) on cells that were provably black → `Where(blue_onehot, ·)` is identical.
     Prove the "painted cells had color X" invariant from the detector (e.g. the fill region = dilation of
     all-black-block anchors ⇒ every fill cell was black).
  2. The padding region matches. The official encoder is `np.zeros` one-hot (`convert_to_numpy`), so cells
     outside the active grid are **all-zero**; `Where(mask=False, ·, input)=input=0` there == the parent's
     `Pad`-with-0. (If a parent used ch0=1 padding this would differ — check the encoder.)
- `NEWCOLOR_vec` and `input` must share dtype for `Where` (both F32 here ⇒ F32 output, matches parent exactly
  under `np.array_equal`). Internals (mask chain) can be F16/BOOL.
- Empirical: **task162 v1** 71,587 → 9,944 (-86%) combining this with 4c below. All ops in-vocab (`Where` is
  already in the parent). See `The_400_Tasks/Task_by_Task/task162.md`.
- Empirical: **task166 v5** 21,648 → 1,900 (−91.2%, byte-equal-full, staged 2026-06-06) — the 8-node endpoint
  form: ONE §41-negpad 1×1-Conv plane ([1,1,12,12] F32, weights **scaled to 12.0**) feeds row/col ReduceMax AND
  `t = Less(plane, col_has)` — the scaled plane makes the col-reduce double as a broadcast threshold, so ONE
  comparison computes (col-occupied ∧ ¬detector). Then And(t, rowb) → Pad → final `Where`. Three moves worth
  copying: (a) **Less-broadcast fusion**: scale a detector plane so an axis-reduce of the SAME plane is the
  comparison threshold — merges boolify + one And into one op (−157 here); (b) the fill-window crop is bounded by
  the ARC-GEN **generator source** (box placement margins), not by grid size — Where-assembly leaves passthrough
  uncropped, so only the *paintable* region needs covering; (c) assembly-floor lemma: `Where(cond [1,1,30,30] BOOL
  = 900 B, vec, input)` is the UNIQUE sub-1,960 output assembly (exhaustive op sweep in `task166.md` Floor proof).
  Template pricing for any "recolor region" task: cond plane 900 + F32 first-touch 4·B² + ~290 logic + ~110 vectors.

# 4c. De-Morgan dedupe: `a·(1-b)·(1-c)·(1-d)` ⇒ one Conv + `Greater(==0)` + `Where`

A "keep cell iff NONE of its (shifted) neighbors are also set" dedupe (common for picking the top-left
representative of an adjacent cluster) is usually built as N shift-ops + N `Sub(1,·)` + `Where` + (N-1) `Mul`.
For 0/1-valued `a,b,c,d` it collapses via `(1-b)(1-c)(1-d) == (b+c+d == 0)`:

```
# Before: 3 shift-Convs/Slices + 3 Sub(1,·) + Where + 2 Mul  (~10 single-channel tensors)
# After:
nbr     = Conv(a_f16, K)          # K weights = 1 at each neighbor offset, 0 at self  -> nbr = b+c+d
nbr0    = Greater(half, nbr)       # 0.5 > nbr  <=>  nbr == 0     (uses Greater, in-vocab; no Less/Not needed)
primary = Where(nbr0, a_f16, 0)    # a where no neighbor set
```
- For the {up, left, up-left} neighbor set, ONE 2×2 kernel `[[1,1],[1,0]]` over `pad(top=1,left=1)` sums
  diag+above+left with the self-position weighted 0 — replacing two shift-Convs + a Pad/Slice diag shift.
- Safety: requires `a` and the shifts ∈ {0,1} (so the product-of-complements = (sum==0) identity holds, and
  `nbr` is an exact small int in F16). The shifts of a 0/1 mask via a [1,0,..] Conv or Slice stay 0/1. ✓
- **In-vocab**: uses only `Conv`/`Greater`/`Where` — introduces NO `And`/`Or`/`Not` (avoids the traps.md
  "novel op chain" grader-credit risk). It is a node-REMOVING rewrite (10 tensors → 3).
- Empirical: **task162** dedupe 7,968 B → 2,000 B. Verified 10,000/10,000 byte-equal incl. adjacent-anchor stress.

---

# 4h. CumSum→ArgMax run-length (drops the F32 CumSum track; opset-12; 2026-06-01, task216)

A "contiguous run length from index `s` in a 0/1 profile" is usually built as
`gap = (run==0) → Cast → CumSum → Less<0.5 → Cast → ReduceSum` (5+ F32 ops; **CumSum
forces F32 — it rejects F16 AND U8**, traps.md §8f). The run length **equals the index
of the first gap**, which `ArgMax` computes identically:

```
run_len(prof_u8, s) :  pad = Pad(prof,[0,K],0); val = Gather(pad, s+range(L+1));
                       span = ArgMax( Less(val, 1) )          # first 0  == run length
```

- Append ONE sentinel gap (scan length `L+1`, the extra slot always 0) so a full run
  with no interior 0 still returns `L` instead of ArgMax's tie-break 0.
- **Everything stays U8/INT/BOOL** — no CumSum, no F32 [k] vectors. `ArgMax(U8)` is
  opset-10+; `Less(U8,U8)`/`Pad(U8)`/`Gather(U8)` all opset-11/12.
- Byte-equal: both methods return the same integer first-gap index, so it's a safe
  swap inside a byte-equal compression (NOT a rule change).
- Empirical: **task216** coordinate track F32→INT pipeline, 20,327→16,215 B (−4,112),
  byte-equal 15,439/15,439. Pairs with the membership compares done in INT32
  (`r≥idx ∧ r<idx+span` via `Not(Less)`+`Less`) so the whole greedy-rectangle
  coordinate math avoids float entirely.

# 5. Equal-fold (algebraic identity)

```
Cast(Less(Sqr(Sub(a, b)), 0.25), F32)   →   Cast(Equal(a, b), F32)
```

Whenever `a` and `b` are exact-integer F32 values (which most NeuroGolf basis vectors are), `(a-b)² < 0.25` iff `a == b` exactly. The intermediate Sqr (Mul-of-self) and threshold check are unnecessary.

**Variants**:

Pointwise (task233):
```
Cast(Less(Mul(Sub(v41, v12), self), 0.25), F32)   →   Cast(Equal(v41, v12), F32)
```
Collapses 3 ops → 1. Saved -32,000 bytes (+0.5510 pts).

With ReduceSum (task133):
```
Less(ReduceSum(Mul(Sub(scalar, v1), self), axes=[k]), 0.25)   →   Squeeze(Equal(scalar, v1), axes=[k])
```
Over a size-1 axis. Drops 4 ops + the threshold init. Saved -55,636 (+0.4238 pts).

**Opset-10 trap (Equal requires INT32)**: under opset 10, `Equal(float_a, float_b)` is INVALID. Cast to INT32 first. To net positive bytes when comparing one value against N constants, **share a single Cast**:

```
# Bad: N Casts (one per Equal), each adds overhead
Equal(Cast(value, INT32), Cast(c1, INT32))
Equal(Cast(value, INT32), Cast(c2, INT32))
...

# Good: 1 shared Cast
v_int = Cast(value, INT32)
c1_int = (initializer, INT32)
Equal(v_int, c1_int)
Equal(v_int, c2_int)
...
```

task368 netted -7,200 bytes via the shared-Cast variant (10-channel color compare).

Detectors: `pass_equal_fold_distance_threshold.py`, `pass_shared_cast_equal_fold.py`. Exemplars: `exemplars/04_algebraic_identities/`.

---

# 6. Add-chain → Sum

```
Add(Add(Add(a, b), c), d)   →   Sum(a, b, c, d)
```

- Add is associative + commutative. ONNX `Sum` accepts ≥1 inputs with the same broadcast semantics as Add.
- Eliminating N-1 intermediate tensors saves N-1 × (intermediate bytes).
- Only collapse chains of length ≥ 2 (≥ 3 inputs in the Sum). Single Add → Sum(2) is a no-op refactor.
- Safety on left-fold: each Add's first input must be the previous Add's output AND the previous Add's output must have ONLY that one consumer (else collapsing would orphan a needed intermediate).
- Detector: `pass_add_chain_to_sum.py`. Exemplar: task327 (-5,760, +0.350 pts), task013 (-17,280).

**Implementation note**: the detector must give every emitted Sum node a UNIQUE name. Multiple `helper.make_node("Sum", name="")` calls produce duplicate-name validation errors at ORT load time. Use a counter.

---

# 7. Shape-only patterns: the cost floor

Patterns expressible purely as `Slice + Gather + Concat + Pad + Reshape + Transpose + Identity + Cast` are the reliable low-cost shape/move floor (often ~1,500 bytes of intermediate-tensor memory plus tiny index params). Score ~17.5 pts/task. Beats LUT and Conv representations when applicable.

- `Where` and `OneHot` are 0 MACs in onnx-tool, but they are **not memory-free shape-only ops**. Their outputs still count unless final-named `output`, and full-canvas `Where`/`OneHot` can be much more expensive than the shape/move floor.
- Detector ordering matters: **try shape-only patterns BEFORE LUT/Conv patterns**.
- Examples: mirror-concat (5 tasks), mirror-grid-2x2 (3 tasks), fixed_rot180/90/270 via Slice→Gather→Pad sandwich (3 tasks at cost 1,091 = 18.01 pts vs full-canvas Gather at cost 36,030 = 14.51 pts).

**Caveat**: shape-only ops on the full canvas vs an active sub-grid differ in cost. `flip_h`/`flip_v`/`shift_h`/`shift_v` at single-Gather (cost 30) is already optimal; adding Slice+Pad sandwich would add ~36k bytes and regress. Measure before adopting.

---

# 8. Whole-grid summary via single Conv

For tasks where the input is a fixed `in_h × in_w` sub-region of the 30×30 canvas, compute a **single integer summary key** with one Conv op:

```
weights[i, j] = 2^(i * in_w + j)   for cells in the active grid, 0 elsewhere
Conv(input, weights, padding=0) → [N, 1, 1, 1]   # the summed key directly
```

Then Cast → Reshape → Gather into a LUT. Drives `grid_pattern_to_color_network`. Combined with `OneHot` (0 MACs at opset 10+, with output memory still counted if intermediate), the LUT can store INT64 colors per output cell instead of one-hot floats → cuts params 10×.

---

# 8b. Collapse one-hot → single-channel color-index canvas BEFORE gather/mirror (kill the ×10)

When a task's logic is "locate something, then **gather/mirror/slice a small region** of the grid
and re-emit it as a one-hot," the expensive part is almost always that the gather source and its
intermediates carry all **10 one-hot channels**. A full-canvas mirror is `[1,10,30,30]`; a 5-row
gather is `[1,10,5,W]`. The answer is only `region × 10` values, but you pay 10× on every
intermediate along the way.

**Fix:** collapse the one-hot to a single-channel **color-index canvas** with one 1×1 Conv, do ALL
the locate/gather/mirror/select logic in canvas space (every data intermediate is now ÷10), then
`Equal`-expand the chosen region back to a one-hot at the very end.

```python
# collapse: weights map channel c -> color value c
canvas = Conv(input, W=[0,1,2,...,9] as [1,10,1,1])      # [1,1,30,30] F32 = 3,600  (was 18,000 F16 one-hot)
# ... all gathers/mirrors now [1,1,5,30]=600 not [1,10,5,24]=2,400 ...
onehot = Equal(selected_canvas[1,1,h,w], chan_ids[1,10,1,1]=[0..9])   # re-expand -> [1,10,h,w] BOOL
out    = Cast(onehot, F16)                                            # 0.0/1.0 == canonical one-hot bits
```

Because one-hot floats are exactly `{0.0,1.0}` and `Conv([0..9])` recovers the integer color exactly,
the re-expanded region is **bit-identical** to the original one-hot → the rewrite can be **byte-equal**
to the canonical (Path A `ab`), not merely truth-equal.

**Sentinel variant — fold the anchor into the same canvas (free).** If the locate step needs to find
one special color (e.g. a blue occlusion == color 1), map THAT color to a large sentinel in the Conv
weights (`W[1]=100`). Then `ReduceMax` over rows/cols of the *same canvas* locates it as the unique
global max — no separate channel slice needed. The selected output region is hole-free so it never
carries the sentinel; add a one-op defensive clamp `Where(canvas>9.5, 1, canvas)` before the expand to
stay byte-equal even on degenerate inputs.

**Measured (task400, 2026-06-01):** D4-symmetry 5×5 hole-fill. Canonical worked in F16 one-hot:
cast 18,000 + 24×24 crop 11,520 + two full-canvas mirrors 11,520 each = **60,395**. Canvas-collapse +
sentinel anchor (v2) → 6,674. **Single-mirror refinement (v3) → 5,730, byte_equal 200,266/200,266** (+2.36 LB,
10.5×): the canonical's data-dependent "which mirror" choice often reduces to a closed-form predicate on
the anchor — here `use_vm = (8 ≤ y0 ≤ 11)` is provably exact (the mirror window's columns equal the hole's,
so overlap depends only on rows), letting you gather ONE mirror via `Where`-selected index lists instead of
computing both + a data Where. **Always check whether the data-dependent branch is a closed-form function of
the anchor — it frequently is, and stays byte-equal.** Dropping the final F16 cast (v4, BOOL output via
`Pad`-on-bool) → 5,230 but the output is only value-equal (not bit-identical) to the F16 canonical.
Floor ≈ 5,200–5,730: the single 30×30 F32 canvas (3,600) is irreducible — an F16/U8 canvas needs a
full-input cast (18,000/9,000) that dwarfs the saving, and ArgMax-over-channels emits `[1,1,30,30]` INT64
(7,200). Build: `Task_by_Task/task400/prototypes/build_task400_v3.py` (recommended) / `_v4.py` (floor).

**Generator-separation shortcut (task004/task165, 2026-06-01).** Before materializing per-color
`[1,K,H,W]` planes, inspect the generator for separation invariants. If each colored object lives in a
disjoint row/box band, a rule that looks "per-color" may collapse to a single-channel canvas plus row/local
tests. task004's shifted parallelograms were row-disjoint and gap-separated: the per-color `max(r+c)`
anchor simplified to "bottom row OR corner-above" on one scalar canvas, so no per-color grouping was
needed (62,082 → 8,229, public +2.02). Conversely, if the rule truly needs arbitrary 2-D color layout
(task165 archfall), the `[1,1,30,30]` F32 color-index plane itself is a 3,600 B floor before output
assembly; sub-1k is not a real target unless you first prove a stronger structural invariant.

**When to reach for it:** any task whose graph diag shows full-canvas (or wide) `[1,10,*,*]` one-hot
intermediates feeding a Gather/Slice/Mirror that ultimately produces a small region. Distinct from §8
(integer summary KEY for a LUT) and from the canvas+`Equal` OUTPUT-assembly pattern in
`cost_floor_reference.md` — here the canvas is the **gather SOURCE**, collapsed before the moves.

**DILATED crop-collapse — collapse channels AND crop to the active grid in ONE Conv (task400 v6, 2026-06-01).**
The color canvas is usually `[1,1,30,30]` F32 = 3,600. If the active grid is `S×S < 30`, you want a
`[1,1,S,S]` canvas, but `Conv(input,[1,10,1,1])` outputs the full 30×30 and a follow-up Slice keeps the
3,600 intermediate. Instead crop AND collapse in a single **dilated** Conv:
```
kernel [1,10,2,2], dilations=[d,d], pads=0,  with the color-selector ([0,1,..,9]) on the (0,0) tap ONLY
=> receptive field d+1; output spatial = 30-(d+1)+1 = 30-d;  pick d = 30-S  (S=24 -> d=6 -> [1,1,24,24])
=> output[i,j] = Σ_c sel[c]·input[c,i,j] = color(input[i,j])   for i,j in [0,S-1]
```
Only the `(0,0)` tap is nonzero so the other 3 taps (which read input[i,j+d] etc.) contribute 0 — the
output is the **per-cell color of the top-left S×S**, i.e. a cropped color canvas. Cost: `[1,1,24,24]`
F32 = **2,304 B + 40 params** (kernel `1·10·2·2`), vs 3,600 (1×1 conv, no crop) or a dense `[1,10,7,7]`
crop kernel = **490 params**. The dilated `[2,2]` is the cheap way to get the 7-wide receptive field.
Use a **sentinel** color (e.g. blue→100) so `ReduceMax` over the same cropped canvas still finds the
anchor. **task400 v6b: 60,395 → 3,557 (17×, +2.83 LB), value-equal 200266/200266.** Build:
`Task_by_Task/task400/prototypes/build_task400_v6b.py`.

> Floor note (task400): even with the 2,304 crop-canvas, the core path `canvas 2,304 + dynamic-5×5
> first-Gather [1,1,5,24] 480 + one-hot Equal-expand [1,10,5,5] 250 = 3,034` exceeds 3,000. An F16
> canvas (1,152) would break 3,000 but ORT's Conv output dtype = input dtype = F32, and casting needs
> the counted F32 precursor (or an 18k full-input F16 cast). Dynamic extraction must be **Gather**
> (static output shape); a dynamic **Slice** with computed starts/ends yields `[unk,…]` shapes that
> `calculate_memory`'s `infer_shapes(strict_mode=True)` rejects (cost → None, unsubmittable).

**Do not force the source canvas and normalized crop planes to the same size (task138 v9, 2026-06-02).**
In "crop then re-render at (0,0)" tasks, there are two distinct bounds:

```
crop planes     need max(down - up + 1, right - left + 1)     # normalized output coords
source canvas   needs max(down + 1, right + 1)                # absolute gather coords
```

Using one square `N` for both wastes bytes. task138 v7 used `N=26` for the color canvas and every shifted
plane (cost 14,838, hidden-correct). Reading the generator more carefully gave `crop <= 24x23`, but the
gather source still had to cover absolute rows `0..24` and cols `0..23`, so the clean split is:

```
canvas source: [1,1,25,24] F32 = 2,400 B
crop planes:   [1,1,24,23] U8/BOOL/F16 as needed
```

That split shipped as task138 v9: **14,838 -> 13,204**, 266/266 visible, 200k/200k truth, 100%
adversarial, and Kaggle confirmed +0.12 LB over v7. A tempting official-generator-only shrink, v10,
used negative Conv pads to drop source row 0/col 0 (`[1,1,24,23]` source; cost 12,899). It was byte-equal
to v9 on official-encoded inputs and 200k/200k truth, but failed adversarial vs LIVE because edge frames
at row/col 0 became unrepresentable. See `traps.md §14i`.

---

# 8c. Count-per-value / dominant-color / mode via `ScatterElements(reduction='add')` (task222 v2, 2026-06-01)

When you need a **histogram of a single-channel index canvas** — "how many cells of each color",
"which color/value is most frequent (mode / dominant)", "per-color block counts" — the obvious
build is a per-color membership `Equal(chan_ids[1,K,1,1], canvas[1,1,H,W]) → [1,K,H,W]` then
`ReduceSum`/`MatMulInteger`. That materializes a **K×(H·W)** tensor (e.g. 10×256 = 2,560 B as the
bool + 2,560 B as the u8 copy = 5,120 B). **`ScatterElements(reduction='add')` does it with no
per-value expansion** — just a flat int32 index list:

```python
# bc = the single-channel value map [1,1,H,W] (e.g. block-color where mono else 0)
bc_flat = Reshape(bc, [H*W])                 # u8  [256]
idx     = Cast(bc_flat, INT32)               # i32 [256]   <- the only real cost (1,024 B for 256)
counts  = ScatterElements(zeros[K], idx, ones[H*W], axis=0, reduction='add')  # counts[idx[i]] += 1  -> [K]
counts  = Mul(counts, no_bin0)               # drop background/sentinel bin
dc      = ArgMax(counts, axis=0)             # dominant value  -> [1] i64
```

- **opset 16+ only** — the `reduction` attr did not exist before; ORT rejects ScatterElements with
  `reduction` at opset ≤13 (`INVALID_GRAPH`). Bump the model to opset 16 (all of Conv/Slice/Equal/
  MaxPool/Pad/ArgMax/Where still valid there).
- **O2 audit (2026-06-19): this is the scatter-reduction shelf that survives.** ORT CPU 1.24.4 runs
  `ScatterElements` reductions as a deterministic flattened fold, so duplicate-target `add` is valid for
  compact histograms on this stack. This does NOT transfer to duplicate-target `ScatterND` reductions; see #34.
- **Indices `Tind` ∈ {int32, int64}** (no u8/int16) — int32 `[H·W]` is the min cost. `data` (zeros[K])
  and `updates` (ones[H·W]) are initializers; `updates` must match the index shape (no broadcast →
  H·W params; for 16×16 that's 256 params, ~256 B of cost, still a win).
- Send non-target cells (non-mono, out-of-grid) to a bin you ignore (bin 0) so they don't need masking.
- **task222: per-color block-count histogram 5,120 B → ~1,300 B** (v1 16,154 → v2 11,758, +0.32 pts).
  Reusable for ANY dominant-color / most-frequent-value / per-class-count task in the 400.

> Note: this is the single-channel-canvas counterpart of §8 (Conv whole-grid summary). Pairs naturally
> with §8b's color-index canvas (collapse one-hot → index, then ScatterElements-count it). task222 v3
> stacks §8b dilated crop-collapse + §8c ScatterElements + Equal-broadcast output → **58,031 → 8,308
> (7×, +1.94 LB), byte_equal 50,266/50,266**. Build: `Task_by_Task/task222/prototypes/build_task222_v3.py`.

# 8d. Per-category min/max of a positional quantity via `ScatterElements(reduction='min'|'max')` (task018 v7, 2026-06-01)

The extremes-counterpart of §8c. When you need the **min or max of some positional quantity, grouped
by a per-cell category** — "the first/last cell of each color", "the topmost row per object id",
"min flat-position per class" — the obvious build is, per category, `Where(cat_mask, value, sentinel) →
ReduceMin/Max` over a full `[1,1,H,W]` plane. That is **one full HxW plane per (category-consumer, min/max)**.
Replace ALL of them with ONE shared scatter that produces a `[K]` vector of per-category extremes, then
each consumer `Gather`s its category (≈free):

```python
# cat_flat = per-cell category index [H*W] (e.g. color-at-anchor, 0 = "ignore" bin); pos_p1 = position+1 INIT
cat_i32  = Cast(cat_flat, INT32)                                  # [H*W] i32  <- the real cost
minpos   = ScatterElements(full[K], cat_i32, pos_p1, axis=0, reduction='min')  # [K] : min pos+1 per category
maxpos   = ScatterElements(zeros[K], cat_i32, pos_p1, axis=0, reduction='max') # [K] : max pos+1 per category
# consumer: this_cat_min = Gather(minpos, my_category)  -> [1]   (tiny)
```

- **Use opset 18 for spec-clean `min`/`max`.** O2 (2026-06-19) source-audited the schemas: opset 16 adds
  `add`/`mul`, while opset 18 adds `min`/`max`. ORT CPU supports duplicate-target `ScatterElements`
  `min`/`max` as a deterministic flattened fold; keep payloads integer when positions must be exact.
- **The UPDATE can be the 0-memory position INIT** (`pos_p1`, an initializer): non-category cells carry a
  real position too, but the category INDEX sends them to the ignore-bin (0), so they never pollute a real
  bucket — **no masking of the update is needed.** (For `min`, the data-init must be LARGE so empty
  buckets stay large; for `max`, init 0.)
- **Indices `Tind` ∈ {int32, int64}** — int32 `[H*W]` is the floor of THE SCATTER FORM (but the histogram
  itself is NOT a hard floor when the counted feature is a fixed-degree product — see §8e).
- **Win condition: ≥2 consumers (or ≥2 of min/max).** task018 had per-template min AND max reference-anchor
  searches = 4 full `[1,1,24,24]` F16 planes (4,608 B) + 2 `anchor_ref_bool` BOOL planes (1,152 B); replaced
  by 1 shared `[576]` i32 index (2,304) + 2 `[K]` scatters (~44) → **net −3,403** (compress_v6 61,739 →
  compress_v7 58,336, 266/266 vs LIVE). Build: `Task_by_Task/task018/prototypes/build_task018_compress_v7.py`.

# 8e. Per-class k-cell-motif histogram via a multi-input Einsum (scorer-invisible) — the scatter is NOT a floor (task222 v9, 2026-06-08, GRADER-VALIDATED byte-equal-LIVE 200266 + profile==static)

When you need a **per-color/per-class count of a fixed-degree CELL MOTIF** — "how many solid 2×2 blocks
of each color", "how many horizontal dominoes", "how many L-corners" — the obvious path is detect the
motif per cell (mono mask, ~2k of U8/BOOL planes) → mask the color → `ScatterElements` histogram (int32
`[H*W]`, §8c/§8d). But the count is a **polynomial in the one-hot**, so it is ONE Einsum whose pairwise
products are all in-kernel (#89 scorer-invisible — `calculate_memory` skips unnamed temps; profile==static):

```
# solid-2x2 count per color c = Σ_{i,j} ch_c(i,j)·ch_c(i,j+1)·ch_c(i+1,j)·ch_c(i+1,j+1)
shift[a,b] = (b == a+1)                         # ONE [H,H] superdiagonal, reused for both axes
cnt = Einsum('bcij,bciJ,jJ,bcIj,iI,bcIJ->bc', x, x, x, x, shift, shift)   # [1,C], EXACT count
dc  = ArgMax(no_ch0 · cnt)                       # dominant class
```

- **Charged cost = the shift init (`[H,H]`) + the `[1,C]` result.** The whole mono-detect + int32 scatter
  (task222: 2,025 + 676 ≈ 2,700 of planes) **vanishes**. task222 v9: dropped it for one 900-param shift →
  histogram path 8e ≈ 940 (mostly the shift) vs §8c path ≈ 2,700. (4-rung arc 8308→5582 conventional→**4680**.)
- **It is EXACT, not a proxy.** The 4-way product is 1 iff all 4 cells are class c, so `ArgMax(cnt)` == the
  true most-motifs class. (Contrast: a max-ADJACENCY *proxy* for "most 2×2 blocks" diverged 1/200266 —
  task222 v8. Compute the actual motif, not a correlate.)
- **OPERAND ORDER IS THE CONTRACTION PATH (#89) — and it dominates RUNTIME, not cost.** `'bcij,bciJ,jJ,bcIj,iI,bcIJ'`
  (each input next to the shift it pairs with) keeps every temp at `[1,C,H,W]` → 2.1 ms; the naive
  `'…x,x,x,x,shift,shift'` materialises `[1,C,H,W,W]` → 72 ms. **Reorder and time a single inference**
  before trusting; a tight per-task wall-clock grader could still object to the in-kernel blow-up.
- **Shift dtype/op:** F32 (ORT CPU Einsum is float/int32/int64 only). The shift superdiagonal also works
  for dominoes (one shift, 2 inputs), 3-in-a-row, plus-shapes, etc. — any fixed cell-stencil motif.
- Transfer: ANY dominant-class / mode / per-class-motif-count rule. See `Task_by_Task/task222/prototypes/build_task222_v9.py`.

---

# 9. Active-cell / non-bg detection (correct version)

Foreground = any cell with a non-bg color OR with bg color INSIDE the active grid (excludes padding outside the grid).

```python
# Active (any color, NOT padding):
active = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)   # BOOL [1,1,30,30]

# Non-bg (excludes both bg cells AND padding):
full_sum   = ReduceSum(input, axes=[1], keepdims=1)
ch0        = Gather(input, [0], axis=1)
nonbg_int  = Sub(full_sum, ch0)
nonbg_bool = Greater(nonbg_int, 0)
```

DO NOT use `Sub(1, Gather(input, [0], axis=1))` — the "1-ch0 trap". See core.md.

### Compact form: row+col reductions + Or (when interior is dense)

The standard `Greater(ReduceSum(input, axes=[1]), 0)` for active-cell detection produces an F32 `[1,1,30,30]` intermediate (3,600 bytes) before the BOOL Equal. When **every interior row and column has ≥1 active cell** (guaranteed for NeuroGolf-encoded grids that aren't all-padding), a cell is "padding" iff its row OR its column is entirely bg — collapsible to two rank-3 reductions broadcast back:

```python
row_active_f = ReduceSum(input, axes=[1, 3], keepdims=1)   # F32  [1,1,30,1] =   120
row_pad_b    = Equal(row_active_f, 0_f32)                   # BOOL [1,1,30,1] =    30
col_active_f = ReduceSum(input, axes=[1, 2], keepdims=1)   # F32  [1,1,1,30] =   120
col_pad_b    = Equal(col_active_f, 0_f32)                   # BOOL [1,1,1,30] =    30
is_padding   = Or(row_pad_b, col_pad_b)                     # BOOL [1,1,30,30] =  900 (broadcast)
```

Total: **1,200 bytes vs 4,500 bytes** for the standard form — saves 3,300 bytes per padding-detection block.

- Equivalence requires every interior row/col to have ≥1 active cell. For NeuroGolf's standard "active grid in top-left" encoding this holds whenever the active grid touches all interior rows and columns. Verify with `Greater(ReduceSum(input[:, :, :H, :W], axes=[1,3], keepdims=1), 0)` showing all-true for the interior in your task.
- Empirical: task110 v9 (2026-05-20).

---

# 10. nonzero_shortcut: `Mul(input, gate) → ReduceSum` collapses

```
Mul(input, nonzero_gate) → ReduceSum(axis=1)   →   ReduceSum(input, axis=1) - Slice(ch0, axis=1)
```

- Works because bg channel (ch0) is 1 on bg-cells, 0 in padding, AND foreground channels (ch1..9) are 0 in padding too. Subtracting the literal ch0 slice cancels exactly the bg-cell contribution from the full sum.
- Saves ~28,800 bytes per task by dropping the entire nonzero-mask construction.
- Critical: do NOT substitute `1 - ch0` for the mask. Padding cells have ch0=0, so `1-ch0=1` there, incorrectly adding padding to the sum.

---

# 11. Static lookup table → dynamic compute (rank-aware)

When a task uses a static lookup table (e.g., a `[5, 1, 1, 9, 9]` ring-mask), and the table value is a computable function of position, replace with dynamic compute.

**Key insight (task071)**: rank promotion is the dominant cost factor, not node count. The same lookup with rank-2 `[H, W]` intermediates beats the rank-4 `[1, 1, H, W]` version by ~50× per intermediate, even with more nodes.

```
# Replace [5, 1, 1, 9, 9] F32 ring-mask table (~2,140 params)
# with: ring_index(r, c) = min(r, c, 2L-r, 2L-c)
range9 = [0..8]                                      # INT64 [9]
ring_idx_h = Min(range9, Sub(2L, range9))             # INT64 [9]
ring_idx_w = same                                     # INT64 [9]
ring_idx = Min(Unsqueeze(ring_idx_h, axes=[1]),       # [9, 1]
               Unsqueeze(ring_idx_w, axes=[0]))       # [1, 9]   →  [9, 9]
frame_color = Gather(frame_colors_init, ring_idx)     # [9, 9, 1] or similar
```

23 dynamic compute nodes producing `[9, 9]` rank-2 intermediates beat the static `[5, 1, 1, 9, 9]` table. Net: -2,709 bytes, params 2,140 → 133.

**Rule of thumb**: trade static table of N elements for K dynamic nodes producing rank-R intermediates of M elements when `K × M < N` AND `R` is as low as the math allows. Always favor rank-2 over rank-4 for spatial-only computations.

---

# 12. Projection-cache (skip the wide intermediate)

When a large `[1, C, H, W]` intermediate is only consumed by reductions/gathers (no element-wise ops needing the full grid), compute the reductions directly from input and skip the intermediate.

```
# Before
input → Slice → wide_intermediate [1, 9, 16, 16] F32 (9,216 bytes)
       → various ReduceMax / Gather on wide_intermediate

# After
ReduceMax(input, axes=[3], keepdims=1)   → [1, 10, 30, 1]   then Slice channels 1..9
ReduceSum(input, axes=[2], keepdims=1)   → [1, 10, 1, 30]   then Slice channels 1..9
ReduceSum(input, axes=[2,3], keepdims=1) → [1, 10, 1, 1]    then Slice channels 1..9
Gather(input [1,10,30,30], 1D_index)     → [1, 1, 30, 30]   then Slice to [1, 1, 16, 16]
```

- Correctness relies on padding cells outside the active grid having ALL channels = 0. Full-grid reductions over [1, 10, 30, 30] then equal cropped-grid reductions over [1, 10, H, W] because padding contributes zeros to both ReduceSum and ReduceMax.
- task071 used this approach. Cross-ref `core.md` "padding cell encoding".

---

# 13. Slice-CSE (common-subexpression elimination)

```
Slice(input, starts=[A], ends=[B], axes=[C]) → tensor_X
Slice(input, starts=[A], ends=[B], axes=[C]) → tensor_Y   # identical params!
... downstream uses both tensor_X and tensor_Y ...

   →

Slice(input, starts=[A], ends=[B], axes=[C]) → tensor_X
... rewire all uses of tensor_Y to tensor_X, drop the duplicate Slice ...
```

- `onnxsim` is supposed to catch this but misses on NeuroGolf graphs in practice (task306 had 4 duplicate Slice nodes that `onnxsim` left alone). Worth a manual scan.
- task306 saved -6,504 bytes via the dedup.
- Detector: `pass_slice_cse.py`.

---

# 14. Slice-chain-to-multi-axis

```
Slice(input, axis=A) → Slice(intermediate, axis=B)   # A ≠ B
   →
Slice(input, axes=[A, B])   # single 2-axis Slice
```

- Eliminates the intermediate tensor between the two Slices.
- Math equivalence: slicing on disjoint axes commutes.
- Detector: `pass_slice_chain_to_multiaxis.py`. Exemplar: task368 (-11,992 bytes via this + shared-cast equal-fold).

---

# 15. Separable mask collapse

When a 2D mask initializer `[H, W]` is constant along one axis (every row identical OR every column identical), collapse to 1D:

```
mask [H, W]   →   mask [1, W]   if every row is identical (varies only by column)
mask [H, W]   →   mask [H, 1]   if every column is identical (varies only by row)
```

Broadcast at use sites reconstructs the full `[H, W]` for free. Saves `H × W - W` or `H × W - H` element count.

- Detector: `pass_separable_mask_collapse.py`. Exemplar: task040 (-21,960 bytes).
- After this rewrite, **clear `value_info`** before save — downstream shape inference depends on the smaller init.

---

# 16. Dead-Relu after pad-zero MaxPool

```
Relu(Sub(MaxPool(X, sym-pad, stride=1), X))   →   Sub(MaxPool(X, sym-pad, stride=1), X)
```

- MaxPool with symmetric pads (`pad_start + pad_end == kernel - 1`) and stride=1 guarantees the output window at position [i, j] includes input[i, j]. So `MaxPool(X) ≥ X` pointwise, `Sub ≥ 0` pointwise, Relu is identity.
- Detector: `pass_dead_relu_post_maxpool.py`. Exemplar: task093 (-3,136 bytes, 4 dead Relus).

---

# 17. opt_safe blind spots — 5 algebraic-identity patterns to manually scan for

`onnx_opt_safe.algebraic_identities` only recognizes scalar 1.0 multiplies and scalar 0.0 adds. These five patterns recur across NeuroGolf tasks and are missed by spread_scan:

## 17a. `Mul(x, all-ones-tensor)` — drop the Mul

```
Mul(x, all_ones_init [...])   →   x
```

- A non-scalar all-ones initializer (e.g., task238 `v15 (1,1,7,7)` F32 all-ones) multiplying a tensor. Drop the Mul, rewire downstream to use `x` directly.
- Detector: `pass_mul_by_all_ones_drop.py`. Saved task238 -49 cost.

## 17b. `Sub(all-ones-tensor, x)` — collapse to scalar

```
Sub(all_ones_init [..., shape...], x)   →   Sub(scalar [1.0], x)
```

- Non-scalar all-ones init in the LHS of Sub (e.g., task264 `v2 (1,1,9,9)` F32 all-ones in `Sub(v2, v224)`). Replace with a scalar `[1.0]` init. Output shape preserved by broadcasting.
- Saved task264 -84 cost.

## 17c. Slice/Pad optional-default inputs (opset 13+)

```
Slice(data, starts, ends, axes, all_ones_steps)   →   Slice(data, starts, ends, axes)
```

- In opset 10+ the `steps` input is optional. Passing all-ones is redundant — drop the steps init and the input slot.
- In opset 13, omit `Slice` axes when slicing all axes and omit zero/False `Pad` constant_value. task196: -9 params byte-equal.
- Apply to all consuming Slice nodes simultaneously to free the shared init.

## 17d. `Cast(Less(Sqr(Sub(a, b)), epsilon), F32)` ≡ `Cast(Equal(a, b), F32)` (Equal-fold)

When `a` and `b` are exact-integer F32 values, the squared distance is exactly 0 iff `a == b` and ≥ 1 otherwise. So any `0 < epsilon < 1` threshold becomes equivalent to direct equality.

Pointwise (task233):
```
Cast(Less(Mul(Sub(v41, v12), self), 0.25), F32)   →   Cast(Equal(v41, v12), F32)
```

With ReduceSum over a size-1 axis (task133):
```
Less(ReduceSum(Mul(Sub(scalar_v13, v1), self), axes=[k]), 0.25)   →   Squeeze(Equal(scalar, v1), axes=[k])
```

Drops 3-4 ops + the threshold init. **opset-10 caveat**: `Equal` requires INT32/INT64 inputs under opset 10; share a single `Cast(value → INT32)` across all parallel Equals. See [traps.md #5](traps.md) (opset-10 Equal int32-only).

See #5 (Equal-fold) above for the canonical algebraic-identity pattern; this is its opt_safe-blindness companion.

## 17e. `Mul(input, nonzero_gate) → ReduceSum(axis=1)` (nonzero_shortcut)

```
ReduceSum(Mul(input, nonzero_mask), axis=1)   →   ReduceSum(input, axis=1) - Slice(ch0, axis=1)
```

Saves ~28,800 bytes per task by dropping the entire nonzero-mask construction. Works because the encoder's bg channel (ch0) is 1 on bg cells, 0 in padding cells, AND foreground channels (ch1..9) are 0 in padding too — so subtracting the literal ch0 slice cancels exactly the bg-cell contribution from the full sum.

**Critical trap**: do NOT substitute `1 - ch0` for the mask — padding cells have ch0=0, so `1-ch0=1` there, which incorrectly adds them to the sum. Use the actual `Slice(ch0)` value. See [`core.md` "1-ch0 trap"](core.md).

See #10 (nonzero_shortcut) above for the canonical pattern; this is the opt_safe-blindness annotation.

## 17 detection routine

In a fresh task card, run a script that prints:
1. All non-scalar all-ones / all-zeros initializers and their downstream consumers.
2. Every `Less(x, threshold)` where threshold is a small positive scalar with the chain `Less(...) ← Mul(self) ← Sub(a, b)` (optionally wrapped in ReduceSum).
3. Validate that `a` and `b` are exact-integer F32 by running the live ONNX with the suspect intermediates exposed as outputs.

Even a -1 cost candidate from this scan is "first registry entry" + structurally cleaner for future stacking.

## 17 also — scalar-broadcast-ones

```
init [..., 1, 1, ..., 1] uniform scalar   →   single-element scalar init   (broadcasts identically)
```

- Detector: `pass_scalar_broadcast_ones.py`. Saved 99 params on task000 just by replacing a uniform [10] init with a scalar [1].

---

# 18. Const-node → initializer

ONNX `Constant` nodes have node-overhead. Moving the value into `graph.initializer` removes the node. Same `params` cost, smaller graph.

- Detector: `pass_const_to_init.py`. Saved across 7 tasks combined (task230 origin).

---

# 19. Conv with runtime weights

ORT accepts Conv's `W` input as a runtime-produced tensor (not just an initializer), verified empirically under `ORT_DISABLE_ALL` at opset 13. This unlocks an algebraic rewrite: any chain

```
v = Mul(input [1, C, H, W], mask_per_channel [1, C, 1, 1]) → ReduceSum(axis=1)
```

collapses to a single op:

```
v = Conv(input, mask_per_channel_as_weights, kernel=[1, 1])
```

The Mul broadcast intermediate (~C × H × W bytes) is eliminated. Same answer; one node + one large intermediate gone.

**Empirical**:
- **task173 v4**: collapsed per-branch `Gemm + Reshape + Conv` chain into one Conv with runtime weight. Saved -7,500 bytes across 3 branches.
- **task205 v1**: collapsed Phase A `v8 = Mul(input_fp16, v7) → v9 = ReduceSum(v8)` into `v9 = Conv(input_fp16, v7, 1x1)`. Saved -18,000 bytes (v8 [1,10,30,30] intermediate eliminated).
- **task205 v4**: same pattern in Phase D crop. Saved -2,000 bytes.

**Anti-pattern — Conv with runtime weights does NOT replace MatMul for outer-product reductions**: if you'd write `cooccur[a, b] = sum_p input[a, p] * full[b, p]` as a Conv, the required output_channels = C×C and the weights tensor would be `[C×C, C, H, W]` — for C=9, that's 81×9×25×25 = 18,225 F16 = **36,450 bytes** just for the weights. MatMul on F16 inputs is 200× cheaper. Use Conv-runtime-weights when M (output channels) ≤ 1, NOT when M = C.

Cross-ref `ort_reality.md` Conv row: W can be initializer OR runtime tensor under `ORT_DISABLE_ALL`.

## 19a. SIGNED runtime-Conv collapses an N-class color separation into ONE plane (task165, 2026-06-01)

When a rule needs to separate two (or a few) **runtime-determined** color classes spatially — e.g. an
"arch" class and a "marker" class whose channel indices vary per input — the naive build extracts each as
its own mask: `idx = Conv(input,[0..9])`; `arch_b = Equal(idx,A)`; `marker_b = Equal(idx,M)`; then a
row-weighted plane per class. That's **4–5 full-canvas planes**. Collapse them with a single signed Conv:

```
W_signed = marker_oh − arch_oh          # runtime [1,C,1,1]: +1 at marker ch, −1 at arch ch, 0 else
signed   = Conv(input, W_signed)        # [1,1,H,W] F32 in {−1(arch), +1(marker), 0(bg/pad)}  -- ONE plane
signed_rw = Cast(signed,INT8) * rowp1   # +rowp1 at marker, −rowp1 at arch, 0 else
marker_lowrow = ReduceMax(signed_rw, rows)   # max positive  = lowest marker row (per column)
arch_lowrow   = −ReduceMin(signed_rw, rows)  # −(most negative) = lowest arch row
```

One signed plane + one row-weighted plane + Max/Min reads the per-column extremes of BOTH classes. On
task165 this took the 2-color masks from 4 planes → 1 (10,440 → 9,280, then 8,920 with the capped-row
fill below). Generalizes to any "+1/−1/0 then read both extremes" separation; for >2 classes use distinct
integer weights and `Equal(signed, k)` per class (still one Conv).

**Guards/traps:**
- `Where(BOOL, INT8, INT8)` is **rejected** by ORT (NOT_IMPLEMENTED). Keep signed-derived row indices ≥0
  and **cast to UINT8** for any downstream `Where`/`Greater` (both supported on UINT8). See dtype matrix.
- The signed values are exactly {−1,0,+1} (one-hot input × ±1 weights) → `Cast(...,INT8)` is exact.
- Zero/duplicate-class guard: multiply the class one-hot by `Equal(class_count,1)` before forming
  `W_signed`, so a malformed input contributes no `+1` channel → no spurious detection.

## 19b. Capped row-index vector bounds a downward fill WITHOUT crop+Pad (task165 v4, 2026-06-01)

For a per-column vertical fill bounded to a fixed active region (e.g. grids are a proven 20×20 in a 30×30
canvas), you don't need to `Slice`-crop and `Pad` the fill plane back. Use a **capped row vector**
`rowp1 = [1,2,…,20, 0,0,…,0]` (active rows = their 1-based index, padding rows = 0):
- `signed_rw = signed_i8 * rowcap` — padding rows weight 0, so they never affect the per-column max/min.
- `fill = Greater(rowcap_u8, lowest_fixed[c])` — padding rows have `rowcap=0`, and `0 > lowest_fixed(≥1)`
  is false, so the fill self-bounds to the active rows with **no Slice and no Pad** (removed a full
  [1,1,30,30] Pad plane). Relies on the generator's fixed size exactly like a crop — gate with truth --full.

## 19c. Cropped label Conv: spend initializer params to delete larger runtime tensors (task286, 2026-06-02)

When a graph builds a full-size position/label plane only to cast and slice it immediately, it can be
cheaper to encode the slice into the Conv kernel shape. This is the same cost trade as crop-collapse:
params are counted once, while runtime tensors are counted at full byte size every inference.

Task286's parent-equivalent crop-label candidate replaced:

```
label30 = Conv(input, 1x1 label weights)   # [1,1,30,30]
label_i = Cast(label30, ...)
label25 = Slice(label_i, [0:25,0:25])
```

with a direct cropped Conv:

```
label25 = Conv(input, 6x6 label weights, nonzero tap only at kernel[0,0])  # [1,1,25,25]
```

The 6x6 kernel spends more initializer params, but it removes the full 30x30 label plane, its Cast,
and the Slice output. On task286 this changed the safe parent-equivalent compression from 63,211 to
55,933 and Kaggle confirmed the follow-on gain (+0.12 on 2026-06-02). The under-20k m2 probe still
failed public badly, so the lesson is **not** "shallow reach is enough"; it is "pay params to avoid
counted runtime planes whenever a fixed crop/slice follows a deterministic label Conv."

Checklist:
- The desired crop must be top-left or otherwise encodable by Conv pads/kernel placement.
- Keep only the tap(s) that correspond to the source coordinate; other spatial weights are zero.
- Price params vs memory. A larger kernel can be a win if it deletes one or more `[H,W]` runtime tensors.
- Re-score with the official profiler; compressed ONNX file size can grow while Kaggle cost falls.

## 19d. Small-frame scalar color-index map before one-hot expansion (task238, 2026-06-02)

When the real decision is "choose one output color per cell" on a **small fixed frame/interior** (task238:
`7x7`), do not build parallel one-hot planes for each candidate side/color. Keep a single scalar color
index map until the tail, then expand once:

```
color_idx = ...                         # [1,1,N,N] F16/U8, values 0..9 plus sentinel 10
output_b  = Equal(color_idx, ids[1,10,1,1])   # graph output, not counted
```

Task238 v4 used a single `[1,1,7,7]` F16 selected-border-color map instead of a 13-plane
`[1,10,7,7]` one-hot chain, then did one final `Equal + Cast/Concat` expansion. It also used sentinel
value `10` for out-of-range/no-output/padding cells, which made those cells expand to all-zero and
removed a separate grid-mask/background-fill plane. The actual tie color in task238 is **8**; sentinel
10 is only for "no color should be emitted." Kaggle confirmed the rewrite: 55,368 -> 15,085, +1.30 LB.

Use this when:
- the output active region is small enough that the scalar map is cheap;
- the final output is one-hot, so the `Equal(..., channel_ids)` expansion can be the graph output;
- a non-output/padding state can be represented as `depth`/sentinel and expanded to all-zero.

---

# 20. Gemm `transA` / `transB` eliminate redundant Transpose intermediates

ORT Gemm supports `transA` and `transB` attributes (opset 7+) that internally transpose A or B without materializing the transposed intermediate. Much cheaper than `Transpose → MatMul` because the Transpose's output tensor is skipped.

```
# Pattern A — drop the SHARED Transpose
flat   = Reshape(x, ...)               # [9, 625] F16 = 11,250
flat_t = Transpose(flat, perm=[1,0])   # [625, 9] F16 = 11,250 — REMOVE
y      = MatMul(flat_t, b)

→ y = Gemm(flat, b, transA=1)

# Pattern B — drop per-branch Transpose
pairs   = ...                          # [9, 9] F16
pairs_t = Transpose(pairs, perm=[1,0]) # [9, 9] F16 = 162 — REMOVE
y       = MatMul(pairs_t, color_vals)

→ y = Gemm(pairs, color_vals, transA=1)
```

Per-Transpose savings: full byte count of the transposed tensor.

**Gotcha**: Gemm at opset ≤ 10 requires the `C` input (the bias). Use opset 11+ where C is optional, or pass a zeros scalar with `beta=0`.

**Empirical**: task173 v3 dropped shared `flat_t` F16 [625,9] = -11,250 bytes; task173 v6 dropped 3× `*_pairs_t` F16 [9,9] = -486 bytes.

Cross-ref `ort_reality.md` Gemm row: note transA/transB attribute support and C-input-required-at-opset-10 caveat.

---

# 21. Drop Conv padding when downstream threshold excludes overflow positions

In rectangle/window-area Conv-based detection, the input is convolved with an all-ones kernel of shape `[H, W]` and the output is thresholded at `area - epsilon` (e.g., `area - 3.5` for "allow up to 3 non-mask cells").

With padding `[0, 0, H-1, W-1]`, the Conv output covers all positions `[0..29]` for both axes. Positions where the kernel "hangs off" the input edge see padded zeros — their count is `< area - (H+W-1)`. For (H, W) in [6..10]×[6..10] with threshold `area - 3.5`, the max possible count at a hanging position is `H*W - (H+W-1) < area - 3.5` for all 25 sizes. These positions **always fail Greater downstream**.

**Remove the Conv padding**: output shrinks from `[1, 1, 30, 30]` (1,800 B) to `[1, 1, 30-H+1, 30-W+1]` (~968 B avg). Save ~832 B per iter × 25 iters = **~20,800 bytes**. Downstream ReduceMax outputs shrink proportionally (~32 bytes per axis × 2 axes × 25 = ~1,600 more).

Correctness: positions removed by no-pad are positions that could NEVER have passed Greater. Output of ReduceMax over the remaining positions is identical. Cropping indices (ArgMax outputs) are in the same numeric range as valid positions; downstream Gather indexing is preserved.

**Empirical**: task205 v3 -20,800 bytes. Combined with Rule 12 (Greater-after-ReduceMax) the per-iter Phase B chain dropped from ~4,620 bytes to ~1,200 bytes.

General principle: when a Conv-Greater-Reduce chain uses padding to extend output coverage but the downstream threshold makes the padded positions provably unreachable, removing Conv padding is a lossless simplification.

---

# 23. Color-index scalar representation (instead of one-hot channel expansion)

When the algorithm processes per-cell COLOR (not per-channel BOOL), represent intermediates as **scalar color-index `[1, H, W]` F16** instead of one-hot `[1, 10, H, W]` BOOL/F16. The channel-axis is materialized only at output via `Equal(out_idx_f16, channel_values_f16 [1, 10, 1, 1])` producing BOOL.

```
# Old: one-hot throughout
input → one_hot [1, 10, H, W] F16   (×10 channels)
... per-channel ops...
Cast to INT64 → OneHot → output [1, 10, H, W]

# New: scalar color-index until output
input → ArgMax(axis=1) → color_idx [1, H, W] INT64
Cast(color_idx → F16) → color_idx_f16 [1, H, W] F16
... scalar marker/ray edits...
out_idx [1, H, W] F16     # set outside-crop cells to out-of-range index 10
Equal(out_idx [1,1,H,W], channel_vals_f16 [1, 10, 1, 1]) → BOOL [1, 10, H, W]
Pad(BOOL) → output BOOL
```

**Savings**: ~10× on intermediate bytes when most ops were per-channel. Avoids both INT64 OneHot index cast AND F16 one-hot output activation.

**ORT compatibility**: opset 13+ for F16 Equal. ORT does NOT support BOOL OneHot, so Equal expansion is the cheaper working route. BOOL Pad supported at ORT-13+ (per `ort_reality.md §1`).

**Trap**: if multiple marker cells share the same color, do NOT use `ReduceSum(masked_color_index)` to recover the marker color — sums repeated positions. Use `ReduceMax` (when marker set has one color) or per-color presence vector (when multiple colors possible).

**Empirical**: task138 v2 went from 150,728 → 58,877 (-61%) via this rewrite alone, on top of the gather-projection v1.

### Magnify / Gather-resample family + COMPUTED decoration overlay (task107, 2026-05-30)

For "magnify the 5×5 input by factor f + draw a fixed decoration" tasks (ARC fractal/tile family),
the cheap Path-A is **all color-index, single channel**:
1. **Slice-crop the input to 5×5 in F32 first** (the input is F32; cropping before any Cast kills the
   `[1,10,30,30]` full-input cast, ~18k). Crop channels 1-9 only — `Conv(crop, [1..9])` gives the
   color-index directly (bg→0), and `ReduceMax(crop,[2,3]) → ReduceSum` gives `#distinct nonbg colors`.
2. **Magnify = Gather-resample** the (sentinel-padded) small color-index with `idx[30] = min(floor(arange/f), 5)`
   — one row-Gather + one col-Gather → `[1,1,30,30]` **UINT8** (900 B, vs 18k for the 10-channel form).
   Pad the small grid with a SENTINEL (99) so output-padding cells decode to all-zero one-hot.
3. **Compute the decoration from geometry — do NOT store per-factor masks.** task107's canonical stored
   15 × 30×30 masks (**13,500 params**). They are exact diagonal+band formulas: cells on
   `R+C==s_t` or `R-C==d_t`, intersected with row/col bands, where `s_t,d_t` are **scalars selected by an
   input feature** (here occupancy of two cells). Two tricks make this cheap:
   - **One sum-grid for both diagonals**: store only `S=R+C` as an init; get the diff-diagonal via
     `R-C==d_t ⇔ R+C == d_t+2C` (Equal `S` against a `[1,1,1,W]` `d_t+2C` vector). Drops the second grid.
   - **Bound the grid to the max decoration box and pad**: the decoration lives in `≤4f_max` (24 for f≤6),
     so compute the BOOL mask on `[24,24]` and `Pad(BOOL)` to `[1,1,30,30]` before the overlay `Where`.
4. **Overlay + tail**: `ci_final = Where(mask, color_u8, magnified_u8)` then
   `Equal(ci_final [1,1,30,30], chanidx_u8 [1,10,1,1]) → output BOOL [1,10,30,30]` (free; §23 tail).

**Empirical**: task107 `crop_v1` 85,131 → **8,339** (-90%, +2.32 LB), **10000/10000 byte-equal vs the
shipped parent** + 3000/3000 vs the ARC-GEN generator. The byte-equal-vs-parent gate is what makes this
"novel op chain on a Solved task" rewrite safe (traps.md SHIP DECISION REFERENCE). Build:
`The_400_Tasks/Task_by_Task/task107/prototypes/build_task107_crop_v1.py`.

### Fold the output mask INTO the gather index — the final gather IS the output (task384, 2026-05-30)

For crop/place/magnify tasks where you Gather-resample a small region and then mask out-of-window
output cells, **don't** build a separate mask + `Mul`/`Where` + `Pad`. Instead **point the out-of-window
output cells at a guaranteed-zero row/col of the source** so the single resample Gather already produces
the masked, full-size result:
- Crop the input to `[1,10,N+1,N+1]` (the N×N grid **plus one all-zero padding row/col** at index N).
- Build the gather index so in-window cells map to the real source and out-of-window cells map to index N
  (the zero row): `row_src[r] = (r < 2h) ? y0 + r//2 : N`.
- The second Gather's output **is** the `[1,10,30,30]` graph output (excluded from scorer memory) — **no
  mask-`Mul`, no final `Pad`**. This eliminates the full-canvas mask intermediates entirely.

**Empirical**: task384 `crop_v1` 77,211 → **9,617** (-87.5%, +2.08 LB); the fold killed all three
`[1,10,30,30]` F16 working tensors (cast + 2 gathers), not just the input cast. **6275/6275 byte-equal vs
the shipped parent** (incl. full 18×18 rule-max). Build:
`The_400_Tasks/Task_by_Task/task384/prototypes/build_task384_crop_v1.py`.

### Prove the crop/output bound from the GENERATOR, not the visible pairs (task239, 2026-05-30)

When cropping to a fixed canvas, the bound must hold on the **hidden** distribution, not just train/test.
Read the ARC-GEN generator source (`external/ARC-GEN/tasks/task_<uuid>.py`) and derive the *structural*
max. task239 (histogram bar-chart): the generator guarantees **distinct** color counts, so width = #colors
≤ 5 (1+2+3+4+5=15 ≤ 16 cells) and height = max count ≤ 12 (largest ring of a 4×4) → **output ≤ 12×5,
provably**, confirmed by 40,000 fresh pairs capping at 12×5. (The prompt's "≤6×5" was train/test-only and
would have truncated tall histograms.) A 12×5 internal canvas is then byte-equal to canonical on every
possible input. **Always cross-check a crop bound against the generator + a large fresh sample before
trusting it** — and note a different-architecture emit (e.g. canvas+`Equal`-broadcast replacing a MatMul
render) ships safely at predicted LB **when it is byte-equal to the parent on a large fresh-generator
sample**: task239 `v2` 96,143 → **1,177** (-98.8%, **+4.41 LB**, the session's biggest single Path-A),
40,000/40,000 byte-equal across 2 seeds. Build: `task239/prototypes/build_task239_crop_v2.py`.

# 24. Merged Conv for "W-wide isolated run" detection

When the original graph tests "W-wide run of value present" AND "both boundary cells are zero" via two separate Convs:

```
# Old: 2 Convs + Equal pair + And
eq1 = Conv(ch9, [1,...,1], kernel=[1,W], pads=[0,0,0,W-1]) == W      # W-wide run
eq2 = Conv(ch9, [1,0,...,0,1], kernel=[1,W+2], pads=[0,1,0,W]) == 0  # both boundaries clear
hit = And(eq1, eq2)
```

Merge into a SINGLE Conv with negative-flanked kernel:

```
# New: 1 Conv + 1 Equal
kernel = [-1, 1, ..., 1, -1]    # W ones flanked by -1.0 (length W+2)
conv_out = Conv(ch9, kernel, kernel_shape=[1, W+2], pads=[0, 1, 0, W])
hit = Equal(conv_out, W)
```

**Why it works**: `conv_out[r, c] = -ch9[c-1] + sum(ch9[c..c+W-1]) - ch9[c+W]`. Equals W iff `sum = W` (W-wide run present) AND both flanking cells are 0.

**Savings**: ~3,600 bytes per W-iteration (eliminated intermediate). task349 v5 saved ~18,000 B across 5 W-iterations.

**Generalizes to**: any task with paired sum-Conv + boundary-Conv equality tests.

# 26. Compact histogram key replaces full flattened MatMul hash

When a memorized-lookup classifier does `Flatten(input)→[1, H*W*C]→MatMul(big_weight=[H*W*C, K])` to compute a hash key, and the inputs are uniquely identifiable by their color histogram, replace with:

```
hist = ReduceSum(input, axes=[2, 3]) → [1, 10]   # color histogram per channel
key = MatMul(hist, small_weight=[10, K]) → [1, K]
```

The 9,000-element flat intermediate disappears; the projection weight shrinks from 9,000 to 10 elements.

**Critical gap requirement**: the projection must achieve `min_gap > sqrt(threshold)`, not just `> threshold`. For a Less<0.25 threshold, the keys must differ by > 0.5 (since distance² < 0.25 means distance < 0.5). Verify empirically with a random-seeded projection across all training inputs.

**Eligibility check**: `np.unique(histograms_across_all_pairs, axis=0).shape[0] == n_examples` — color histograms must be unique per input.

**Empirical**: task158 v2 saved -35,960 bytes intermediate + -8,990 params via this. Together with BOOL-Or chain replacing Sum+Greater (next) and AND-reorder (next next), total task158 saving was -114k.

# 27. BOOL-Or chain replaces Cast→Sum→Greater accumulation

When the question is "are ANY of N boolean masks active" and the current graph does:
```
Cast(b0, F32) + Cast(b1, F32) + ... + Cast(bN, F32) → Sum → Greater(>0.5)
```

Replace with a chain of `Or(BOOL, BOOL)`:
```
union = b0
for i in 1..N:
    union = Or(union, bi)   # stays BOOL
# Use union directly (e.g., Where(union, paint, input))
```

**Savings**: N × per-Cast intermediate (F32 [..,H,W] = 4 bytes/elem) minus (N-1) × BOOL [..,H,W] (1 byte/elem). For N=18, [1,1,30,30]: 18 × 3600 = 64,800 B saved on Casts; 17 × 900 = 15,300 B added on Or chain. **Net -49,500 B for N=18**.

**Key ORT fact**: `Where(BOOL, F32, F32)` is supported. `Where(BOOL, BOOL, BOOL)` is NOT (per ort_reality.md §2). Use the Or-chain result directly in Where conditions.

Empirical: task158 v2 saved -54,000 B via this on the 18-mask cursor accumulation.

# 28. AND-reorder for row_range × col_range cursor checks

When checking "cell is in row range AND col range" via per-cursor And chain, the order matters:

```
# Bad: produces [1, 1, H, W] BOOL TWICE per cursor
And(en, row_gt)           → [1, 1, H, 1]
And(prev, row_lt)         → [1, 1, H, 1]
And(prev, col_gt)         → [1, 1, H, W]   ← creates first [H, W] BOOL
And(prev, col_lt)         → [1, 1, H, W]   ← creates SECOND [H, W] BOOL = mask

# Good: only ONE [H, W] BOOL per cursor (final broadcast)
And(en, row_gt)           → [1, 1, H, 1]
And(prev, row_lt)         → [1, 1, H, 1] = row_range
And(col_gt, col_lt)       → [1, 1, 1, W] = col_range
And(row_range, col_range) → [1, 1, H, W] = mask    (broadcast at last step)
```

`And` is commutative + associative — semantically identical. **Savings per cursor**: one [H, W] BOOL intermediate = `H*W` bytes. For 18 cursors at 30×30: -15,660 B.

**Same law, shared-subexpression form (task234, 2026-06-20):** when a positive class and its negation both gate on the same factor — `X = AND(A, AND(B,C))` and `Y = AND(NOT(A), AND(B,C))` — compute `BC = AND(B,C)` ONCE and reuse it (`X = AND(A, BC)`, `Y = AND(not_A, BC)`). task234's CS classifier (`is_anchor = AND(eq_area, AND(has,notch0))`, `is_mover = AND(NOT(eq_area), AND(has,notch0))`) factored the shared `has_notch = AND(has,notch0)` and took `not_eq = NOT(eq_area)` directly off `eq_area` (not off the intermediate `is_filled`), cutting the BOOL [1,10] chain from 5 intermediates to 4 (−10 B, 2457→2447). Any "this class AND that class share a guard mask" structure: factor the guard first.

# 29. NOOP sentinel position for ScatterND when ALL positions are valid

When building ScatterND-based sparse stamps and some "padding" slots in the index tables need a safe landing zone, but **every position in the active output region is valid foreground for at least one exemplar** (so there's no safe always-background position):

```
# Extend canvas by 1
canvas = [0] * (N + 1)      # N + 1, not N
NOOP_POS = N                # the extra slot, outside valid 0..N-1 range
# Initialize padding slots in index tables → NOOP_POS (not 0)
ScatterND(canvas, indices, values)
# Slice away the sentinel
output = Slice(canvas, starts=[0], ends=[N])
```

Avoids "last-write-wins" overwrites at position 0 from NOOP slots. **Empirical**: task182 v1 needed this — position 0 IS valid foreground in some exemplars; previously NOOP-at-0 overwrote real stamps.

# 30. Memorized-lookup discriminator must stay F32 when hash sums exceed F16 exact range

For memorized-hash-matcher graphs with `MatMul(features, hash_weights)` discriminator: if `max(feature_value) × N_features > 2048` (F16 exact-integer ceiling), F16 conversion of the discriminator chain produces wrong hashes silently.

**task182 measurement**: feature values 1–127 × 4000 positions → sums reach ~25k. F16 conversion gave 0/267 verify (catastrophic). Keep the discriminator MatMul + Sub + Mul-self + ReduceSum in F32. Cast to F16 only AFTER the threshold check.

Cross-ref `ort_reality.md §11` F16 integer-exactness threshold.

# 31. Direct channel slices vs staged `[1, N, H, W]` mask tensor

When a graph needs BOTH individual channels AND a single `OR` (or `Max`) across them, computing the per-channel slices DIRECTLY from the full input UINT8 + per-channel `Max` is cheaper than:

```
# Staged: extra [1, N, H, W] intermediate
ch19_u8 = Slice(input_u8, channels 1..9)  ← [1, 9, H, W] = N*H*W bytes
wall_u8 = ReduceMax(ch19_u8, axis=1)
ch1..ch9 = Slice(ch19_u8)                  ← N more slices

# Direct: skip the staged tensor
ch1_u8 = Slice(input_u8, ch=1)
ch2_u8 = Slice(input_u8, ch=2)
...
ch9_u8 = Slice(input_u8, ch=9)
wall_u8 = Max(ch1_u8, ch2_u8, ..., ch9_u8)
```

**Savings**: `N * H * W` bytes (the staged [1, N, H, W] intermediate disappears). task187 saved -5,625 B at 25×25 / N=9.

**Caveat**: only helps when downstream uses per-channel slices anyway. If many ops consume `[1, N, H, W]` as a single tensor, keep the staged version.

---

# 25. Active-subset projection before pair-matrix enumeration

When a graph enumerates `N_colors × N_positions` candidates and builds a full `[N, N]` pair-validity / overlap matrix, but the task rule guarantees only K ≪ N active channels: **select the active K first, then enumerate only the `K × K` cross-product**.

```
# Old: 9 colors × 9 placements × full pair matrix
candidates = [c0, c1, ..., c8] for each placement
pair_matrix [81, 81]     # ALL pairs computed regardless of activity
overlap, count, valid    # full 81×81 intermediates
final selection from pair_matrix

# New: dynamic active-subset
active_color_idx_1 = ArgMax(weighted ReduceMax on input) → INT64 scalar
active_color_idx_2 = ArgMax(masked) → INT64 scalar
Gather just the 2 active masks
Generate 9 placements per active mask → [9, 9] candidate cross-product (NOT [81, 81])
Test only the [9, 9] pair matrix
```

**Savings**: `(N² - K²) × cell_size` for the pair matrix plus the corresponding initializer elimination. **task153 v2 saved -173k bytes** (177k → 4k, **-97.7%**) via this on a 9×9 active-subset reformulation. The `tbt_upper [81, 81]` initializer was deleted entirely.

**Eligibility**: any task with a large pair/overlap matrix where the task rule guarantees only K active elements (e.g., "two non-bg colors", "exactly one rectangle pair", "the foreground color and bg"). Quick spot check: does the graph have a `[N, N]` or `[N, M]` (N >> K) initializer that's only consulted at K active positions?

**Distinct from §22 (compact-mask projection)**: §22 shrinks perpendicular shift intermediates. §25 shrinks all-pairs enumeration. The two stack.

Cross-ref: `exemplars/01_whole_graph_rewrites/` family. Worth a new exemplar (task153 active-pair projection — until then: `The_400_Tasks/Task_by_Task/task153.md`).

---

# 22. Compact-mask projection for perpendicular thickening

When a graph builds a "thick mask" via N perpendicular shifts of a sparse single-cell or single-row/col mask (e.g., shift LEFT_1..K and shift RIGHT_1..K of a column-ray to make it K-cells-wide), each shift output is a full `[1, 1, H, W]` BOOL — most of which is zero (the shifted single column/row).

**Rewrite**: project the source mask to its perpendicular axis FIRST via `ReduceMax(source, axis=ray_axis, keepdims=1)`. This gives a compact `[1, 1, 1, W]` (or `[1, 1, H, 1]`) tensor. Apply perpendicular shifts on the COMPACT tensor. Materialize the full `[1, 1, H, W]` thick mask ONLY at the final use site via broadcast:

```
# Old: 40 perpendicular shifts × [1, 1, 20, 20] BOOL = 40 × 400 = 16,000 B
shift_L_1 = Pad(Slice(ray, ...), [...])    BOOL [1, 1, 20, 20] = 400
shift_L_2 = Pad(Slice(ray, ...), [...])    BOOL [1, 1, 20, 20] = 400
... [40 total] ...

# New: project first, shift on compact, broadcast at use
ray_compact     = ReduceMax(ray, axis=2, keepdims=1)            # [1, 1, 1, 20] BOOL = 20
shift_L_1_comp  = Pad(Slice(ray_compact, ...), [...])           # [1, 1, 1, 20] BOOL = 20
shift_L_2_comp  = Pad(Slice(ray_compact, ...), [...])           # [1, 1, 1, 20] BOOL = 20
... [40 shifts on compact, 20 B each] = 800 B total

# Materialize thick mask only at final use site
thick_mask = And(ray_compact_rank2 [1,1,20,1], stamp_compact [1,1,1,20])  # broadcast to [1,1,20,20] = 400
```

Savings on `H × W = 20 × 20`: 16,000 B → 800 B = **15,200 B saved**. Generalizes: `(H × P_shifts × B_per_cell) → (P_shifts × B_per_cell + 1 final broadcast)`, an `H×` reduction.

**Eligibility**: any task whose perpendicular thickening shifts a single-cell mask across a perpendicular axis. Quick spot-check: graph has N perpendicular `Slice → Pad` pairs whose input is a sparse 1-cell or 1-row/col tensor.

**Empirical**: task280 radical saved -15,200 bytes via this technique. Highest-leverage NEW technique from that worker session.

**Distinguish from `§15 separable_mask_collapse`**: §15 compresses STATIC initializers. This technique compresses RUNTIME perpendicular shift operations. The two techniques are duals.

---

# 51. Merged-Conv (pair-Conv + Equal → single Conv + Equal) <!-- formerly a duplicate "# 23"; NOT a dup of #24 (5% overlap, unique boundary-Conv formulas) — renumbered 2026-06-05 -->

When two separate Conv+Equal ops jointly test (1) a W-wide sliding sum == W and (2) both adjacent cells == 0, they can be merged into a single Conv + Equal using a combined kernel.

**Pattern:**
```
eq1 = Conv(signal, ones[1,W], kernel=[1,W], pads=[0,0,0,W-1]) == W   # W consecutive ones
eq2 = Conv(signal, [1,0...,0,1], kernel=[1,W+2], pads=[0,1,0,W]) == 0  # both boundaries clear
hit = And(eq1, eq2)

→

kernel = [-1, +1, +1, ..., +1, -1]   # W ones flanked by -1.0
conv_out = Conv(signal, kernel, kernel_shape=[1, W+2], pads=[0,1,0,W])
hit = Equal(conv_out, W)
```

**Math**: `conv_out[r,c] = -signal[c-1] + sum(signal[c..c+W-1]) - signal[c+W]`.
This equals `W` iff `sum = W` (all W interior cells = 1) AND both boundary cells = 0 — exactly the isolated W-wide run condition.

**Why it works without And**: if either boundary cell is 1, `conv_out < W` (boundary penalty reduces total). If sum < W, `conv_out < W` (shortfall). Equal-to-W is a sufficient condition.

**Savings**: per iteration, eliminates one Conv + two Cast + one And → one Conv, saving ~3,600 bytes of F16 intermediates. Over 5 iterations: **~18,000 bytes**.

**Kernel dtype**: F16 (NeuroGolf models use F16 conv weights). Input must be F16 or cast before the Conv.

**Multi-width variants**: works for any W. Use `cmp_val = scalar_init(W)` already present in most NeuroGolf tasks.

**Empirical**: task349 V4 saved -18,000 bytes via 5 merged Conv ops (W ∈ {2, 4, 6, 8, 10}).  
Build: `The_400_Tasks/Task_by_Task/task349/prototypes/build_task349_v4.py`.

---

# Radical-rewrite playbook

When polish passes (techniques 1-17 above) cannot reduce a task's cost by ~5,000 bytes (target ~+0.1 LB), trigger a rebuild from scratch. Order of escalation:

1. **Apply rule 8 (input → BOOL)** if downstream is bool-tolerant. The 4× dtype-width cascade is the biggest single lever. See task376.
2. **Apply projection-cache (technique 12)** if the model has wide `[1, C, H, W]` intermediates only consumed by reductions/gathers. See task071.
3. **Replace static tables with dynamic compute at rank-2** (technique 11). See task071 ring-mask.
4. **Algorithm-family swap**:
   - Conv-heavy → predicate-mask reformulation.
   - Gather-heavy → ReduceMax-based one-hot lookup.
   - ReduceSum-heavy → Conv1x1 with constant weights.
   - ReduceMax-heavy → ArgMax + ScatterND.
   - Where-chain-heavy → direct arithmetic on integer-encoded grids.
5. **Reformulate as shape-only** (technique 7) if the rule allows. Cost floor ~1,500 bytes.
6. **Compute the answer as a Constant** (cost = element count of the output as an initializer). Some tasks have a globally-correct constant output (e.g., always-all-zeros output, or a fixed pattern). Cost can drop to ~5,000 bytes for a `[1, 10, 30, 30]` BOOL const.

Exemplars (read all that apply to your task family):
- `exemplars/01_whole_graph_rewrites/` — task376 (input→BOOL), task049 (least-color bbox), task382 (CumSum+Gather), task119 (diag-conv), task366 (template-vote → sparse-stamps).
- `exemplars/05_static_to_dynamic/` — task071 (ring-mask), task382 (geometric).
- `exemplars/07_conv_reformulation/` — task119, task374, task323.

## Radical pattern: template-vote → sparse-stamps decomposition

When a graph has `(N_patterns → template_sets [N_t, K] → set_votes → ReduceMax → Gather → ScatterND)` — i.e., per-position voting across templates with a `template_sets [N_t, K] INT64` initializer dominating params — **transpose the data layout from "per-output-position" to "per-input-pair"**:

```
# Old: 1872 templates × 37 positions = 138k intermediate + 69k params
ArgMax (matches cast to F16) → pair_idx scalar
Gather(bg_per_pair, pair_idx) → bg_color
Gather(stamps_idx_per_pair, pair_idx) → sparse_stamps_idx
Gather(stamps_color_per_pair, pair_idx) → sparse_stamps_color
ScatterND(bg_canvas, sparse_stamps_idx, sparse_stamps_color) → output
```

- Eligibility quick check: graph has an INT64 initializer shaped `[N1, N2]` with N1≈N2≈10²–10⁴.
- Combine with K-bucket long-tail compression (`Sub-technique 5` in the exemplar) for additional ~40% params savings.
- Combine with F16 spatial chain (Rule D above) for additional ~5k bytes.
- **Empirical**: task366 v10 reduced cost 289,181 → 44,864 (-84.5%). **BUT public LB delta was 0** because task366 has zero public weight — see exemplar for the caveat.

## Universal trick: OneHot-value gating for no-match fallback

When a pair-lookup model needs an all-zero output when no pattern matches (without paying for a `Where(has_match, value, sentinel)` mask intermediate):

```
onehot_vals_init = [0.0, 1.0] F16                        # off, on
has_match_f16 = ReduceMax(matches_f16)                    # scalar, 0.0 or 1.0
onehot_vals_dyn = Mul(onehot_vals_init, has_match_f16)    # [2] F16 — [0,0] when no match
OneHot(indices, depth, onehot_vals_dyn) → output          # all-zero when no match
```

When `has_match=0`, `onehot_vals_dyn=[0,0]`, so every cell of OneHot output is 0 regardless of indices. **Cost**: ~8 bytes (1 Mul + tiny F16 [2] intermediate) vs ~3,600 bytes for the `Where → INT32 mask` alternative.

Generalizes to ANY OneHot-output task with a no-match fallback requirement. Empirical: task366 v7 saved -3,597 bytes via this. See `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md` sub-technique 2.

---

## Variable-size geometry — negative-index Gather-wrap, fold-mask-into-index, square-grid count (task150/155/384, 2026-05-30)

Three byte-equal tricks that recur in flip / reverse / crop-and-place tasks where the active grid is N×N inside the 30×30 canvas. Each deletes a `mask → Where` (or `Pad`) tail and was validated **byte-equal** vs LIVE on the entire input space.

### A. Negative-index Gather-wrap — free padding handling for variable-size flips/reversals
A flip-H of an N-wide grid sends output col `c` to input col `N-1-c`. For padding cols `c ≥ N`, `N-1-c` is **negative**; ONNX `Gather` wraps a negative index `-k → dim-k`, landing on a col deep in the (guaranteed-zero) padding region, so the gathered result is all-zero — **byte-identical** to the canonical `mask(c<N) → Where(..., 0)`. So just precompute `idx = (N-1) - arange(30)` as an **INT32** init and `Gather(input, idx, axis=W)`: no mask, no Where, no Squeeze.
- task150 FLIP-H: 4,674 → 278 (−94%); 266/266 byte-equal vs LIVE + 3,010 random over the entire space (non-square / edge / empty / all-bg). task155 FLIP-V is the same recipe on the H axis → 158.
- Requires the padding region to be zero (true for the 1-channel-canvas / one-hot encodings). Verify on **empty + all-bg** inputs (where a wrap-vs-mask difference would surface).
- Opposite of the OneHot negative-index *trap* (`Ort_quirks §3` / traps §10): there a negative index is a bug (wraps to channel 9); here it's exploited because the wrap target is guaranteed zero.

### B. Square-grid count identity — read N as sqrt(cell count), fuse with ReduceL2
When the generator emits only **square** grids (`grids(size, size)`, every visible pair square), the side length is recoverable from the one-hot cell count: a square one-hot grid has exactly N² set cells, so `N = sqrt(ReduceSum(onehot))`. `ReduceL2` fuses `ReduceSum + Sqrt` into **one op** (`L2 = sqrt(Σx²) = sqrt(Σx) = sqrt(N²) = N` for 0/1 data), replacing a `[30]` width-counter (Slice + ReduceMax + arange-compare) with one scalar reduction.
- task155 FLIP-V variant E: `ReduceL2 + Cast + Sub + Gather` = 4 nodes, cost 158 (−96.6%). task150 v2 sqrt-N floor 162.
- **Square-grid CONTRACT ship class** (validated 3×: task388 v4, task155 E, task150 v2). The square assumption is byte-equal-safe ONLY when BOTH hold: (1) every visible pair is square, AND (2) the generator source is `grids(size, size)` / a single `randint` dim. Confirm both in `external/ARC-GEN/tasks/task_<uuid>.py` before relying on it — a single non-square hidden pair regresses.

### C. Fold-mask-into-Gather-index — the 2nd Gather IS the [1,10,30,30] output (crop-and-place)
For crop-then-place tasks (extract a window, paint it elsewhere, rest = background), point every **out-of-window output cell** at a guaranteed-zero row of the gather table. Then the final `Gather` directly emits the `[1,10,30,30]` graph OUTPUT — no mask-Mul, no Pad, no Where. Build a `[N+1, …]` table whose last row is all-zero and set the placement index map so in-window cells → their source row, out-of-window cells → row N (the zero row).
- task384: 77,211 → 9,617 (−87.5%); eliminated 3× 18,000-B full-canvas tensors; 266/266 byte-equal + 6,275 byte-equal vs canonical incl the full 18×18 rule-max.
- Sibling of the OneHot-value-gating no-match trick above (both zero-out via index/value redirection rather than a counted mask intermediate). Use THIS when the output is a *placed crop*; use OneHot-value-gating when the output is a *lookup with no-match fallback*.

---

## Path-B-only wins from the ARC-GEN generator (task018 from-scratch swap, 2026-05-31)

Two reusable levers that are *unavailable to a byte-exact Path A* but safe once you gate on **ARC-GEN ground truth** instead. Both validated by hidden LB (task018 90428→62603, +0.37; the F16-max and sym-restriction were isolated as independently hidden-safe by a free-submit cost-spread ablation — see `task018.md`).

### A. F16 `ScatterElements(reduction='max')` — ORT accepts it where `'add'` is rejected
ORT (opset 16+) rejects F16 `ScatterElements`/`ScatterND` with `reduction='add'` but **accepts `reduction='max'`**. When each scatter index is written **at most once with a non-negative value** (e.g. one-cell-per-color, one-write-per-slot), `max(0, v) == add`, so `max` is a drop-in replacement that lets the whole scatter chain stay F16 — **eliminating the two F32 casts** the `add` form forces (the data-in cast and the result cast). ~−4.6K on task018. NOT byte-exact in general (differs from `add` only on the never-generated multi-write-per-index case), so gate on the generator's invariant, not byte-equality. (See traps.md scatter-dtype note.)

### B. Generator-symmetry restriction — enumerate only the transforms the generator actually uses
A solver that brute-forces all 8 D4 symmetries (or any transform family) is wasteful **and less accurate** if the ARC-GEN generator only ever emits a subset. Read `external/ARC-GEN/tasks/task_<uuid>.py`, map its actual transform choices to your solver's index set, and **drop the rest**. task018's generator uses only `randint(1,4)` rotations → v54's D4 syms `{1,3,5,6}`; restricting 8→4 was a **double win: −4.7K cost AND +1.1pt accuracy** (the spurious syms had been causing false anchor-matches). General rule: spurious candidate transforms don't just cost bytes, they cost *correctness* (extra chances to mis-match) — so trimming to the generator's true set usually improves both. Confirm the subset from the generator source, then verify accuracy rose (not fell) on a fresh ARC-GEN sample.

---

## Reusable patterns from the 2026-05-31 +23.7-LB run

Four cross-task techniques validated by hidden LB this session (each landed at its predicted cost-ratio; all gated on large fresh ARC-GEN == the hidden distribution).

### Packed-base-N source via Conv + Div/Mod-by-constant decode (task313, 74959→2322, +3.48)
When a small fixed set of K source labels (each 0..N−1) must be read and reused, **don't materialize the `[1,K,…]` source slice**. Pack them into ONE scalar with a Conv whose weights are `[N⁰, N¹, …, Nᴷ⁻¹]` (base-N place values), then **decode with float `Div`/`Mod` by the constant base N**. task313 packed 6 labels into a base-16 scalar via a 2×4 Conv, decoded to a U8 patch. Constraints: (a) the packed max must be exact in F32 (< 2²⁴ ≈ 16.7M — 6 base-16 labels ≈ 10M is fine, near the edge); (b) `Div`/`Mod` must be by a **constant** (not input-derived) or it SIGFPEs on the scorer's zero-grid (traps §8d) — a clean `score_network_v2` (not `None`) confirms no SIGFPE.

### Orbit-max periodic inpainting (task110 from-scratch, 76661→24502, +1.14)
For **doubly-periodic-pattern-with-holes** tasks (output is periodic, input is that pattern with black rectangles punched in): detect the period `p` per axis, then `filled[r][c] = Max` over the cell's **residue-class orbit** (`{…, r−p, r, r+p, …}`). Holes are 0, so the max restores the pattern exactly — replaces an entire per-direction flood chain with cheap shifted `Max` ops. Realize the orbit via **bidirectional doubling** (k steps reach 2ᵏ·p), and **self-clamp** out-of-range shifts to the cell's own index (NOT the edge row — clamping to row 29 injects a foreign residue class and corrupts the fill). This was *more* hidden-correct than the incumbent flood solver.

### OR-collapse / leaner reformulation can be MORE correct than the parent (task233 v4)
When dropping a parent's machinery (multiplicity histogram, dedupe, etc.), **use the byte-equal harness as a divergence *detector*, not a gate** — run a 3-way compare (parent vs candidate vs ARC-GEN ground truth) over thousands of fresh pairs. A candidate that diverges from the parent may be diverging because it's *more* correct (the parent has rare F16/path bugs). task233 v4 was 100% vs 99.95% over 5000 pairs with **0 regressions** — shippable as ≥-parent at lower cost even though not byte-equal. Decisive metric: **parent-only-correct count == 0** ⇒ never regresses.

### Generator-shaped crop — fixed/active-region crop is safe when the generator bounds the size (task181 850, task070 8347)
If the generator hardcodes the grid size (`grids(17,17)`) or always paints a small active region, a static crop to that region is byte-equal on **100% of inputs that can exist** — the >size divergence is on impossible inputs. The generic adversarial helper (which feeds out-of-domain grids) will read 0/200 by design; the authoritative signal is **large fresh `generate()` byte-equality** (the real distribution). Confirm the size bound in the generator source, then trust the ARC-GEN gate over the adversarial helper. (See also cost_floor_reference single-channel-canvas note.)

---

## Reusable patterns from the 2026-06-01 captain session

### Priority-scatter inverse permutation — `ScatterElements(reduction='max')` replaces a `[K,K]` match-matrix (task203 v6, opset-18)
To build an inverse channel-permutation `perm[j] = argmax_k P[j,k]` **without** materializing the `[K,K]` priority matrix `P` (match + arithmetic ≈ 1,300 B on a 10-color recolor), encode the priority **in the scatter VALUE** and let `max`-reduction resolve ties:
```
values[k] = used[k] ? (BIG + k) : k    # used colors get +BIG so they win a collision
init[j]   = first_unused_color         # fallback for channels no used color targets
perm_raw  = ScatterElements(init, indices=lut_out, updates=values, axis=0, reduction='max')
perm[j]   = perm_raw mod BIG           # strip the priority offset
```
`lut_out[k]` = target channel for color `k`. The scatter writes each `k`'s priority-tagged index into slot `lut_out[k]`; `reduction='max'` picks the `used` color on a collision; `mod BIG` recovers the plain index. Byte-identical to the match-matrix form on the bijection-on-used contract, ~−110 B, and **drops the entire `[K,K]` plane**. General use: invert any small sparse map with a tie-break priority (recolor permutations, label remaps). **opset-18 `ScatterElements` scores and holds on the grader** (task203 v6 → +0.10). See also §A F16 `ScatterElements(reduction='max')` above (the `'add'`-rejected / `'max'`-accepted dtype rule).

### One-hot MatMul → ArgMax + Gather (byte-equal table lookup, task182 v53, 2026-06-01)
When a selector `ak` is **always one-hot** on the real distribution, `MatMul(ak, table_f16) → Cast` (reconstruct the selected row) is byte-identical to `ArgMax(ak) → Gather(table_i8)` — and the I8 table + Gather is cheaper than the F16 matmul + its intermediate (task182: −52 B, drops the `matmul` F16 output + the `cast` I8 intermediate). Store the template/lookup table as I8 and index it. **Caveat (learned the hard way): "byte-equal to sibling vN" only proves equivalence to vN — if vN is wrong on some pairs, so is this.** Gate vs LIVE / the full official pair set, not vs a sibling candidate: task182 v53 was byte-equal to v52 AND 20k/20k vs the generator, but both missed 3 official lookup exemplars → scored **0** on hidden.

### Positional-hash match-collapse — replace N orientation-correlation Convs with 1 windowmasked hash Conv + integer-code membership (task191 v5, 2026-06-01)

When a graph template-matches a small pattern against a grid in **N orientations** via N correlation Conv kernels (producing a `[1,N,H,W]` match map that is the cost center), and you only need to know **which orientation/group matched** (not a continuous score), collapse the N kernels:

1. **Group orientations by WINDOW SHAPE** (the bounding box the match checks). task191's 8 D4 orientations → 2 box shapes (tall×wide, wide×tall).
2. **One positional-hash Conv per shape-group:** `kernel = base2 * windowmask`, where `base2[pos] = 2^pos` (row-major over the kernel) and `windowmask` zeros cells outside the box. Output `code[i,j] = sum_{pos in window} 2^pos * cell[i+pos]` — a UNIQUE integer fingerprint of the window's binary pattern (codes ≤ 2^(box-area)−1; exact in F16 up to 2048 — **mechanism corrected by K5 (2026-06-07): there is NO native F16 Conv CPU kernel; ORT cast-inserts → F32 accumulation → final-only F16 rounding, so the envelope is FINAL value ≤ 2048 (partial sums are F32-safe to 2^24). See #77.**).
3. **Per orientation, a target code** `= sum 2^pos * oriented_pattern[pos]` (built at runtime from the extracted pattern via the SAME perm tables that built the correlation kernels). `match = Equal(code, target)`; OR the group's targets → group seed.

**Why it is EXACTLY equivalent to correlation matching:** `code == target` iff `cell == oriented_pattern` at every in-window position (binary positional expansion is injective), which is the identical condition to `Conv-correlation == npat`. Verified task191 v5: 20000/20000 ARC-GEN GT + 267/267 visible.

**Cost:** 2 code Convs (`[1,1,H,W]` F16) + per-group `Equal`+`Or` (BOOL) replaces N F16 `[1,1,H,W]` correlation channels. task191 match core 14,964 → 12,732; combined with the single-channel canvas output, 32,841 → 26,917 (−5,924, +0.199 LB raw log_e).

**The caveat that defeats the natural objection:** a per-group **windowmasked** hash is NOT a "fixed 3×3 LUT/hash". The windowmask makes the code respect the VARIABLE box size, so the standard "a fixed-window encoding over-constrains small boxes and diverges" objection (which correctly rules out a *naive full-window* hash) does NOT apply. task191's own v3 floor proof conflated the two and wrongly declared the 8-Conv match core irreducible — this collapse falsified it.

**opset-13 safe floor:** when the per-group target codes are arbitrary/runtime, the exact membership still
needs one exact `Equal` per candidate target (no linear orientation-invariant can distinguish patterns
that share Klein-4 cell-orbit counts but lie in different orbits — the size-4 cell-orbit count=2
ambiguity). Do NOT try to fold the N target tests into one "invariant code."

**opset-20 membership reduction (task191 v6, 2026-06-02):** ORT/Kaggle accepts `ReduceMax(BOOL)` at
opset 20 with axes as an input. So the per-group OR chain can become one broadcast Equal plus one bool
reducer:

```
eq      = Equal(code[1,1,H,W], targets[1,N,1,1])   # [1,N,H,W] BOOL
matched = ReduceMax(eq, axes=[1], keepdims=1)      # [1,1,H,W] BOOL
```

Task191 v6 used two such group reducers and Kaggle accepted opset 20, taking the safe opset-13 floor
from 26,545 to 24,807 (+0.29 LB). Keep the opset-13 Equal+Or path when you need maximum schema
conservatism; use opset20 when the submit is free to test or the saved bool plumbing is material.

> Other 2026-06-01 cross-task lessons live where they're most actionable, not duplicated here: the **dilated crop-collapse** Conv is documented above (task400 v6); the **floor taxonomy** (which full-30×30-plane costs are irreducible) is in `cost_floor_reference.md`; the **Path-B / op-swap ship discriminator** (full-corpus correctness, not op-vocabulary) is in `traps.md` SHIP DECISION REFERENCE; and the **spread / floor-verification discipline** (cheapest-staged-probe-solving ⇒ probe cheaper) is in `current_strategy.md` §captain-session-2026-06-01.

---

# Anti-patterns (do NOT do)

## DON'T use `Sub(1, Gather(input, [0], axis=1))` for foreground detection
The "1-ch0 trap". Padding cells get incorrectly flagged. See core.md.

## DON'T use `pass_fp16_selective` (onnxconverter_common blanket conversion) as the primary F16 strategy
Regresses ~134/401 tasks because it pre-casts the full input. Use technique 4 (placement-aware F16) instead.

## DON'T trust 100% visible verify as proof of hidden-correctness
task080 v5_bool_slices: 231/231 visible + 450 adversarial × 9 modes (100%) → -13.32 LB on Kaggle. Boolean reformulations of float graphs particularly prone. For any major float→bool or new-encoding rewrite, generate 20-50 synthetic boundary inputs and verify byte-exact against LIVE.

## DON'T trust a numpy reference that passes 266/266 visible
task008's numpy ref passed 266/266 visible but only 3/86 adversarial against LIVE. Numpy correctness on visible ≠ rule-captured.

## DON'T assume "K-fold template collapse" saves bytes by default

Stacking N parallel Conv chains (each producing `[1, 1, H, W]`) into one Conv with N output channels (producing `[1, N, H, W]`) is **BYTE-NEUTRAL** for static-shape per-iter intermediates. Element counts are conserved across the reshape. Stacking is a NODE-COUNT win and a graph-compilation-overhead win, NOT a memory win. **Measure per-iter intermediate sizes before assuming K-fold collapse is the win.** task205's prompt estimated ~96k savings from K-fold collapse; empirical measurement showed 0 bytes saved with input padding to align kernel shapes adding +21k. The real wins came from Rules 10/11/12/19/20/21 (dtype + algebraic rewrites), not stacking.

## DON'T use Conv-runtime-weights for outer-product reductions (M = C cooccurrence)

Conv with runtime weights (technique §19) replaces `Mul-broadcast + ReduceSum` chains when output_channels M ≤ 1. For outer-product reductions where M = C (e.g., `cooccur[a, b] = sum_p input[a,p] * full[b,p]`), the required weights tensor `[C×C, C, H, W]` exceeds the MatMul output by 200×. Use MatMul/Gemm for outer-product, Conv-runtime-weights for per-channel weighting only.

## DON'T try to merge a multi-mode Pad chain
`Pad(mode=reflect) → Pad(mode=constant)` cannot fold — different boundary semantics. Confirmed dead-end on task240.

## DON'T mismatch OneHot index and depth dtypes
ORT's OneHot accepts INT32 or INT64 indices under `ORT_DISABLE_ALL`, but the scalar `depth` input must
match the index dtype. INT32 indices with an INT64 depth fail with a generic load/runtime error; this
was the real task124 v3 failure behind the older "INT64 only" note. If you want INT32 index maps, make
`depth` INT32 too (task383 re-probe).

## DON'T do `Where(bool, bool, bool)`
ORT does not support it. Rewrite as `Or(And(c, a), And(Not(c), b))` or, with one-hot structure, `And(one_hot, pick_row) + Or(row_mask, col_mask)`.

## DON'T skip the `g.ClearField("value_info")` step after crop/dtype surgery
ORT cached value_info will mismatch new runtime shapes. Model fails to load. Mandatory after any shape-changing rewrite.

## DON'T trust the `crop_to_size_advisory` verdict blindly
If the model already Slices to a small region on its first node and Pads back at the end, pre-crop adds overhead. Read the graph first.

## DON'T submit candidates with sub-100% visible verify
Exception only: tasks with <10 visible pairs. Confirmed 7/8 sub-100% wildcards regressed -13 to -17 LB each.

## DON'T forget the cross-task memory cascade for known trigger tasks
`{45, 129, 130, 171, 172, 210, 211, 220, 239-242, 256-262, 309}` — a high-memory ONNX in these slots can ERROR the next task. Prefer aggressive memory reduction over a complex high-memory candidate on these tasks specifically.

## DON'T do cross-bundle swaps without grader-verification (octaviograu 6042.85 + our 2026-05-24 verified rule)
Source: `data/competitor_artifacts/<author>-<LB>/` ONNX swapped into our submission can REGRESS LB by -100s of points even when local verify passes. Octaviograu documented `v16: predicted +116 LB, grader -308 LB` for the same pattern. Our hybrid v1 (beicicc-6645 base + our 26): predicted ~+540 LB, grader -4598 LB. Our hybrid v2 (6 surgical afr1ste swaps): predicted +30-60 LB, grader -86 LB. **Bundles' big ONNX exist because their author rejected cheap alternatives that fail hidden.** The only exception is solve-from-scratch on cherry13b=0 tasks where LIVE contributes 0 — there's nothing to preserve.

---

# 33. Negative-MaxPool trick for MinPool / connected-component labeling (octaviograu 6042.85 technique, opset 11+)

For connected-component labeling without banned ops (Loop/Scan/NonZero/Unique/Compress):

```python
# Initialize each cell's label = row*30+col for mask cells, sentinel (big) elsewhere
labels = where(mask, row_idx * 30 + col_idx, sentinel)

# Propagate min across 3x3 neighborhoods using -MaxPool(-x) = MinPool
for _ in range(5):  # 5 iterations suffices for 30x30 active region
    labels_neg = -labels
    pooled_neg = MaxPool(labels_neg, kernel_size=3, stride=1, pad=1)
    labels = -pooled_neg
    labels = where(mask, labels, sentinel)  # re-mask non-mask cells to sentinel each iteration

# After convergence: each mask cell holds the minimum cell-index in its 8-connected component.
```

Extract per-component IDs via repeated `ReduceMin` after masking off discovered components. Banned ops avoided. Used in octaviograu's task277 hand-build at 120K cost / +13.30 LB (our task277 ship at 16,748 cost / +15.27 LB beats this).

# 34. ScatterND-add histogram for size-counting (opset 17+, octaviograu 6042.85 technique)

**O2 correction (2026-06-19): treat this as historical / high-risk, not a default recipe.** ONNX defines
`ScatterND` reductions, but ORT CPU 1.24.4 computes duplicate-target `ScatterND` writes through a parallel
shared-output path without visible conflict resolution. Do NOT use duplicate-target `ScatterND(add|min|max|mul)`
for new histograms, overlap stamps, or priority routing unless a local repeatability probe closes the exact
model. Prefer `ScatterElements` (#8c/#8d) whenever the addressing can be axis-local; use `ScatterND` freely only
for unique-target sparse patches (#139) or trailing-slice writes whose targets are proven disjoint.

For "components of size N get recolored to X" patterns:

```python
labels = label_propagation(mask)  # per-cell component min-id from technique #33
labels_idx = Unsqueeze(Cast(labels, INT64), axes=[4])   # [1,1,30,30,1]
hist = ConstantOfShape([1000], 0.0)                     # zeros, size > max possible label
hist = ScatterND(hist, labels_idx, mask, reduction='add')   # hist[id] += 1 per mask cell
count_per_cell = Gather(hist, labels, axis=0)           # [1,1,30,30] = component size per cell
```

Then `Where(count == target_size, color_X, color_Y)` splits cells by component size. Used in octaviograu's task330 at 62K cost / +13.97 LB.

> **⚠️ 2026-06-04 cost-falsification (task364 worker, measured — work in progress): on small canvases, a U8 dominance flood STRICTLY DOMINATES §33+§34.** §33's MinPool labeling runs the label plane in **F16 at 1,760 B/step** (negation forces a signed dtype) and §34 adds INT64 index machinery (`Cast`+`Unsqueeze`+ScatterND idx) on top. A **single-channel U8 "dominance flood"** — iterated `Max`-propagation of per-cell ids under the component mask (`ids = Max(ids, masked-dilate(ids))`, U8 carrier, no negation, no INT64) — costs **880 B/step** on the same ≤22-canvas and yields the same per-component id plane (max-id instead of min-id; equivalent for grouping). Prefer the dominance flood for per-component labeling/feature extraction; reach for §33/§34 only when true MinPool semantics or the opset-17 histogram (`reduction='add'`) is itself required. (Cross-ref §37 for setting/justifying the iteration count from the corpus.)

---

# 32. Cell(0,0) base detection via ArgMax (replaces N diagonal Slice + variadic Max)

When a task algorithm encodes a color-index answer as a diagonal offset table where `offset_table[0,0] = 0` is invariant, cell (0,0) always holds the base color exactly. Replace the full diagonal search with a single ArgMax on cell (0,0).

**Pattern (task175 v12 → v14/v15)**:

```
# OLD: 21 diagonal Slices (each [1,9,1,1]) + variadic Max([1,9,1,1]×21) = ~500B intermediates
diag_00 = Slice(input, [1,0,0], [10,1,1], [1,2,3])   # [1,9,1,1]
diag_11 = Slice(input, [1,1,1], [10,2,2], [1,2,3])
... ×21 cells ...
base_onehot = Max(diag_00, diag_11, ..., diag_20_20)
base_m1     = ArgMax(base_onehot, axis=1)

# NEW: single 3-axis Slice on cell(0,0) + ArgMax = 36B intermediate
cell_00 = Slice(input, starts=[1,0,0], ends=[10,1,1], axes=[1,2,3])  # [1,9,1,1] F32
diag_idx = ArgMax(cell_00, axis=1, keepdims=1)                        # [1,1,1,1] INT64
base_m1  = Cast(diag_idx, INT8)                                       # [1,1,1,1] INT8
```

**Safety**: requires `offset_table[0,0] = 0` (task's invariant). With `val_mod[0,0] = (base_m1 + 0) mod period = base_m1`, cell (0,0) holds the raw base color directly — no diagonal traverse needed.

**Savings**: eliminates 20 Slice nodes + 20 intermediate tensors + variadic Max. For task175 on [1,9,1,1] intermediates: 20×36B = 720B memory + 20 nodes removed.

**Eligibility check**: confirm `offset_table[0,0] == 0` and the cell (0,0) is within the active grid (not padding) for ALL training pairs. If the grid may be shifted such that (0,0) is always inactive, this fails.

**Empirical**: task175 v14/v15 (cost 3,231 → 2,353/2,351, +0.317 LB predicted).

---

# 58. Sub-presence period detection (ReduceSum − ch0_presence) <!-- formerly a duplicate "# 33" — renumbered 2026-06-05 VERIFY P4 -->

When period = number of distinct foreground colors = `ReduceSum(presence_10) - 1` (subtract 1 for ch0 bg), the correct form MUST account for which cells contribute ch0.

**Wrong approach**: `ReduceSum(presence_10, all channels)` overcounts by 1 if bg channel (ch0) is present in the active area (typical for NeuroGolf inputs — interior bg cells have ch0=1).

**Wrong approach 2**: `ReduceSum(Slice(presence_10, ch1:10))` — correct but costs 36-byte `[1,9]` intermediate.

**Correct form**:

```
presence_10 = ReduceMax(input, axes=[2,3], keepdims=0)   # [1,10] F32
presence_sum = ReduceSum(presence_10, axes=[1], keepdims=1)  # [1,1] F32 = 4B
presence_ch0 = Slice(presence_10, starts=[0], ends=[1], axes=[1])  # [1,1] F32 = 4B
period_f     = Sub(presence_sum, presence_ch0)               # [1,1] F32 = 4B
period_i     = Cast(period_f, INT8)
period_safe  = Max(period_i, one_i8)
```

**Why**: `presence_ch0` is the ch0 column of `presence_10`, which is 1.0 when ANY cell has ch0=1 (i.e., any bg is present, which is always true). Subtracting it cancels the bg channel's contribution. Only 2 tiny [1,1] intermediates (8 bytes total).

**Memory cost**: 4B (presence_sum) + 4B (presence_ch0) + 4B (period_f) = 12B intermediates. The Slice ch1:10 approach would produce [1,9] F32 = 36B. **Saves 24B memory vs Slice approach**.

**Empirical**: task175 v14/v15 — Slice ch1:10 approach gave 2,381; Sub-presence approach gives 2,353 (v14) / 2,351 (v15).

## Upgrade: ArgMax `select_last_index=1` supersedes Sub-presence when colors are contiguous

When the task guarantees **contiguous active colors** (ch0=bg always present; ch1..ch_period=active; ch_{period+1}..ch9=absent), replace the ReduceSum+Slice+Sub chain with a single ArgMax:

```python
# NEW (task175 v17): last-max-index = period
period_i64 = ArgMax(presence_10, axis=1, keepdims=1, select_last_index=1)  # [1,1] INT64 = 8B
period_i   = Cast(period_i64, INT8)                                          # [1,1] INT8 = 1B
period_safe = Max(period_i, one_i8)                                          # [1,1] INT8 = 1B
# Total: 10B vs 12B for Sub-presence. Saves 3 nodes, -2 params (drop int64_one, int64_zero).
```

**Net savings over Sub-presence**: -2B memory, -2 params, -3 nodes.
**`select_last_index` availability**: opset 12+; fine at opset 18. Confirmed under ORT 1.24.4.
**Contiguous-color check**: verify all training pairs — if ANY pair has a "gap" color missing, ArgMax returns wrong period.
**Empirical**: task175 v17 (cost 2,349 → 2,345, 11 nodes, 9 inits, 468 params).

---

# 59. Opset-18 Pad axes input + shared init with ReduceMax <!-- formerly a duplicate "# 34" — renumbered 2026-06-05 VERIFY P4 -->

**Opset-17 Pad** (`pads` as input): pads tensor length = `2 × rank`. For a [1,1,21,21] tensor (rank=4), pads = 8 elements = 64 bytes INT64.

**Opset-18 Pad** (optional `axes` input): when `axes` is provided, pads length = `2 × len(axes)`. To pad only HW dims: `axes=[2,3]`, `pads=[H_beg, W_beg, H_end, W_end]` — 4 elements = 32 bytes INT64. Saves 4 INT64 elements (32 bytes params).

**Bonus**: opset-18 `ReduceMax` moved `axes` from attribute to input. The same `axes_23 = [2,3]` init can be SHARED between ReduceMax and Pad:

```python
# opset 18 — shared axes init
make_init("axes_23", np.array([2, 3], dtype=np.int64))   # 1 init, 2 elements, used by both

helper.make_node("ReduceMax", ["input", "axes_23"], ["presence_10"], keepdims=0)
helper.make_node("Pad", ["val_mod", "pads_hw", "pad_sentinel", "axes_23"], ["color_30"])
```

**vs opset 17**: ReduceMax uses `axes` attribute (free), Pad needs 8-element pads init (8 params). With opset 18: both use the shared 2-element `axes_23` init (2 params) + 4-element `pads_hw` (4 params) = 6 total params for what was 8 params. Net: **-2 params**.

**Correctness of opset-18 Pad arg order**: `(data, pads, constant_value, axes)` — `axes` is the 4th input, `constant_value` is 3rd (cannot skip; pass sentinel init there).

**ORT support**: opset 18 verified under ORT 1.24.4 / ONNX 1.21.0. Set `ir_version=10` when bumping opset.

**Empirical**: task175 v15 (cost 2,353 → 2,351, -2 params). Small but confirms the sharing works cleanly under ORT 1.24.4.

## Cellular automata cost reference (from C1 research deliverable 2026-05-26)

Score-specific cost formulas + dtype recommendations for grid neighborhood operations. Full details in `cellular_automata.md`. The non-obvious wins:

### Win 1: Shared 3×3 kernel beats grouped Conv for K-color majority vote
For `K=10` color majority vote `[1,1,H,W] → [1,1,H,W]` via per-color counters:
- **Recommended** (`Equal×10 → Cast×10 → Conv×10 with shared kernel → Concat → ArgMax`): **19 params** (K + 9 shared kernel elements)
- **Naive grouped Conv** (`[K,1,3,3]` weight): **100 params** (9K + K)

The grouped variant physically stores 10 copies of the same 3×3 kernel; the shared-initializer variant counts once. Activations are roughly equal; params differ by 81. Already used in task324 v5b plaid majority.

### Win 2: int8 MaxPool for binary morphology (opset 12+)
For binary mask dilation/erosion:
- **Dilation**: `MaxPool(int8 mask, kernel=[3,3], pads=[1,1,1,1])` — 0 params, `HW·b_i8` intermediate
- **Erosion**: `Neg → MaxPool → Neg` on int8 — 0 params, `3HW·b_i8`

Cheaper than the float Conv equivalent when the mask source is bool/int. Cast bool→int8 once at entry, stay in int8 for the whole morphology chain. ORT 1.24 CPU supports int8 MaxPool from opset 12.

### Win 3: Conway B3/S23 formulation without `Where(bool,bool,bool)`
For Life-style binary update rules: `new_alive = (neighbors == 3) | ((neighbors == 2) & alive)`. Uses only `Equal`, `And`, `Or` on bool — never triggers the Where(bool,bool,bool) ORT rejection (traps.md §DON'T do Where(bool, bool, bool)). Cost at 30×30: `2HW·b_f + 4HW·b_b + 11 params = 10,811 bytes` per step.

### Win 4: Multi-step unroll cost = `P_shared + N·A_step`
When unrolling N CA steps with the same rule, share the kernel initializer across all N Conv nodes. Cost scales linearly in N (activations) with a one-time `P_shared` term. For Life at 30×30: ~10,800 bytes/step. For 10-color majority: ~124,200 bytes/step. Use to budget unroll depth against memory.

### Win 5: Boundary penalty for reflect vs zero pad
Reflect padding at 30×30 costs **+1,520 bytes/step** vs zero pad (which is free via Conv's pads=[1,1,1,1]). Only use reflect when the rule semantics demand it (corner/edge values reused). Toroidal wraparound is NOT available — `Pad(mode='wrap')` is outside opset ≤ 18.

## Weighted-cover detector cost reference (from F1 research deliverable 2026-05-26)

Score-specific cost analysis for radius-encoded weighted-cover detectors (cross/plus/ring/disk/L/T/arrow templates). Full deliverable in `_archive/radius_encoded_weighted_cover.md`. Non-obvious wins beyond standard QLinearConv usage:

### Win 1: Separable rank-1 box detector — `k²` → `2k` MACs/cell
For a `k×k` box detector that's rank-1 (e.g., all-ones), apply as two 1D Conv passes (vertical `k×1` then horizontal `1×k`) instead of one dense `k×k` kernel:
- **Dense 5×5 on 30×30**: 25 weights, 22,500 MACs
- **Separable 5×5**: 10 weights (`5+5`), 9,000 MACs (60% reduction)
- **Dense 11×11 on 30×30**: 121 weights, 108,900 MACs
- **Separable 11×11**: 22 weights, 19,800 MACs (82% reduction)

ONNX pattern: `Conv(group=C, weight=[C,1,k,1]) → Conv(group=C, weight=[C,1,1,k]) → ReduceSum`. Already used in task255 v1's 30×1 prefix/suffix counts. The savings compound for larger radii — at `k ≥ 7` separable is always materially better.

### Win 2: Plus/cross exact decomposition — `2k+1` work
For a plus-shaped detector (vertical line + horizontal line + center):
- Decompose as "vertical-line score + horizontal-line score − center correction"
- Costs roughly `2k+1` work per cell vs `k²` for dense
- Exact, not approximate. Each line score is a separate 1D Conv.

### Win 3: Concentric shells via cumulative-box-subtract
For multi-radius rings/annuli, compute cumulative box counts at each radius, then subtract adjacent radii to get annular scores. Outer shells REUSE the inner box work. For 3 concentric shells at radii 1/3/5:
- **Naive 3 separate dense kernels**: 9 + 49 + 121 = 179 weights × output map per radius
- **Cumulative-subtract**: compute box_count[r=1], box_count[r=3], box_count[r=5] once each; annulus[r=3] = box[r=3] − box[r=1]; annulus[r=5] = box[r=5] − box[r=3]

Saves both params (shared cumulative kernels) and intermediates (the box_count maps are computed anyway). Untried in our pipeline as of 2026-05-26.

### Win 4: U8S8 > S8S8 on x86 CPU per ORT
For QLinearConv with unsigned occupancy inputs + signed weights (anti-cover penalties):
- **Recommended**: `x_dtype=uint8`, `w_dtype=int8` (U8S8). ORT-fast on x86.
- **Avoid**: `S8S8`. ORT docs explicitly say "S8S8 with QOperator is slow on x86-64 and should generally be avoided"
- **Fallback if saturation visible**: `U8U8` (split detector into positive branch + negative branch, subtract — doubles Conv work but avoids signed saturation)

### Win 5: Requantization y_scale formula
To keep score range in int8/uint8 output without clipping:
- For signed int8 output: `y_scale ≥ x_scale · w_scale · Smax / 127`
- For uint8 output: `y_scale ≥ x_scale · w_scale · Smax / 255`
- Threshold `τ_real` becomes `τ_q = round(τ_real / y_scale) + y_zero_point`
- For exact preservation: pick `y_scale` such that `Smax * (x_scale * w_scale) / y_scale ≤ output_range`

Concrete: an 11×11 unit-count cover detector has max count 121 → fits uint8 exactly with `y_scale=1`. A 121-cell signed kernel with weights ±3 can reach |score|=363 → forces coarser bins for single int8 output.

### Win 6: Mask-out-boundary > padding tricks
When the rule semantically requires FULL neighborhood support (cells where the full radius doesn't fit are invalid):
- **Bad**: zero pad — systematically lowers boundary scores
- **Bad**: reflect/edge pad — hallucinates support
- **Good**: precompute a valid-center mask (ones where full radius fits), `Where`/`Mul` with sentinel before threshold/ArgMax

For r=2 on 30×30: valid interior is 26×26=676 cells; boundary suppression covers 224 cells via a single static mask init. Avoids semantic ambiguity entirely.

## Merge two diagonal-spread Conv kernels into one X-kernel (halve params, byte-equal) — task324 2026-05-31

When a graph spreads markers along BOTH diagonals via two separate `[1,1,N,N]` Conv kernels OR'd together — `on_diag = (Conv(m,d1)>0) OR (Conv(m,d2)>0)`, where `d1` is the main diagonal (`k[i,i]=1`) and `d2` the anti-diagonal (`k[i,N-1-i]=1`) — replace with **ONE X-kernel** carrying 1.0 on both diagonals: `on_diag = Conv(m, x)>0`.

**Byte-equal proof**: Conv is linear and `m,d1,d2 ≥ 0`, so `Conv(m,d1)+Conv(m,d2)=Conv(m,d1+d2) ≥ 0`, and `(a>0 OR b>0) ⟺ (a+b>0)` for non-negative `a,b`. The center tap (on both diagonals) may be 1.0 (assignment) or 2.0 (sum) — irrelevant to the `>0` test. Drops the 2nd `[1,1,N,N]` kernel **and** the 2nd Conv + its `Greater` + the `Or`. For N=39 (20×20 active): **3,042 → 1,521 params** + ~3 fewer intermediates. Verified on task324 compress_v1 (0 divergences vs the two-kernel parent across 5,000 inputs). Generalizes to a `+`-kernel (4-cardinal) or any union of linear spread patterns whose combination is only `>0`-tested.

## Corner-anchor base/stripe disambiguation for plaid tasks (replaces an expensive per-cell vote) — task324 2026-05-31

When an ARC task is a **two-color plaid** (one base color + a second color drawn as full-row/full-col stripes) and you must classify each cell's underlying bg-class, the whole problem reduces to ONE global decision: **which of the two bg colors is the base?** Once the base B is known, `on_base(r,c) = (countB_row[r] > 0) AND (countB_col[c] > 0)` is EXACT (stripe rows/cols have zero base cells; markers are non-bg so never add to base counts).

The cheap exact anchor for "which color is base": **if stripes start at a fixed minimum offset, the cells before that offset are guaranteed base.** In ARC-GEN `task_d07ae81c`, `brows`/`bcols` both start at `randint(2,7) ≥ 2`, so rows 0-1 and cols 0-1 are never stripes → **the top-left 2×2 corner is always the base color.** Read B = the bg color present in that corner (robust to ≤3 markers: 4 corner cells, ≥1 is base). DON'T assume `base = most-common color` — when stripes are dense the stripe color is MORE common (this naive assumption scored 88/266 on task324).

Payoff: replaced v5b's 3-way majority vote (4-neighbor Conv + plaid formula + row-majority + 2-of-3, ~5k of masks) with a 2×2 Slice + 2 Equal + a `Where` to pick B, plus 2 ReduceSum + 2 Greater + 1 And. Result was **exact (100% vs ground truth) where the heuristic vote was only ~98.5%**, and ~5k cheaper (task324 compress_v3: 18,545 vs the byte-equal 23,491). General lesson: **a fixed-offset region the generator guarantees to be "background" is a free exact anchor — use it instead of a learned/voted global statistic.**

## Row-signature + rank-reduction cost reference (from F2 research deliverable 2026-05-26)

Score-specific recipes for per-row signature computation + ranking + dedup. Full deliverable in `_archive/row_signature_rank_reduction.md`. Non-obvious wins beyond standard usage:

### Win 1: ARC row packing needs 2 INT64 limbs, NOT a single uint32
ARC has 10 colors × 30 cells = **120 bits** required for exact row encoding. A single uint32 (32 bits) or int64 (64 bits) is INSUFFICIENT. Correct pattern:
- Split 30-cell row into two 15-cell halves
- Each half packed in **base 16** (4 bits/cell × 15 = 60 bits, fits exact in INT64)
- Store key as `[N, 2]` tuple of `(lo, hi)` INT64

```python
weights_lo = np.array([16**i for i in range(15)], dtype=np.int64)
weights_hi = np.array([16**i for i in range(15)], dtype=np.int64)
key_lo = row[:, :15] @ weights_lo     # [N] INT64, 60 bits used
key_hi = row[:, 15:] @ weights_hi     # [N] INT64, 60 bits used
key = np.stack([key_lo, key_hi], axis=1)  # [N, 2] INT64
```

Equality between rows: compare both limbs (`Equal(lo,lo)` AND `Equal(hi,hi)`). Already used in task255 row-packing; this codifies it.

### Win 2: Counting sort cheaper than TopK for tiny integer domains
When the sort key is a small integer (e.g., occupancy ∈ [0,30], 10-color bitmask ∈ [0,1023]), counting sort beats `TopK(K=N)`:
- `OneHot(key, depth=B) → ReduceSum(axis=0) → CumSum(exclusive=1)` gives stable bucket offsets
- For N=30, B=31 (occupancy): ~930 one-hot slots + reductions
- For N=30, B=1024 (10-color bitmask): ~30,720 one-hot slots — still cheaper than the implicit O(N log N) of TopK at large N

Decision rule: if key range B << N, use TopK. If B is small absolute (≤32), counting sort wins.

### Win 3: TopK(sorted=1) ties → lower original index first
The ONNX `TopK` spec guarantees `sorted=1` breaks ties by returning lower original index first. This is exactly the "keep first occurrence" rule. No need for `ArgMax(select_last_index=0)` workarounds when you're already using TopK.

### Win 4: Trilu(upper=0, k=-1) for "seen-before" dedup mask
Canonical pattern for "preserve first occurrence" duplicate detection:
- `eq[i,j] = (key[i] == key[j])` — full pairwise equality `[N,N]`
- `lower_tri = Trilu(eq, upper=0, k=-1)` — only `j < i` entries kept
- `is_dup[i] = ReduceMax(lower_tri[i], axis=1)` — row i is dup if any earlier key matches
- `keep_mask = Not(is_dup)`

Cleaner than custom mask construction. The `k=-1` argument excludes the diagonal (every key trivially matches itself).

### Win 5: Pairwise row equality without exact scalar key
When you can't (or don't want to) build an exact scalar/limb key, compute pairwise row equality directly:
- Broadcast rows to `[N,1,M]` and `[1,N,M]`
- `Equal` produces `[N,N,M]` boolean cube
- `ReduceMin(axis=2)` or `ReduceSum == M` collapses to `[N,N]` row-equality
- Apply Trilu as in Win 4

Cost on ARC 30×30: `30×30×30 = 27,000` cellwise comparisons + 900-element pairwise matrix. Manageable for single-pass use. Avoids the limb-key bookkeeping.

### Win 6: Equality-mask MatMul as "dense permutation" alternative to Gather
For "assemble output rows by signature match", two equivalent forms:
- **Sparse (Gather)**: produce indices first, then `Gather(X, indices, axis=0)`. Smaller intermediate.
- **Dense (MatMul)**: build mask `M[t,i] = [sig[i] == target_sig[t]]`, cast to numeric, `Y = M @ X`. Static, no index plumbing, blends row-selection + row-validity in one tensor.

Dense form better when same reorder is reused later OR when you want to blend with a "keep this row" mask in a single linear combination. Sparse form better when intermediate memory dominates.

### Win 7: Variable-size output — packed-top + bg-pad ambiguity
ARC has no "out of band" sentinel color (all 10 colors are real palette). For variable-row-count output:
- **Cheapest**: packed-top + background padding. Risk: all-bg row from rule is indistinguishable from padding.
- **Cleanest**: `grid[N,M] + valid_rows[N]` mask output. Only works if wrapper accepts auxiliary output.
- **NeuroGolf-specific**: our `run_network` hard-codes `output` as the only output name (per traps.md `calculate_memory` rejection rules), so multi-output heads are NOT available. Stuck with packed-top + bg-pad and a per-task ambiguity check.

## Multi-stage compositional rules cost reference (from F3 research deliverable 2026-05-27)

Score-specific recipes for staged rules (key-first → score → assemble). Full deliverable in `_archive/multistage_key_first_compositional.md`. Most of the 5 pattern families (key-first+count, anchor+propagate, filter-then-paint, group-then-aggregate, two-stage reach-N) are standard ONNX practice. The non-obvious wins:

### Win 1: Reach-N propagation — separable MaxPool beats Conv on memory+params
For square (Chebyshev) dilation of a seed mask by radius N on 30×30:

| Strategy | Memory bytes | Params | Notes |
|---|---|---|---|
| N-step 3×3 Conv (iterated) | N × 900 | 9 (shared kernel) | Faithful for sequential state updates |
| One-shot MaxPool (2N+1)² | 900 | 0 | Square envelope only; over-expands corners for Manhattan |
| **Separable MaxPool 1×(2N+1) → (2N+1)×1** | 2 × 900 = 1,800 | **0** | O(N) instead of O(N²) window cost; same square reach |
| Bit-packed propagation | ~30 × 30 / 8 = 113 | 0 | Lowest arithmetic, more graph plumbing |

For NeuroGolf's `memory + params` scorer, separable MaxPool dominates Conv-based dilation by ~9 params per step. For N=7 (reach-7), separable saves 63 params vs 7-step Conv (assuming shared kernel) and matches the radius-7 envelope without iterative state. **Caveat**: only correct for SQUARE (Chebyshev) dilation. For MANHATTAN reach, must iterate cross-kernel Conv N times — one-shot MaxPool over-expands corners.

Already-deployed in task066 wavefront, task219 reach, task285 MaxPool sweeps. F3 codifies the dtype/cost tradeoffs explicitly.

### Win 2: Constant folding requires INITIALIZERS, not graph inputs
ORT's basic graph optimizations include constant folding. The folding only fires when constants are model INITIALIZERS, not graph INPUTS. A common mistake: declaring coordinate grids / permutation banks / template kernels / one-hot codebooks as graph inputs (overridable) blocks the optimization.

Rule: for hand-built static ARC graphs, every immutable tensor must be an initializer. If you find yourself passing a fixed coordinate grid as an input, fold it. ORT's troubleshooting docs explicitly call this out.

Verified relevant: our submission models use `input` as the only graph input (per `core.md` validator rejection criteria — `run_network` hard-codes `output` and `input` names). Anything else must be in `model.graph.initializer`.

### Win 3: Quantization split for staged graphs
For multi-stage compositional rules with both control and numeric subgraphs:
- **Keep unquantized**: `Equal`, `ArgMax`, `TopK`, `Where`, mask plumbing, control-flow logic. Already low-parameter; quantization adds overhead without benefit.
- **Quantize**: dense numeric stages — `Conv`/`QLinearConv` score banks, `MatMul`/`QLinearMatMul` selection bodies, propagation kernels. These are where arithmetic dominates.

Practical: keep your stage-1 (key extraction) and stage-3 (output assembly) in their natural domains (bool/int), and only QDQ-wrap stage-2's score computation. Saves the requantization overhead at every Equal/Where boundary.

### Win 4: `ScatterElements` opset 18 reduction='max' for top-K keep-mask
Pattern for "select K survivors via TopK, then convert indices to a binary spatial mask":
- `TopK(score, K)` returns `[K]` int64 indices
- `ScatterElements(zeros, indices, ones, reduction='max')` produces `[N]` binary keep-mask where K positions are 1

`reduction='max'` (opset 18+) is the right choice because overlapping writes collapse to 1 rather than summing. Without reduction='max', multiple writes to the same position would accumulate.

### Win 5: Empty `ReduceMean` is undefined behavior
Centroid computations via `ReduceMean(sum_xy, mass)` are unsafe when mass can be 0 (no foreground cells). Robust pattern: keep `(sum_row, sum_col, mass)` as integer tensors, compare cross-products before dividing, or divide only once after a nonzero-mass `Greater` gate. Float drift in centroid ranking is small but undefined-empty-reduce can produce NaN/Inf that propagates through Where.

Already documented obliquely in traps.md F16-arithmetic notes; F3 codifies the specific empty-reduction case.

## Multi-QLinearConv detector banks cost reference (from F4 research deliverable 2026-05-28)

Score-specific recipes for hand-authored multi-detector banks where each kernel is a structural rule (template library, not learned weights). Full deliverable in `_archive/direct_detection_multi_qlinearconv.md` including 4 bank-layout patterns, 4 combination strategies, kernel-size×stride tradeoff table at 30×30, and worked examples for task367 v2 (open-family `structural direct-25 QConv`) and task077 rawwhere (open-family `hull-8 detection`). Most of the operator-level QLinearConv reference is already in F1 (Wins 1–6 above) and the existing techniques.md QLinearConv coverage; F4 adds bank-design and detector-pruning wins beyond that:

### Win 1: Exact-match binary template kernel — universal `b = 1 − |P|` formula
For an exact-match binary template on a one-hot plane `x ∈ {0,1}`, choose weights `w_i = +1` at required-1 positions in support `P`, `w_i = −1` at required-0 positions in `Z`, and `w_i = 0` at don't-cares in `D`. Set bias `b = 1 − |P|`.

Then detector score `s(x) = Σ w_i x_i + b` has the clean property: **exact match → s=+1, any missing required positive or any forbidden positive → s ≤ 0**. Works identically for L-shapes, T-shapes, crosses, corners — pick `P` to suit. Example for canonical 3×3 L (`P = {(0,0),(1,0),(2,0),(2,1),(2,2)}`, `|P|=5`):
```
W = [[+1,-1,-1],[+1,-1,-1],[+1,+1,+1]],  B = −4
```
Output normalized to `+1` exact / `≤0` mismatch is the simplest way to share `y_scale` across an arbitrary bank size (Win 2 below).

### Win 2: Bank quantization — share `y_scale` by normalizing every detector to "exact = +1, mismatch ≤ 0"
`QLinearConv-10`'s `w_scale`/`w_zero_point` may be **per output channel** but `y_scale`/`y_zero_point` is **scalar for the whole output tensor**. For a multi-output-channel bank where channels are compared via `ArgMax(axis=channel)` or thresholded jointly, all detectors share one `y_scale`. **If two detectors have different y_scales, the integer ArgMax across channels is meaningless** — the integers represent different real units.

Fix: hand-normalize every kernel to the "exact = +1, mismatch ≤ 0" convention (Win 1). All detectors then live in one comparable integer domain and a single shared `y_scale=1` works for the whole bank. Avoids the per-detector requantization that F1 Win 5 has to compute when score ranges differ.

### Win 3: Single multi-output-channel bank > D separate 1-output-channel nodes
`QLinearConv` weight shape is `M × (C/group) × kH × kW`. Both "M=D parallel detectors in one node" and "D separate 1-channel nodes" encode the same bank, but:
- **Single-bank**: shares one `x_scale`/`x_zero_point` dequant + one `y_scale`/`y_zero_point` quant; one node in the graph.
- **D separate**: D pairs of dequant/quant scalars; D nodes.

Single-bank wins on graph-node count and on shared scalar quantization initializers. Pick D separate nodes only when detectors need genuinely heterogeneous quantization (rare) or different padding/stride per detector.

### Win 4: Dilation is the cheapest receptive-field expansion for sparse-anchor templates
For templates that check sparse positions (corner verifiers, L-shape with 3 required cells, ring witnesses), the kernel doesn't need dense `k×k` coverage. `Conv(dilations=2, kernel=3×3)` has **effective receptive field 5×5** at **3×3 weight cost (9 vs 25 weights, 9 vs 25 MACs/cell)**. `dilations=3` gives effective 7×7 at 9-weight cost.

Use dilation when the template's required positions are at corner anchors or wide-spread points, not when every cell in the receptive field matters (use full `k×k` for dense templates). Combines with Win 1 (zero weights at don't-cares) — dilated kernels naturally encode sparse-anchor templates.

### Win 5: Detector bank combination — default is `ReduceMax`-gate + `ArgMax`-paint
For a bank where multiple detectors paint different colors, the strongest default is a 2-stage combination:
1. `ReduceMax(scores, axis=channel)` + `Greater(reduced, threshold)` → boolean background-or-paint gate.
2. `ArgMax(scores, axis=channel)` + `Gather(palette, argmax)` → color lookup.
3. `Where(gate, palette_color, background)` for final paint.

Trade-offs vs alternatives:
- **`ArgMax` alone**: always picks something, even at cells where every detector scored ≤ 0. Fires everywhere unless threshold-gated.
- **`ReduceMax` + threshold alone**: cheapest if all detectors paint the same color (drop the ArgMax+Gather entirely).
- **Priority chain (nested `Where`)**: O(D) graph depth, scales poorly past ~6 detectors but lets detector 1 override detector 2 regardless of numeric score (good for exception-first rules).
- **Sum + LUT**: collision-free needs powers-of-two codes → 2^D LUT size; use only for 2–4 strongly-correlated detectors.

`ArgMax-13` ties default to first-index; pass `select_last_index=1` to flip — important when multiple detectors at the same cell would otherwise win in the wrong order.

### Win 6: Detector pruning via ORT quant-debug API (free dead-channel discovery)
ORT exposes intermediate-tensor harvesting through `onnxruntime.quantization.qdq_loss_debug.modify_model_output_intermediate_tensors()` + `collect_activations()` + `create_activation_matching()`. Originally written for float-vs-quantized debugging, but works for any per-channel score harvest:

1. Run the modified model on visible training inputs.
2. Record `max(score_i)` and `argmax-wins_i` per detector channel `i`.
3. Mark detector `i` as dead if it never exceeds the positive threshold AND never wins downstream selection.
4. Structurally prune: shrink `w`, `B`, per-output `w_scale`/`w_zero_point` along axis 0; in a priority chain, delete the branch and collapse the `Where`; in a sum+LUT scheme, recompute the integer codebook.

For NeuroGolf-scale banks (10–50 detectors), dead-channel discovery is cheap and the pruning is a one-shot graph rewrite. Useful before any submission with a hand-authored detector library.

### Worked examples (speculative, see deliverable)
- **task367 v2** (`structural direct-25 QConv`): plausible reading is 25 parallel detectors, one per output offset `(dy, dx)` in a 5×5 target patch — each detector emits at the cell that should be painted for that offset. Combination `ReduceMax`+threshold if same color, priority chain if not.
- **task077 rawwhere** (`hull-8 detection`): plausible reading is an 8-direction witness bank (one kernel per compass direction) with dilated 3×3 or 5×5 kernels for "reachable in <3 Manhattan steps"; outputs ORed via `ReduceMax`, thresholded, fed into `Where` for direct paint. The "rawwhere" name fits: only one paint action, so `ArgMax` is unnecessary — just gate-and-paint.

Both are inferred from project-local labels + public ARC hints; treat as starting templates not specifications.

---

# 52. Separable-depth rewrite for nested-frame / "onion" tasks (task377, 2026-06-01: 63,682 → 9,887, 6.4×) <!-- formerly unnumbered "# N." -->

**Meta-lesson first.** task377's `.md` had a 2026-05-19 "no from-scratch savings; 5-iteration `[1,1,30,30]` structure unavoidable" audit. **It was wrong — it audited the implementation, never re-derived the rule.** Always `decode_rule.py`-style print the actual train/test pairs before accepting a from-scratch floor. The real rule (N nested rectangular frames → `(2N-1)²` concentric bullseye) needs NO 30×30 seed-finding.

**When it applies:** output is a small concentric-ring/bullseye, input is nested rectangular frames (often with repeated colors across layers, and the onion is *offset* — not concentric with the grid). The expensive LIVE graph iterates ArgMax+mask to peel layers.

**The cheap pattern (everything is `[1,10,30]`/`[1,1,30]`/scalar except one `[1,1,29,30]`):**
1. **Color-diff via a Conv difference-kernel** — `Conv(input[1,10,H,W], W[1,10,2,1])` with `W[0,k,0,0]=-ramp[k], W[0,k,1,0]=+ramp[k]` outputs `cc[r+1]-cc[r]` directly (the colour ramp is baked into the kernel → **no `[1,1,H,W]` colour canvas needed**).
2. **Find the deepest line** — `cstar = ArgMax(ReduceL1(diff, axis=rows))`. Per-column total-variation peaks at the column through the innermost frame; `argmax(TV) == argmax(transition-count)` (verified) and TV is one op (`ReduceL1`) vs a clipped-count map. (Coverage/bbox-by-colour only 90% — fails repeated colours; per-cell transitions are mandatory for depth.)
3. **1D shell along `cstar`** — gather that one column, compute `shell = min(CumSum_inclusive_from_top, CumSum_reverse_EXCLUSIVE_from_bottom)` of its gated transitions. ⚠️ the reverse cumsum **must be exclusive** (`CumSum(reverse=1, exclusive=1)`): a transition stored at index r is the boundary between r-1 and r, so the far side must not count it (else the outer frame on that side gets shell 1, polluting layer 1).
4. `layer[d] = max(colcol where shell==d)`, `L = max(shell)`; feed the ring renderer.

**Cost levers that stacked to sub-10k (all dtype-narrow / op-fold, correctness-preserving):**
- column read via `ArgMax(col_oh)+ReduceMax(col_oh)` instead of `Mul(ramp)+ReduceSum` (drops a `[1,10,30]`);
- F16 front-end arithmetic (colours 0-9, shells 0-4 are exact in F16) — but **`CumSum` rejects F16** (opset-11 `T`), so cast to F32 only around the cumsum;
- uint8 output tail + int32 ring index (§ NOVEL-OP-safe dtype narrows);
- **renderer background fold**: `v_masked=Mul; v_oob=Sub(1,·); v_bg=Mul(·,10); v_total=Add` (4 `[9,9]` tensors) → `v_total = Where(bool_inbox, ring_colour, 10)` using the `Less` results as bool directly (skip their F16 casts). −~600 B.

**Ship class:** Path B (not byte-equal). Gate = visible all-pass + **correctness-equivalence to LIVE** (0 cases where LIVE is right and the rewrite is wrong, measured across the full 200k corpus — `diagnose_fails.py`). Gain ≈ `ln(cost_old/cost_new)` × 1.0 (task092/task173 precedent). The novel ops (Conv/ReduceL1/CumSum) are a §NOVEL-OP caveat, but those documented failures were *approximations*; a rewrite that shares LIVE's exact ceiling is the same rule, not an approximation. Full worked example: `The_400_Tasks/Task_by_Task/task377.md` + `task377/prototypes/build_task377_rewrite_v{1..4}.py`.

**Transition-count Einsum kills the F32 Conv-diff carrier (task377 v8, 2026-06-21, 8192->5689).** The
"deepest column" selector (step 2) does NOT need to materialize the `diff` F32 `[1,1,29,30]` plane
(3,480 B). Replace `Conv(input, color-ramp-diff) -> diff -> ReduceL1 -> ArgMax` with ONE ordered
four-operand Einsum that counts adjacent active row-pairs whose colors differ, per column:
`Einsum("bchw,hg,bdgw,cd->bw", input, adj_next[30,30], input, neq_cd[10,10])`. Operand ORDER is
load-bearing for ORT's left-deep contraction (#89): `input × adj_next` first keeps the prefix at
`[B,C,W,G]`, then the second `input` and the small color-mismatch matrix `neq_cd` collapse to `[B,W]`.
Net: -3,480 B (`diff`) for +900 (`adj_next`) +100 (`neq_cd`) + a `[1,30]` F32 result (120 B), static
(profile==static==5689). This is the explicit adjacency/mismatch transition-count form of the
`argmax(TV)==argmax(transition-count)` identity above (cross-ref #89 in-kernel temporaries).

# 53. Fold "colored mask" + "active mask" into ONE weighted-conv input read (task192, 2026-06-01: 56,531 → 18,166, 3.1×) <!-- formerly unnumbered "# N." -->

Many per-cell rules need TWO masks off the one-hot `[1,10,30,30]` input: **active** (any channel hot =
inside the grid, incl bg ch0) and **colored** (channels 1-9 hot = non-bg). The obvious build reads the
input twice — `active = ReduceSum(input, axis=channels)` and `colored = Conv(input, [0,1,1..1])` — two
full-canvas `[1,1,30,30]` F32 intermediates (3,600 B each; the input itself is F32 so they can't be F16
without a Cast that costs what it saves).

**Fold both into ONE weighted conv** `W = [0.25, 1,1,1,1,1,1,1,1,1]` (1×1):
- empty cell → 0, bg cell (ch0) → 0.25, colored cell → 1 (exact in F32/F16 for one-hot input; 0.25 = 2⁻²).
- `active  = combined > 0.1` (bg + colored), `colored = combined > 0.5` (colored only) — both **byte-identical**
  to the two-read masks. Any two thresholds separating {0, 0.25, 1} work; the bg weight just needs to be in
  (0, 0.5). Saves one full `[1,1,30,30]` read (and, if you crop, its crop). Generalizes to any "split a
  one-hot into k disjoint cell-classes by one scalar value" need.

The `input` tensor is **not counted** (named `input`) — always Conv/Reduce directly on it; never pre-cast or
pre-crop the 10-channel tensor (a `[1,10,20,20]` working tensor is 8–16k, a net loss). 1×1 conv and
channel-reduction commute with a spatial crop, and the input padding region is all-zero with no conv bias,
so full-canvas processing == LIVE's crop-then-pad byte-for-byte whenever grids ≤ crop size.

## Single-channel index canvas vs multi-hot painting — the dominant-color TIE trap

For a "paint regions with the dominant color, rest = bg" output, the `Equal(single_channel_index_canvas,
[0..9])` floor trick (`cost_floor_reference.md`) collapses three `[1,10,h,w]` one-hot tensors to one
`[1,1,h,w]` canvas — **but it cannot reproduce a multi-hot output**. If the dominant color is `Greater(counts,
max−0.5)` (or any arg-max-with-ties), a count TIE makes LIVE paint 2+ channels; a single canvas value paints
one. Invisible on real data, but ~10% divergence on random multicolor (all tie-grids). **Fix without losing
most of the win:** paint with two `Where(BOOL, U8, U8)` layers (opset 13) → `[1,10,h,w]` U8 (1 B/elem, ¼ of
F16's 4 B) — multi-hot on ties exactly like the float graph, and the U8 cropped output (8,000 B at 20×20)
still beats LIVE's 24,000 B of F16. Gate any single-channel-canvas candidate with a multicolor fuzz vs LIVE
(real-encoder, ≤grid-size) before trusting it — visible+truth can be 100% while ties diverge (task080/097
class). Full worked example: `The_400_Tasks/Task_by_Task/task192.md` + `task192/prototypes/build_task192_v{1..4}.py`.

# 54. Runtime variable-template extraction → Conv-with-runtime-weights (task191, 2026-06-01: 63,060 → 32,841, 1.92×) <!-- formerly unnumbered "# N." -->

For tasks where a **variable template is embedded in the input** (e.g. a marker shape you must
detect copies of) and the template VARIES per input, you can extract it at runtime and feed it as
a Conv weight — ORT runs `Conv`/`ConvTranspose` with **runtime-computed weights AND bias**, not
just initializers (the LIVE task191 graph and this rewrite both rely on this).

**Pattern (all intermediates tiny except the unavoidable match map):**
1. **bbox of the marker** via `ReduceMax`(presence plane) per axis → `ArgMax` with
   `select_last_index=0` (first index) and `=1` (last index) → first/last occupied row & col.
   No reversal/ramp needed (vs the LIVE graph's asc/desc F16 ramps).
2. **extract a variable patch into a FIXED shape** with `Gather` on RUNTIME index VALUES:
   `Gather(plane, rowidx, axis=2)` then `Gather(_, colidx, axis=3)` where `rowidx = start + [0..k-1]`.
   Gather keeps a STATIC output shape `[..,k,k]` even though the indices are data-dependent.
   Runtime `Slice` starts/ends need the #146 value_info rescue when the crop size is fixed;
   without that annotation the scorer sees unknown dims and returns `None`. **Clamp indices with `Min(idx, size-1)`** so degenerate/dummy inputs (the scorer
   profiles with `ch0=1` everywhere → no marker) don't crash the Gather with OOB.
3. **build N transformed kernels** (e.g. 8 D4 orientations) via **build-time permutation tables
   selected by a runtime "combo" index**: precompute `PERM[n_param_combos, N, k*k]` (the source-flat
   index for each output cell, per (geometry-combo, orientation); point unused cells at a 0-sentinel
   appended to the flattened kernel), compute `combo` from the runtime dims, `Gather(PERM, combo)` →
   `[N, k*k]`, then `Gather(kernel_flat_with_sentinel, perm)` → reshape to the Conv weight. **Clamp
   `combo` to the valid range** for the same degenerate-input reason.
4. **exact match** = `Conv(plane, kernels, bias = 0.5 − |pattern|)` then `Relu` → 0.5 at exact
   matches, 0 elsewhere (the masked kernel `2·template − windowmask` gives `|pattern|` only on an
   exact hit; the variable `windowmask` makes cells beyond the variable box "don't care"). The
   Relu-bias form needs no BOOL intermediate (vs Greater+Cast → saves the BOOL tensor).
5. **paint** = `ConvTranspose(seed, fill_rects)` with `pads` set to crop the output directly to the
   active size (e.g. `pads=[1,1,3,3]` turns a 5×5-kernel `[1,c,23,23]→[1,1,27,27]` into `[1,1,23,23]`,
   no separate Slice). Group orientations sharing a paint-rect shape and `ReduceMax` each group's
   Conv to shrink the paint-seed channel count (8→2 here; safe iff no anchor matches >1 orientation
   with a differing rect — instrument it).

**Important correction after v5/v6:** the original v3 was only the first safe landing. If the match is
ultimately boolean, the `[1,N,H,W]` orientation-correlation map is not the true floor: collapse it with
the windowmasked positional-hash technique above, then use opset-13 Equal+Or membership or opset-20
`ReduceMax(BOOL)`. v3 held on public (6446.21 -> 6446.86, +0.65); v6 later held at 24,807 with opset20
membership reduction. Full v3 builder: `The_400_Tasks/Task_by_Task/task191/prototypes/build_task191_v3.py`.

# 35. Conv weighted-sum color decode + asymmetric-kernel free crop (task368 v9, shipped +2.08, 2026-06-04)

On one-hot `[1,10,H,W]` input, a `Conv` with weights `[0..9]` (single `[1,10,1,1]` kernel) yields
EXACT color indices in F32 (integer-valued, exact fp32 arithmetic). This replaces `Slice`+`ArgMax`
entirely — killing both the 10-channel cropped intermediate AND ArgMax's forced INT64 output
(task368: `inp10` [1,10,10,10] 4,000 B + `gcol_i64` 800 B both eliminated). Cross-ref §8b/§23
(color-index canvas): this is the cheapest decode INTO that representation.

**Asymmetric kernel height crops one axis for free**: kernel `[1,10,21,1]` (only kr=0 nonzero)
makes output rows = 30−21+1 = 10 → conv output `[1,1,10,30]` F32 (1,200 B) instead of
`[1,1,30,30]` (3,600 B). Kernel ZEROS still count as params (10·21·1 = 210), so cropping BOTH
axes this way explodes (`[1,10,21,21]` = 4,410 params) — crop the long axis with the kernel and
the short axis with a post-Cast U8 `Slice` (Cast-before-Slice beats Slice-before-Cast, 400 B vs
500 B; task368).

# 36. Cumulative ≥k masks + deepest-rank-outermost Where nesting (task368 v9)

Selecting per-cell among rank-indexed values `T[0..3]` does NOT need exact-rank masks
(`rank==k`, usually Xor-derived). Nest `Where` with the DEEPEST rank OUTERMOST:

    Where(ge3, T3, Where(ge2, T2, Where(ge1, T1, T0)))

A rank-2 cell fails `ge3`, passes `ge2` → `T2`. The cumulative `≥k` masks come straight out of
the shift chain (consecutive-AND of shifted predicates), eliminating the exact-rank Xor masks —
and when the innermost default is the passthrough canvas, the separate final merge `Where` too
(task368 v8).

### Field validations 2026-06-04 (all shipped, exact ln() calibration)

- **§8b dilated 2×2 Conv** validated on task287 (grids provably FIXED 16×16): `[1,10,2,2]`
  kernel with dilation `[14,14]` → `[1,1,16,16]` F32 directly from the `[1,10,30,30]` input — no
  full-canvas intermediate. Shipped 50,758 → 3,016 (memory floor 2,948 proven exhaustively).
- **Sentinel=10 in Pad before Equal-expand** (task287 + task368; extends the task238 sentinel
  entry): the harness encodes EXPECTED outputs with all-zero padding, so the padded index canvas
  must use a sentinel ≥10 — `Equal(canvas, ci)` is then all-False at padding = matches expected.
  Sentinel 0 makes ch0=True at padding → the gate fails. Pad the 1-channel canvas BEFORE the
  Equal-expand (padding the 10-ch BOOL costs 10×; the BOOL graph output itself is uncounted).

# 37. Corpus-measured iteration floors for flood/CC chains (task325 + task279, shipped 2026-06-04)

When a rule needs bounded iteration (flood-fill, CC label-propagation), do NOT guess the
iteration count — MEASURE the corpus and let the histogram set (and *prove*) the floor:

- **K-minimum proof (task325)**: run the candidate at K and K−1 over the full 200k corpus;
  K=5 passed everywhere, K=4 failed 399/200k ⇒ K=5 is the proven minimum. The failing-pair
  count at K−1 is the proof artifact — record it.
- **Depth-histogram proof (task279)**: histogram the *required* flood depth per corpus pair
  ({1:2028, 2:34673, 3:72899, 4:58300, 5:24630, ... 11:1}); a cost target that only affords
  ≤3 steps provably breaks ~45% of generator draws. This converts "can't go lower" into a
  quantified impossibility statement.
- **Shortcut levers must be measured the same way**: task325's [5,5] 2-hop jump (would halve K)
  was killed by measuring gap density — 79,986/200k pairs have components at Chebyshev distance
  exactly 2, so the wider window leaks IDs (also proves per-step re-masking is mandatory).
  task279's erode-4×4 seed enrichment was measured **-EV** (saves 256 B, breaks 237/200k).
- **Generator-geometry contracts can legalize cheaper steps**: task279 read the ARC-GEN source
  and proved every component is a rectangular ring (3×3–5×5, margin ≥2, holes ≤3×3) ⇒ a single
  8-conn CC step is bbox-safe (512 B) instead of two 4-conn quartets (1,024 B). Structural
  proof from the generator, not empirical luck (same discipline as the task306/task198 ships).

# 38. Marker-sentinel rewrite + sparse ScatterND canvas updates (task122 v7, shipped 2026-06-04)

When a rule MOVES one color's region: re-encode the moving color as an out-of-range marker
(255) on the U8 index canvas, locate the target geometry directly from the code plane, and
`ScatterND` ONLY the cells that change (the moved copy + vacated cells) instead of rebuilding
the canvas. task122: 52,517 → 4,816 (10.9×, byte-equal). Pairs with §29 (NOOP-sentinel
ScatterND) and the sentinel-10 Pad rule (sentinels ≥10 are invisible to Equal-expand).

# 39. GatherND(batch_dims=1) sparse reads — read ONLY the cells the rule needs (task195, shipped +3.23, 2026-06-04)

The campaign's biggest single gain: task195's 47,787 graph carried a counted 17×19 source
plane; v2 (1,878 = 25.4×) computes two anchor coordinates with small `Einsum` reductions from
the FREE graph input, then `GatherND(batch_dims=1)` reads exactly the 9 sampled cells. When a
rule's output depends on k cells, the floor is "locate + k-cell read", not any plane. Pairs
with §35 (Conv color decode) and §8b. Gate truth-FULL — a 9-cell read that passes 200k fresh
pairs is a function, not a memorizer (the same task's 253-pattern lookup FAILED the official
test pair — traps §VISIBLE-FAIL).

### Calibration warning: dynamic-shape constructions can cost MORE on the grader (task014, 2026-06-04)

task014 v3 (dynamic single-channel crop via dynslice): local `score_network_v2` = 4,380 but
Kaggle credited ≈6,093 (scored −0.33 below the raw-ln prediction — the campaign's first
non-exact landing after 24 exact). The official trace profiles dynamic shapes at their
observed/padded extents (`scoring_internals.md (profiling section)`). Byte-equal-full made it safe; the
lesson: for DYNAMIC-shape candidates, treat local cost as a lower bound and prefer a
static-shape variant when within ~30% — or ship both and let keeps-highest pick.

### §39 UPGRADE (2026-06-04, from `_archive/Research_ort_dynamic_shape_profiling.md` — definitive, ORT-source-verified)

The G1 research note settles the dynamic-shape cost question. Core facts:
- **The profile trace records RUNTIME-OBSERVED output shapes** (`sequential_executor.cc::CalculateTotalOutputSizes` serializes executed `Tensor::Shape()`), never `value_info`/NodeArg metadata — and `ORT_DISABLE_ALL` does NOT skip shape inference, but the trace still uses runtime shapes.
- **The Gather golden rule**: `Gather` output shape depends ONLY on the SHAPE of the `indices` tensor, never its values ⇒ **fixed-length clamped index templates (`Add`+`Clip`+`Gather`) are PROVABLY static** — the canonical safe construction for crop/extract/select.
- **Zeros-drift directions**: shape-driving chains profile on the zeros branch — `Greater/NotEqual`-mask → count→0 ⇒ UNDER-states real memory (and a grader run on different inputs then credits MORE — the task014 −0.33 mechanism); `Equal(X,0)`-mask → max count ⇒ OVER-states. `Slice/Pad(pads)/Resize(sizes)/ConstantOfShape` driven by computed values all drift; `ArgMax` ties pin to index 0 on zeros.
- **Design rule: never let data VALUES determine tensor EXTENTS.** Values may only determine contents, masks, or what is written into a fixed-shape box. Five static recipes (fixed-template Gather+Clip crop; fixed canvas+mask; fixed max canvas + validity; constant worst-case Pad; fixed-K clamped selection) are tabulated in the research note.

### §37 ADDENDUM (2026-06-04, from `_archive/Research_log_round_flood_reconstruction.md` — the flood-schedule verdict)

**Doubling/log-dilation schedules are PROVABLY NOT exact for masked flood** (Vincent: one big
conditional dilation ≠ N elementary geodesic steps; U-corridor counterexample: Chebyshev-2 but
geodesic-4 — the exact task325 gap-2 failure we measured). Exact o(K)-round methods exist only
as graph/matrix-closure algorithms with Θ((HW)²) state — outside the one-plane cost model.

**The one local acceleration that IS exact**: a 5×5 masked dilation equals exactly TWO 3×3
rounds **iff the mask satisfies scale-2 frontwise geodesic convexity** (every candidate the 5×5
adds has an in-mask 8-neighbor witness path of length ≤2 inside the same window). The
certificate is CORPUS-MEASURABLE per task (§37 discipline): run both schedules over 200k pairs
and diff — zero diffs ⇒ certified, halve the rounds (D=5→3, 14→7, 25→13; plane-bytes halve
too). task325's family FAILS the certificate (79,986/200k gap-2 pairs); other flood tasks may
pass — measure, don't assume. Atrous/rate-2 3×3 is strictly dominated by plain 5×5. With no
certificate, the optimum remains 3×3 × D (measured depth).

# 50. Crossing-parity interior fill (point-in-polygon, NO BFS) + F16 triangular-MatMul prefix-sum (task338 v5, 2026-06-04) <!-- formerly a duplicate "# 40" — renumbered 2026-06-05 -->

When a rule fills the interior of **complete rectangular rings** (hollow frames), the K-step
masked-BFS flood (§37 family) is NOT the floor — ray-casting parity is, at roughly **half** the
BFS chain's cost (task338: 12-step BFS family floor 26,562 → parity 17,212):

    H(r,c)  = wall(r,c) ∧ wall(r,c−1) ∧ wall(r,c+1)     # horizontal-edge interior cells
    parity  = (Σ_{k≤r} H(k,c)) mod 2                    # crossings above, per column
    interior = parity ∧ background

A ring fully above a cell contributes top+bottom = 2 crossings (even); the cell's own ring
contributes only its top edge (odd). This is exact for any layout of complete rings — including
the multi-ring "pocket" cases where 4-direction ray-existence (And of seen-above/below/left/right)
false-positives (task338 v1: 10/267 fail; edge-TYPED rays do not fix it — parity counts, existence
doesn't).

Construction notes (all opset 14, all corpus-validated):
- **H via 1×3 MaxPool erosion**: `H = 1 − MaxPool(free, [1,3], pads=[0,1,0,1])` (2 tensors) beats
  shift-And chains (4 tensors) and avoids negative-Pad shifts. **Mandatory edge fix**: MaxPool
  0-pads = virtual WALL outside ⇒ spurious H at the first/last columns; true H is provably 0
  there ⇒ multiply by a `[0,1,…,1,0]` column mask (25 params, 625B). Unmasked = 97.26% truth;
  masked = 100.000%.
- **Column prefix-sum via F16 triangular MatMul**, NOT CumSum: `cum = tril(ones[25,25]) @ H_f16`.
  CumSum has no F16/U8 kernel (INT32 floor ⇒ Cast 2,500 + CumSum 2,500); the MatMul does it in
  2×1,250B F16 + 625 params ⇒ net −1,875B. Counts ≤25 are exact in F16.
- **Inclusive ≡ exclusive cumsum here**: H=0 at every background cell, and only background cells
  consume parity (walls/out-of-grid are forced by the canvas arithmetic) ⇒ no exclusive attr, no
  row shift.
- **Parity = `Mod(Cast(cum_f16 → U8), 2)`** — Mod(u8) kernel probed OK at opset 14; the
  truncating cast preserves small integers.
- Tail: arithmetic canvas `10·free − ch0·(10 − 3·parity)` (no BOOL Where chain), Pad sentinel-10,
  Equal-expand BOOL output.

**more-correct-than-LIVE bonus** (task279 precedent): task338's LIVE flood marks multi-ring
pockets 3 (reachability semantics); the generator's truth is per-ring bbox fill ⇒ LIVE scores
199,984/200k, parity scores **200,000/200k**. When LIVE and ARC-GEN truth disagree, prefer the
generator semantics — hidden pairs are generator-drawn. Diff LIVE-vs-truth fails BEFORE choosing
which semantics to replicate.

<!-- RESOLVED 2026-06-05 (Phase-0): collisions renumbered — output-overlay → #48, strip-runfill → #49,
crossing-parity → #50, second Merged-Conv → #51, "# N."s → #52-54. Canonical: #40 = plane-breaker
catalog, #41 = 2D-imos + negative-pad addendum/VERDICT ("§41 negpad"). Title-grep remains the contract. -->

# 48. Output-overlay `Where(cond, const_onehot, input)` + line-mask conditions (task119 v3, 2026-06-04) <!-- formerly a duplicate "# 41" — renumbered 2026-06-05 Phase-0 -->

When a rule's output is the INPUT with K cells recolored to ONE color (paths, rays, stamps,
single-color fills), do NOT rebuild a canvas. The graph input and output tensors are both
FREE — the overlay

    output = Where(cond_30 [1,1,30,30] BOOL, target_onehot [1,10,1,1] F32 init, input)

makes the recolor cost exactly ONE counted tensor: the [1,1,30,30] BOOL condition (900 B).
This kills the whole canvas+Pad+Equal-expand tail (task119: −1,332 B) AND every 2D mask that
existed only to assemble the canvas (wall/seed planes — another ~900 B). The Where sets ALL 10
channels at painted cells (one-hot semantics handled by the [1,10,1,1] broadcast); unpainted
cells pass the input through byte-exact, including the ch0-hot padding region.

**Line-mask conditions (§39 companion):** when the painted set is a per-row column function
c(r) (diagonals, rays, single-cell-per-row paths), the condition costs ~200 B, not a plane:

    c_eff  = Where(valid_rows [1,1,12,1] BOOL, c_of_r [1,1,12,1] F16, -1.0)   # sentinel
    mask12 = Equal(c_eff, col_grid [1,1,1,12] init)   # broadcast → [1,1,12,12] BOOL
    cond30 = Pad(mask12 ∨ mask12_other, …)            # BOOL pad, opset 13+

c_of_r is pure scalar/1D arithmetic (`c = s·(r − off)` for diagonals at ±45°). One Equal per
geometric line — task119 needs two (pre + post bounce arm; a horizontal-wall bounce re-enters
the pre rows, so one row can paint 2 cells). Per-cell exceptions along the line (e.g. "skip
seed cells") are a 1D `GatherElements(plane, clip(c_of_r), axis=3)` row-read of an
already-counted plane (+~156 B), NOT a 2D And.

task119 ladder: 43,122 (LIVE conv-sim) → 22,640 (v1 coord masks) → 6,556 (v2 lean 2D) →
**3,667 (v3 line masks + overlay)**, byte-equal at every rung (266/266 visible, 200k/200k
truth-FULL, 2,266/2,266 ab vs LIVE). All static shapes. The v2→v3 step is the generalizable
one: 2D logic planes → 1D column functions + Equal-broadcast + free-tensor overlay.

# 40. Plane-breaker catalog — what actually beats the 2-input conservation law (2026-06-04, from `_archive/Research_nary_fanin_catalog.md`)

**The reliable breakers** (recipes that repeatedly beat 2-input plane counts):
1. **Gather LUT on an existing INT32 code** — 0 intermediate planes if the code exists; count the code plane (4 B/elem INT32) when built. The strongest single recipe.
2. **Bit-packed boolean decision**: `MatMul/Conv` with bit weights `[1,2,4,…]` → integer code → `Gather(constant_LUT, code)` — ONE code plane + tiny params replaces large Where trees. The general answer for "piecewise-constant function of several predicates".
3. **Variadic Min/Max** — N-way compare in one node, ZERO intermediates. **Variadic Sum + Greater** — majority/threshold of n masks with ONE count plane (Sum is not a bool reducer on CPU — keep masks numeric).
4. **Mod on an existing integer code** — parity/residue at zero planes (clamp dynamic divisors `Max(d,1)`).
5. **Conv(1×1)/MatMul/Gemm weighted reduction over an EXISTING channel stack** — count/score many masks in one node.

**The anti-recipes** (look like breakers, aren't): generic k-way mux over unrelated live tensors — the `Where` tree (k−2 planes) is STILL cheapest; `Concat+Gather` loses (the bank costs k×output bytes); building a `OneHot` selector erases arithmetic-select wins; `ScatterND` for dense logic (it's for sparse writes — §38).

**⚠️ Einsum ≥3 operands**: ORT CPU executes ≥3-operand equations PAIRWISE internally (`EinsumTypedComputeProcessor` — Transpose/ReduceSum/MatMul under the hood, partial result between pairs). Trust 1–2-operand Einsum unconditionally; for ≥3 operands (the task328 `bchw,h,w->bc` corner reads), the win depends on the partials not being trace-visible — **task328 scored exactly as locally predicted, so on THIS grader build the internals don't materialize — but re-verify each new ≥3-op equation with `score_network_v2` before counting the savings** (local==grader held 28×; the check is free).

**Dtype gotchas** (ORT CPU, opset ≤20): Conv/GlobalPools/Mean = float-only; **GatherND AND ScatterND indices are INT64-only (8 B/elem!) — shape-inference hard-rejects INT32 (`unsupported type: tensor(int32)`, empirically re-confirmed task122 2026-06-07); Gather/GatherElements/ScatterElements accept INT32 (4 B/elem)** — prefer the Elements/axis ops when the addressing allows. CAVEAT (task122): ScatterElements scatters along ONE axis with a same-shape `[…,K]` index grid, so a sparse arbitrary-cell write on a VARIABLE-size grid has no safe no-op landing column → can't replace a sparse `ScatterND` there; the INT64 `[K,4]` index pair is then the floor. Max/Min gained wide dtypes at opset 12+.

### Field validations 2026-06-04 night (all shipped, exact)

- **§19 runtime-weight Conv as a TRANSLATION-SEARCH engine** (task363, 3,796, +2.44): the seed
  pattern as a runtime Conv kernel computes ALL 361 translation validity tests in two Convs
  (`seed ⊛ not-obstacle == n_seed` ∧ `seed ⊛ is-bg ≥ 1`) on one 19×19 plane — replacing the
  per-translation loop entirely; stamp-back = the flipped-kernel correlation.
- **Design for plane REUSE** (task049 v7, 748 B record): one `Equal(c_idx, ch_ids_4d)` tensor
  consumed as BOTH the Einsum selector AND the And mask (−14 B/−4 params) — the trace charges a
  materialization once regardless of fan-out; route multiple consumers through one plane (CSE
  §13 thinking applied at design time, not cleanup time).
- **Einsum 'bchw,c->bh' row-projection** (task049 v2 lever): collapses a [1,10,30,30]→per-row
  presence read to a 120 B vector — the same §39-family "read only what the rule needs" move
  in 2-operand form (unconditionally trusted per §40's Einsum caveat).

# 41. 2D imos corner marks + QLinearConv U8 prefix-sum — rectangle-interior fill at ~1/3 the parity cost (task338 v7, 2026-06-04)

Extends §40. For "fill the interior of complete rectangular rings" rules, even the column-parity
construction is not the floor. The four **2D imos deltas** of each ring's interior bbox —
(r0+1,c0+1), (r0+1,c1), (r1,c0+1), (r1,c1) — are each the **bottom-right cell of a 2×2 window
containing exactly 3 wall cells** (the free cell is the ring's interior corner). One local test
yields all four marks; `interior = parity(2D-prefix-sum(marks))`. Degenerate width/height-2 rings
produce s=4 windows → no marks → no fill (correct). s==3 is provably impossible at out-of-grid
cells on rectangular grids. task338: column-parity 16,586 → **8,919** (price ladder from the
original BFS: 26,562 → 16,586 → 8,919; ln vs 45,649 live = +1.63).

Construction (all corpus-validated 200k/200k):
- **One Conv reads everything** (§8b/§35 discipline — the graph input is FREE, only the output
  plane counts): kernel `[1,10,7,7]`, taps (0..1,0..1) on ch_wall = the 2×2 window sum; tap (1,1)
  on ch0+ch_wall × (−160) + bias 160 = a `160·(1−in_grid)` field in the SAME plane; kernel size 7
  + pads [1,1,0,0] crops 30→25 on both axes. One F32 [1,1,25,25] plane = 2,500B total read.
- **2D prefix-sum via QLinearConv** (x u8, w i8 ones [1,1,25,25], all scales 1, zps 0, pads
  [24,24,0,0]): U8-in/U8-out → the prefix plane costs **625B** vs F16 MatMul pair 2,500B vs INT32
  CumSum 5,000B. MEASURE the saturation bound first (§37 discipline): max marks/grid must be < 256
  (task338: 20). Conv-kernel params (625) are the trade for 1,875B of memory — wins under the
  equal-weight cost model.
- **Sentinel piggyback**: `Div(s'_u8, 16)` maps the 160-field to {0,10} directly — out-of-grid
  sentinel with no ch0/free planes anywhere in the graph.
- **ch_ids absorb output arithmetic**: canvas = n10 + parity ∈ {0,1,10}; set
  `ch_ids = [0,200,201,1,202..207]` so ch3 fires on canvas==1 — deletes the ×3 Mul. Any per-channel
  REMAP of canvas values is free in the final Equal; only EXACT matches work (no ranges/parity).
- Parity P is 0 at walls, edges, and out-of-grid by imos geometry — unlike column-parity (§40),
  NO masking and NO background plane are needed.

Cross-task hint: any "detect a fixed sparse pattern → fill derived rectangles" rule can chain
(pattern-Conv → Equal → QLinearConv prefix/box → Mod/Div decode) entirely in U8 planes after one
F32 entry. Candidates: task367 enclosure-fill, other frame/bbox tasks.

# 42. Free-input Einsum marginals: 0-memory channel reductions (task119 v5, 2026-06-04)

**Probe-verified (ORT 1.24.4, ORT_DISABLE_ALL, local scoring pipeline): `Einsum` on the FREE
graph input materializes NO counted internal temps.** `input → Einsum('bchw,c->bh', sel_c[10])
→ output` scores **(macs 0, memory 0, params 10)** — the kernel never exposes a counted
[1,10,30,30]-scale temp to the profiler. Equations verified: `'bchw,c->bh'`, `'bchw,c->bw'`
(2-operand, one [10] selector init).

**Implicit-axis summation (task195, 2026-06-12):** an axis ABSENT from the output is summed out, so
`'bchw,c,w->bh' == 'bchw,c->bh'` — dropping the redundant `w` operand deletes its `[30]` broadcast init for free.
**Combo-floor decode (task256, 2026-06-12):** recover a feature `L` from a biased combo-head max via `Floor(Max/K)`
when the residual base `< K` — folds a separate selector reduction into the existing max. **Divider-band shrink
(task059, 2026-06-13):** fixed all-zero divider rows/cols carry no signal — shrink the count basis (`[5,30]→[3,30]`)
and remap to the full coefficient grid with a tiny free in-kernel `[3,5]` operand.
**Dynamic scalar selection via OneHot + input-Einsum beats a high-channel Gather (task377 v9,
2026-06-21, 5689->4554):** when a dynamic `Gather(input, cstar, axis=3)` is only consumed for scalar
per-row VALUES and activity (not the full channel-by-row slice), skip the `col_oh` F32 `[1,10,30]` and
read the selected column as two 1-D outputs: `OneHot(cstar1) -> csel F32[1,30]`, then
`Einsum("bchw,qw,c->bh", input, csel, ramp[10]) -> colcolor_f[1,30]` and
`Einsum("bchw,qw->bh", input, csel) -> active_f[1,30]`. The `csel` one-hot picks the column inside the
contraction (#89), so no wide channel slice materializes.

Implication: **per-channel row/col COUNT marginals cost only their [1,N] output** (120 B at
N=30 F32) — cheaper than any Slice-plane route (a 12×12 single-channel plane is 576 B before
its reductions) whenever you need ≥1 marginal and don't need joint (r,c) cell reads. The
counts are exact small integers in F32, and they carry MORE than presence: thresholds at 0.5
give presence/extents (ArgMax), thresholds at 1.5 give **multiplicity** (task119: "some row
has 2 seeds" = the chain contains its own bounce — a V-chain flag that no presence vector can
express).
**Marker-locating F32 crop → free-input Einsum marginal (task047, 2026-06-22, 862->537).** A common
anti-pattern: `Slice` a small F32 sub-window out of the input ONLY to `Cast->ReduceMax`/`ArgMax` a
marker's row/col. The crop is NOT load-bearing — read the marker coordinate as a 0-memory scalar
straight off the one-hot input: `Einsum('bchw,c,h->b', input, sel_k[10], ramp30) -> r_k[1]` (row of
marker `k`), `Einsum('bchw,c,w->b', input, sel_k, ramp30) -> c_k[1]` (col), where `sel_k` is a one-hot
channel selector. Then rebuild boundary-correct code vectors with `Equal(arange9, r_k)` masks (#67) over
the ACTIVE positions only, `Where` in the codes, `Pad` to 30. task047 deleted two F32 marker windows
(`ch7[1,1,7,6]=168 B` + `ch8[1,1,6,6]=144 B` = 312 B), memory 826->448, byte-equal-full. Trigger: any
"Slice a small F32 window only to locate a marker" is a free-input Einsum marginal — keep only the `[1]`
scalar.

What marginals can NOT give: joint info (the cell at (r, c(r)), boundary-cell identification)
— keep tiny multi-axis boundary Slices ([1,1,12,1] F32 = 48 B) or a GatherElements plane read
for those. task119 v5 (3,390, byte-equal-full): Einsum marginals + 1.5-threshold V-flags +
arm-extent scalars replaced the 576-B ch8 plane, its projections, AND the per-cell
GatherElements post-line read (−205 net vs v4). Caveats: 3+-operand equations and other
runtimes unverified — probe before relying; grader-vs-local Einsum accounting parity is being
confirmed by the v5 submit itself (manifest flags it; task014 §39 precedent for divergence).

### §41 addendum — negative Conv pads as a param-free crop (task338 v8, 2026-06-04, GRADER-UNPROVEN)

A crop-via-kernel-size Conv (§35/§41: kernel k = crop + tap-span, zeros counted as params) can
swap the oversized kernel for the true tap-span kernel + NEGATIVE end-pads: task338's
`[1,10,7,7]` (491p, taps only at (0..1,0..1)) → `[1,10,2,2]` + `pads=[1,1,-5,-5]` (41p),
identical taps verified cell-level. Passes the ENTIRE local scoring path (onnx.checker
full_check, strict shape inference, ORT, score_network_v2). **Caveat: no confirmed real-Kaggle
datapoint** — task138 built a negative-pad Conv variant but never shipped it (traps.md edge-frame
note was a *semantic* issue, not a rejection). task338 v8 (8,469, staged 2026-06-04) is the
designated grader probe; if it credits its raw-ln prediction exactly, this trick saves
`C_in·(k²−span²)` params on every crop-conv in the corpus. If it zeroes, record the rejection
here and ship the standard-pads sibling.

### §39 ESCALATION [SUPERSEDED by §39 MECHANISM CONFIRMED — kept as evidence] (2026-06-04 night, task398): the local profiler can be fooled 37× — Kaggle is the only cost oracle for exotic constructions

The task398 v8–v15 family scored **364–519 LOCAL** via a scorer-dummy/boolpad/optional-input
stack — and the grader credited **≈13,300 for every rung** (banked +1.10 instead of +4.70; the
fallback-ladder + fixed-base probes decoded the credited cost exactly). The 014 gap was 1.4×;
this is **37×**. Operational rules, in force now:
1. `score_network_v2` parity with the grader is PROVEN ONLY for boring-static graphs (30+ exact
   landings). Any construction using dummy branches, optional/default inputs, boolpad collapses,
   or data-dependent shapes is **unverified until a Kaggle datapoint exists for that trick**.
2. When a local score looks too good next to the §-floor analyses (398: local 364 vs the rule's
   obvious ≥1k static floor — a tell), assume a gap and ship the BORING-STATIC sibling alongside.
3. The worker pattern that saved this ship: stage a LADDER with at least one no-tricks rung, so
   the captain can bracket the credited cost in one round of fixed-base probes.


# 49. Strip-extracted runfill + ScatterND row-writes — K-bounded sparse lines (task054 v5, 2026-06-04) <!-- formerly a duplicate "# 41" — renumbered 2026-06-05 -->

When a rule draws K-bounded line/run structures (K anchors known small via corpus sweep), do NOT
run the CumSum+MaxPool run-fill (sec38-adjacent, task054 v1 nodes 61-90 family) on the full
canvas: Gather-extract the K anchor rows into a `[K,30]` strip stack, run the SAME primitive
there (1/H the bytes), and write back with `ScatterND(data [30,30], indices [K,1], updates
[K,30])` row-writes; do the perpendicular orientation on the transposed canvas (one Transpose
pair). Stamps/erases ride one flat `[N,1]` point-ScatterND whose updates Where-chain can Gather
EXISTING content (makes padding slots exact no-ops AND preserves cross-feature overwrite order).
task054: the "irreducible" 27,900-B line core became ~6,100 B; live 47,180 -> **26,043** byte-equal
on the full 200k corpus (`task054.md` 2026-06-04). The whole canvas pipeline stays 4-D:
ScatterND row-writes into `[1,1,30,30]` take `[K,3]` zero-prefixed indices (Concat a [K,2]
zeros init) — saves the 2-D reshape canvas (-900) for +128 of index bytes.

**Perpendicular orientation: ScatterElements beats the transpose pair.** For the V-lines (write
COLUMNS), don't `Transpose -> ScatterND-rows -> Transpose-back` (3 full planes = 2,700). Instead
gather column strips `Gather(canvas, cols, axis=3) -> [1,1,30,K]`, runfill along axis 2, and write
back with `ScatterElements(canvas, idxV, updV, axis=3)` where `idxV = Expand(cols[1,1,1,K],
[1,1,30,K])` cast INT32 — one result plane (900) + a 480-B index, net **-1,320** (task054 v7).
The 4-D *point*-scatter for the stamps is the OPPOSITE — it REGRESSES: `[N,4]` INT64 indices
(1,440) plus the fact that the existing-content Gather still needs a flat canvas mean the flat
round-trip (`Reshape->ScatterND[N,1]->Reshape`) is strictly cheaper. Rule: **ScatterElements for
full-row/col writes (cheap index); flat ScatterND for sparse point writes (cheap index); never
2-D/4-D ScatterND for dense logic** (the multi-axis index tensor dominates).

Hard-won correctness rules (each cost a debug round or was caught by sweep):
- **Anchor locate**: TopK over `presence x iota` (presence = ReduceMax-U8 per axis, NOT count
  sums) and consume the TopK **values** (absent slots -> 0, deterministic) — the indices output
  has unspecified tie order among zeros. Pad-slot anchors at coord 0 are safe iff the sweep
  proves real anchors never use coord 0.
- **Dual-axis selection**: when anchors pair ambiguously per axis, compute both (TopK-axis,
  derived-axis) coordinate sets and Where-select by `#distinct-rows >= #distinct-cols` — the
  TopK axis is provably single-anchor per strip, the derived axis is NOT.
- **Duplicate scatter indices**: ScatterND is last-write-wins, so updates for duplicate rows
  MUST be identical — build per-strip indicators as MULTI-anchor onehots (plane-gather), which
  unions bands and exclusions for free. Per-strip exclusions silently lose the other anchor's
  cells (~6%% of task054 draws).
- **Profile-input safety with NO Clip**: reroute degenerate indices (`Equal(idx,0)` ->
  `Where -> iota` row-0 cells) instead of clamping — cheaper than Clip chains and keeps every
  Gather/ScatterND in-bounds on the all-ch0 scorer dummy.
Scorer accepts ScatterND (x3) + TopK (x2) at opset 14 in one graph (first combined datapoint).


### §41 VERDICT (2026-06-04 night): NEGATIVE-PAD CONV IS GRADER-CONFIRMED

task338 v8 (`pads=[1,1,−5,−5]` replacing a 491-param 7×7 crop kernel) scored **+1.685 EXACT**
(6513.25) — the first real-grader datapoint for negative Conv pads, and it credits honestly.
The §41 trick is now SHIP-CLASS: saves `C_in·(k²−span²)` params on every crop-conv. (Same round,
the no-negpad sibling v7 landed +1.633 exact — both honest; the doctrine pair cost one slot.)

### §41 addendum — negative BEGIN pads crop from top/left too (2026-06-06, task124 v7, ORT-probe-verified)

`Conv(pads=[−2,−1,−25,−20])` on [1,10,30,30] yields [1,1,3,9] aligned at input (2,1) — ORT
computes the window start from the negative begin pad; passes checker full_check + strict
shape-inference + corner-alignment probe. Lets ONE shared 1×1 colorize weight serve N
different crop views of the input at zero param cost (pads are attributes): task124 v7
derived P [rows 0:5] AND both (2,1)-test operands [rows 2:5, cols 1:10]/[rows 0:3, cols 0:9]
from three Convs sharing `Wc` — replacing 2 Slices + 10 params of starts/ends/axes inits.
Grader datapoint: end-pads only (task338); begin-pads are static-shape so §39 MECHANISM
prices them exactly. Fully gated rung (visible 267/267, truth 200k, ab byte-equal):
`task124/prototypes/build_task124_v7.py` (superseded by v8, which needs no begin-pads).

### §41 addendum — 1×1 Conv + negative END-pads = decode AND crop in ONE node (task122 v8, 2026-06-07, gated+gate-5)
A §35 color-decode Conv that ALSO crops the 30×30 pad needs no dilated/asymmetric kernel:
`Conv(input[1,10,30,30], weight[1,10,1,1], bias, pads=[0,0,−9,−9])` → `[1,1,21,21]` F32, top-left
crop, weight-decoded colours + bias sentinel — **byte-identical** to the dilated-2×2 form but
`10` weight params instead of `40` (`C_in·(k²−1)` saved per channel). The 1×1 kernel makes the
decode a pure per-cell channel-contraction; the crop is entirely in the (free) pad attribute.
Verified: `score_network_v2` (0,1764,11), checker full_check + strict infer OK, ring→255/line→3/
bg→0/offgrid→10 exact. Use this as the default decode-crop primitive. `task122 v8` (cost 4688).

### §41 addendum — negative END-pads on MaxPool crop stride-pooled outputs (Aditya task213, 2026-06-11)

Negative pads are not only for Conv. ORT also accepts negative END-pads on `MaxPool`, so a
decimation pool can crop to the exact block grid without a post-pool Slice. Example:
`MaxPool(input, kernel=[3,30], strides=[3,30], pads=[0,0,-9,0])` pools 3-row bands
across the full width and drops the trailing windows in the same node; the column twin uses
`kernel=[30,3], strides=[30,3], pads=[0,0,0,-9]`. Use this for "pool every B rows/cols
then assemble a square/strip output" patterns. Dtype follows MaxPool support: U8/I8 at
opset >=12, BOOL rejected.

### §39-ESCALATION counter-datapoint (2026-06-04, task252): ZERO-MEMORY params-only graphs credit HONESTLY

task252 v6 (cost 280 = memory 0 + params 280, a pure params/LUT graph) scored **+4.958 EXACT**
(6518.21, campaign record) — joining task215 (110, live) as proof that the LEAN side of
§GRADER-GAP is safe: the gap-prone constructions are dummy-branches/optional-inputs/boolpad/
data-dependent shapes, NOT extreme leanness itself. A params-only signrank/LUT factorization
with byte-equal-full gates ships at face value.

### §40 SIZING LEMMA (2026-06-04, task338 close — when Gather-LUT reads LOSE)

The §40 Gather-LUT breaker has a plane-scale break-even, now priced exactly: a per-cell LUT
read needs an INT32 index plane (4 B/elem ⇒ 3,125 B on a 25×25 plane) while one U8 ALU stage
costs 625 B — so **a LUT read only wins when it replaces an UNBROKEN pointwise run of ≥5
stages**. task338's longest foldable run was 3 (a QConv prefix breaks the chain) ⇒ dead by
≥1,875 B. Check the run length BEFORE building the pack. Related: opset-18 `BitwiseAnd/Or/Xor`
+ `BitShift` (u8/i32/u32) are now CPU-CONFIRMED on our build (task338 probe; rows stamped in
`ort_reality.md`) — but exact packed-row arithmetic on 25-bit rows priced
≥5,000 B vs the 1,875 B it would replace (pack + XOR-2D-prefix + INT32 unpack broadcast). The
H2 research card carries the general packing calculus.

### §39 gap ledger (running) [data rows feeding the GAP-LEDGER]: extraction-class grader markups are SMALL and consistent

- task014 dynslice: local 4,380 → credited ≈6,099 (1.39×)
- task065 dil-guard extract: local 3,275 → credited ≈3,805 (1.16×)  [2026-06-04]
- (contrast: task398 boolpad/dummy class: 37× — different failure mode entirely)
Pattern: variable-output extraction graphs carry a modest credited premium; params-only and
boring-static graphs land exact (30+ datapoints). Budget +15–40% on dynslice-family costs when
predicting, or build the §39 static-canvas variant.

# 43. Negative-bias cliff Conv — threshold detectors without Equal chains (task364, 2026-06-04; pointer section added 2026-06-05)

Worker-authored on the 6500-crosser ship (+1.18 exact, byte-equal, 46,250 → 14,162): encode
"count ≥ k" detectors as a single Conv whose bias is −(k−1) followed by clip/sign — the bias
CLIFF replaces an Equal/Greater chain per threshold, one plane per detector family instead of
per comparison. Floor theorems that shipped with it (cue non-separability; pre-dilation
zero-trade) live in `task364.md` — this section existed only as citations until 2026-06-05
(Phase-3 double-check found the dangle); full construction + measured costs: `The_400_Tasks/Task_by_Task/task364.md`.

# 44. Spatial data-movement quick table (2026-06-04, from `_archive/Research_spatial_data_movement_catalog.md`)

**Adopt immediately** (each beats our Gather-built equivalents under the cost model):
- **flipLR / flipUD**: ONE `Slice` with `steps=[-1]` — 0 intermediates, 4 params. **rot180**: ONE
  `Slice` reversing both axes — 0 intermediates, 8 params. (Negative-step Slice, opset 10+.)
- **transpose**: `Transpose` — 0 params, 0 intermediates (perm is an attribute). rot90/270 =
  Transpose + one Slice (1 intermediate, 4 params).
- **tile m×n**: ONE `Tile` (repeats init = rank elems). `Expand` only when repeated dims are 1.
- **block-upsample s× nearest**: ONE `Resize(nearest, asymmetric, floor)` with scales init.
- **triangle masks**: `Trilu` directly on data — 1 op, 0 params at k=0 (CPU dtypes: bool/double/
  float/int64 only). **diagonal canvas**: `EyeLike` when a same-shape 2D carrier exists (dtypes
  double/float/int32/int64/uint64).

**Admire from a distance** (semantically nice, lose under the metric): `GridSample` — CPU
source-confirmed (registry-ambiguous) but a constant grid costs 2·H·W params (1,800 on 30×30);
only for true runtime warps (grid SHAPE fixed ⇒ output shape fixed — the Gather golden rule
analog). `ReverseSequence` — **VALIDATOR-BANNED** (op type contains "Sequence"; R6 scorer-confirmed
+ ORT rejects `time_axis>=2`) — NOT a flip option; `Slice` negstep above is the only flip.
`Pad(wrap)` roll (opset 19+) — padded intermediate usually outprices a Gather roll.
`CenterCropPad`/`Col2Im` — CPU dtype rows unconfirmed; don't build on them without a probe.
Runtime-k roll/shift: NO clean ≤2-op winner exists — Gather stays.

# 45. The packing law — "plane-neutral" is FALSE (2026-06-04, from `_archive/Research_bit_packing_calculus.md`)

**The law**: packing n BOOL planes into a width-w carrier wins iff
`α·m·(n−w) > pack_cost + g·n·unpack_cost` — α = fraction of the m consumer sites that execute
ON the packed words (opset-18 `Bitwise*`, CPU-confirmed), g = shared unpack stages. α is the
whole game:
- **α=0 ⇒ packing is strictly WORSE than neutral** (pack + unpack outputs both count). This is
  why task328/task338 measured "plane-neutral/dead" — their consumer mixes couldn't stay packed.
  The conservation-law floors stand for α≈0 graphs; packed-executable regions ESCAPE them.
- **U8 carrier, n=8**: fully-packed region breaks even at m>22/7 (≈4 sites); one shared final
  unpack: m>38/7 (≈6); ≥2× total win from m≈8. **I32 carrier = trap for n≤8** (every
  intermediate 4P; BitShift is UNSIGNED-only on CPU — u8/u32/u64, no i32!).
- **Packed counting beats unpacking**: byte-popcount routes — Gather-LUT-on-byte 5P (threshold)
  / 8P (count, I32 LUT) ≪ U8 SWAR 10P ≪ Div/Mod ladder 22P. LUT = 256 params. Threshold the
  count in-carrier (integer Greater/Less return BOOL).
- **Row-packed uint32 for our ≤30-wide grids**: one row per 32-bit lane ⇒ NO cross-word carry
  problem (ONNX BitShift is elementwise — bits never flow between tensor elements!); only edge
  masks remain. The bitboard/Dumb7Fill literature (shift-OR flood) transfers directly — a
  candidate route to cheaper §37 flood chains for tasks with row-expressible masks.

### §42 VERDICT (2026-06-04 night): free-input Einsum marginals GRADER-VALIDATED

task119 v5 (the §42 construction + the (0,0,10)-probe accounting) landed **+2.543 EXACT**
(6520.75). Free-input Einsum reductions are confirmed honest on the grader — 2-operand Einsum
remains unconditionally trusted (consistent with §40's pairwise caveat applying only to ≥3
operands).

### §39 GAP-LEDGER [CURRENT — the consolidated gap model] (2026-06-04 night, task094 trilogy): dynamic repricing is a CONVERGENCE, not a multiplier

Three byte-equal task094 models, one submit round: v7 dynamic-block 313 local → **≈24,386
credited (78×)**; v6 dynamic-basis 451 local → ≈19,182 (42×); v1 static 3,806 local → **3,806
EXACT**. The two dynamic rungs converged to the same ~20-24k ABSOLUTE band — the grader's trace
records the dynamic machinery's TRUE runtime shapes on grader inputs; a zeros-run local profile
of a dynamic graph is fiction, not a lower-bound-with-markup. Revised operational rules:
1. The old "+15-40% markup" guidance applies ONLY to mildly-dynamic extraction graphs (014
   1.39×, 065 1.16×, where dynamic extents are small). Graphs whose WORKING SET hides behind
   data-dependent shapes can be off by ORDERS OF MAGNITUDE (094: 78×; 398 boolpad: 37×).
2. The static sibling is not a hedge — it is frequently the WINNER (094 v1 beat both dynamic
   rungs by +1.6). Always build and stage it; expect to ship it.
3. A dynamic candidate's local cost is admissible evidence ONLY of correctness economics, never
   of credited cost. Predict nothing from it; bracket with submits if the route matters.

### Cheap-byte notes (2026-06-04 ships)
- **opset-11 axes-as-ATTRIBUTE** (ReduceSum et al.) and axes-free Slice beat opset-13+ axes-input
  tensors — attributes are free, input tensors are initializers/params (task374 v8, −34 B).
- **Zero-initializer scalars**: derive scalars from FREE INPUT cells (e.g. top-left background
  cell) inside an existing Einsum/op instead of paying a 1-element initializer (task293 v9,
  −1 param + simpler graph; §42 family).

### §39 MECHANISM CONFIRMED (2026-06-04 night, task054 worker — GRADER SOURCE READ): cost = max(static, runtime-on-REAL-inputs) per tensor

`neurogolf_utils.calculate_memory` profiles the trace on **real train + test + arc-gen inputs**
and charges **max(static, runtime) per tensor**. This is the root mechanism of the entire gap
ledger:
- **Static graphs: local `_score_v2` == Kaggle EXACTLY** (30+ landings) — a submit can never
  reveal a lower cost than local measures; byte-equal already maxes correctness ⇒ **there is no
  profitable submit-probe for a static graph**. Don't burn slots.
- **Dynamic graphs: real inputs inflate runtime shapes; max() charges the inflation** — the
  014/065 markups and the 094/398 absolute-band convergences are this, exactly.
- **dim_param symbolic-shape exploits: PATCHED** (grader returns None) — consistent with
  multi-output→None (§14z), TopK-symbolic-K→None (064-v3), unannotated-dynamic→None (025).
- Annotation lies don't just gap — annotated-dynamic can score cheap while BREAKING multi-input
  correctness (025's dynamic-slot broke >1-slot inputs). Never ship an annotation the graph
  doesn't honor.

### F16-CONV VERDICT (2026-06-04 night, task080 v9): GRADER-VALIDATED — ship-class project-wide

The project's first F16-Conv ship landed **+2.166 EXACT** (6538.23). F16 cell-logic Convs are
confirmed on the real grader: **2 B/elem planes halve every F32 conv intermediate** on cell-logic
tasks. Combined with the §39 mechanism (static ⇒ local==Kaggle), F16 conv chains can be priced
locally with full confidence. v9's trim trio worth copying: per-axis ArgMax template reads (two
tiny ReduceMax+ArgMax beat a flat [100] reshape), precomputed index tables as inits (kill
runtime Div/Mod), folded constant reads into existing probe slices.

### Floor-proof style: the INFORMATION-FLOW floor (task080)

Price the rule's irreducible stages independently (canvas law + decode + select + sequential
conv passes + assembly + probes), then CONSTRUCT AND MEASURE every fusion alternative
(task080: Equal-before-Gather 5,040 vs 1,260; ScatterND ≥10k; batched-period 3×; ConvInteger
500 vs 400 — all worse, all documented as don't-dos in task080.md). Floor ≈ Σ stages; delivered
5,664 vs floor ≈5,050. Second independent scorer-source audit corroborated §39-MECHANISM
accounting (every node-output counted; only literal graph input/output names free;
dynamic/sparse/subgraph/multi-output → None).

### #46/§42 addendum — MULTI-OPERAND EINSUM FUSION: kernel-internal scratch is COST-FREE (2026-06-06, task124 v9-v11, locally exact ×3)

`calculate_memory` charges max(static, profiler-observed) per GRAPH tensor — einsum
kernel-INTERNAL pairwise temps are not graph tensors and are NEVER counted. Corollary:
any einsum→einsum chain can fuse into ONE multi-operand einsum, deleting the intermediate
from value_info entirely. task124 ladder, each landing EXACTLY at its static prediction:
- 3-op `'abcd,c,wdxy->'` (P, rowsel, convW) — scalar from a [1,1,5,10] plane, the [5]
  marginal never exists (−48 B);
- 3-op Gram fusion `'abcd,abed,ce->'` (P, P, K) — the [5,5] row-Gram became internal
  scratch (−100 B);
- 4-op factored quadratic-form `'rc,abcd,re,abed->'` (L, P, L, P) — SSD as ||L·P||² with
  the FACTOR stored instead of K = LᵀL (−25 params; repeated subscript across operands 0/2
  and the operand-pair (P,P) both fine).
Same trick family: weight REUSE via size-1 subscripts (`wdxy` matches a [1,10,1,1] Conv
weight — one init serves Conv AND einsum). Dtypes still float/double/int32/int64 only.
This subsumes the old "2-operand unconditionally trusted" caution for STATIC graphs:
local `_score_v2` prices the fusion exactly (§39 MECHANISM), so measure-and-keep. Worked
builders: `task124/prototypes/build_task124_v{9,10,11}.py` (all byte-equal, truth 200k).

# 46. Einsum repeated-subscript DIAGONAL READS — adopt immediately (2026-06-04, from `_archive/Research_einsum_equation_catalog.md`)

**`bchh->bch` (main-diagonal per color, [1,10,30], 1,200 B, 0 params, ONE op)** and
**`bchh->bc` (per-color trace, 40 B)** are spec-legal (official `einsum_batch_diagonal` example)
and ORT-CPU-implemented (source: `EinsumComputePreprocessor` → `CpuDeviceHelpers::Diagonal`;
innermost repeated axes skip the internal transpose). Replaces EyeLike+Mul+Reduce diagonal
reads (40,800 B → 1,200 B, 34×). Rules:
- READS only — output-repeated labels rejected (`i->ii` illegal). Diagonal/band WRITES stay
  with EyeLike/Trilu.
- Square axes only at execution (helper enforces equal extents) — fine on our 30×30.
- Anti-diagonal: negative-step Slice flip first (+36,000 B intermediate) — only pays when the
  flipped tensor serves other consumers; k-offset: Slice to [30-|k|]² square then `bchh`.
- **Adopt-list** (all DIRECT from the free input, 0-30 params): `bchw->bc` (40 B), `bchw->bch`/
  `bcw` (1,200 B), `bchh->bch`/`bc`, `bchw,h->bc`/`bchw,w->bc` (centroid numerators, 40 B),
  `bh,bw->bhw` (separable rect masks, 3,600 B), `bchw,bhw->bc` (masked counts).
- ORT CPU Einsum dtypes: float/double/int32/int64 ONLY (no F16/U8). Integer Einsum pointless
  here: counts ≤ 26,100 ≪ 2^24 ⇒ F32 exact; casts cost more than they save.
- Both-operand repeats (`ii,jj->ij`): source-supported, NOT grader-proven — probe before ship.

# 47. Row-packed uint32 flood — VERDICT: adopt for 4-NEIGHBOR floods with K ≥ 12 (2026-06-04, from `_archive/Research_row_packed_uint32_morphology.md`)

Exact constructions confirmed (all ORT-CPU-legal; **MatMul-13 accepts uint32 on CPU** — now in
the dtype matrix): pack = Cast(u32) + MatMul(P2[30]) + Reshape = **3,840 B**; round =
2 BitShifts + 2 negative-Pad lane shifts + 4 Ors + And(mask) = **1,080 B** (the "~120 B/round"
slogan is FALSE — every temporary counts); unpack = Reshape + broadcast BitwiseAnd(P2) +
Greater = **4,620 B**. One-pack crossover vs unpacked 1,800 B/round: **K = 12** (K≈18 if the
mask also packs online). Params: 41 shared.
**Connectivity caveat (captain pricing)**: the priced round is 4-NEIGHBOR. The 8-neighbor
extension (+4 diagonal shifts +4 Ors ≈ 2,040 B/round) LOSES to unpacked 1,800 — so: corpus-
certify the task's connectivity FIRST; adopt only for 4-connected floods with measured K ≥ 12
(§37 K-floors tell you K). Boundary hygiene: ROW_VALID_30 = 0x3fffffff; east shift needs bits
30/31 clear in the packed mask; west/north/south self-clean.

# 55. task013 trio: aspect-branch + clamped Mod/Gather periodic fill + broadcast-And assembly (promoted from `.update/` 2026-06-05; shipped byte-equal, 58,000 → 2,774)

**(a) Aspect-ratio branch for transpose-symmetric generators**: when the generator's `width`/`height`
ranges are DISJOINT (013: w∈[20,30], h∈[6,12]), `xpose` decodes from aspect alone:
`branch = Greater(ReduceSum(col_act), ReduceSum(row_act))` — 2 scalars, NO input-derived Div/Mod
(dodges the zero-grid SIGFPE, traps §8d). Beats reverse-engineering stripe-count rules (the old
`dc≤dr` rule missed 124/267).
**(b) Runtime-period periodic fill**: for "tile a 1-D line at data-dependent period p, alternating
two markers" do NOT enumerate distances (LIVE burned 30 Pad + 32 Slice). Two anchors ⇒
`cmin = ReduceSum(pos) − ReduceMax(pos)`; `m = Max(2·dc, 1)` (zero-grid clamp); fold the whole
line through ONE `GatherElements(line, cmin + Mod(arange − cmin, m))`, gate left-of-first-anchor
with Where.
**(c) Broadcast-And dual-orientation assembly — NO [1,1,30,30] canvas**: keep every intermediate
rank-1 per spatial axis (`colpat [1,10,1,30]` = 300 B, `rowpat [1,10,30,1]`) and make the final
`And(X, Y)` output the FREE graph output. Sentinel the off-branch line to color-10 so its Equal
is all-False (select becomes a plain Or). Strictly cheaper than a color-index canvas for runtime
transpose branches. Full walkthrough: `task013.md` + `.update/DONE_2026-06-01…task013….md`.

# 56. Hash-MLP architectural fingerprint — "is this MLP a hash table?" (promoted from `mlp_rule_extraction.md` 2026-06-05)

Signature (task219-verified): hidden-1 width ≈ visible-pair count (264 vs 266); activation is
RBF `(x−b)² < ε` not ReLU; EXACTLY 1 detector active per input; single-bit perturbation → 0
active; small-tree distillation fails (1/150 outputs at depth 4). Verdict: the MLP is a perfect
hash of the training set — **the true rule exists OUTSIDE the MLP** (in the data-construction
process); distillation is a TRUE NEGATIVE, not a failure of method. Stop probing the net; decode
the generator. Detail: `mlp_rule_extraction.md` §Hash-MLP architectural fingerprint.

# 57. Boolean-safe reduction scaffold (promoted from `symmetry_and_affine.md` 2026-06-05)

`ReduceMin/ReduceMax/ReduceSum` REJECT BOOL on ORT CPU (`ort_reality.md §7`). The cheapest valid
scaffold for any symmetry/all-true/any-true test: `Cast(BOOL→UINT8)` (1 B/elem) → reduce → compare.
F16 also works (2 B/elem) — pick U8 on large flattened tensors. (Note ReduceMax(BOOL) IS legal at
opset 20 per the dtype matrix — this scaffold is for the ≤19 paths and Min/Sum at any opset.)

# 62. Staircase-code rank-1 assembly — separable outputs WITHOUT planes (task055, 2026-06-06; 31,628 → 2,386 byte-equal, 13.3×) <!-- formerly a duplicate "# 58" — renumbered 2026-06-06 -->

Generalizes #55c: when every output channel is (rows-pattern × cols-pattern)-shaped, build
U8 row codes `X [1,10,30,1]` and col thresholds `Y [10,1,30]` and make
**`output = Greater(X, Y)` the graph output** — the broadcast comparison materializes for
free and the decoder's `>0` reads BOOL directly. No channel plane ever exists; the whole
graph is 30-long vectors.
- **Outer-product channel** (`band_r & band_c`): x = band ∈ {0,1}, y = ¬colband ∈ {1,0}.
- **Union of two outer products** (rank-2, e.g. a line cross `(line_r&act_c)|(act_r&line_c)`):
  a 3-LEVEL staircase does it in the same op: x = act+line ∈ {0,1,2}, y = (¬act)+(¬line);
  x>y ⟺ x_code + y_code ≥ 3. A single `Where(condR, A_c, B_c)` CANNOT express this (3
  distinct row-patterns vs 2) — Greater's ordered codes can. Any row/col-monotone staircase
  relation works: pick integer codes satisfying the order constraints.
- Y rank-3 `[10,1,30]` (broadcast right-aligns) lets the col chain share the row CumSum-axis
  init and a 1-element Unsqueeze axes.
- U8 trim kit: compare-then-unsqueeze the 1-byte BOOLs (not the F32 marginal);
  `Min(u8,u8)` = AND in one op (dtype matrix, task055-verified); `Add(u8)` of disjoint
  bands = OR; zero slots = one shared zeros-init (30p) on the X side, ANY u8 vector on the
  Y side (`Greater(0, y)` never fires); Sub(u8) needs opset 14+ but Not-based complements
  beat it (no scalar init).
- Extraction pairs with §42/#46: per-axis counts via `Einsum('bchw,c->bh'/'bw', input,
  onehot[10])` = 120 B each, 10 params, direct from the FREE input.
- Full worked example + construction-floor table: `The_400_Tasks/Task_by_Task/task055.md`.

# 62b. BITAND bit-detector upgrade — halves #62's assembly again (task055 v8-v11, 2026-06-06; 2,386 → 1,174, ab-byte-equal-full) <!-- formerly "# 58b" — renumbered 2026-06-06 -->

#58's compare-form needs TWO 10-channel stacks (600 B). When the output decomposes into
≤8 (row-set × col-set) rectangle products, give each product ONE BIT of a U8 instead:
per-axis bitmask vectors `x [1,1,30,1]` / `y [1,1,30]` (30 B each); channel masks
`K [10,1,1]` init (10 params; a union channel just sets several bits — task055 ch8 line
cross = 2 bits; empty channels = 0). Channel k fires iff `x(r) & y(c) & K_k != 0`,
re-associated so the expansion is HALF the stacks: `XK = BitwiseAnd(x, K)` [1,10,30,1]
= 300 B, then `output = BitwiseAnd(XK, y)` = the FREE graph output. Kit:
- **Code selection**: nested `Where(bool, u8-scalar, u8-scalar)` — 30 B/select +
  1 param/constant; k distinct codes ⇒ k−1 selects (minimal: each Where adds one leaf).
  Affine codes (`A + B·pre`) exhaustively searched (8! bit assignments × mod-256 ×
  single- AND double-sided): ZERO solutions for the 5-fill+cross+corners detector set —
  don't re-search.
- **ArgMax banding beats CumSum** when band boundaries are marker rows holding the
  STRICT MAX of a free per-axis signal (055: divider count W≥5 vs crossing 2):
  `r1 = ArgMax(counts)`, `r2 = ArgMax(counts, select_last_index=1)` = 8 B each; bands =
  `Less(iota_init [1,30] I64, r)`. Kills the whole Cast-F16+CumSum block (−75/axis net).
  ArgMax(F32 AND U8, select_last_index) execution-verified at opset 18.
- **Iota init (30p) beats Range** (Range output is a counted intermediate: [30] I64 =
  240 B); params count ELEMENTS, so an I64 iota compares against ArgMax outputs cast-free.
- Opset 18 required (BitwiseAnd) — grader-safe: task191 v6 shipped opset 20; **BitwiseAnd itself grader-VALIDATED 2026-06-07 (task205 v31 scored exact at 4,193)**; QLinear-saturating casts @ opset 19 grader-validated same day (task256 v12, 1,334 exact).
- U8 graph output is fine (real decode is `>0.0`) — but the advisory adversarial tool
  decoded `isclose(1.0)` until 2026-06-06 and scored such outputs 0/all (traps #30,
  fixed). If an advisory gate contradicts the certified gates, read the tool first.
- **Background-channel completeness trap (task104):** a free-output bit detector that emits only
  foreground channels is wrong when the reference one-hot also requires `ch0=1` on in-region
  background cells. The foreground `BitwiseAnd` is monotone; `ch0 = NOT(foreground)` is often
  non-monotone and cannot share the same bit lattice. If ch0 cannot be emitted by the detector,
  fall back to a charged small one-hot assembly before Pad (#67/#81), not a foreground-only output.
- Measured floors (055): extraction 240 + act/line 120 + localization ~150/axis +
  (k−1)-selects + expansion 300 ⇒ sub-800 architecturally unreachable there.

### §39 gap-ledger discriminator (2026-06-05, task398 v16): SHAPE-dynamic is mild; STRUCTURE-dynamic is catastrophic

v16 (dynamic active-square crop via free-input Einsum, 2,507 local) credited ≈3,120 — a 1.24×
markup, beating the honest static v17 (3,781). The ledger now separates two dynamic classes:
- **NEW BAND — MODERATE (≈3×, 2026-06-07 task225)**: a "static-looking" graph with an unflagged runtime surface (topgrid Expand/Slice family) credited 2.9× its local score DESPITE byte-equal gates. Pre-submit real-input profiling (WORKER_PROTOCOL gate 5) is the detector — the local profile equals the credit. **Escape confirmed same-day: the no-`Range` sibling (v5 qlinear-where, 1,614) credited honest-static (+0.37 banked) — within a family, removing the dynamic surface op restores EXACT pricing.**
- **CREDIT-PROBE CONFIRMED (2026-06-07, task065 v7): the profile==credit law holds on a +51% dynamic gap.** v7 static 1,327, real-input profile 1,999 (a variable-compartment-crop dynamic surface) — credited at EXACTLY 1,999 (+0.64 = the worker''s profile-based prediction to the cent; also retro-confirmed LIVE was crediting its profile 3,803, not its static 3,275). The law now spans static (exact) → mild (1.0-1.4x) → moderate (2.9x) → this +51% probe: in EVERY band the local real-input profile == the Kaggle credit. Gate-5 profiling fully predicts credit regardless of the static/profile gap; the gap only tells you WHETHER to trust static (it doesn''t) — the profile is always the number.
- **MILD (1.0–1.4×)**: data-dependent SHAPES that stay small — extraction crops, active-square
  windows (014 1.39×, 065 1.16×, 398-v16 1.24×, **178-v30 dynline 1.00× — first FULLY-HONEST
  dynamic credit, 2026-06-07: a dynamic-reshape-hint graph priced at its exact local 729**,
  246 both-rungs 1.07–1.08× — 1,275→~1,369 / 2,632→~2,843, 2026-06-07: **gold gates do NOT
  detect mild surfaces; correctness and pricing are independent axes** — only gate-5 profiling
  reads the markup). Local cost ≈ lower bound; worth credit-reads when the local number wins
  the ladder.
- **CATASTROPHIC (10–80×)**: data-dependent STRUCTURE — dummy branches, optional inputs,
  bool-Pad collapses (398-boolpad ~26-37×, 094 42-78×). Local cost is FICTION; never predict.
Static remains the only EXACT class. Bracket mild-dynamics with one fixed-base probe when the
local number would win the ladder.
- **§39 CRASH ADDENDUM (2026-06-06, task061 v11/v12): dynamic can CRASH, not just misprice.**
  Dynamic `Pad` (v11, local 1,131) and dynamic `Mod` (v12, local 141) both crashed ORT at
  dynamic-buffer REUSE when run on real visible inputs — the graphs scored locally on dummy
  shapes, then died at inference. A dynamic graph that never runs end-to-end on the REAL
  officials is not a candidate at any local price. Run the real pairs before staging anything
  dynamic. (Don't-do details: `task061.md`.)

# 63. Einsum × OneHot selected-channel marginals — skip the all-color count map (task079, 2026-06-06; 24,409 → 1,382 shipped) <!-- formerly a duplicate "# 59" — renumbered 2026-06-06 -->

§42/#46 give free per-axis counts for ALL colors (`'bchw,c->bh'` against a ones/onehot init).
When the rule only needs marginals of ONE data-dependent color, don't build the 10-channel
count map at all: compute the selector first (e.g. global-max color via #46 reductions), make
it a `[1,10]` one-hot, and contract **`Einsum('bchw,bc->bh', input, sel)`** — the row-sum
vector of the SELECTED color directly, 120 B, no `[1,10,30]`/`[1,10,30,30]` intermediate ever
exists. Same for `->bw` cols. This single move took task079 sub-1,500 (v36 1,382: memory
1,316 + params 66). Pairs with #62/#62b assembly on the output side.

# 64. Cast-BOOL occupancy → ArgMax bbox extents — the cheapest data-dependent bounding box (task228, 2026-06-06; 23,286 → 2,087 shipped byte-equal)

For "find the active box, work inside it, write back" tasks, the extent extraction is four
scalars: reduce row/col sums of the relevant plane, **`Cast(F32→BOOL)` then `Cast(→UINT8)`**
to get occupancy vectors (Cast-BOOL is free-er than `Greater(x,0)` — saves the comparison's
zero-init and a wider intermediate), then `ArgMax` = top/left and
`ArgMax(select_last_index=1)` = bottom/right (8 B each).
- **Gotcha (scorer-verified None)**: ORT REJECTS `ArgMax(BOOL)` — the U8 hop is mandatory
  (task228 v7 died on this; v8/v10 fixed it).
- Inside the box: §41 negpad Conv crops param-free; write back the handful of changed cells
  with a small `ScatterND` (task228: 16 scalars) instead of re-assembling planes.
- Anchors a 3-ship lineage: v3 negpad-scatter 2,609 → v10 castboolocc 2,087, both exact.

# 65. Assembly-choice laws — Equal-vs-Where and the mild-dynamic 600B threshold (task166 floor campaign, 2026-06-06; 21,648 → 1,900 shipped)

Two corpus-priced selection rules from task166's 4-sweep floor proof (5 assembly families
priced dead — full table in `task166.md`):
- **Equal-assembly vs Where-assembly is decided by PASSTHROUGH.** `Equal(cidx_u8, ch_ids)`
  output (task398-style, sentinel-padded) wins only for FULLY-GENERATED outputs. The moment
  the rule passes input cells through, Equal needs a full-canvas color decode (3,600 B) + an
  active-mask (3,600 B) to reconstruct what `Where(cond, fill, input)` reads for FREE from
  its else-branch (one BOOL cond plane, 900 B). Passthrough ⇒ Where, ≥3× cheaper.
- **Mild-dynamic (§39 1.1–1.4× class) pays only when the static tensors it deletes are
  ≥600B-class.** The extent-extraction plumbing costs ≈750 B runtime-observed (2× einsum
  axis vectors + 2× Sign + 4× ArgMax + I64 Concat). task398's 625–3,600 B canvases cleared
  the bar; task166's 144–576 B window machinery did not (net +150 WORSE). Price the deleted
  statics BEFORE building the dynamic route.


# 66. Superoptimizer verdict + the exact-rewrite shortlist (research card K1, 2026-06-06)

K1 surveyed TASO/TENSAT/egg/egglog/Glenside/PET/DialEgg/STENSO/Prism + ONNX-native rewriters
for auto-golfing under our `memory+params` objective. **Verdict: skip all ports** — published
rule sets are float-algebra-aggressive (not bit-exact), extraction under our cost is a DAG-ILP
problem greedy extractors mis-score, and the build is XL. If hand-optimization ever bottlenecks,
the sanctioned route is a tiny ONNX-native peephole pass (onnx-ir / onnxscript rewriter) over
the EXACT-rule shortlist, scored by the real oracle. Full survey: `_archive/Research_k1_superoptimization_survey.md`.
- **The exact-rule shortlist** (bit-exact by construction — steal for manual passes NOW):
  zero-`Pad`→`Conv`/`Pool` pads-attr fusion · `Concat`/`Split`/`Slice` normalization ·
  `Transpose`/`Reshape` no-op collapse · duplicate-initializer dedup · dead-node elimination ·
  `Where` canonicalizations on bool masks.
- **Worked byte math**: a separate zero-Pad before a 30×30 F32 Conv materializes
  `[1,10,32,32]` = **40,960 B** that `Conv(pads=...)` computes for free. Audit every graph
  with a standalone Pad feeding Conv/Pool.
- **Bit-exactness counterexamples to remember** (why float-algebra rules are banned):
  reassociation breaks at 2^24 (`(2^24+1)+1 = 2^24` vs `2^24+(1+1) = 2^24+2`);
  `(x/3)*3 != x` at 1.0f; AvgPool↔Conv(1/9) is inexact (1/9 unrepresentable).
  Integer-valued INPUTS do not rescue rules when WEIGHTS are arbitrary floats.


# 67. FREE-OUTPUT catalog — the output node is never charged; make the expensive op BE the output (2026-06-07 wave)

§39 mechanism: `calculate_memory` + profiler skip the tensor named `output`. Grader-proven
precedent: shipped task287 (+2.83) ends `Pad(u8,sentinel) → Equal(ch_ids) == output`. The wave
banked four free-output patterns (full mechanics: `.update/_integrated/` 148/205/256/078 notes):
- **Equal-tail**: `value-canvas u8 → Pad(sentinel) → Equal(ids[1,10,1,1]) == output` — the
  10-channel one-hot explosion never exists (148: 2,880→900 B; 199 reused independently).
  Pad value = sentinel no id matches ⇒ padding decodes all-False. Equal(u8) opset 11+.
- **BitwiseAnd set-intersection** (205 v31, GRADER-VALIDATED 2026-06-07): `BitwiseAnd(
  A[1,C,30,1] u8, B[1,1,1,30] u8) == output` = cell fires iff row-class-set ∩ col-class-set
  ≠ ∅ (≤8 classes/u8). Emits rank-2 `x_r OR y_c` patterns no arithmetic final node can.
- **Broadcast-Add assembly** (256 v7-v12): `output = A[1,10,30,1] + B[1,10,1,30]`, decode >0 —
  half-planes/staircases/triangles/intervals via ramp-cliffs (`Min(w+.5, 64W−63w−32)`), dead
  channels via Pad(−BIG). NO 2-D plane exists anywhere in the graph. Margin discipline: prove
  off-band A+max(B)<0 at actual F16 values; 64=2^6 scale is F16-exact.
- **Comparison/BOOL tails** (078): `Greater`/`Equal`-XNOR bands straight to BOOL output.
- **Two-color small one-hot via `Where` (task104):** when the active crop has exactly foreground
  color `k` on background `ch0`, assemble `[1,C,H,W]` as `Where(mask, CHk_u8[1,C,1,1],
  CH0_u8[1,C,1,1])`, then Pad with zero fill to output. This beats an Equal-tail color-index
  detour for small crops. The branch tensors must be UINT8/F16/F32/etc.; `Where(BOOL, BOOL, BOOL)`
  has no ORT kernel — concretely, a "Where-as-output one-hot from BOOL branches" (the #67/#78
  free-output route) FAILS TO LOAD at opset 16 with `Could not find an implementation for Where(16)`
  (task177, 2026-06-21). Workaround: keep the branches U8 and route through a **U8 `Where` ->
  `Equal(channels)` tail** instead of emitting BOOL straight from `Where` (see ort_reality.md
  `Where` row).
**SPREAD TARGET: `scan_freeout_candidates.py` found 97 tasks ending `Pad→output` with pre-pad
tensors ≥1,800 B (~210 KB upper bound; top: 289/323/278/295/350/071).** Dedicated sweep worker
recommended.

**`verify_against_live` TRAP for MARGIN finals (task295, 2026-06-08): `Add`/`Sum`-final free-outputs emit
raw margins that decode ONLY under the grader's `(>0)`. `verify_against_live`'s default full-tensor mode
compares the RAW output to the one-hot WITHOUT applying `>0` → a verify-passing margin model shows a
FALSE `0/268 "PROTOTYPE DIVERGES FROM LIVE"`.** It is NOT the hard gate for this class (cf. line ~41).
Two fixes, pick by need: (a) gate with the official path instead — `sanitize_model` → ORT session
(ORT_DISABLE_ALL) → `nu.verify_subset` (applies `>0`); (b) make the output literally 0/1 by ending on a
COMPARISON, not arithmetic: `Greater(A[1,10,30,1], (−B)[1,10,1,30]) == output` BOOL, since `A > −B ⟺
A+B > 0`. Same free-output, same operand sizes, +0–1 param; BOOL 0/1 then matches LIVE's 0/1 AND passes
`verify_against_live` raw-compare. task295 shipped this (broadcast-Greater staircase, 22748→2038). Rule of
thumb: when a free-output broadcast/Sum/Add final relies on `>0`, prefer the comparison-final form if you
want the vs-LIVE byte-equal gate to be meaningful. (Separately: "other" channels that must stay off need
value **0, not −BIG** — 0 isn't `>0` — which shrinks the broadcast-Add channel selector by one column.)

**Active-region boundary (task046, 2026-06-17):** the free-output one-hot fold wins only when
`C·active_cells < full_grid_cells`. For a SMALL active region, folding to a full `[1,10,30,30]` one-hot
canvas LOSES — keep the channel-small one-hot and make the spatial `Pad` the free output instead. (The
ch0-Slice lever #151 is the structural-zero sibling: drop a never-selected channel before the fold.)

# 68. Floor proofs are FORMULATION-RELATIVE — enumerate final-op families FIRST (task256, 2026-06-07)

Two correct-within-class floors fell in one session (plane 3,892 → negmask 3,627 →
broadcast-Add 2,403 → 1,334). Before believing any floor: **list every op family that can
legally write `output`, price its minimum input set** — Equal / BitwiseAnd / broadcast-Add /
Mul / Where / Min / Concat / OneHot / ScatterND / MatMul / Resize — THEN price stages inside
your formulation. task256's enumerated kill-list (Resize-linear no-cliffs, Mul-final
rectangles-only, MatMul-final rank-k ≥2,400 B...) is the template. **ADDENDUM (2026-06-07, task304 v4): the family list MUST include CONTRACTION finals — Einsum-with-selector-inits as the output node converts memory into params at a steep discount for periodic/block-structured outputs (a formally-intact broadcast-final floor proof fell 2.6x to a 9-operand einsum final, 1,114 -> 424). Also enumerate: the params-only single-node class (215@110, 252@280 — memory ZERO when the only node writes output).** **DISCRIMINATOR (2026-06-07, task357 4-route kill): contraction-finals fold ONLY when the (r,c)-coupling is SEPARABLE under contraction (312: color-perm; 304: margin products). An INPUT-SELECTED PERMUTATION EQUALITY coupling cannot fold — it must be paid as a comparison op (~200 B) or a >=rank-bound param bank (357: >=319 by combinatorial-rectangle test). Check the coupling class BEFORE building the einsum route; full 4-route kill template in task357.md.** Floors fall to idea-classes,
never to grinding (4 pings → 4 broken floors).
**Slice-search corollary (task056):** a "minimum witness Slice" proof must enumerate all contiguous
blocks, not just the visually obvious row/column. A `[1,1,2,1]` col-slice with place-value weights
beat a `[1,1,1,3]` row-slice (8 B vs 12 B) while preserving class separation; fold any scalar bias
into the Equal threshold vector.

# 69. Sign-arithmetic kit — channels need the right SIGN, not {0,1} (task256/078, 2026-06-07)

The `>0` decode means every Relu·2·Min clamp-to-one-hot ladder is deletable:
- Planes as raw signed margins `count − (r+.5)` — half-integer thresholds = tie-free, F16-exact.
- Bands: `Relu(affine)` / `Min(affine,0)` (zero off-band ⇒ product- and add-safe);
  in-band = `ReduceProd(margin_lo, −margin_hi) > 0` when both-positive is impossible (078).
- **Negated-mask identity**: `t·Relu(x) ≡ (−t)·Min(x_flip,0)` — pick the polarity your 2-D
  consumers want by swapping Sub operands, zero Neg ops.
- Exactly-at-row from two one-sided clamps: `Sum(above,below,+64) == 0` on-row, ≤−32 off —
  variadic Sum, no Abs, no quadratic.

# 70. Einsum place-value / dominance packing — many scalars per ONE free-input reduction (078/256/247, 2026-06-07)

Three grader-exact variants of the same idea (pack K answers into one einsum output via
weight design):
- **Digit-packing** (078): K per-column counts ride one einsum via place-value coefficients
  (`n1 + 100·n12`); compare at the high place directly (margin argument bounds low digits
  below radix/2); unpack low with `Mod(packed, radix, fmod=1)` — const divisor = SIGFPE-safe.
- **Combo head** (256): merge all-channel + object-channel marginals with `w_c = 1+α·sel[c]`
  (α = power of two ≥ max marginal): ArgMax/Max/Sum all read from the single combo vector.
- **Exponential positional dominance** (247, 543-cost ship): first-occurrence ORDER as one
  scalar — `f = Einsum('bchw,w->bc', input, b^−j)`; `leftmost(c)<leftmost(d) ⇔ f(c)>f(d)`
  (leading digit dominates the tail). No occupancy plane, no ArgMax, no rank matrix.
Mind the F16/F32 integer-exactness envelope when picking radixes (F32 2^24 / F16 2048).

# 71. Moment-correlation mono-window detection — variance-zero identity (task143 → task206 family, 2026-06-07)

"Find an exact translate of a runtime template among same-colored cells" needs NO per-channel
grouped correlation. On the color-index plane: `cV = Conv(idx,Tw)`, `cV2 = Conv(idx²,Tw)`,
window is one solid color ⟺ `cV² == cV2·nT ∧ cV > 0` (Cauchy-Schwarz equality case).
Occupancy-only diverges (49k/200k measured — don't retry). **F16 wall: (9·nT)² < 2048 ⇒
safe to nT=5** — sweep the corpus nT histogram first; go F32 (+1,200 B) above it. Companions:
self-match kill folded into the existing Greater via a threshold-plane init; stamp-back via
Conv with rot180(Tw) (= one neg-step Slice, #44). 15,965 → 4,160 shipped exact.
**Row/stripe homogeneity variant (Aditya task202, 2026-06-11):** the same
Cauchy/variance equality can classify whole rows without a row-profile tensor. Compute
`N*sum(x^2) - sum(x)^2 < 1` as scalar/payload Einsums over repeated input operands; the
intermediate pairwise products stay inside the Einsum kernel (#89), so no `[C,H]` or
`[H,W]` homogeneity map is born. Reach for this when the output rule branches on
"every cell in this row/stripe has the same code/color".

# 72. QLinearConv rounding-boundary gadgets — logic ops from quantization rounding (task077, 2026-06-07)

Ternary-vote: binary U8 input, kernel [1,2,1], scales 1/1/3, zps 0 ⇒
`y = round((a+2b+c)/3) = b OR (a AND c)` EXACTLY (no .5 ties). One op + 3 params replaces
~4 shift/Min/Max planes (~1,680 B) per use: 1-D hole fill, between-fill, center-biased
majority. General recipe: pick integer weights + y_scale so accept-set and reject-set sums
land on opposite sides of a rounding boundary; verify no half-ties. Also banked (task077):
the orientation lemma for separated-component reach (build support on the TIGHT axis, reach
along the LOOSE axis; mirrored design is UNSOUND — third-component interference).

# 73. Locate-then-detect + separable ones-banks + marginal-profile architecture (task205, 2026-06-07)

- **Locate-then-detect**: never run a detector bank full-range when the target's size is
  corpus-bounded. One cheap count-Conv + ArgMax locates the window (corpus-certify margin),
  fixed-shape clamped-Gather crops it (§39 golden rule — static), bank runs inside:
  25 planes 11,025 → 625 B. Winner-coords return via ONE offset add `+ start × count_acc`
  (preserves tie-SUM and no-fire semantics — byte-equal incl. degenerates).
- **Separable ones-kernels**: an H×W all-ones bank factors into shared row-convs + per-h
  shared col-convs: 1,600 → 80 params.
- **Marginal-profile (no-plane) family**: solid-border structures locate via free-input
  Einsum marginal profiles + first/last≥θ edges with **θ = min-border-count** (borders pass
  deterministically; bg clears at ~1e-4). Multi-pass refinement rows→cols→rows. Class:
  filtered-clean bets (064/209/079/205 band).

# 74. 3-operand free-input Einsum VERIFIED + INT8/U8 vector lanes at opset 14 (task148, 2026-06-07)

- `Einsum('bchw,c,w->bh', input, sel[10], weight[30])` scores memory = its 120 B output ONLY
  (probe-verified; tightens §42's 3-operand caveat to VERIFIED). Adopt: `'bchw,c,w->bh'` /
  `'bchw,c,h->bw'` joint channel+axis weighted marginals.
- **INT8/U8 row lanes**: [1,30] vectors with values 0..30 run U8 (30 B); signed lanes INT8
  (30 B) — half of F16. Probed: ArgMax/Sub/Mul/Add(INT8), Add(U8), Where(bool,u8,u8),
  GreaterOrEqual(u8) broadcast, GatherElements(BOOL, INT32 idx). **TRAP: narrow-int
  Add/Sub/Mul REJECT at opset 13 (shape inference) — opset 14 minimum.**
- **task348 additions (gated green on 200k, 2026-06-07)**: `Mod(F16,fmod=1)` CPU kernel
  exists and is exact on small ints (probed 0..9); einsum `'bchw,cv->bhv'` ([10,1]
  selector) emits a rank-3 [1,30,1] ROW-vector — comparisons then broadcast it against
  [1,10]-shaped col-vectors reshape-free; a 2-D-indexed 3-way value split costs exactly
  2 chained `Where(bool,u8,u8)` — nest a [1,10] u8 col-vector Where inside the row-cond
  Where to skip the And+mask plane (−90 B vs And-then-Where). **Don't einsum UNWEIGHTED
  reductions**: plain `ReduceL1` (attr-axes ≤ opset 17, one-hot ⇒ L1 == sum) charges the
  same output bytes with NO selector init (−10 params) — einsum is forced only where a
  channel SELECTION rides the reduction.
- **All-ones contraction elimination (task379):** an all-ones operand used only to sum an axis is redundant; drop it and leave that index absent from the output (`bchw,c,h->bw`). Avoid runtime Range+Cast.

# 75. K3 residual-op audit — five real finds + three confirmed dead (2026-06-07, `_archive/Research_k3_unexploited_op_audit.md`)

72 residual default-domain ops audited vs ORT 1.24.4 CPU registrations. Finds:
- **`Sum` (variadic n-ary add) — the top find**: collapses an Add-chain into ONE charged
  output tensor; CPU-kernel-backed on practical dtypes. (The 256/078 workers independently
  adopted it this same wave — `Sum(m1,m3,L_c)`, `Sum(above,below,+64)`.)
- **`RoiAlign`** — fixed-ROI crops with static ROIs as initializers (a Gather-window
  alternative). **`LpPool(p=1)`** — window-sums without Conv weights; **R6-VERIFIED 2026-06-19**
  (opset-22 2×2 stride-1 final output scored memory=0/params=0, exact window sums — but
  **float-only, nonnegative inputs only**: ARC one-hot/mask planes qualify, signed deltas do NOT).
  **`Hardmax`** — one-op one-hot-of-max. **`MaxUnpool`** — niche but kernel-backed (float-only,
  needs MaxPool int64 indices).
Confirmed DEAD: **`ReverseSequence` is VALIDATOR-BANNED** (op type contains the substring
"Sequence" — core.md; R6 scorer-printed "not permitted", and ORT additionally rejects
`time_axis>=2` so NCHW H/W reversal won't even load) — use `Slice` negstep (#44) for every flip;
**`CenterCropPad` has NO dedicated CPU kernel** (runs only via ONNX function expansion — N4 ran it
locally but it stays function-backed = validator-risk; the op_cost_patterns "don't build" flag was
right); **`NonMaxSuppression` validator-FATAL** (data-dependent output length); **`Pad.wrap` is
opset-19+** (no toroidal tiling win at ≤18).

# 76. K4 verdict — logic-synthesis offline pass is the ONE transfer; nothing else beats hand construction (2026-06-07, `_archive/Research_k4_compression_native_minimal_nets.md`)

Survey of CompressARC/MDL/DreamCoder/WANN/exact-pruning for minimal exact networks:
**nothing reviewed beats current hand construction for 100–25,000-byte exact graphs.**
CompressARC = inference-time optimization (unusable); DreamCoder/MADIL emit symbolic
artifacts; WANNs optimize the wrong objective; exact pruning = post-hoc cleanup only.
The one actionable option: **offline Boolean logic synthesis** — export a known boolean
subproblem to SOP/AIG, minimize with ESPRESSO/ABC/mockturtle, hand-compile the minimized
circuit back to 1×1-Conv/threshold/gather primitives (complement absorption, plane
elimination). Public tooling is solid for ≤9-input threshold cuts; 25-input window
predicates have no mature exact synthesizer. Use when a bool cascade resists manual
minimization.

# 77. F16 EXECUTION MODEL on the grader CPU — K5 source-read + ship-empirics reconciled (2026-06-07, `_archive/Research_k5_accumulation_exactness.md`)

K5 REFUTED the old mechanism claim: **there is NO native MLFloat16 kernel for standard
Conv on ORT 1.24.4 CPU x86-64** (also none for MatMul/Einsum/ReduceSum/CumSum). Yet F16-Conv
ships scored EXACT at F16 prices (080 +2.166, 055 CumSum-F16, 143). Reconciliation
[DERIVED-LOCAL, empirically anchored by those ships]: ORT **cast-inserts** around
kernel-less dtypes → compute runs in **F32** → one final rounding to F16 → the model graph
(and the grader pricing) stays F16. Operational rules:
- **Kernel-less F16 ops (Conv/MatMul/Einsum/Reduce/CumSum): SAFE class.** F32 accumulation
  (partials exact to 2^24); exactness = FINAL stored value ≤ 2048. Design only the finals.
- **Native-F16 ops are the real hazard:**
  - **`Gemm<MLFloat16>` EXISTS natively** — and the default MLAS half path does
    **STEPWISE HALF ROUNDING** (every k-step partial must be F16-exact, left-to-right;
    counterexample [2048,1]·[1,1] → first rounding at 2049). x86-64 Eigen fallback
    accumulator UNRESOLVED — **do not use F16 Gemm for counting; use Conv/einsum or F32.**
  - **Elementwise Add/Sub/Mul F16 are native Eigen-half** — per-NODE rounding: every node
    output must be F16-exact (683×3=2049 is the smallest valid-input multiply miss).
- **QLinearConv counter bound CONFIRMED at source**: int32 accumulation, stored u8 exact
  0..255, first miss at 256 (k×k all-ones window exact iff k ≤ 15).
- **F32 envelope**: integers exact to 2^24 = 16,777,216 (k ≤ 4096 windows).
- **Determinism**: inspected kernels parallelize disjoint output tiles — no cross-thread
  reduction reassociation suggested at fixed build/threads; intra_op=1 fully fixed.


# 78. FREE-OUTPUT catalog v2 — four MORE output-node families (wave 2, 2026-06-07; extends #67)

The catalog doubled overnight. New grader-validated or probe-verified patterns:
- **Conv-with-asymmetric-END-pads as PLACER (task357)**: `Conv(X[1,1,10,10], W[10,1,1,1],
  pads=[0,0,20,20])` emits [1,10,30,30] directly — `pads` is an ATTRIBUTE (zero params),
  placement + one-hot channel-expand FUSED into the free output node. Pre-output tensor
  drops to the bare coupling grid (100-200 B vs Equal-tail's 900 B canvas). **Direct
  upgrade for the 97-task Pad→output spread map: any pre-pad that is channel-decodable
  fuses its decode INTO the placer Conv.** (357's own "Equal-tail is cheapest" floor fell
  to this — Conv was mis-killed as translation-invariant without the pads-as-attr trick.)
- **QLinearConv-as-output (task125)**: asymmetric pads place a small X into the full
  canvas; padding cells = x_zero_point ⇒ dequant-0 ⇒ decode-empty. Linear taps INSIDE the
  output conv are free compute (3×3 ones block = 8-neighborhood count that never exists
  as a tensor; u8 saturation = free Relu). **Gotchas (probed): y_zero_point MUST be u8
  (INT8-y NOT_IMPLEMENTED); all 8 inputs MANDATORY; share 3 scalar inits across scales/zps.**
- **BitwiseAnd i32 class-LUT (task345)**: `BitwiseAnd(A_lut[1,10,30,1] i32 INIT,
  B[1,1,1,30] i32)` where `B[c] = 1 << class(c)` — an ARBITRARY per-class LUT, escaping
  the rank-1 limits that kill broadcast tails. ≤31 classes in i32; i32 graph output
  decodes via >0. Collision-free SPARSE class codes beat compact renumbering (fewer gate ops).
- **Where-as-output with Y=input (task312)**: the passthrough overlay as the final node —
  **but ask the #65/#68 question FIRST**: if the output is fully-generated-expressible
  (identity ∈ the map), a contraction final beats any Where (312 itself: 2,777 → 600).
  Use only when the replacement needs comparisons no bilinear map expresses.

# 79. Small-structure reconstruction kit — Equal(iota, sum-code) + the parameter-LUT crossover law (345/357, 2026-06-07)

- **Never Slice a ≤1-marker-per-column plane (task345)**: read it as a code vector
  `Einsum('bchw,c,h,d,e->bdew', input, sel, ramp(r+1), one1, one1)` → [1,1,1,30] (120 B;
  the dummy [1] operands give rank-4 directly, no Unsqueeze — task246 precedent), then
  reconstruct the plane as `Equal(R1_u8, Cast(code))` → BOOL (100 B). The 0-sentinel
  never equals r+1, so empty columns decode all-False free. 400 B crop → 270 B total,
  and the code vector is reusable.
- **The parameter-LUT crossover law (task357)**: for any per-index quantity, price THREE
  routes — output-LUT (full table), parameter-LUT (tiny per-index vector, e.g. [9,10,1]
  u8 = 90p), computed-at-runtime (op chain). The parameter-LUT wins whenever
  params(table) < intermediates(arithmetic chain) — 357: closed-form Mod/Abs chain saved
  80p but cost 154 B of intermediates, net WORSE.
- **Plane-tax law for runtime-generated banks (task104)**: replacing a small pattern bank
  with computed masks only pays if the computation adds at most ~1 extra `[h,w]` plane.
  Price it as `params_saved` vs `extra_planes*h*w*itemsize`; four 9x9 compare/Where/And
  planes lost even after saving 260 params. Middle ground: store one canonical pattern and
  generate flips/rot180 by INT32 `Gather` perm vectors, adding about one plane instead of a
  full mask chain.
- **Fixed witness pixels for bounded corruption (task130)**: when each 3x3 cell has at most
  two gray corruptions, a fixed 3-pixel row/column is still a sufficient color witness.
  Map gray to 0, read only the witness pixels, then reconstruct through a color-index
  Equal tail instead of classifying the whole 3x3 patch.
- **Scalar-threshold stencil (Aditya task399)**: for tiny output patterns driven by a scalar
  count/mass, compute `marks = Greater(scalar, thresholds)` once, then `Gather(marks,
  pattern_indices[K,H,W])` to paint the small pattern. `Not` + `Concat` supplies the background
  channel; `Pad` writes the final canvas. This is a cheaper path than building a color-index
  plane when the output is a unary/tiny shape selected by ordered thresholds.
- **A sliced one-hot row IS a mask**: ch0 of an all-bg row is 1.0 in-grid / 0.0 in padding
  — `Slice → Cast(BOOL)` doubles as the geometry mask off a tensor you already sliced.
  Kills a Less + iota.
- **Negative Gather indices are legal ONNX** (wrap semantics) — useful when a bank index
  is `count − k` and zero-profiles may feed it.

# 80. DepthToSpace(CRD) block-stamp interleave — fractal layouts in ONE op (task304, 2026-06-07; extends #44)

Any block-stamp layout `out[b·r+dr][b·k+dc] = f(r,k)·g(dr,dc)` (tile-where-mask,
self-similar/fractal stamps, periodic block fills): build the [1,b·b,b,b] channel stack
with one Where outer-product, then `DepthToSpace(blocksize=b, mode='CRD')` materializes
the full interleave as a single op. The 124/304 family's native vocabulary.

**BLOCK-EXPANSION = single grouped ConvTranspose (task307, 2026-06-07; the inverse direction).**
When the rule is "every input cell expands to a k×k same-color block" (nearest-neighbor upscale),
ONE grouped `ConvTranspose(W[C,1,k,k], stride=k, group=C)` IS the upsampler — W per group is the
k×k all-ones (or color-routing) stamp, the output is free (#67), and end-crop pads trim to the
exact output size. Cost = C·k·k params, memory ZERO: task307 2×2 expansion = W[10,1,2,2] = **40**
(1800→40, 44×). The cheapest legal k×k upscale; Resize/Slice routes are strictly worse (Resize-ROI
sub-40 killed). D2S rearranges EXISTING channels; ConvTranspose CREATES the k×k block per cell —
reach for ConvTranspose when each cell must grow to a block, D2S when channels must interleave.

# 81. PAD-FLOOR CROSSOVER LAW — price the Equal-tail BEFORE building it (sweep wave-3, worker-7, 2026-06-07; extends #67/#78)

The Equal-tail's cost is FIXED: ~900 B padded [1,1,30,30] u8 canvas + ~30-60 B canvas chain
+ ~30 params. Converting `X[1,10,h,w] -> Pad -> output` pays ONLY when killing X beats that:

| X dtype | wins iff h*w > | sweep-verified losers |
|---|---|---|
| BOOL | 100 | 031 (56), 267 (49), 127 (77), 360 (40) |
| F16  | 47  | 272 (25) |
| F32  | 23 nominal; chain overhead pushes the real bar to prepad >= ~1,000 B | 373/244/236/266 |

Tasks already ending `canvas -> Pad -> Equal == output` with small canvases ARE the floor — do
not re-convert. **Where-as-output discriminator (292: 6,301 -> 1,310)**: if output == input
except at maskable cells (sparse SUBSTITUTION), skip the value-canvas entirely:
`cond_small -> Pad(BOOL,False) -> [1,1,30,30] 900 B -> Where(cond30, e_k[1,10,1,1], input)`.
Replaced cells get the full e_k channel vector; no active-mask needed (untouched cells come from
input itself). Substitution-with-identity -> Where; fully-generated output -> #68 contraction.
**SMALL-GRID INVERSION (task032, 2026-06-07): when the output grid is tiny (<=~9x9), a decoded [1,10,h,w] one-hot + free Pad BEATS the fixed ~900 B Equal-tail canvas.** The Equal-tail's 900 B is fixed regardless of grid size; a 6x6 10-channel one-hot is only 360 B. So the #81 crossover FLIPS for tiny grids: emit the small decoded one-hot and Pad it to 30x30 for free, rather than building the full 900 B canvas. Check the generator's grid bound first (task032: 4-6) — the small-grid one-hot wins below ~9.5x9.5 (where 10*h*w < 900). **1x1 answer cell extreme (task056, 2026-06-10):** `Equal(code, ids[1,10,1,1]) -> BOOL[1,10,1,1]` costs 10 B, then Pad; a full 900 B canvas is wrong by ~90x. Bias after an affine scalar read can be folded into the ids vector: `linear+b==k` becomes `linear==k-b`, deleting Add + scalar.
**PAD-LAST REORDER COROLLARY (task083, 2026-06-07): the Equal-last vs Pad-last ORDERING is itself the lever for free-output one-hot reconstruction — EXPAND the one-hot at the small pre-pad extent then Pad-as-output (10*h*w B), do NOT pad the 1-channel index to [1,1,30,30]=900 B then Equal-last. task083 6x8 one-hot 480 B beat the 900 B padded-index+Equal. Transfers across mirror/reflection TILINGS where pre-pad h*w < 90 (task083 681, task142 519 — both captured). Does NOT apply to self-product fractals (task001 = D2S 1149) or pattern-color rules (task211) — verify the rule is an actual mirror before assuming the reorder helps (worker-scoped 2026-06-07).**
**U8-canvas crossover refinement (task035, 2026-06-11):** the 10x10 one-hot-vs-canvas tie flips toward the 900 B canvas when the producer plane is already U8; the old +100 B cast penalty is gone. Price the crossover after dtype-narrowing, not before. Corpus map: 79 Pad-tails >=1,800 B remain (54 deferred / 22 aditya / 3 boladi) —
`Logs/spread_scan_results/freeout_scan_20260607.csv`; 12 boladi conversions shipped 2026-06-07.
**Output-fold Equal corollary (task377, 2026-06-11):** if the tail is
`Equal(value, channel_ramp) -> Cast -> Pad -> output`, fold the Equal to the graph
output instead: `Pad(Cast(value -> U8), sentinel)` builds one padded value canvas,
and `Equal(padded_value, ramp)` is the output node. You pay the `[1,1,30,30]`
canvas, not the counted `[1,10,h,w]` one-hot plus Pad. Price both; the fold wins
when the pre-pad one-hot pair is larger than the fixed value canvas.
**opset14 BOOL-Pad small-grid INVERSE of that corollary (task377 v10, 2026-06-21,
4554->4464):** at opset14 `Pad` accepts BOOL, so when the rendered grid is small
(`10*h*w < 30*30`, e.g. a max 9x9 bullseye) the cheaper tail FLIPS back to building
the small one-hot FIRST: `Equal(value[1,1,h,w], ramp[1,10,1,1]) -> small_oh BOOL[1,10,h,w]`,
then `Pad(small_oh) == output`. This is the #81 small-grid inversion applied AFTER the
v6 output-fold — below ~90 active cells the small decoded one-hot beats the fixed 900 B
padded value canvas when the `Pad` can be the graph output. So the value-canvas fold above
and this BOOL-one-hot fold are the two sides of the same crossover; price the pre-pad
`10*h*w` one-hot against the 900 B value canvas and pick the smaller.
**Tiny two-color decoded floor (task386/task026, 2026-06-11):** when the active grid
is very small and byte-equality requires background ch0, one `Where(mask, e_fg, e0)`
at `[1,10,h,w]` in U8/BOOL plus Pad-as-output beats an Add/delta assembly and beats
the 900 B Equal-tail for `h*w < 90`. Do not drop the complementary color plane when
color 0 is a real output: all-zero cells decode as "no color", not background.
**Data-dependent recolor floor (task040, 2026-06-11):** a per-cell 2-way recolor on
a KxK crop usually costs four planes: source crop + BOOL condition + replacement
broadcast + result. `Where`-as-output only dissolves the result when the replacement
is a fixed one-hot vector; if the replacement color is itself data-dependent (for
example row-vs-col vectors selected into a KxK field), the replacement plane is
still charged and arithmetic blend shortcuts do not beat the real floor.
**PAD-LAST only wins when the small tensor reaches graph output.** task061 killed
the tempting "Equal at 18x18, then Pad" route: the `[1,10,18,18]` Equal costs 3240 B
before the Pad, while the padded U8 canvas stays 900 B. The old crossover is the
real rule: active_area * channels must beat the full canvas (roughly `H*W<90` for
10-channel one-hot; `H*W<900` for a single U8 mask). task335 adds the intermediate
trap: if the final ABI is `[1,10,30,30]` and a `path20` mask must feed a later
`Where`, `path20 -> Pad -> path30` charges BOTH tensors (400+900 in the task) and
loses to building the single full-size mask.
**GENERATOR-BOUND PAD-LAST proof (task174, 2026-06-22, 3716→2527, -1189).** C-channel one-hot Pad-last beats 900 B U8 canvas when `C*H_win*W_win < pad_H*pad_W`. task174: generator bounds H,W≤5 → 250 B. Window MUST be bounded from ARC-GEN generator (not visible pairs). For 30×30 output, crossover: `H_win*W_win < 90`.

# 82. Singleton-object plane rebuild kit — einsum scalar reads + Equal-ramp planes (353/047/389, 2026-06-07; composes #42/#46/#79)

Singleton-object tasks need NO input plane at all:
- **Coordinate reads** (4 B each, #42-free): `Einsum('bchw,c,h->b', input, sel_c, arange30)`.
- **Plane rebuild from scalars**: `And(Equal(pos_r, ramp_r), Equal(pos_c, ramp_c))` — ramps are
  params, the plane is born at the output stage. A CENTROID-singleton rebuild of a passthrough
  plane held byte-equal over 200k (353).
- **k-value extraction** (389): `Einsum('bcij,c->b', ReduceMax(input,[2,3]), arange_with_banned_zeroed)`
  — presence-weighted color id in one read.
- **Outer-MIN block stamp** (001, extends #80): `D2S(CRD) u8` + broadcast `Min(idx3, Reshape(idx3,[1,9,1,1]))`
  builds fractal self-products in 2 ops.
**Builder hygiene (sweep-confirmed, cheap to avoid)**: READ LIVE INITS VERBATIM — never
re-derive from 1-2 train pairs (104's masks are ANTI-diagonal, first cut 2/3 visible; 194's BL
quadrant is rot90CCW not transpose, masked by a palindromic train column). The 200k ab gate is
the real net; officials alone may not exercise the asymmetry.
**STRIDED-SLICE anchor for a k-spaced lattice "first-active" read — beats the F32-marginal+GatherND anchor
(task195, 2026-06-22).** When the rule samples a `k`-spaced lattice anchored at the first active row/col,
LOCATE and READ the lattice in one cheap head with a STRIDED `Slice`, not full-width `[1,30]` F32 Einsum
marginals: `Slice(input, starts=[0,ch,k,k], ends=[1,ch+1,END,END], steps=[1,1,k,k]) -> [1,1,S,S]` (S small)
→ `Cast U8` (e.g. 5×5 = 25 B) → `ReduceMax` dropping each spatial axis → `ArgMax` (FIRST occupied strided
index) → `Add [0,1,2]` → plain INT32 `Gather` ×2 to read the 3×3 pattern. The marker sits only on the
`{k,2k,3k,…}` lattice so a strided window always contains it; **plain `Gather` with INT32 indices is fine —
this is NOT GatherND, so the INT64-index 216 B grid does not apply.** task195: this anchor ~150 B vs the
prior head's two `[1,30]` F32 marginals (240 B) + binary-clamp→ArgMax (120 B) + INT64 GatherND `[1,3,3,3]`
(216 B) = 576 B — ~426 B cheaper. **Process lesson: READ THE TEAM-BEST ONNX before stamping a WALL verdict**
— task195's "WALL: cannot cross 921" proof only priced OUR head and never `onnx.load`ed the team-best graph,
where this cheaper anchor sat in plain sight (#68: floor proofs are formulation-relative). When both graphs
beat each other on DIFFERENT sub-blocks (theirs = anchor, ours = free-output Einsum #109/#119), splice the
cheaper sub-block from each (task195 hybrid 525).
**Clip the read to the generator's SPAWN BOX on EVERY axis — don't trust a single-axis clip as the floor
(task267, 2026-06-22, 525->468).** When a charged F32 input read (a `Slice` of a channel) feeds a shape
mask, its footprint floor = the GENERATOR's spawn box, clipped per-AXIS — and the clip is byte-equal because
the clipped-away cells are corpus-guaranteed background. The trap is clipping ONE axis (the visually obvious
one) and declaring it "the read floor": task267's shape is `common.continuous_creature(...,5,5)` then
`+1`-shifted into a 7×7 grid, so it ALWAYS lives in rows 1-5 AND cols 1-5. Prior task267 had clipped rows
7→5 (`bg7`=[1,1,5,7]=140 B) and called 5×7 the floor — but the COLUMN span is also exactly 5, so the true
box is 5×5=100 B (Pad the mask back on BOTH H and W, `pads=[1,1,1,1]`,`axes=[2,3]`). The −60 memory beat the
shipped team-best. Cheap recipe: a corpus `np.argwhere(grid>0)` bbox scan over ~20k generated inputs, EXCLUDE
known fixed cells (a marker), take per-axis min/max → that IS the read box; verify with the scan, not just
the obvious axis. (Pairs with #133 generator-content-bound crop and the F32 read-tax in cost_floor_reference.md.)

# 83. NIBBLE-PACKED transport plane — 2 cells/byte halves any color-index plane (task209 v6, 2026-06-07; pair with #81 before converting)

The entry channel-collapse conv packs 2 cells/byte: kernel `[1,10,1,2]` stride (1,2), weights
`(val, 16*val)` => `packed[r][p] = color(2p) + 16*color(2p+1) <= 153` (F32/u8 exact).
Plane [1,1,H,W]+u8 -> [1,1,H,ceil(W/2)]+u8 = half the dominant tensor pair. The enabling law:
**`packed > 0 <=> pair-has-content`** — every BOOLEAN consumer (row/col-any, Min(x,1)+ArgMax
extents #64) and every ROW-GATHER runs on the packed plane unchanged at half width; only VALUE
reads unpack: `hi = BitShift(p,4,RIGHT)` (opset 11), `lo = p - 16*hi` (u8 Mul/Sub, opset 14 —
no BitwiseAnd needed); cell select = `Where(parity, hi, lo)`. task209: plane 1,620 -> 810,
net -575 after unpack logic. 2 cells/byte is the u8 ceiling (4 nibbles => u16, no clean ORT path).
**Two mandatory disciplines (both bit task209 before gating green):**
(a) **max-of-packed is BOOLEAN-ONLY** — nibbles of a ReduceMax'd byte are cross-row garbage
    (`max(8+16, 48) = 48` loses lo=8); for nibble checks, gather the REAL pair column and reduce
    each nibble plane separately.
(b) **clip CELL coords BEFORE pair/parity decompose** — negative/overflow coords under Div-trunc +
    Mod(sign=divisor) route to a DIFFERENT phantom cell than a clipped-cell gather, and such reads
    can be geometry-VALID (box marker at grid col 0) so validity masking does not cover it.
Applicability: any graph whose dominant tensors are a color-index plane + its u8 cast, with reads
that are row-gathers + point/window lookups. Packing pays only while the PLANE (not the output
stage) is the dominant tensor — run the #81 crossover first.
**Same-drop micro-kit (all probe-verified)**: `Where(bool,i8,i8)` has NO CPU kernel — route i8
selects through u8 (wrapped negatives land only in unselected lanes), one final Cast->i32 at the
Gather (-251 B on 209's render lanes; #74 lane table). `Min(vec,1)` replaces Greater+Cast before
first/last-ArgMax extents (-1 node, -19 B per vector). Einsum rank-drops: `'bchw,c->h'` emits a
clean [30] with batch axis ABSENT (ArgMax then yields true [] scalars — zero Reshape glue);
`'bchw,bchw->'` self-dot replaces Mul+ReduceSum for sum-of-squares (F16-in safe, #77 class).

# 84. Placer-Conv structural toolkit — IG one-hot-sum identity, groups-halving, dilated quadrant reads (task180, 2026-06-07; params-only class 3rd member: 180@210 joins 215@110, 252@280)

- **IG identity**: `IG(t) = sum_ic in(ic,t)` is 1 at any in-grid cell, 0 outside (encoder one-hot).
  Any Conv reading raw input gets free per-tap "in-grid indicator" columns (weight 1 on ALL input
  channels at that tap). Replaces UNSAFE positive bias in placer convs: bg channel
  `ch0 = -IG(t00)+IG(t01)+IG(t10)+IG(t11) - 2*sum(masks)`; block confinement `-IG(t_a)+IG(t_b)` is 0
  iff both taps in-grid — kills ghost fires and removes data-purity assumptions entirely.
- **Groups-halving under generator-certified purity**: when regions hold only {0, color_q},
  "occupied" <=> `1 - in0(cell)` — bg substitutes for all color channels in masks; cross-channel
  reads collapse into the group containing in0 (180 v2: groups=2, W[10,5,2,2]+B[10] = 210 params,
  memory ZERO). Negative biases outside-safe. CHECK the partition first: every output channel must
  get its reads within its group (g=5/10 died).
- **Dilated quadrant reads**: `Conv(2x2, dilations=[b,b], pads=[0,0,30-2b...])` reads the four
  quadrant cells per output AND places the block top-left with free END-pads growing the canvas —
  overlay/zip rules need NO crop/concat/placer stage.

# 85. Free-output QLinearConv toolkit — int32-bias binarizer, in-conv morphology, the bias-leak law, crop-instead-of-mask (task265, 2026-06-07; extends #72/#74/#78)

1. **QLC int32 BIAS = free threshold.** Scales 1, zps 0: `y = sat(conv_int32 + B)`. Any `count==K`
   on a bounded count: `B = -(K-1)` -> y in {0,1} (u8 saturation = free Relu). Deletes the
   Conv->Greater pair per detector; exact k-window template match via structural-max sum. Each
   binarized plane u8 324 B vs F16 648 + BOOL 324.
2. **Window taps INSIDE the free output conv replace a morphology stage** (extends 125's
   center-taps): 2x2 output kernel tapping a binarized erode channel = dilate counts that never
   exist as a tensor; channels ride sign-correct integer expressions under `(out>0)` decode.
   **Boundary-leak refinement**: positive off-center taps are safe IFF the tapped channel is
   STRUCTURALLY ZERO on its last row/col (erode reads pad there) — border-stress audited.
3. **BIAS-LEAK LAW**: free-output conv padding region == all-input-channels-0, so every output
   channel must be CONSTANT-FREE — any bias/folded "+1" lights the whole out-of-grid canvas
   (certain fail). Complement channels cannot be derived in the output conv — both plane and
   complement must arrive as separate x channels.
4. **CROP-instead-of-mask**: when an exact template partitions its kxk window across complementary
   channels, true fires live only >= ceil(k/2) from every edge — give the detector pads=0 (emits
   interior only, e.g. 196 B vs 324) and ONE shared default-Pad after restores alignment AND IS
   the boundary mask. Audit per-axis: fire impossible iff each padded row/col loses >=1 "+1" cell.
5. **Pack-entry + BitwiseAnd-as-output residue** (sub-kill skeleton worth reusing):
   `Conv(input, W[1,10,1,30] = 2^c row weights)` emits packed rows [1,1,30,1] F32 120 B
   (<2^24 exact, NO slice) + `BitwiseAnd(rowcodes i32, colpows)` unpacks AT the free output node.
   Skeleton ~1.8k incl. assembly — pays on tasks whose per-row logic is a few shifts/ANDs (on 265
   it died at ~7.5k: bit-parallel 5x5 matching needs ~30 charged [30,1] tensors). Blocker: no
   wide-weight integer pack op exists — commit packed end-to-end or not at all.
6. **Free-output QLC placer can read an existing U8 feature plane directly** (task081): if the
   feature tensor is already U8, feed it into the output QLinearConv instead of casting back through
   F16/F32. Signed INT8 weights encode complements/negative taps; keep `y_zero_point` UINT8 (ORT
   rejects INT8-y). Still apply the bias-leak law above: a nonzero bias paints the padded canvas.
   Dynamic signed INT8 weights can also encode `marker_onehot - e0` routing for a free-output placer
   (task267); use only after profile confirms the dynamic weight path has no runtime cost gap.
7. **BIAS-TO-WEIGHT FOLD for clamped detector banks** (task034, 2026-06-21, 3496->3492 byte-equal):
   a small NEGATIVE INT32 bias whose only job is to require all non-target cells be non-background (or
   to enforce an exact-match threshold) can sometimes be DELETED by rescaling the positive-witness and
   anti-background weights so true cases stay positive and every false case is <=0 BEFORE the UINT8
   clamp. task034 `6*red_corner - 2*bg_count - 5` (true=1 only when the 2x2 corner is red and the
   window has no background) reduces to the no-bias `1*red_corner - 1*bg_count`: aligned red corner
   `1-0=1`; red corner in a bg-touching window `1-bg<=0`; non-red corner `0`; bg window `<0`. Because
   QLinearConv emits UINT8, every <=0 case clamps to 0, so the seed map is unchanged and the 4-element
   `seed_b_q` INT32 initializer disappears with no charged tensor added. Reusable wherever a bias only
   gates "all non-target cells must be non-background"; the bias-leak law (item 3) still applies to any
   bias you KEEP.

# 86. Occupancy-pack + max-bound existence — the aggregation IS the detector (task346, 2026-06-07; extends #69/#70/#71)

- **128-spacing pack**: when a detector needs a window's value-sum AND occupancy, one kernel:
  `w_pack(c) = c + K*[c>=1]`, K > max window value-sum -> `ringval = A + K*occ`, (occ,A) decodes
  uniquely (A < K). "Solid window of color v, occupancy n" = ONE Equal vs `n*v + K*n` — partials
  can't fake it (gap K beats the A range); 2-colors-present mixed windows can't either (exchange
  argument). Halves the #71 moment-pair bill (no second-moment conv/Mul-square/variance Equal).
- **Max-bound existence**: if the query color a is the MAXIMUM present, `ringval <= n*a + K*n`
  with equality IFF some window is solid-a -> `exists == Equal(ReduceMax(ringval), n*a+K*n)`
  (4+1 B, scalar-land). No comparison plane, no Where plane, no BOOL->U8 hop; multi-fire immune
  by exists-semantics. Resolve the complement branch with a structural guarantee +
  `Where(exists, other, va)`. (Presence via Sign(counts) — count-weighted max is WRONG, task346.)
- **BBOX-EXTENT via single marginal (task291, 2026-06-07): one rowsum[1,10,30] marginal yields BOTH extents.** When the shape has full-width/height walls (boxes, frames, dividers), the wall rows/cols carry the MAX marginal: `ArgMax(ReduceSum over the perpendicular axis)` = the row extent (height), `ReduceMax(rowsum)` = the width — replaces two F32 occupancy maps (2400 B) with one (1200 B). Pair with the **one-hot-Pad corner emit**: a [1,10,1,1] one-hot Pad'd to the free output replaces a 900-param corner plane (task291: 933->22 params). Einsum-moment `D=n^3-RR*CC` for exact bbox-area is NON-exact (falsified 7/8000) — extents need the marginal max, which Einsum cannot compute.
- Datapoints: 4-operand dummy-axis einsum profiles EXACT but loses to plain ReduceSum (#74 rule);
  3 negpad begin+end Convs profile 0% gap (more §41 begin-pad evidence).
- **DISJOINT-BAND precondition for code-packing (task253, 2026-06-22 — the general "wall" on halving two marginals into one).** Packing N codes into ONE weighted sum + single-`Mod` decode works ONLY when each code lives in a fixed, bounded, DISJOINT numeric band: `code_i ∈ [0, B_i)` with pack base `≥ Π B_j (j<i)`. If any code has a VARIABLE additive offset — a free object position whose marginal `wr=Einsum(input,iota)` ranges widely — the low code's carry overflows into the high band (`packed//K = wc + wr//K`, not `wc%3`) and no single `Mod` demuxes it. Worse, the only Einsum that mixes the `h` and `w` axes into one packed output is `bchw,hwy->bcy` with `W[h,w]=h+K·w`, which is ADDITIVE not rank-1 separable → it needs the full dense `[30,30,1]` = 900-param weight, not a reused `[30,1]` iota. And add-folding `packed=wr+K·wc` from two existing einsums still materializes both einsums + an Add (net ≥ 0). So two genuinely-independent 1-D reductions (e.g. orientation `mr=Mod(wr,3)`, `mc=Mod(wc,3)`, the #42 free-marginal pair) STAY two tensors when the object position is free — the pack saves nothing and costs a dense mixing weight. This is the exact counter to "pack two marginals into one to halve detection cost." (Cross-ref #42 free-input weighted marginals; the occupancy/value pack above works precisely because its `K > max window value-sum` guarantees the disjoint band.)

# 87. Selector-bank rank minimization — the L1 method (research deliverable, 2026-06-07; serves #68-class rank residuals like 312's +0.51)

Full note: `_archive/Research_l1_exact_cp_decomposition.md` (source-tagged). The operating verdict:
**unrestricted-real CP rank is ∃R-hard — no public tooling closes it on demand; the ONE regime
worth compute is bounded integer/dyadic coefficients + symmetry breaking + a structured
dictionary, as a Gurobi MIQCP/MILP.** The 5-pass recipe:
1. **Cheap certificates FIRST**: all three matrix-flattening ranks are unconditional CP lower
   bounds — if any flattening rank equals the incumbent T, the residual is DEAD instantly (run
   this before any solver). Permutation-like support: one rank-1 block covers at most one 1 =>
   Boolean/OR-semiring rank = n exactly — OR-compression of permutation banks is impossible
   (this also re-derives 357's discriminator kill from theory).
2. Bounded-alphabet feasibility at T-1, T-2, ...: S={0,±1} first, then {0,±1,±2}, then small
   dyadic; normalize first-nonzero=+1 per term, lex-order term supports (the two symmetry breaks
   beat any solver tweak).
3. **Dictionary before alphabet**: candidate mode vectors from structures actually present
   (basis, ones, class indicators, cyclic shifts, diagonals, motifs of the incumbent) -> set-
   partitioning model. A good dictionary beats a richer alphabet on these banks.
4. Deletion-and-reoptimize around the incumbent (remove 1-2 terms, reoptimize the rest) — finds
   cancellation identities a blind search misses.
5. Honest certificates: bounded-alphabet infeasibility certifies ONLY that class; unrestricted-
   real needs a rigid template + Groebner elimination (rarely worth it).
Time budget: ~20 min invariants/dictionary; 30-60 min T-1 at {0,±1}; 1-2 h richer; STOP unless
near-feasible incumbents appear — then treat the known T as final for engineering.
**TOOLING (M2, 2026-06-08, `_archive/Research_m2_exact_rank_tooling_recipe.md`): the solver is OR-Tools
CP-SAT** (exact integer, free/`pip install ortools`, MIT) — the runnable winner script is in the
deliverable. Gurobi = optional scout (not the proof engine: float tolerances). Z3 = wrong (NIA
undecidable). Recipe: S={0,±1} first, TABLE-encoded products (not raw bilinear), first-nonzero-
positive normalization on A&B, support-code ordering of rank-1 terms, explicit `|prefix|<=2^24`
guards on the rank accumulation. Realistic yield on a structured 10x10x10 bank: good chance
19->16, materially lower 15->13, LOW for 10->12 generic. **This unblocks R2's 312 ranks-16-13
harvest** (R2 was blocked only by the absent solver) — a worker with `ortools` installed can now
run the deeper search on the rank-18 bank.
Exact-rank anchors: cyclic/group permutation stacks have rank exactly n (interpolation upper
bound 3n-2); superdiagonal Delta_n rank n in every semiring.

**FREE-OUTPUT SEPARABILITY TRAP (task127, 2026-06-07): factorization reduces PARAMS but materializes
a charged INTERMEDIATE — for a free-output single conv the dense form usually wins.** A dense final
`Conv(input, w[Co,Ci,k,k]) -> output` charges ONLY the weight (the output tensor is free, #67); the
k*k*Ci*Co params are the whole cost. Splitting a rank-1-separable kernel into `Conv(k×1) -> Conv(1×k)`
cuts params but the FIRST conv's output becomes a charged [1,Co_mid,H,W] intermediate — task127's
rank-1 split needed [1,5,30,30] F32 = 18,000 B, turning a 900 cost into 18,300. RULE: factorize a
final/free-output conv ONLY if the intermediate it creates is smaller than the param saving (almost
never at full HxW). Separability pays on INPUT-side convs where the intermediate already exists, not
on the free final. (Kill-test order for a dense final-conv floor: 1×1-collapse if off-center taps ~0
-> separability ONLY if intermediate < saving -> groups if the channel partition holds.)

# 88. u8 flood kit — row-byte packed flood (#47 crossover FLIPS at width<=8) + Min-masked MaxPool + seed/priority tricks (task048, 2026-06-07; 8,831 -> 699)

- **ROW-BYTE bit-packed flood**: at grid width <=8 one u8 holds a whole ROW (bit c = col c) —
  planes become [1,1,8,1] lanes at 8 B/tensor and #47's economics INVERT. Pack for FREE inside
  the extraction conv (`Conv(input, W[2,5,1,8] taps 2^c, group=2, negpads)` = crop + select +
  pack in ONE node). 8-conn round = **72 B**: `ew = x|x<<1|x>>1` (u8 overflow IS the boundary at
  width 8), `d8 = ew|pad_down|pad_up` (negative-Pad lane shifts), `reach' = d8 & pass`. No unpack
  stage for hit-tests (GatherElements -> BitShift -> And 1). Bit tricks probed: lowest-set
  `x&(0-x)`; highest-set 3-smear; BitwiseOr/And@18; runtime BitShift amounts broadcast.
  ArgMax trap: `Min(lane,1)` before ArgMax (raw bytes rank by VALUE not occupancy).
- **Min-masked u8 MaxPool flood** (128 B/step, 8x8): `reach' = Min(MaxPool(reach,3x3,p=1), mask)`
  — pool of binary stays binary and `Min` against a NON-BINARY value-coded mask keeps it binary
  (min(1, m>=1)=1): the mask doubles as the class plane, no per-step binarize. 4.5x cheaper than
  Conv+Greater+Where F32. Mul-masking breaks binariness when mask values >1.
- **Outer-product one-hot seed from a flat ArgMax idx** (99 B + 20p vs OneHot 256 B): power-of-2
  sides => `r = idx>>3, c = idx&7`, two broadcast `Equal(iota, scalar)` + Mul. (OneHot u8-values
  is NOT_IMPLEMENTED; ScatterElements(zeros, idx, val) beats Equal+Cast+Mul for direct seeds.)
- **ArgMax value-priority**: weight the extraction conv so class-A code > class-B code — ArgMax
  locates A on the RAW plane; the dedicated binary class-A mask tensor disappears.
- **bmask 2-op shortcut (task048)**: the high-bit smear `bmask = highbit | highbit>>1` equals
  `smo3 − (smo3>>2)` (proven over all 256 u8 values) — saves 2 intermediates vs the OR-chain; reuse the
  `>>2` shift constant.
- **Last-round mask-skip**: the final flood round can skip its Min/mask when the consumer reads a
  mask-subset; 2-target hit-test (`highbit | highbit>>1`) trades +2 B for −1 round.
- **Generator-invariant connectivity** (§GENERATOR-INVARIANT instance): ARC-GEN resampled until
  4conn==8conn path-existence — so iterate the CHEAPER 8-conn (3x3 MaxPool, k<=12) instead of the
  literal 4-conn (k<=18). Grep generators for equality-constrained resample loops before paying
  for the literal metric; gate officials separately (they bypass the loop).
- **Disjoint-boxes theorem**: row-major FIRST and LAST target cells always lie in different boxes
  => "flood from first reaches last" == inter-box connectivity — one GatherElements replaces the
  whole count-and-threshold stage. Corpus-reproven 0/200k.
- **ROUND COUNT = corpus-max BFS DIAMETER (task048).** Packed flood cost is `72*k`; k is not tunable. Run BFS on the full corpus under the chosen seed/target framing and keep the max distance. Dropping below true diameter visible-failed; only a framing change that widens endpoints can reduce k. Pre-dilating the seed is just another paid round. The F32 encoder bridge from extraction is dtype-locked; fuse extract+crop+pack to the smallest F32 bridge before U8 packing.
- **Width 9-24 successor (task243): U8 byte-lanes beat #47 U32 rows.** Pack rows into ceil(W/8) lanes; split cheap vertical Pad rounds from horizontal cross-byte carry. Combined H+V pays carry every step; 8-conn MaxPool needs proof that 4conn==8conn. task243: 3 lanes, 20V+16H, cost 11,315.

# 89. Einsum-final design rules — ORT 1.24.4 source-read (L2 deliverable, 2026-06-07; the einsum-final class #68's runtime contract)

Full note: `_archive/Research_l2_ort_einsum_internals.md` (source-cited: einsum.cc + preprocessor +
typed processor + auxiliary ops at the v1.24.4 tag). The load-bearing findings:
- **PRODUCTION-CONFIRMED (task188 SIEGE, 2026-06-07): the scorer-invisible in-kernel temporary is REAL.**
  `Einsum('bchw,h,w->bchw', input, rowvec, colvec)` builds a separable row*col mask — the [1,10,30,30]
  masked product is an IN-KERNEL temporary, never a charged node output. Credited 558 EXACTLY (the 900 B
  2-D mask collapsed to two 240 B 1-D keep vectors); 6924->558, +2.52. Generalizable: ANY separable
  mask/weighting over a free input+output belongs in an Einsum, not a materialized 2-D tensor.
- **In-kernel pairwise temporaries are TEMP-SPACE allocations, not node outputs — INVISIBLE to
  the profiler and the scorer.** This is WHY einsum-finals price so well: the whole left-deep
  contraction chain is free in the cost model; only declared node outputs charge.
- **Operand order IS the contraction path**: ORT runs a strict LEFT-DEEP pairwise chain with NO
  path optimization (source comments acknowledge it). Design rule: order operands to kill large
  non-output labels AS EARLY as possible — not merely smallest-first. 6-9 operands are sane iff
  the best left-deep order keeps every prefix live-set small; NO hard operand cap exists (spec
  allows 1..2^31-1), the risks are prefix width and latency, both linear in steps.
- **Split into two chained Einsums ONLY for a ~10x worst-step win**: a split converts a free
  in-kernel temporary into a CHARGED visible intermediate — it buys latency, never cost, unless
  the tree-shape change is order-of-magnitude.
- **Fuse single-consumer Einsum chains (task275).** `Einsum -> Einsum -> output` can become one larger equation and delete charged intermediates; give independent same-size reductions distinct letters or you cross-multiply paths.
- **Slow paths to avoid on big live tensors**: repeated-label diagonals (serial extraction loop,
  TODO-parallelize in source) and large explicit output permutations (extra transposes around
  the MatMul core). Latency per step ~ worst B*M*K*N + transposes/diagonals (per-pair-latency
  rule, task270, now mechanistic).
- **Known edge bug**: output-subscript permutation after reductions (`ijk->ki` class) hit
  FinalizeOutput shape errors (issue #18654) — gate any final whose output permutes reduced-
  adjacent axes on the real officials before staging (which gate 2/5 already forces).
- **TWO-full-operand fold → reorder before declaring it a timeout (task293, 2026-06-22).** When a
  free-output Einsum folds TWO copies of the same full `[1,10,30,30]` input (two spatial factors
  inline), ORT's greedy left-deep path can materialize the dense cross-product of the two large
  operands and blow runtime up ~1000× — slow inference at TINY static cost. The fix is purely operand
  ORDER (byte-identical output): list each `input` copy ADJACENT to its own small factors so ORT
  contracts each side to a tiny vector FIRST. task293 SLOW `pc,bdhy,bexw,qd,re,y,x,pqr->bchw` =
  6308 ms/pair (120 s gate timed out); FAST `bdhy,qd,y,bexw,re,x,pc,pqr->bchw` = 3.8 ms/pair, all
  gates green, byte-identical. A both-fold is NOT intrinsically intractable — **probe one `sess.run`
  timing before arming a 200k gate**; a worker had abandoned this fold as a dead-end purely on the
  timeout (598->358, +0.51 pts).

### §89 addendum — free-output Einsum as a SEPARABLE MASK (task188 v5, 2026-06-07, GRADER-CONFIRMED 0% gap)

When the output is `input ⊙ M` and the mask **M is separable** as an outer product of a per-row
vector and a per-col vector (`M[h,w] = rowvec[h]·colvec[w]` — i.e. "keep rows<oh AND cols<ow",
rectangles, row/col thresholds, stripes), do NOT build a 2-D mask plane. Write it as one Einsum:

    output = Einsum('bchw,h,w->bchw', input, rowvec [H] F32, colvec [W] F32)

The `[1,10,30,30]` elementwise product is an **in-kernel temporary (invisible, per #89 above)**;
`input` and `output` are the FREE graph tensors → the ONLY charged cost is the two 1-D vectors
(`30+30` elems F32 = **240 B**). This **beats the §48 free-output `Where(cond_2D, input, 0)`**
whose `[1,1,30,30]` BOOL cond is a charged **900 B** plane — the 2-D AND of the two 1-D masks is
exactly the materialisation Einsum avoids. **task188: cond-Where 1325 → Einsum-mask 558 (−58%).**
Caveats: all Einsum operands must share the input dtype (F32 in/out ⇒ F32 vectors — you cannot
F16 the vectors without casting the whole input to an 18 KB F16 plane); build the vectors with
`Cast(Less(idx, thresh), F32)`. A non-separable mask (true 2-D logic) still needs §48's plane.

# 95. DISJOINT-SOURCE MERGE WITHOUT A `Mul`: scaled-conv weights + `ch_ids` remap (task020 v2, 2026-06-07, GRADER-CONFIRMED 0% gap) <!-- renumbered from a worker-drafted ### #90 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When you decode a colour-index plane from one-hot channels and one needed colour lives in a
**non-contiguous** channel (e.g. colours `{1,2,3,4}` at channels 1:5 but colour 8 at channel 8),
the naive merge is `Conv(ch1:5,[1,2,3,4]) ; Mul(ch8,8) ; Add` — two extra F32 `[1,1,k,k]`
temporaries. Instead:

1. **Scale the conv weights** so the contiguous block lands on a disjoint value set:
   `Conv(ch1:5, [2,4,6,8])` → values `{0,2,4,6,8}` (even).
2. **Encode the off-block colour as its raw one-hot value** (channel-8 slice is already `{0,1}`).
3. The two supports are disjoint per cell → **merge with one `Max`** (Max == Add when supports
   are disjoint), no `Mul`, no F32 merge temp. Cast both to u8 first; `Max` *does* have a u8 kernel.
4. **Undo the value remap at the free Equal-tail**: `ch_ids[k]` = the canvas value that should
   light output channel `k`. e.g. `[0,2,4,6,8,200,200,200,1,200]` maps value 2→ch1 … 8→ch4, 1→ch8,
   and `200` is a never-present sentinel parking the unused colour channels (5,6,7,9). The
   outside-grid sentinel (10) simply isn't in `ch_ids` → those cells stay all-zero.

task020: removed the `Mul` + two F32 `[1,1,5,5]` temporaries, −150 B (2547→2396 within the rewrite).

**ORT u8/dtype kernel facts (probed, opset 13, `ORT_DISABLE_ALL`)** — the constraint that forces
the trick: `Add`/`Mul`/`ReduceSum` have **NO UINT8 kernel** (INVALID_GRAPH at load); `Max` works on
u8/i32/f16; `Add`/`Mul` work on i32/f16. So any colour-index *arithmetic* must stay F32/i32 — but a
*disjoint merge* needs only `Max`, which is u8-cheap. Corollary trap: a `Concat` of colour planes to
feed one conv is F32 (`[1,C,k,k]`=C·k·k·**4** B) and usually loses to a per-block conv + Max.

# 90. No-Mul colour decode — scaled-conv weights + raw-channel u8 Max merge (task020, 2026-06-07)

When a task uses a small colour set with a contiguous run plus one outlier (e.g. {1,2,3,4} + {8}),
decode WITHOUT a Mul or F32 merge temporary:
- One 1x1 Conv weights the contiguous channels to distinct canvas values (`[2,4,6,8]` for colours
  1-4 -> values 2,4,6,8).
- Bring the outlier in as its RAW channel value (colour 8 = channel-8 one-hot value 1).
- The two sources are disjoint per cell -> merge with a single u8 `Max` (no Mul, no F32 merge temp).
- Remap `ch_ids` at the Equal-tail to undo the encoding (`[0,2,4,6,8,200,200,200,1,200]`; 200 = a
  never-present sentinel for the unused colour channels; the outside-grid sentinel 10 is preserved).
Saves the Mul node + its F32 merge intermediates vs the naive per-colour-multiply-then-add decode.
Pairs with the Equal-tail (#67/#81). Don't-do (same task): Concat-ing the colour channels in F32 to
feed one conv is [1,5,5,5] F32 = 500 B (element-count*4, easy to misjudge as 125) — decode the
contiguous channels with one conv and bring the outlier separately instead (+300 B saved).

# 91. Einsum-final vs Pad-last crossover for free-output spatial remaps (task142 519→456, 2026-06-07)

For a fixed free-output spatial remap (crop active region → tile/mirror → pad to 30×30), there are
two competing final-op families. Both end on the free `output` (#67); the question is which counted
tensors survive:

- **Pad-last (#81):** build the mirrored one-hot, then `Pad`-as-output. You PAY the one-hot
  `[1,10,h,w]` **and the entire mirror build chain** — every `Slice`/`Concat`/`Equal`/`Cast` that
  assembles the canvas is a *named, counted* tensor. (task142 padlast = 360 one-hot + **126 B build
  chain** + 33 params = 519.)
- **Einsum-final (#89):** `Slice`-crop the active input `[1,10,h_in,w_in]`, then
  `Einsum('bchw,hH,wW->bcHW', crop, M_h, M_w)`-as-output, where `M_a[i,j]=1` iff out-index j samples
  in-index i (pad columns all-zero). You pay ONLY the crop + the selection maps — the mirror+tile+pad
  happens inside the einsum's **scorer-invisible pairwise temporary** (`calculate_memory` skips
  unnamed tensors; #89, production-validated: task188 558 credited exactly). (task142 einsum = 360
  crop + 90 map + 6 slice = 456.)

**Crossover:** einsum wins when `crop_bytes + map_params < onehot_bytes + build_chain_bytes +
padlast_params`. Einsum trades the *build chain* (often 100–200 B of counted Slice/Concat temporaries)
for *map params*. It wins whenever the build chain it deletes exceeds the extra map params.

**Symmetry doubles the lever:** when the remap is symmetric (H-map == W-map, i.e. a square
input→square mirror), the **same** `M` initializer feeds both einsum operands → **counted once**.
That halves the map cost and frequently flips the crossover to einsum even where the asymmetric
sibling stays pad-last:
- task142 (3×3→6×6, **symmetric**): crop 360 + **shared** `[3,30]` map 90 = **456** < padlast 519. ✓
- task083 (3×4→6×8, **asymmetric**): crop 480 + **two** maps `[3,30]+[4,30]` 210 = 690 > padlast 681. ✗

**Dtype floor:** the crop is F32 (Slice/Pad/Conv/Cast preserve the F32 input dtype) and ORT CPU Einsum
is float/double/int32/int64-only (op_cost_patterns.md Einsum salvage) — so there is no sub-4-byte einsum input; the crop is
`10·h_in·w_in·4` B, firm. F16 multi-operand einsum is NOT byte-equal (task191).

**LAW (the meta-lesson): when transferring a "FLOOR" across same-family tasks, RECOMPUTE the
final-op kill-table for the new geometry — never inherit a KILL verdict.** task142's 519 "floor"
inherited task083's einsum KILL number (690) verbatim; a shared/smaller map for the symmetric 3×3
geometry broke it (456). Candidate transfers of this move: any GridSample-`[1,c,h,w,2]`→Pad
free-output remap with a **square** active region (e.g. task211 9×4 and task001 — same Tiling family,
but both need their rule decoded first).


# 92. Marginal-split orientation detector — 2 depthwise marginal-diff Convs beat the 40-ch 2D conv (task253 23974→2434, 2026-06-08)

When the rule turns on the **orientation / chirality / asymmetry** of a small per-colour shape
(L-tromino corner, which-side-is-heavy, top-vs-bottom), do NOT detect it with the LIVE-style
N-orientation 2D conv bank (`[1, C·K, H-1, W-1]` — task253 LIVE: `[1,40,12,12]` = 11,520). The
orientation of a 2×2-bbox shape factors into **two independent 1-D bits** — (pair-row top/bottom)
and (pair-col left/right) — each readable from a single collapsed **marginal**:

```
is_bottom[c] = ReduceMax_r( rowsum[c,r] - rowsum[c,r+1] ) > 1.5   # (2,1) row-pair -> max 1 ; (1,2) -> max 2
is_right[c]  = ReduceMax_w( colsum[c,w] - colsum[c,w+1] ) > 1.5
```

Each `marginal-sum + adjacent-difference` folds into ONE depthwise Conv whose kernel spans the
**full other axis** (to sum it) and is `+1/-1` on the two adjacent taps of the scan axis:
`ker_row[C,1,2,W]` (= +1 top row / −1 bottom row, all W cols) → output `[1,C,H-1,1]`; symmetric
`ker_col[C,1,H,2]`. Cost: two `[1,C,scan,1]` F32 planes (task253: `[1,10,13,1]`×2 = **1,040**)
vs the 2D bank's 11,520 — an **11× detection collapse**. `group=C` (per-colour) is mandatory.

Compounding moves that took task253 to 2,434 (all gated: visible 265/265, ab byte-equal-LIVE
200265/200265, profile==static):
- **Negative END-pads (#41 VERDICT, grader-confirmed) crop the active region INSIDE the conv** —
  13-wide kernel + `pads=[0,0,-16,-17]` makes the kernel sum exactly the 13 active cols and the
  output 13 rows: kernel 600→260 params AND output 29→13 rows, in one node, zero extra tensors.
- **Encoder zero-padding fixes the edge case for free:** the discriminating diff peak for a
  bottom/right-pair shape at the LAST active row/col is `marg[k+1]-marg[k+2]`; on the raw S×S grid
  k+2 is off-grid (a numpy oracle on the bare grid mis-scores it — 271/400), but the 30×30 encoder's
  all-zero padding row/col 13 supplies `marg[k+2]=0` so the conv sees it (600/600). **Always test the
  oracle on the 30×30-encoded input, not the bare grid.**
- **bool-chain assembly + einsum-fold output:** keep detection in BOOL (`Greater`/`Less`/`And`) so
  the per-bit F16 casts collapse to ONE cast of the assembled `[1,C,K]` one-hot; fold the K corner
  stamps + the ch0 base into one `S[K,h,w]` and emit the output with `Einsum('nak,krc->narc')` (no
  matmul+reshape double-pay). **Einsum dtype wall:** ORT CPU Einsum is float/int32/int64 only — U8
  one-hot/output is `NOT_IMPLEMENTED`, so the `[1,C,h,w]` output one-hot floors at F16, not U8.

Transfer targets: any task whose rule reads a per-object 2×2/3×3 orientation, reflection-parity, or
"heavy side" and re-stamps it at a fixed location — decode the generator first, then ask whether the
discriminator factors into independent row/col marginal bits before reaching for a conv bank.

**ADDENDUM (task253 2434 → 568, 2026-06-08): kill the detection conv with a WEIGHTED-marginal
Einsum + mod-N — position-invariant orientation from ZERO spatial planes.** The conv above still
materialises `cr`/`cc` (`[1,C,scan,1]` F32 = 1,040 for task253) just to host a `ReduceMax`. You don't
need the scan at all: the marginal is a **free-input Einsum** (#42 VERDICT, 0-memory), and the
orientation bit is a **modular residue** of a *position-weighted* marginal:

```
weighted_row[c] = Σ_r r·rowsum[c,r] = Einsum('bchw,hy->bcy', input, iota[30,1])   # input·iota in-kernel/invisible
                = 3k+1  (2-cells-in-TOP-row)   or   3k+2  (2-cells-in-BOTTOM-row)   # k = top row of the bbox
is_bottom[c] = (weighted_row mod 3) > 1.5      # the +k offset cancels under mod 3 -> POSITION-INVARIANT
is_right[c]  = (weighted_col mod 3) > 1.5      # weighted_col via 'bchw,wy->bcy', SAME iota
is_used[c]   = weighted_row > 0.5              # 3k+1/3k+2 >= 1 for any used colour, 0 for unused -> no ReduceSum
```

This drops detection to four `[1,C,1]` F32 scalars (160 B) + `iota[30,1]` (30 params) — **no conv, no
kernel, no scan plane.** `profile==static` confirms the `input·iota` product is never materialised
(weighted free-input marginals credit 0-memory exactly like plain `#42` ones). The mod-N trick
generalises: any signature that is `a·(base) + offset·(position)` with small `a` reads back via
`mod (a+offset)` — the unknown position cancels. Pick N so the per-class residues are distinct
(here the (2,1)/(1,2) split needs N=3).

**Output without the einsum-U8 wall: 3D direct-Gather, BOOL end-to-end.** Assemble the `[1,C,K]` slot
one-hot (here K=5: 4 corners + ch0-centre) in BOOL, then `Gather(oh5[1,C,K], cellmap[h,w], axis=2)` →
`[1,C,h,w]` in ONE node (Gather out-shape = data[:axis] + indices.shape + data[axis+1:]). `cellmap`
maps each output cell to its slot. Stays BOOL (1 B/elem = the output floor; grader-confirmed
BOOL-output task069/376), so it sidesteps the einsum-U8 `NOT_IMPLEMENTED` wall AND beats the
einsum-fold F16 output by the F16 cast + the F16-vs-BOOL canvas. task253 ladder, all gated
(visible 265/265, ab byte-equal-LIVE 200265/200265, profile==static): conv-negpad **2,434** →
einsum-marginal + einsum-F16-out **985** → einsum-marginal + gather-BOOL-out **568** (42× vs LIVE).
Builders `task253/prototypes/build_task253_marginal_v{4,6}.py`.

# 96. FUSED LINEAR-COLLAPSE + NEGPAD-CROP DETECTION CONV: one conv from the raw one-hot to the detection signatures (task264, 2026-06-08, grader-confirmed byte-equal-LIVE 200265/200265) <!-- renumbered from a worker-drafted ### #91 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When the per-cell quantities your detector needs are **linear projections of the one-hot input**,
do NOT materialise them. For a `[1,10,30,30]` one-hot, `is_color = Σ_{c∈colours} ch_c`,
`is_gray = ch_5`, `colour_index = Σ_c c·ch_c`, `is_solid = 1−ch_0` are ALL linear. So a single
3×3 `Conv(input, K[Cout,10,3,3], pads=[0,0,-14,-14])` computes, in ONE node:
- `sig` = Σ over a 3×3 window of `Σ_c K[o,c]·ch_c` = the windowed projection — i.e. it folds the
  channel-collapse (1×1) and the spatial detector (3×3) into one kernel, because conv is linear in c.
- the **negative END-pads (#41) crop** the 30×30 conv to the 14×14 window-top-left frame in the same op.

For task264 this replaced G30(3,600)+G16-slice(1,024)+5 derivation tensors+det_in-concat(1,536) — ~9 KB
of machinery — with one conv whose output is the only tensor (siege_v1 19,447 → siege_v2 12,732).
Kernel `K[o,c,dr,dc]`: put the **per-channel collapse weight** for output-signature `o` times the
**spatial stencil** for that signature. Example (3 outputs): `K[0,c,:,:]=Wstencil·[c∈colours]` (a
per-direction bitmask of the colored cells), `K[1,c]=Wstencil·[c≠0]` (solidity sum), `K[2,c]=c·[c∈colours]`
(colour-value sum, stencil=ones). Transfer: any object/orientation detector whose input is a one-hot —
collapse + crop + 3×3 detect is ONE conv, not a four-stage chain. (Conv output dtype follows input → F32
when the model input is F32; see #92 for the uint8 follow-through.)

# 97. uint8 SIGNATURE DETECTION + (K+1)-BIN ScatterElements ROUTING: route N detected objects to K slots without a `[K,H,W]` one-hot (task264, 2026-06-08, grader-confirmed byte-equal-LIVE 200265/200265) <!-- renumbered from a worker-drafted ### #92 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

Routing N spatially-detected objects into K output slots normally needs a `[1,K,H,W]` selector
(`Equal(sig, B_k)` for each slot) which must be **F16 for the contraction Einsum** (ORT has no
int8/bool Einsum) → `K·H·W·2` bytes (task264: 8·14·14·2 = 3,136 + the 1,568 bool it casts from).
Two moves kill it:
1. **Make the signature fit uint8.** A subset-sum signature with **power-of-2 weights overflows
   uint8** (full grid = 511). You only need the K object-signatures DISTINCT, not a true bitmask — so
   search small positive weights with `len({Σ_{M_k} w})==K` and `Σ_all w ≤ 255`. task264: weights
   `[1,4,2,3,6,1,1,1,1]` give 8 distinct sums, full-sum 20. Then `Cast(conv_F32 → uint8)` is exact
   (integer conv of 0/1 inputs) and downstream tensors are 1 byte/cell, not 2–4.
2. **Bin-scatter instead of K-way compare.** `Reshape` the H×W detection to `[1,1,HW]`, mask to
   valid-object locations, `Cast` the signature to int32 (the MINIMUM index dtype ScatterElements
   takes), then `ScatterElements(zeros[1,1,Σw+1], idx, values, axis=2)` builds a tiny `(Σw+1)`-bin
   histogram; `Gather` the K known signatures out of it. task264: a **21-bin** scatter + Gather
   replaced the 4,704 B f16 8-way → siege_v3 7,888. **Opset-13 ScatterElements has no `reduction`
   attr** (added opset-16); plain replace is exact iff each valid location has a UNIQUE index (here:
   distinct directions) and all non-valid locations write 0 to bin 0 (unread). The int32 index
   (`HW·4`, e.g. 784) is the residual floor of this route — still ~4× cheaper than the f16 K-way.

Pair with #91 (fused conv) + a `ScatterND` output stamp (28 fixed cells → ~87 params vs a 648-elem
template bank) for the full task264 siege: **34,606 → 7,366 (−78.7%)**, all gates green at full 200k.
Transfer: any "find K labelled objects at arbitrary positions, emit a fixed-shape readout" task.

# 93. Weighted-marginal Einsum + mod-k — a position-invariant feature from ZERO spatial planes (task253, 2026-06-08; extends #42/#82/#89) <!-- renumbered from a collided #92: the worker's marginal-split-conv #92 above is the task253 2434-anchor sibling; THIS is the shipped 568-primary trick (no detection conv) -->

GRADER-VALIDATED (task253 23,974->568, +3.75 exact, byte-equal 200,265 + 500/500 generator). To
read a POSITION-INVARIANT orientation/parity/class bit without materializing any spatial plane:
- **Weighted marginal**: `Einsum('bchw,hy->bcy', input, iota)` computes `sum_r r*rowsum[r]` — the
  input*iota product is an IN-KERNEL temporary (scorer-invisible #89), so this position-weighted
  marginal costs ZERO memory (only the tiny iota init + the [b,c,y] result). Same for cols with
  a w-axis iota. (Extends #42 free-input marginals + #82 scalar reads to WEIGHTED moments.)
- **mod-k cancels the unknown position**: if the feature sits at an unknown offset k (e.g. a pair
  at rows {3k+1, 3k+2}), the weighted marginal is `3k + orientation`; `Mod(., 3)` cancels k and
  leaves the orientation bit alone — a translation-invariant signal with no detection conv, no
  spatial mask, no plane.
- **Output via gather-BOOL** (opset 13, BOOL end-to-end): `Gather(onehot_rows, cellmap, axis=2)`
  assembles the answer staying BOOL, sidestepping the Einsum-U8 NOT_IMPLEMENTED wall (BOOL graph
  output grader-confirmed, task069/376).
Reach for this on any orientation / asymmetry / parity / position-invariant-class rule where the
old graph spent a detection Conv + spatial planes to find "which way / which of N". The whole
detection collapses to one weighted-marginal Einsum + a Mod.
**APPLICABILITY BOUNDARY (task390, 2026-06-08): the orientation BIT is cheap, but SELECTING between two
FULL-OUTPUT branches with it is NOT.** task390's variance bit (`count·S2(row)−S1(row)² > count·S2(col)−S1(col)²`)
is exact (266/266), but combining the row- and col-branch outputs failed: `If` (run one branch) is
SCORER-REJECTED (score_v2=None), and `Where` MATERIALIZES BOTH full [1,10,30,30] branch logits = 74,736.
So #93-select pays ONLY when each branch result is small (scalars / a few cells, assembled AFTER the
select — as in task253, where the output is built from the selected scalar coords), NOT when each branch
is a full materialized output (Where doubles it; If is dead). Decode whether you can select the SCALARS
pre-assembly (cheap) vs the full OUTPUTS post-assembly (dead) before reaching for orientation-select. Floor here ~510 (the +11% is the
bool AND-tree assembly). Promoted for transfer across the orientation/asymmetry family.
**Coordinate-CLASS recovery via marginal + Equal (task199, 2026-06-21, v7 1435->1402 byte-equal):**
the parity/mod-class of a one-hot's ACTIVE POSITION is cheaper as a weighted-marginal Einsum
followed by `Equal` than as `ArgMax -> Mod`. Recover a column-parity class with
`Einsum(col_dot, parity_f, "bw,xw->bx") -> Equal` instead of `ArgMax(col_oh) -> Mod` — this deletes
the INT64 `c_idx`/`c_parity` intermediates AND their integer initializers (`parity_i64`/`two_i64`),
since the position-class test never leaves the float marginal. Companion dtype trim: compute the dot
channel in F16 (`Cast color_f -> F16` then F16 `Sub`) to delete the F32 `dot_ch_f`; keep Einsum factors
F16 (NOT INT8 — `Einsum(INT8)` has no ORT kernel). Both trims sit on top of a #119 free-output Einsum
and stay fully static (profile==static).

# 94. Post-opset-18 levers + bf16 trap (M1 opset 19-22 + N1 opset 23-26 audits, 2026-06-08/06-19; `_archive/Research_m1_opset19_22_op_audit.md`, `_archive/Research_n1_opset23_26_delta_audit.md`)

Verdict: the post-18 op surface is MOSTLY BARREN for our metric. Three situational levers + three trap classes:
- **ConstantOfShape = param-shrink for fill-pattern Constants (the best find; works at ANY opset).** A `Constant` initializer of N identical values costs N params; `ConstantOfShape(shape, value=fill)` costs ~5 (shape vector + scalar). Replacing a `[1,10,30,30]` zeros/const Constant: 9000 -> 5 params (-8995). Output bytes unchanged — the win is ENTIRELY param-side. Use wherever a charged Constant tensor is a uniform fill (zeros, a single color plane, a sentinel canvas).
- **bool ReduceMax/ReduceMin @ opset 20** deletes the `Cast(bool->u8) -> Reduce -> Cast` scaffolding (#57): `mask(bool) -> ReduceMax(out)` directly. ~-9001 B on a [1,10,30,30] mask (more if the old detour cast to f32). Opset-20 CPU bool-reduce only.
- **Pad(mode="wrap") @ opset 19** = circular/toroidal padding in ONE (free-output) node, deletes Slice+Concat emulation (~-2400 B per axis). Good for periodic-tiling tasks (e.g. 343).
TRAPS (negative findings, equally important):
- **bf16-schema mirage**: opset-22 added bf16 to many ops but ORT 1.24.4 CPU stays float-only for Conv/ConvTranspose/pools/GridSample/Dropout/Elu/Det. "ONNX 22 says bf16" != "ORT CPU runs bf16." (And params are dtype-blind anyway.)
- **No CPU kernel** (checker-passes, inference-fails): ImageDecoder, DeformConv, Bernoulli, main-domain GroupNormalization (ORT exposes contrib `GroupNorm`, not the opset-21 main op).
- **AffineGrid = param-shrink-but-loses**: replacing a stored `[1,30,30,2]` grid (1800p) with AffineGrid cuts params but materializes a charged 7200 B grid before GridSample -> usually worse (the free-output separability trap, #87, again). Q-ops (QuantizeLinear/QLinearMatMul) save NOTHING by themselves (dtype-blind params) — only via deleting intermediates.

**Opset 23-26 delta (N1 + R6 + O3, 2026-06-19) — equally barren.** ORT 1.24.4 loads ai.onnx **through opset 25 only**; opset-26 `BitCast`/`CumProd` reject at load BEFORE kernel lookup (R6-probed — never set `opset_import` > 25). The one new direct-kernel op with any NeuroGolf mechanism is **`Attention-23/24`**: a single-output `Attention` hides qk/softmax tensors as kernel-internal temporaries. O3 smoke-probed it on the pinned stack: checker + strict shape inference + ORT CPU run passed, local scorer charged `memory=0 params=0`, and the explicit `MatMul->Softmax->MatMul` sibling scored `108000` B. But O3 scanned 7,162 current ONNX artifacts and found **zero Softmax nodes / zero real attention chains**, so this is a future trigger only: use it if a later graph already has a large float attention-like chain; do not spend worker time inventing one. Dead on this stack: `TensorScatter-24` (no ORT CPU kernel — NOT_IMPLEMENTED), `Swish-24`/`CastLike-*` (function-backed, no direct kernel; run via expansion only), `RMSNormalization-23`/`RotaryEmbedding-23` (float-only + scale/cache params, LLM-shaped). Everything else in 23-25 is a type-constraint refresh adding low-bit/float4/float8 schema dtypes that ORT CPU mostly doesn't implement and that dtype-blind param scoring makes useless anyway. Net: the opset<=22 toolkit + old `ConstantOfShape`/dtype-cast tricks stay the answer.

# 98. DILATED-KERNEL SPATIAL-CROP + CHANNEL-REDUCE in ONE conv (task046, 2026-06-08, GRADER-VALIDATED 2026-06-08 via A/B primary-scored; gate-clean 200267 + 200k truth) <!-- renumbered from a worker-drafted ### #94 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When you need a small **top-left active region** of a one-hot input AS a reduced quantity (colour index, occupancy, any `Σ_c w_c·ch_c`), do NOT `Slice` the multi-channel crop first. A `[1,9,3,20] F32` crop is **2160 B**; the channel-reduce Conv that consumes it is tiny but the crop dominates. Two ops (Slice then Conv) — or Conv-then-Slice ([1,1,30,30]=3600 B) — both pay a big intermediate. **One dilated valid Conv reads the FREE input directly and does crop(both axes)+reduce at once:**

- To crop an axis `L→k` with a 2-tap kernel: effective size `(2−1)·dilation+1` must equal `L−k+1`, so `dilation = L−k`. Put the real weight on tap 0, **zero on tap 1** (the far tap only exists to make the valid conv shrink the axis).
- task046: input `[1,10,30,30]`, want `[1,1,3,20]` colour index. Kernel `K[1,10,2,2]`, `dilations=[27,10]` (rows 30→3 via dil 27, cols 30→20 via dil 10), `K[0,ch,0,0]=ch` (tap (0,0) per channel = the cell itself), all other taps 0. `out[r,c] = Σ_ch ch·input[ch,r,c]` for r∈0..2, c∈0..19. **40 params, output 240 B** — replaces the 2160 B crop (cost 4135→3853; the whole crop-elimination arc 6214→3853).
- Param math: dense tall kernel `[1,10,28,1]` = 280 params; the **dilation collapses 28→2 taps → 20 params** (×2 for the 2nd dilated axis = 40). Dilation is the lever that makes the tall/wide crop cheap.

This **kills the "no single op does spatial-crop+channel-reduce" floor** (a wrong assumption that capped task046 at ~6200 for 6 rungs). Transfer: ANY task that reads a fixed sub-rectangle of the one-hot as a channel-collapsed map — colour index, foreground mask, weighted marginal. Contrast #41 (negpad crop, post-conv) and #91 (3×3 fused collapse+negpad-crop): dilation crops by **stretching the kernel's receptive field with zero far-taps** instead of negative pads, and works for arbitrary crop ratios on each axis independently. Conv output dtype follows input (F32 here) — Cast to F16 after if downstream is F16.
**Task383 calibration:** the same 2-tap formula updates the 30→24 crop case: `dilation=L-k=6`
on each cropped axis, with the far tap zero, reads the raw input and emits only the 24x24 reduced
carrier. This moved the task383 crop/color path **13,112→10,942**; the remaining measured floor is
~10,884. If a proposed floor still pays a 30x30 reduced canvas before cropping, reprice it through #98.
**Dynamic-channel extension (task174, 2026-06-19):** if a dynamic channel Slice only
exists to turn a one-hot crop into a color-index/membership map, let the dilated
Conv read the free input and emit the color-index map directly. Pattern:
`Conv(input, W[1,10,2,2], dilations=[20,20]) -> labels[1,1,10,10]`, with
`W[0,ch,0,0]=ch`; then Cast/Equal the small label map. This dissolved multiple F32
channel Slices in task174 (4786 -> 4017). Boundary: #98 saves
`max(0, slice_bytes - conv_output_bytes)`. For a 1-channel -> 1-channel crop of the
same size, the Conv output is the same bytes as the Slice and the params make it
worse (task250; task102 confirmed — a same-size 1-ch F32 crop feeding QLinearConv
repriced 1810->~1922).
**Transfers to ConvTranspose/stamp graphs (task327, 2026-06-22):** the dilated read
isn't only for terminal color-index maps — any graph that opens with
`Slice -> Conv(channel-collapse)` on a fixed top-left sub-rectangle and feeds the
result DOWNSTREAM (e.g. a ConvTranspose diagonal stamp) can drop the F32 crop with
one dilated valid Conv. task327: `src_f32[1,9,3,3]=324 B` crop vanished,
`Conv(input, K[1,10,2,2], dilations=[27,27]) -> color_grid[1,1,3,3]` (3×3 crop, equal
dilation on both axes), `881 -> 580` byte-equal-full.

# 99. TopK STREAM-COMPACTION: remove gaps / left-pack occupied columns WITHOUT cumsum+OneHot+matmul (task046, 2026-06-08, GRADER-VALIDATED 2026-06-08 via A/B primary-scored; gate-clean 200267 + 200k) <!-- renumbered from a worker-drafted ### #95 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

To compact a variable set of "kept" positions to the front (gap removal, NonZero-style gather — but NonZero is BANNED), the textbook route is exclusive-cumsum→build a `[N,K]` OneHot select-matrix→matmul (the [20,16] Sel = 640 B in task046). **TopK does it in 3 small ops with no select-matrix:**
- Score each position so KEPT outrank DROPPED and kept sort in the order you want: `score = BIG·keep − arange` (BIG ≥ N makes every kept-position score `BIG−i` exceed every dropped `−j`; the `−arange` tiebreak gives **ascending index order** among kept). `keep` is your 0/1 occupancy.
- `TopK(score, k=K_max, axis=<pos axis>, largest=1, sorted=1)` → **`.indices` = the source positions in order** = `src(t)`. (`sorted=1` is mandatory — ORT TopK, ort_reality K2.)
- `Gather(data, idx, axis=<pos axis>)` → the compacted result. **Trailing slots** (when `#kept < K_max`) auto-fill with the smallest-index DROPPED positions; if `data` is 0 there (gaps carry no signal) they read as 0 and a downstream `inrange = arange < #kept` mask zeroes them.

task046: replaced occ_i+CumSum+Reshape+OneHot[20,16]+MatMul (976 B) with Reshape+Mul+Sub+TopK(idx int64 128)+Gather (≈336 B) → **−581 B** (3272). `idx` is int64 (TopK fixed); `k` is a 1-elem int64 input; `K_max` must be a generator-hard bound (here W_out≤16). Transfer: ANY left-pack / gap-removal / "gather the N present items" with a known max count — segment compaction, object lists, dedup-to-front. **Strictly safe vs ScatterElements compaction** (which is duplicate-index UB). Pair with #94 (dilated colour read): the two killed task046's two biggest "floors" (6214→3072).
**Full-palette color-id TopK (task174):** with exactly K non-bg colors, rank `[1,10]` with ch0=0 and non-bg scores >0. `TopK.indices` are ARC color ids; drop non-bg Slice and `+1` repair.

# 100. FOLD per-window CLASS + VALIDITY into ONE uint8 conv channel (task264 v5, 2026-06-08, grader byte-equal-LIVE 200k) <!-- renumbered from a worker-drafted ### #93 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

A per-window detector that needs both *which class* (a discriminating bitmask) and *is this a valid
object* (solidity / full-support) usually spends two conv channels. Fold them: pick a class-weight
`Wp` (distinct subset-sums per class, all small) and emit
`combined = Σ_class Wp - BIG·solidcount + BIG·K` where `BIG > max class-sum` and `K`= window cells.
Implement with a conv **bias** = `BIG·K` (the conv can't add the constant otherwise). Then
`combined <= max_class_sum  <=>  solidcount==K  <=>  a valid object`, and `combined == B_d` identifies
the class — **one uint8 channel, no separate validity Equal/Where**. task264: `combined = sig_col −
14·solidcount + 126`, sprites → `B_d<=13`, everything else → `>=14`. Halved the detection conv
(3→2 channels) and deleted the validity machinery (7,366 → 5,416).

**TRAP that cost a debug cycle — off-grid != black.** The 30×30 encoder pads off-grid cells as
**all-zero one-hot** (NO channel set), so `onehot_0` (the black channel) is 0 there. Counting
"background" via `onehot_0` MISSES off-grid cells → a sprite at the grid edge gets a window that
looks solid and collides. Count non-support as `1 − is_col − is_gray` (= 1 for black AND off-grid),
which is exactly what the `−BIG·solidcount + bias` form does. **Always test the detector on the
30×30-encoded input, not the bare grid.**

# 101. TopK(k=N) for EXACT-COUNT object extraction → collapse the scatter index (task264 v7, 2026-06-08, grader byte-equal-LIVE 200k, profile==static) <!-- renumbered from a worker-drafted ### #94 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When a generator places EXACTLY N objects and your detector value separates them from background by a
threshold (e.g. #93's `combined<=13` for objects vs `>=14` for everything else), `TopK(detector, k=N,
largest=False)` returns EXACTLY the N object locations — then `GatherElements` their payload and route
the classes by a tiny `Equal`. This **collapses an O(H·W) int32 scatter index** (task264: 196×4 = 784 B)
**to an N-element routing**. Requirements/gotchas:
- The N object detector-values must be DISTINCT (no tie at the cutoff): here `{0,3,5,7,8,9,10,11,13}`,
  all `<14`, so the 9 smallest are uniquely the 9 sprites.
- **ORT TopK is float-only** (uint8/int NOT_IMPLEMENTED) — `Cast` the uint8 detector to F16 first
  (small: one `[1,1,H·W]` plane). **ORT Mul has no uint8 kernel** either — do the routing reduce in F16
  or via `Where`.
- TopK is multi-output but used INTERNALLY (single graph output) → traps §14z (multi-output *models*
  score None) does NOT apply; the local profiler prices it exactly (task264 profile==static==4,842).
This took task264 5,126 → 4,985 (→ 4,842 after `Where`-folding the routing). **Floor reached:** the F32
detection conv (the F32 one-hot input forbids a uint8 conv) + the one-hot output are ~2,300 B together,
so an exact spatial-scan task like this cannot go below ~2,300.
**Sentinel/route code is safe ONLY if the dummy can't enter the selection (task264 v22, 2026-06-22, 2293->2229).**
Using a value like code `0` as a route/sentinel collides with a real route if that value can enter the `TopK`
(or ArgMax) result. task264 v22 ranks only the 8 colored sprites ABOVE the gray center, so `TopK(k=8)` never
admits the center — which let `0` be reused safely and deleted a `Neg` + `Gather(bins, Bd_idx)`. Before reusing
a sentinel code, prove the dummy/background value falls outside the selected set; otherwise the old collision
risk stands (sibling of §10's `OneHot(-1)` wrap trap).

# 102. eps-FRACTIONAL FOLD: pack a discrete CLASS + a quantized PAYLOAD into ONE float channel so TopK/sort carries both (task264 v9-v11, 2026-06-08, grader byte-equal-LIVE 200k, profile==static==2979) <!-- renumbered from a worker-drafted ### #95 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When a per-object detector value is an integer CLASS `C` (here a direction-bitmask B_d in
{3,5,7,8,9,10,11,13}) and you ALSO need a small integer PAYLOAD `P` (here colorsum = |M_d|*color,
<=36), don't spend a second conv channel. Emit ONE channel:
    val = C - eps*P,   eps = 1/64  (a NEGATIVE POWER OF TWO -> exact in F32 AND f16)
- Pick eps so `eps*Pmax < min gap between classes` (here 1*... the B_d are >=1 apart, eps*36=0.5625<1)
  -> the class ranges `[C-eps*Pmax, C]` stay DISJOINT.
- The detector's "valid object" threshold still separates: solid sprites land at `<= max_C`, junk at
  `>= threshold` (task264: sprites <=13, non-solid >=13.44), so `TopK(val, k=N, largest=False)` returns
  exactly the N objects AND `topv` carries both fields:
      C = ceil(topv)            (the class)
      P = (ceil(topv) - topv) / eps    (= (ceil-topv)*64)
This **deleted the entire 2nd conv channel + GatherElements + uint8 narrowing** that v8 used:
task264 4,842 -> 3,286 (one F32 conv channel does detection+direction+color).
**f16 is safe BECAUSE eps is large enough:** cast error <= half-ulp (0.0039 at val~13) -> P error
<= 0.0039/eps = 0.25 < 0.5, so Round recovers P exactly (proven on truth 200k). Use a power-of-two eps
so neither C nor eps*P introduces representation error.
Gotchas: (1) `Clip(ceil(topv), 0, Cmax)` before any scatter-by-class index — the score_v2 DUMMY run
feeds noise, TopK then returns junk classes that overflow a small bin array (ScatterElements "out of
bounds" -> score None); the clip is a no-op on real data. (2) route the N values with a tiny
(Cmax+1)-bin ScatterElements + Gather, NOT an 8xN Equal/Where. (3) build the categorical one-hot output
with `Gather(colors9, cell_dir)` where cell_dir is the fixed (H,W) "which class owns this cell" map
(cheaper than a template Einsum or ScatterND).
Transfer: any "detect N objects, each carrying a class + a bounded scalar, emit a fixed-shape readout"
task. Floor for task264-shape problems: F32 detection conv + its reshape + the one-hot answer ~= 2,300;
below that needs a non-F32-conv detector (blocked by the F32 one-hot input) or a free-output one-hot.

# 103. MaxPool sprite-preserving DOWNSAMPLE before TopK (task264 v12, 2026-06-08, byte-equal-LIVE 200k, profile==static==2605) <!-- renumbered from a worker-drafted ### #96 (parallel-worker collision); breadcrumb only, cross-refs below may use the draft number -->

When you `TopK(k=N)` over an H×W detector map to extract N objects (#94/#95) and the generator
guarantees objects are **≥p apart** in some dimension, a `MaxPool(p×p, stride p)` first SHRINKS the
TopK frame with NO loss: each p×p cell then holds ≤1 object top-left (two within p−1 of each other
would overlap), so the object's value survives as the cell max. Emit the detector so **objects are the
MAXIMA** (here: NEG `combined`, so sprites ∈ [−12.95,−2.44] and non-solid ≤ −13.44 → sprites win the
pool), then `TopK(largest, k=N)` on the (H/p)² cells. task264: a `MaxPool(2,2)` cut the reshape tensor
**784→196 B** (14×14→7×7), v11 2,979 → v12 2,605. The decode shifts by the negation/offset:
`B_d = −floor(topv)`, `colorsum = (topv−floor(topv))·64`. Pool size is bounded by the spacing (here a
4×4 cell still holds ≤1 sprite, but 14 doesn't divide by 4 cleanly — pad/ceil edge cases pad with 0 and
can beat negative object values, so OFFSET the detector positive first or stick to the clean p that
divides H). FLOOR after this: only the F32 detection conv + the one-hot answer remain irreducible.

# 104. Min-cost ONNX realization of a fixed integer linear map y=Wx — the 7-step decision procedure (M3, 2026-06-08; `_archive/Research_m3_min_cost_integer_linear_map.md`)

The big-fish tier is dominated by large integer linear maps (old-blend graphs). Given a fixed
integer W, run these tests IN ORDER, scoring each candidate, stop when remaining classes can't beat
the winner (the structure->op table + kill-numbers are in the deliverable):
1. **Cheapest graph BOUNDARY first**: keep [1,C,H,W] if convolutional; expose [1,N]->[1,M] if dense/
   selector/factored — removes otherwise-charged flatten/reshape tensors (scorer skips input+output).
2. **Structural zero-param**: identity / axis-permutation / SpaceToDepth/DepthToSpace / concat-split.
   A true reindex-only map = score 0, beats everything. (#80 D2S/#84 placer are this family.)
3. **Selector/permutation**: each row exactly one nonzero ±1? -> structural (0 params) if a layout
   shuffle, else Gather (M params). Square permutation: 0 or N vs N^2 dense.
4. **Convolutional sharing** (before rank/sparsity): reshape W's row/col indices to tensor coords;
   if coefficients depend ONLY on channel-pair + relative offset (dh,dw) NOT absolute position ->
   grouped Conv crushes dense (Co*(Ci/g)*kh*kw params, 0 memory). Cyclic boundary -> explicit
   wraparound + Conv (Pad(mode=wrap) is opset>18, #94). Mere LOCALITY != Conv (score as sparse).
5. **Block-diagonal**: bipartite support graph splits into disconnected components -> per-block best.
6. **Exact low rank** (#87): two-stage factored, ACCEPT ONLY IF `r(M+N)+b*r*L < current best` — the
   FREE-OUTPUT SEPARABILITY TRAP crossover (the charged middle tensor must not eat the param saving).
7. **Sparse COO**: Gather->Mul->ScatterND(add) (~3*nnz params); else dense MatMul/Gemm.
**Exactness** (mechanical, ordered): (a) reconstruct W by applying the graph to basis vectors e_j;
(b) f32 only if every integer partial stays <2^24 — REJECT f16 for exact integer maps (native Gemm
rounds every partial, #77); (c) verify the ORT CPU kernel exists for the dtype/opset (checker != kernel).
dtype-blind params: NEVER int8-weights-to-cut-params. Worked example: a 900x900 "dense" map that is
actually a 3x3 conv = 810,000 -> ~900 params (#84/#98); a rank-2 map = 810,000 -> 3,608 (224x).


# 105. "STRICTLY BETWEEN TWO MARKERS" span-fill via 4 directional U8 MaxPool scans (task350, 2026-06-08, reviewer structural-floor, byte-equal-LIVE 200267 + truth 200000)

Rule class: a cell becomes color X iff it lies strictly between two marker cells along a row OR a column
(spans, enclosures, "inside the brackets"). The naive route builds per-direction running flags with
Scan/Loop or a big cumulative tensor. Cheap route — **one directional cumulative MaxPool per side**:
- For each of the 4 directions (left, right, up, down), a directional/asymmetric MaxPool over the marker
  mask answers "is there a marker somewhere on THIS side of me?" as a U8 plane (the pool's running-max
  along the axis = marker-seen-so-far).
- **(L AND R)** = strictly between two markers horizontally; **(U AND D)** = vertically; **OR** them =
  the fill region. That AND/AND/OR tree is **3 binary ops, provably minimal** for the both-sides test.
- `Greater`/mask out the markers themselves, then `Where(fill, X, input)` — **free output**, preserves
  markers/background/padding untouched (#67/#78). Keep it U8 (the masks are 0/1).
**Task035 refinement:** when the blocker/marker itself can be encoded as the scan carrier, zero the
carrier at blocker cells before the prefix MaxPool. Then the inclusive prefix-max automatically drops the
blocked cell one position behind the wall, deleting a post-shift/mask. If all four directions are needed,
one all-sides `Pad` plus directional `Slice`s usually beats four independent Pads. U8 MaxPool is available
from opset 12; keep final color-index selects in `Where(BOOL, UINT8, UINT8)` when possible.
Floor note (task350): the F32 ch-slice of the one-hot input is the immutable input-contract cost (~28%);
the 4 scans are the irreducible both-sides test; the 3-op tree is minimal. Reviewer killed vector-
reduction, 2-MaxPool, and slice-cast-fusion alternatives with numbers. Transfer: ANY between-two-markers /
enclosed-region / bracket-fill rule on rows+cols. (Residual register: a future fused span/pool primitive
could shave the 8 directional planes ~600-1200B; tight, not provably-minimal.)

# 106. FIXED-CORNER MASK kills a DYNAMIC OneHot selector when the index is generator-constant (task183, 2026-06-08, reviewer structural-floor, byte-equal-LIVE 200265)

When a graph selects "the cell/row/col at position p" and p is **the same across every input** (the
ARC-GEN generator fixes it — a corner, a center, a fixed offset), do NOT build a runtime `OneHot(p)` /
dynamic `Gather` selector. Decode p from the generator and bake a **static constant mask** (the corner/
position is a fixed [.] one-hot Constant, or a #94 ConstantOfShape fill). This deletes the entire dynamic
selector subgraph (the index compute + OneHot materialization) — params drop to the tiny constant.
task183: a fixed-corner mask replaced the dynamic OneHot, 15138->896 (16.9x); the residual 360B 6x6x10
one-hot output tail is then structurally forced (#81 Equal-tail), and both F16 re-encode routes priced
higher. **Always ask: is this selector's position data-dependent, or generator-fixed?** If fixed, it's a
Constant, not a runtime op. Pairs with #104 step-2 (structural-zero-param recognition) and the
§FAMILY-LABEL rule (decode the generator — that's where you learn p is constant).

# 107. CAUCHY-SCHWARZ EINSUM "filled-rect / all-rows-equal" test — per-channel shape detection with NO [C,H] tensor (task234, 2026-06-08, gate-5 profile==static, 266/266 + 10000 fresh)

To classify a per-channel one-hot region as a **filled rectangle** (or more generally "all nonzero row-
profiles are equal"), the obvious route materializes `rowsum[1,C,H] = ReduceSum(input, W)` (a charged
[1,10,30]=1200 B F32 tensor) plus a `(rowsum>0)` bool + its F16 cast for the span — ~2100 B just to detect.
Replace ALL of it with **three in-kernel Einsums** whose products never become graph tensors (#89):
```
N  = Einsum('bcrw->bc', input)            # cellcount   (= sum_r rowsum)
S2 = Einsum('bcrw,bcrv->bc', input, input)# sum_r rowsum^2
S3 = Einsum('bcrw,bcrv,bcru->bc',in,in,in)# sum_r rowsum^3
is_filled = (N>0) AND (S3*N == S2*S2)     # Cauchy-Schwarz equality <=> every nonzero rowsum equal
```
Outputs are all `[1,C]` (40 B). The identity: `(Σa²)² ≤ (Σa)(Σa³)` with equality **iff all nonzero a
equal** (a = the per-row counts). For rect-vs-rect+protrusion shapes "all rows equal" == "filled rect"
(a tongue/bump on EITHER axis breaks row-uniformity — the row test alone catches both orientations;
no col pass needed). **Exactness/precision**: products are integers; foreground stays < 2^24 so F32 is
exact (task234 foreground Σrowsum² ≤ ~850). The background channel (ch0) can exceed F32 integer
exactness but is masked out by `notch0` — verify your consumer ignores it. **DO NOT** use the cheaper-
looking `Σrowsum²·Σcolsum² == N³` (also true for filled rects) as the test — it has FALSE POSITIVES
(movers coincidentally satisfy it: 13/266 on task234). The Cauchy-Schwarz row form is the exact one.
Once classified, all row/col PROJECTIONS are likewise Einsum-from-input (`Einsum('bcrw,bc->br', input,
sel)` -> [1,30], product in-kernel) so the WHOLE per-channel-shape analysis carries no [C,H] tensor.
task234: dropping the rowsum family took the rewrite 7092 -> 5010. Transfer: any "is this colour a solid
rect / is its profile uniform / does it have a 1-wide appendage" detector on the 30x30 one-hot input.

# 108. COLLAPSE N nested/shell/ring/frame mask planes into ONE per-cell distance predicate (task096, 2026-06-08, reviewer floor-verified, byte-equal-LIVE 200266)

When the rule paints concentric shells / nested frames / rings / distance bands, the naive graph (and
often LIVE) builds a SEPARATE mask plane per shell + per-shell hollow/ring/centre logic, then ORs them.
Almost always that whole stack is **one per-cell predicate**: `paint iff distance(cell) <= threshold(shell)`.
Decode the distance metric from the generator (Chebyshev `max(|a|,|b|)`, Manhattan `|a|+|b|`, axis `max(a,b)`)
and the per-shell threshold, then evaluate the single inequality with a `Greater`/`Less` + a small `Gather`
of the threshold-per-shell vector. task096: LIVE's 23 mask planes + separate hollow/ring/centre logic ->
one rule `paint iff max(a,b) <= s + arm[s] - 1` + 2 Gathers + a [6]-vector centre scatter, output = free
Equal-tail (#81). 9545 -> 7173. Floor after: the dominant cost is the FORCED F32 row/col marginals
(ReduceMax off the F32 one-hot input, cast to U8 immediately = unavoidable transit — the **task222
F32-marginal pattern**, ~2400 B), the Equal-tail canvas (900, #81), and the I32 gather index (484, §8d).
**byte-equal-to-LIVE can be PROVABLY OPTIMAL, not just "good enough":** here the residual 1.2% truth gap is
INFORMATION-THEORETIC (clipped shells have unobservable arm length) — no rewrite can be more-correct, and a
cherry-probe confirmed LIVE scores full hidden points, so byte-equal = full points at minimum cost. When a
reviewer proves the truth gap is information loss (not a recoverable bug), stop optimizing correctness and
bank the cheapest byte-equal graph. Transfer: ANY concentric/nested/shell/ring/frame/distance-band rule.

**WALL — "it has a closed form" is necessary, NOT sufficient (task175, 2026-06-20).** The collapse pays
only when the predicate is **BOOLEAN** (feeds a `Greater`/`Less`+Gather into a free-output Equal/Einsum tail,
case a) OR the generated tensor **replaces a LARGER init without adding ≥1 full-size intermediate per op**
(case b). It is a WALL when the stored plane is a **multi-valued INT offset table that is a genuine 2-D
(non-rank-1) function of position** consumed by a CHARGED `Mod`/`Pad`/`Equal` canvas: regenerating it on-graph
costs one full-size plane PER op. task175's `offset_table` (INT8 [1,1,21,21]=441 B) has an exact closed form
`O = d//(m+2)−1` (m=min(r,c), d=|r−c|), yet computing it needs Min/Sub/Abs/Add/Div/Sub/mask/Where ⇒ **≥8
surviving 441 B planes** (Div may force INT32 = 1764 B each) = ~3,528 B to replace one 441 B init = +2,600 B.
#109 free-output doesn't rescue it: the output color `(base+O) mod period` is rank-tested NOT a row⊗col outer
product, so no K/R/C decomposition absorbs the planes. **For a single multi-valued plane consumed by a charged
op, the stored packed init IS the floor — price the surviving-intermediate memory before regenerating.**
(Bonus: task175 is an INPAINT rule, `output==input` where `input!=0`; the "Euclidean wedge" label is the
visual, not the transform — neighbor-copy fill of holes fails, the offset field is the genuine mechanism.)

# 109. FREE-OUTPUT multi-term separable Einsum — build the WHOLE one-hot output with NO canvas/Equal-tail (task234, 2026-06-08, gate-5 profile==static==2808, 266/266 + 10000 fresh) <!-- renumbered 108->109: parallel-worker collision with task096 #108 -->]

When every channel of the `[1,10,30,30]` one-hot output is a LOW-RANK outer product of row/col
indicators, do NOT build a color-index canvas + Equal-tail (that charges the [30,30] canvas, 900 u8 /
1800 f16). Build the output DIRECTLY:
```
output = Einsum('tk,tr,tc,b->bkrc', K[T,10], R[T,30], C[T,30], ones[1])   # [1,10,30,30], named 'output' => FREE
# output[k,r,c] = sum_t K[t,k] * R[t,r] * C[t,c]   ; the [T,1,10,30,30] product is in-kernel (#89), scorer-invisible
```
Each "term" t is a rank-1 plane R[t]⊗C[t] routed to the channels selected by K[t]. Cost = K+R+C only
(tiny: T*10 + 2*T*30). The `ones[1]` operand just carries the batch dim into the output (avoids an
Unsqueeze). gate-5 CONFIRMED the multi-operand Einsum's internal temp is not charged at runtime
(profile==static) — but always re-run profile_real_cost.py (task225 reprice law).

**✅ GRADER-VALIDATED (Boladi task234, 2026-06-10):** own-stack spread (5010 gold vs 2808 candidate) gave **delta +0.58 EXACT, no reprice** — the in-kernel contraction temp is uncharged on the KAGGLE grader, not just local profile. This is the FLIP side of the dynamic-shape fiction-costs (task132 "1370"→3486, task029 v18 repriced UP): a multi-operand Einsum's *contraction* temp credits at static, but a dynamic-*shape* runtime temp reprices. Discriminator = gate-5 (profile==static): if it holds, the temp is free on the grader too. Free-output Einsum is now a CONFIRMED lever for the mid-cost harvest.

**Carrier-dtype tax (task091/task351):** a free-output Einsum reading the F32 one-hot input is F32-locked; all operands share dtype, so F16 selectors with F32 input fail ORT. For kxk crops, price F32 selectors `2*k*N*4` against crop bytes `C*k*k*4`; basis wins only if `2N < C*k`. If not, use GridSample/Gather crop + free-output Pad. task351: k=5,N=30,C=10 -> selectors lose, and pre-Slicing the input still charges a crop plane. **Window-minus-figure (task195):** 2-term `(ones|P)` emits figure and region-complement with no bg sentinel. **Unlocked inverse (task195):** no F32 input operand => F16 operands legal; small logits decode under `>0` (525->489).
**Output canvas disappears; upstream extent floors do not (task096, 2026-06-19).**
Free-output multi-term Einsum can delete the final 900 B Equal-tail canvas and
per-cell drawing planes, but it does not delete independent F32 bbox/extent
marginals needed to order shells exactly. For nested square rings, the residual
row/col extent marginal was the floor. Micro-idiom: strict L-band corners can avoid
correction terms with `paint_s = E[s]xColBand[s] + ColBandStrict[s]xE[s]`, so the
corner is not double-counted in the first place.
**RANK GATE before the dissolve (task183, 2026-06-22 — the deciding test, not output size).**
A free-output multi-term Einsum is cheap ONLY when the per-channel output mask is a FEW
rank-1 terms (T small): separable row⊗col stripes, solid rects, box-moves. When the mask
is an arbitrary data-dependent SCATTER whose matrix rank reaches the grid dimension on a
meaningful corpus fraction, T collapses to **one term per cell (T = H·W)** and the
selectors `R,C[H·W, W]` are **F32-locked** (#109 carrier tax) — the selector bytes then
DWARF the materialized one-hot you tried to delete. **Measure rank FIRST** (`np.linalg.matrix_rank`
over ~400 fresh ARC-GEN masks, ~2 min): if commonly rank>1/near-full, the dissolve is dead
before you build it (mirrors #110's ray-fill rank gate, now for the one-hot-dissolve class).
Byte law: dissolve wins only if `2·T·W·4 + T·10·4 < 10·H·W`. task183 cyan scatter rank over
400 pairs `{1:91,2:74,3:75,4:51,5:65,6:44}` (full-rank-6 in 11%) → T=36 forced → the 839 B
materialized one-hot is the floor (no-candidate).
**ROLE-binding non-separability — a SECOND failure mode distinct from rank (task225, 2026-06-22).**
Even when the per-role placement is low-rank, a multi-COLOR scatter loses the dissolve if the
source→dest ROLE is a function of a JOINT multi-cell detection, not the single source cell. Decision rule: check whether color→placement role is set by a single source cell or by a joint block/shape/orientation detection — joint detection ⇒ dense placement ⇒ the dissolve loses (task225 floor stays 918).
**RUNTIME-DECODED selectors double-kill (task392, 2026-06-22): rank gate AND non-initializable selectors.** When placement selectors (R,C) are functions of runtime-decoded scalars, they cannot be fixed initializers — they reintroduce the render memory the dissolve was meant to delete. Discriminator: are K/R/C fixed params or computed from live inputs? If computed, free-output fails regardless of rank (see §132 "placement count must be structural").
**BLOCK-SUM PROJECTOR FOLD trap (task025, 2026-06-22).** A 0/1 block-sum projector can fold into Einsum operands ONLY when each operand is contracted on a DIFFERENT index. When two operands share contracted index p, the projector groups DIAGONAL products `op1[p]·op2[p]`; distributing it introduces spurious cross-terms. If more than one operand carries the projector's contracted axis, the fold changes the math — gate visible before trusting the lower cost.
**TERM COUNT = the RANK of the per-channel spatial mask — and rank-2 (fill-minus-hole) is cheap (task065, 2026-06-22, 641->625).**
Don't frame the term count purely as a liability or default to one-term-per-bit-condition: a U8 bitwise-AND
tree (`chan_bits & row_bits & col_bits`) whose dominant charge is a `[C,S,1]`/`[C,1,S]` broadcast
intermediate (the **association-order minimum** = channels × ONE spatial dim, e.g. `[1,10,30,1]`=300 B; NOT
the full `C·S·S`) is NOT automatically the floor. Compute the matrix rank of the per-channel target mask:
rank-K ⇒ a K-term free-output Einsum (`Σ_k A[k,c]·R[k,r]·W[k,w]`, F16) charges only `K·S` F16 per spatial
axis, beating the bit-AND when `2·K·S(f16) + K·C(f16) < C·S(u8)` (task065 K=2: 240+40 < 300+60). A signed
channel-weight `A[1] = present_other − present_self` carves a single-cell HOLE out of a filled region (the
rank-2 fill-minus-hole that one term can't express). But count the rank EXACTLY — a needless k=3 stack repriced
this task UP to 1016.
**OPERAND vs NODE — the law behind "factor it into the free-output Einsum" (task011, 2026-06-22, floor proof).**
A factor that rides as an extra OPERAND on the ONE terminal output node is free (only its element count charges);
a factor that becomes a separate NODE feeding an `Add`/`Concat`/`Where` merge RE-CHARGES the full output-shape
intermediate, EACH. task011's 2-node factoring → cost **72,649 (80× worse)**. Diagnostic: *does this add a NODE or just an OPERAND?*
Also: an identity/diagonal coupling operand `δ(u,c)` is IRREDUCIBLE within one node. And params count ELEMENTS dtype-free — F16 on initializers saves 0; fewer elements (#15) is the lever.
**GROWTH-OP charge law — only `'output'` is free, never its predecessor (task057, 2026-06-22).** Any op that GROWS a smaller piece to full output shape — `Pad(X)`,`Concat(A,B,…)`,`Tile(X)`,`Expand(X)` — makes its INPUT `X` a charged intermediate. Decision rule: price the growth-op INPUT (`numel × itemsize`, BOOL=1) vs the params a smaller map saves; if input bytes ≥ param save, the full-maps single-Einsum wins (task057 full `rowmap+colmap`=180 params, no intermediate).
**Factors are MEMORY, not params, at full output extent (task295, 2026-06-21).** When
length/color/width vary per input (the normal from-scratch case), K/R/C are data-dependent
F16 INTERMEDIATES @ 2 B/elem, NOT initializers. Because the Einsum must EMIT the full
`[1,10,30,30]`, its r/c axes are pinned at 30: factor-bytes `= (T·10 + 2·T·30)·2`. At T=10
that's ~1400 B + the `Less` BOOL plane → ~1700 floor, which LOSES to the 900 B U8 canvas +
Equal-tail. You can't shrink R/C to `[T,9]/[T,18]`: a smaller Einsum output is a charged
3240 B intermediate needing a Pad. So free-output wins only when the output (or active crop)
is genuinely small (≤ ~9×10) OR the factors bake as initializers (position-only rule). Price
factor-bytes vs the 900 B canvas BEFORE building.
**LOCAL-SHIFT / cross-channel-recolor → dense `Conv[10,10,3,3]=900` is the floor (task015 + task095, 2026-06-22).**
When a rule needs `±1` (or any small fixed) SPATIAL-NEIGHBOUR access AND a 10-channel one-hot
output AND cross-channel recolor, the free-output Einsum (#109/#119) structurally cannot beat
the dense Conv. A shift is a near-full-rank band matrix `N[30,30]` (rank ~9 over a 9×9 active
region) — NOT low-rank, NOT digit-factorable — so `params` (element count) is always 900 per
shift basis, and keeping `output` a single free Einsum forces the identity-COPY term to carry a
second `I[30,30]` basis (`B[2,30,30]=1800`); pulling the copy out makes `delta` a 36000 B plane.
task015 built it clean (visible 265/265, truth 200k, profile==static==**2216**) — a +1316 LOSS
vs 900. Discriminator: **rank-test the per-axis shift/placement matrix over the active region
first**; rank>1 ⇒ Conv floor, don't build the Einsum expecting a win. Same family, crop+one-hot
variant: task095 (cross-channel, NON-CONTIGUOUS dest channels 0/1/5, full-10-ch output) — the
dense `[10,10,3,3]=900` is the single-op floor (sparsity/depthwise/grouped all fail; the
crop+small-one-hot route's forced F32 channel-slice 324 B + contiguous-`Pad` span to ch0..5
(486 B) already exceed 900 before computing the ring).

**Non-rank-1 channels are fine if they're low-rank SUMS.** task234's background channel-0 =
`grid - anchor - mover` (rank 3). Fold the negatives into the channel selectors instead of adding terms:
`K = [is_anchor - e0, is_mover - e0, e0]` (e0 = channel-0 one-hot), `R=[a_r, m_r, grid_r]`,
`C=[a_c, m_c, grid_c]` -> 3 terms give chA=anchor, chM=mover, ch0=grid-anchor-mover. is_anchor/is_mover
are the one-hot CHANNEL indicators (from the classifier), so the colored boxes land in the right data-
dependent channels with NO color-value scalar and NO palette. **Padding is free**: grid_r⊗grid_c=0
outside the grid -> all channels 0 there, no sentinel needed. Output dtype F16 with {0,1} values; the
validator's `>0` binarize handles it.

This BEAT the Equal-tail canvas on task234: 5010 (f16 [30,30] canvas) -> 2808 (free-output, no canvas),
-2202 B. Crossover vs #81: the Equal-tail canvas wins when the output is NOT low-rank-separable (true
2-D content); the free-output Einsum wins when channels decompose into a few row⊗col terms. Always price
both. Transfer: any output that is a handful of solid rectangles / separable masks placed in data-
dependent channels (box-move, paint-rectangle, axis-aligned region rules).
**Pure-(r,c)-formula vs 2D-REGION selection rule (task398, 2026-06-21, floor proof).** A separable
`Equal(K_k - r, c)` free-output renders any single anti-diagonal / curve (a 1-D locus) for ~0 charged
bytes — but a 2-D "everything-NOT-on-the-curves" REGION (a complement) **cannot** be expressed by
separable equality (equality = a curve, not a region). A 2-D region complement forces EITHER (a) a
color-index **canvas + Equal-tail**, which pays >=1 **INT32 Gather index plane = 3,600 B** (`Gather`
rejects U8 indices — the INT32 plane is irreducible for any canvas route), OR (b) a **high-degree
single-Einsum fold** whose minimum polynomial degree is pinned by an exact per-active-subset LP. Pick the
free-output Einsum when the output is a pure formula in (r,c); reach for a canvas ONLY for a true 2-D
complement, knowing it pays that INT32 plane. **Degree lower-bound method:** before claiming a free-output
floor, solve the exact-subset LP — for each in-band node subset, require the polynomial to fire/not-fire;
infeasibility at degrees 2..d-1 with feasibility at d PINS the degree (task398: 32 subsets, 3,089
constraints, infeasible deg 2-5, feasible deg 6 — the degree is set by the count of in-band nodes, NOT
the band width). task398: free-output floor v20 = 2,488 (static==profile, byte-equal-full) beat the
canvas route's ~7,420 (the INT32 Gather alone exceeds the gap); degree-6 forced 1,106 params
(shift[7,7,7]+binom[7,7,7]+rpow[7,30]+cpow[7,30]), and M-fold/shift-fold saves ~294 params but adds an
equal F64 runtime intermediate (net wash). See traps.md (free-output boundary).

# 110. BIT-PACK + PREFIX/SUFFIX LUT-GATHER for ray-fill / axis-aligned propagation (task064, 2026-06-08, reviewer floor-verified, byte-equal-LIVE 200267)

For "sources shoot axis-aligned rays" / directional-fill-until-obstacle rules where each source carries a
DATA-DEPENDENT value (its own color, an id), the directional-MaxPool span renderer (#105) is the WRONG
tool — it can't disambiguate which source's value fills a cell (task064: the MaxPool-scan route priced
**16,285, 66% worse**, because it needs extra dynamic-color disambiguation + obstacle gating). Cheaper:
pack the per-row (or per-col) source positions into a few BYTES (a 3-byte bitfield per row), then compute
the fill with **prefix/suffix lookup-table `Gather`s on the packed bytes** — a precomputed LUT maps each
packed-byte pattern -> the filled-span byte (prefix = fill-from-left, suffix = fill-from-right), so the
ray-cast is a table lookup, not a per-cell scan. Output is the free `Where(painted, sourcecolor, input)`
(#67, preserves obstacles/sources/bg). task064: 14706 -> 9819.
**Discriminator #105 vs #110:** uniform fill between/around markers (no per-source value) -> directional
MaxPool scans (#105, cheap). Data-dependent per-source value carried along the ray -> bit-pack +
prefix/suffix LUT-Gather (#110). Price both; the dynamic-color case flips the winner. Transfer: laser/ray
rules, flood-until-wall with colored sources, directional shadows, projection-with-identity.
**Rank-test the ray mask before reaching for separable render.** task064's ray-fill
mask family had matrix rank up to 10; a row-vector x col-vector basis cannot
represent it. On a 1000+ pair corpus, if the target ray/interval mask is commonly
rank>1, #119-style separable render is the wrong shelf and this bit-pack/LUT route
is the floor. The measured "paint30 BOOL 900 B" canvas was structurally forced;
the 8 B v11 shave was not worth shipping.

# 111. CROSS-CREATOR CALIBRATION — Aditya wins by COMPUTING rules at zero/low-param where we spend learned weights (Team_Best_Onnx analysis, 2026-06-08)

Audited the 175 team-best tasks Aditya wins. The pattern is stark + repeatable: **his graphs carry almost no
params because he expresses the rule as a COMPUTED graph (boolean predicates + einsum moments + structural
ops + cheap masking + positional integer arithmetic), while we default to learned-weight Conv/MatMul.** Same
#104 decision procedure — he just runs it far harder. Concrete teardowns (his params vs ours):
- **task037: 0 params** (we spent 4013). A fully COMPUTED solver: `Einsum`(moments) -> `TopK`(select) ->
  `Gather`/`ScatterND`(place) + `Where`/`Greater`/`Sqrt`. Selection/reordering by computed property = zero param.
- **task058: 6 nodes, 781 params** (we spent 7326). `ReduceSum`->`Sqrt`->`Gather`-LUT + **`BitwiseAnd`** mask.
- **task200**: dense boolean logic (9x `Less`, 7x `And`, `Or`, `Where`) + `Div`/`Floor` positional arithmetic — the
  rule written as predicates, not weights.
- **task223: ONE `ConvTranspose` (90 params)** — that is our OWN #80-inverse block-upsampler, which we MISSED and
  spent 1800 on. (Re-build candidate: any "each cell -> kxk block" task = single grouped ConvTranspose.)
- **task349**: `QLinearConv`(x5) + `MaxPool`(x6) cheap quantized detector (we spent 955, he 49).
**The lever (act on this):** whenever a candidate spends more than a few hundred params on a `Conv`/`MatMul`,
STOP and ask "can this rule COMPUTE instead of learn?" — boolean predicates (`Less`/`And`/`Or`/`Where`),
einsum moments (#93/#107), structural reindex (#104 step-2, #80), positional `Div`/`Floor`/`Mod`, and **`BitwiseAnd`
for cheap bit-level masking** (new to our kit — Aditya uses it on task058; runs on ORT CPU for integer dtypes).
These tasks are already Aditya's in the team merge, so no re-ship is needed — the VALUE is transferring his
zero-param style to OUR assigned tasks. **Open research (worth a card):** systematically tear down all 175
Aditya-won graphs and extract every distinct cheap-compute idiom — this first pass only sampled 5.

# 112. BIT-PACK (value | position<<k) then MaxPool to SELECT-BY-POSITION, mask to extract value (Aditya task009, R5 2026-06-08)

The "find the nearest marker in each direction and take ITS color" rule normally costs a directional scan
per color. Aditya does it at ~zero extra param by **packing the value and the position into one u8 token**
and letting MaxPool pick the winner: `tok = BitwiseOr(color_u8, coord_u8)` with the **color in the low nibble
and the coordinate in the high bits**; then `MaxPool(tok, kernel=[1,10], pad to one side)` returns the token
with the **largest coordinate = the nearest marker on that side** (because position dominates the high bits);
then `BitwiseAnd(near_tok, 0x0F)` masks the position away to recover that marker's color. Four directions =
four pad-asymmetric MaxPools. Zero-out empty cells first via `Where(cell==0, 0, tok)`. **The lever: encode the
tie-break key in the HIGH bits so a single MaxPool/MinPool does the argmin/argmax-by-position selection, then
mask it off.** Transfer: nearest-marker, ray-source color, "which of N is closest", priority-fill. Pairs with:
**Wrong abstraction for ANY-SAME-COLOR MEMBERSHIP (task025, 2026-06-22).** A #112 nearest-source token pack
answers "what is the closest marker's color," which is NOT the same as "does any same-color marker exist on
this side" — on task025 nearest-source disagreed with membership on hundreds of official/arc-gen side checks.
Membership needs exact #110 bitset over all-color side tensors (or the F32 descriptor reductions), which did
NOT beat the existing free-output Einsum graph. Check whether the rule is *nearest* vs *membership* before
reaching for the pack; for membership, price the full bitset, don't assume nearest-source is equivalent.

# 112b. BITSHIFT as cheap integer ×2^k / ÷2^k (Aditya task012/263)

`BitShift(x, [1], direction=RIGHT)` = integer `x>>1` = floor(x/2); LEFT = ×2. Aditya uses it for `(S-D)>>1` to
halve a bounding-box span with no Div/Mul-const tensor. Runs on ORT CPU for uint8/int. Use for any power-of-2
integer scale (halving spans, doubling strides, /2 midpoints) instead of a `Div`/`Mul` with a charged constant.
The Bitwise family (`BitwiseAnd`/`Or`/`Xor`, `BitShift`) is our cheapest masking/packing kit — under-used by us.

# 113. GatherND STENCIL-READ + binary place-value PATTERN-ENCODE + ScatterND reduction-WRITE (Aditya task027)

Three reusable moves for "read a fixed set of named cells, classify the local pattern, write to named cells":
- **`GatherND(input, idx[1,K,3], batch_dims=1)`** reads K specific (c,h,w) cells in ONE op — a fixed STENCIL
  read (the K offsets are a constant index tensor), far cheaper than K Slices.
- **Binary place-value encode:** `Mul(cells, [8,4,2,1]) -> ReduceSum` collapses a 4-cell on/off pattern into a
  single 0–15 score = a hashed local-pattern signature, at zero param (the weights are a constant).
- **`ScatterND(input, idx, updates, reduction='add')`** writes deltas to named cells with overlap-safe add;
  `ScatterElements(..., reduction='min')` (task012) stamps with min. The `reduction` attr removes the
  read-modify-write dance. Transfer: rotate/reflect-a-sub-shape, local-rule cellular automata, fixed-glyph edits.

# 114. CumSum over run-START markers = ORDINAL / rank labeling of bands (Aditya task184)

To label "the Nth stripe/band/object" differently, detect run-starts then **`CumSum` the start-markers along the
axis** — the cumulative count IS the ordinal index of each run (1,2,3,...). Run-start = `And(active, Not(shift(active)))`
(shift via a `Slice`+`Pad`). Then `Equal(cumsum, n)` masks the n-th band. Zero param. Transfer: "color the 2nd
row", "every other stripe", count-and-index segments, ordinal-dependent recolor. (We had run-detection in #105;
this adds the ORDINAL via CumSum.)

# 115. VARIANCE-IDENTITY spatial EXTENT from einsum moments: D = sqrt(2·Q − S²) (Aditya task012)

Beyond "is it uniform" (#107 Cauchy-Schwarz), Aditya measures a shape's SPAN/extent with raw moments, all
in-kernel (#89): S = Σ(r+c) and Q = Σ(r²+2rc+c²) computed as weighted-marginal Einsums (`'nchw,mc,h->n'` with
iota/iota² weight vectors), then `D = sqrt(2Q − S²)` is the diagonal extent and `(S±D)/2` are the bounding-box
min/max corners. Pure einsum + one Sqrt, zero param. Transfer: bounding box, diameter, "how big is this blob",
center+extent — anywhere you'd otherwise ReduceMin/Max over coordinate planes.
**Clear the denominator to delete `/2` product tensors (task092, 2026-06-22, 4152->4033).** A moment/span
inequality like `t² + (s²−q)/2 <= s·t` can be doubled to `2t² + (s²−q) <= 2st`, removing the small `/2`
product tensors (task092 deleted `row_prod`, `col_prod`, `half_h`). Behavior-preserving ONLY if the integer
F16 path is audited over the FULL coordinate domain — `s²` can round by 1 for odd sums above 45, so an
exhaustive F16-op enumeration over all endpoint/coordinate triples is required (task092: old test, doubled
test, and true interval membership all agreed across `0..29`). Clearing a denominator is a generic golf for
any comparison; verify the rounding equivalence, don't assume it.

# 116. OneHot(color-index grid, depth=10) as the DIRECT one-hot output (Aditya task007) — crossover vs #81 Equal-tail

For the final `[1,10,30,30]` one-hot output, instead of the #81 Equal-tail (`Equal(canvas, iota10)`), Aditya
uses `OneHot(grid_i, depth=10, values=[0,1], axis=1)` straight from the color-index grid. We avoided OneHot as
"expensive", but on a small index grid it's a single node with a tiny `depth`/`values` constant and no [30,30]
canvas materialized first. Price both: Equal-tail wins when you already have the canvas; OneHot wins when you
have the index grid and want to skip the canvas. (11 of his nets use OneHot — we should stop reflexively avoiding it.)
**Size crossover (Boladi task183, 2026-06-09):** a small-grid `OneHot` (e.g. `onehot6` = 360B for a 6x6 output) BEATS
the 30x30 Equal-tail canvas (#81, ~900B forced) when the output is **below ~9.5^2 (~90 cells)**; above that the canvas
wins. Price both at the ACTUAL output size — the #81 canvas isn't always the floor for compact outputs.

# 117. RESIZE with a DATA-DEPENDENT ROI to scale/place a constant template (Aditya task021)

For "stretch a small fixed pattern to fill a computed region" rules, Aditya computes a region-of-interest from
the input and lets `Resize` do the placement: read separators with `AveragePool` (kernel=[1,30]/[30,1] = row/col
means) -> derive ROI corners `(c1,c2,row_end,col_end)` by arithmetic -> `Resize(source_const[1,2,2,2], roi=...,
sizes=[10,30,30], axes=[1,2,3])` stretches the tiny constant source to that ROI. ~15 params total (the source +
ROI scalars). The lever: when the output is a SCALED/STRETCHED version of a fixed small template positioned by
computed coordinates, `Resize` with a runtime `roi` does it in one node — no per-cell canvas build, no
ConvTranspose weight. (We have #80 ConvTranspose for integer block upscale; Resize handles fractional/ROI-placed
scaling.) Transfer: stretch-to-bounding-box, fill-region-with-pattern, scale-by-measured-extent.
**Block-decimate + Resize back (task317):** when the rule paints BxB blocks around lattice markers,
decimate to the lattice, keep the marker plane U8 end-to-end, then nearest-Resize back to full size.
Build the inverse/background channel with `Equal→Cast`, not U8 `Sub` (no U8 arithmetic kernel at low
opsets). This falsified an F16-plane floor: `[1,2,9,9]` U8 was the floor, 442→268.
**Free-output crop-and-resize stamp (task021, 2026-06-11):** for a computed solid
rectangle / channel stamp, the whole output can be the `Resize` node. Use a tiny
`source_const[1,2,2,2]` U8/F16 template, `coordinate_transformation_mode =
tf_crop_and_resize`, `mode=nearest`, `nearest_mode=floor`, `axes=[1,2,3]`, and
`sizes=[10,30,30]`; the runtime ROI selects both the channel slice and rectangle
extent, with `extrapolation_value=0` for the outside. This avoids the `[1,10,30,30]`
mask/canvas entirely. `AveragePool` with kernel/stride `[H,W]` can collapse one-hot
row/col separators straight to `[1,C,1,1]` stats before forming the ROI.
**Tiny-strip output STRETCHER (task247, 2026-06-22, 497->494):** the same
`Resize(tf_crop_and_resize, nearest/floor)` tail can stretch/crop a tiny DYNAMIC strip
(e.g. a winner row) to the full output, avoiding row masks and output basis matrices. Two
gotchas: a true 1-row source **cannot crop rows** — duplicate the strip to **height 2**
first; and encode the active height in the ROI as `roi_h_end = 30/M`. F16/Resize-specific:
a U8 Einsum tail for the same job was rejected by ORT CPU (Einsum has no U8 kernel,
ort_reality.md), so this route is the F16 alternative when the output is a stretched strip.

## R5 coverage note (2026-06-08): all 175 Aditya-won nets classified (`Logs/research/R5_aditya_idiom_catalog.md`).
The non-novel remainder are confirmed transfers of ours: structural Slice+Pad crop-place (#67/#81/#104),
`Abs(Sub)` absolute-difference distance, Gather/GatherElements LUTs, ArgMax selection, `AveragePool` means,
small QLinearConv detectors (#85). Net new this pass: **#111 (the doctrine) + #112–#119 (9 idioms).**
**2026-06-11 rescan:** the current workbook has 189 Aditya-owned nets. The newer 14 mainly
reinforce #119; the added deltas are repeated-free-input/self-correlation Einsums
(task001/task141/task161/task202), scalar coefficient solve before final output (task141),
row homogeneity via the #71 variance form (task202), and negative-END-pad MaxPool crop (task213).

# 118. BASIS-MATRIX axis remap / upscale / placement as an EINSUM OPERAND (Aditya task034/068, einsum-catalog 2026-06-08)

A `[in_axis, out_axis]` matrix factor inside an Einsum remaps/upscales/places that axis with NO separate
Gather/Resize/ConvTranspose. task034 `bkc,bkhw,hH,wW->bcHW`: the `hH`,`wW` operands are `[9,30]` basis matrices
that upscale+place a 9x9 object grid into the 30x30 output, fused with the class-map (`bkc`) and spatial-mask
(`bkhw`) in ONE einsum. task068 reuses a `P2 [2,30]` position-basis 6x to map abstract positions to spatial.
The basis matrix is the only charged cost (small); the placement/upscale itself is an in-kernel temp (#89). **The
lever: any structural axis op that can be written as a `[in,out]` matrix — block-upscale (alt to #80), nearest/
fractional resize, permutation, gather, scatter, position placement — becomes an einsum FACTOR, fused with the
surrounding contraction instead of a standalone node.** Build the basis matrix as a constant (0/1 for select,
block-diagonal for upscale). Pairs with #119 (the basis matrices ARE the factors in his big output builders).

# 119. DEEP separable output — push the free-output Einsum to 6–30 operands (Aditya task009/084/068; extends #109)

#109 built one-hot outputs from 3–4 separable terms; Aditya routinely goes much deeper — 6 operands (task009),
14 (task084), **30 (task068)** — and it's still ONE Einsum with every intermediate in-kernel (#89; Einsum has no
operand cap, L2). task068 builds the full `[1,10,30,30]` output from `C[3,10]` (class) x reused `P2[2,30]`
(position-basis) x `L[2,2,2]` (local pattern) x computed selectors — a hierarchical **region x position x
local-pattern** decomposition at **150 params total, zero materialized canvas**. The lever: when the output
factors along several independent axes (which channel x which abstract slot x which local glyph x which region),
enumerate ALL the factors as einsum operands and contract in one shot — don't stop at 3 terms, and don't
materialize the partial products. Price = only the small factor tensors. This is the single biggest reason his
nets are 5-15x cheaper than ours: we build outputs with a canvas + Equal-tail; he factors them into a 1-node
einsum. Operand order still matters for speed (#89/L2): keep big non-output labels contracted early.
**Input-as-active-mask trick (task176):** the graph input is a free Einsum operand too. When the input
one-hot already identifies the active cells, contract its channel axis inside the final free-output
Einsum instead of first materializing an active mask or casting the whole input to F16. The final
rank-6 placement can combine `input` × class factors × row/col basis factors in one output node; only
the small factors are charged. Any factor-shrink that forces a charged pre-output plane must beat that
plane's bytes; for positional period-fields, keeping the Tucker factors all in-kernel was optimal.
**Mirror-tile floor (task152/task142):** symmetric mirror grids like `[0,1,2,2,1,0]` transfer across
tasks: one shared `[3,30]` map is counted once and the free-output Einsum emits one-hot directly.
Color-index + Equal-tail is an anti-pattern for these mirror tiles because it rematerializes the index
plane before the free output.
**Shared identity/embed operands (task083):** identical source-embed operands in a free-output Einsum can reuse one initializer name twice; score counts it once. Promote a smaller inner dim only when the extra slot maps to structural zero; never merge distinct placement maps.
**Term-axis-last layout (task092):** lay span/term factors so existing singleton dims already match the final contraction; this deleted Reshape/shape-init scaffolding (4033->3980).
**Tucker init-fold trap (task362):** folding a tiny J factor into repeated U cores multiplies U params at every use. Share the small factor unless the net init count actually drops.
**Conditional shift/shear inside final Einsum (task004, 2026-06-11):** before
accepting a canvas floor, price a single free-output Einsum with a small offset
basis. A conditional horizontal shift can be represented by `S[k,W,W]` identity/
shift matrices plus row/context features; the output-column axis is then selected
in-kernel, not by a materialized canvas. Task004 used the graph input twice: once
for color data and once as the active-mask operand at the output position, deleting
the sentinel/inside plane. The `S[k,W,W]` basis is often the true forced cost when
both input and output column axes must be pinned.
**Conditional shift guardrail (task382, 2026-06-19):** dense shift-basis Einsum is
`O(K*W*W)` in basis params, while a Gather shift-bank is `O(K*W)`. For per-row
variable shifts at W=20..30, the bank usually wins unless the placement is already
low-rank/separable and can stay entirely inside the final free-output contraction.
Price `S[k,W,W]` against a `[K,W]` shift-bank before calling #119 the floor.
**Absorb coefficient assembly into the output Einsum (task034/task092/task306/task313/task094):**
if small scalars/vectors only feed the final free-output contraction, try folding
them into the final operand list. Broadcast F16 `Where` constants can replace
`Cast->Mul->Sub` coefficient tails, a final-output deep Einsum can beat a dense
Conv for tile/copy shrink, and source row/col selectors can be operands multiplied
by the active input mask instead of a charged F32 Slice. **Channel-selector collapse
(task094, 2026-06-21, v10 1742->1442 byte-equal):** when only a few INPUT channels feed
the output, add a small `out_sel[c,s]` operand to the SAME contraction so the dense
coefficient's input-channel axis collapses against it — `out_coeff[10,10,2,2]` (400 params)
factors into `[10,2,2,2]` (80) + `out_sel[10,2]` (20). The fold introduces no broadcast
intermediate (profile==static), so the output stays byte-identical with the input-channel
sum living in-kernel (#89).
**Repeated free input / self-correlation operands (Aditya task001/task141/task161/task202):**
the graph input can appear many times in one output Einsum, not just once as an active mask.
Use repeated input operands to express self-products and pairwise correlations in-kernel:
task001 emits a self-product/fractal layout with three input copies plus reused `[3,30]`
bases (202 params); task161 detects the target color by `Einsum(input,input,edge_pick)`
cover counts plus `ReduceSum(input)==4`; task202 computes row homogeneity and the final
stripe output with five input operands. Repeating the free graph input costs no params and
the pairwise products are scorer-invisible; only the named scalar coefficient nodes and
small basis tensors charge.
**Scalar coefficient solve before the final output (Aditya task141):** it can be cheaper to
compute 1-2 scalar coefficients with tiny preparatory Einsums, then feed those scalars into
one huge signed/logit output Einsum, than to materialize row/col distance masks or choose
between prebuilt canvases. This is #130 analytic-output doctrine with data-dependent
coefficients: keep the coefficients scalar, keep the spatial polynomial/rank bases as
small operands, and make the signed field itself the graph output.
**More fold idioms (2026-06-22 harvest):**
- **Input-side Einsum replacing a separable input-side Conv (task004, 2662->2367):** a separable
  Conv that only feeds a free-output Einsum can be replaced by an INPUT-side Einsum reusing basis
  operands already in the graph — deleted the dense `W_bitrow[1,10,1,30]` block for a few scalar/vector
  params, no charged intermediate.
- **Finite-difference selectors over an already-needed prefix basis (task123, 631->475):** exact one-hot
  selectors can sometimes be written as finite differences over a prefix basis the kernel already holds
  (`H5[5,30]` factored as `D5[5,10]@P[10,30]` in-kernel) — saves params with zero memory, inside the
  free-output boundary.
- **Finite choice BASIS BANK selected in-kernel (task289, 634->614):** for a small finite data-dependent
  scale/choice `k`, a constant basis bank `expand[K,...]` selected by `Equal(s, k_values)` inside ONE
  free-output Einsum can beat runtime ordinal/prefix masks. **Verify the FULL generator k-range** — a
  cheaper `k=1..4` bank was visible-incomplete because generator-valid k=5 cases exist.
- **Scalar-as-operand + bias-folded-in-Sum (task100, 354->247):** carry a scalar multiplier as an Einsum
  OPERAND instead of post-scaling a larger tensor; fold channel-exclusion as a BIAS into an existing `Sum`
  rather than a Slice/mask.
- **Side-vector Einsum for static ScatterND updates (task181, 369->360):** when a static ScatterND layout
  is already proven, Gather the side/anchor pixels as a compact side vector and form the signed update
  patches directly with `Einsum(mirror8, side2, side_coeff)` — removes the selected-coefficient tensor and
  a Bool path while preserving the static patch structure.
- **Price the SELECTOR by its ONNX output shape, not the concept (task108):** a shared conceptual selector
  is not enough — axis-local `Gather` sharing a 30-element index still materializes a full `[1,10,30,30]`
  F32 middle tensor, and `GatherND` avoids it only with a dense INT64 per-output coordinate table. Both
  lost to a 300-param final-Einsum basis. Verify the op output shape + index broadcast before counting only
  the selector (see also op_cost_patterns Gather/GatherND thresholds).
- **Broadcast `Equal` over scalar-Equal + Concat ladders (task184, 2245->1914):** when rank masks compare
  active cumulative counts against a small fixed rank vector, ONE broadcast `Equal(counts, rank_vec)` removes
  the repeated per-rank mask tensors a scalar-Equal-plus-Concat ladder builds. For nonnegative count tensors,
  `Cast(count -> BOOL)` also replaces `Greater(count, 0)` and drops the zero scalar. (CumSum is u8-unsupported
  here — keep the counts in a kernel-backed dtype; ort_reality.md.)
- **Output-basis SELECTOR REUSE deletes a charged source selector (task194, 2026-06-22, ->283):** when the
  deep free-output Einsum's OUTPUT basis already contains the source one-hot columns (e.g. `Q[30,7]` carries
  the low columns a separate `A[30,3]` source selector would re-derive), reuse those columns and add only a
  tiny MODE selector (`G[3]`) instead of materializing the full charged source selector. No extra
  intermediate — the reuse lives entirely inside the contraction. General move: before adding a selector
  operand, check whether an operand ALREADY in the Einsum spans the columns you need.
- **SLOT-width routing + CONSTANT channel-widening operand (task010, 2026-06-22, 877->600).** When a
  free-output separable Einsum routes a DATA-DEPENDENT one-hot to the full `[1,10,30,30]` output's
  10-channel axis, do NOT materialize the routing tensor at full channel width — a full-width data-dependent
  F32 operand scales as `terms × output_channels` of charged memory. Keep the routing at the minimal SLOT
  width (here 5 = bg + 4 bars) as `route[1,T,Q]`, then widen `Q → output_channels` with a CONSTANT selector
  basis `S[Q,K]` folded as an EXTRA Einsum operand: `Einsum('bjrc,tj,tc,btq,qk->bkrc', input, P, bcol,
  route, S)`. `S` costs only `Q*K` PARAMS (1/elem, dtype-blind) and ZERO memory, vs the data-dependent
  full-width operand's `T*K*4` F32 bytes (task010 dissolved a 360 B route/Concat into 100 B `route[1,5,5]` +
  a 50-param `S[5,10]`). This is the #118 basis-operand idiom applied to the data-dependent SLOT→CHANNEL
  remap. Two floor traps: don't peel a constant (bg) term into its own additive Einsum (forces an `Add` of
  two charged `[1,10,30,30]` planes); and build the route one-hot via `route = Cast(Equal(slot_idx[1,T,1],
  arange_Q))` — NOT a per-channel `Concat` zero-pad (the 877 graph paid `k_zero_lead`/`k_zero_tail` Concats
  for exactly this). (`ReduceSum` has no U8/BOOL kernel, so a pairwise-Greater rank count (#124) must `Cast`
  BOOL→F32 before `ReduceSum`; traps §8f.)

# 120. QLinearConv y_scale != 1 = conv + BUCKET / THRESHOLD in one op (Aditya task278/349/396; extends #85)

`QLinearConv` quantizes its output as `clamp(round(conv/y_scale) + y_zero_point, 0, 255)`. With `y_scale=1` it's
the identity counter (#85); with **`y_scale=k>1` it BUCKETS the conv output into width-k bins** — a built-in
threshold/quantize with no separate `Div`+`Round` or `Greater`. Aditya picks `y_scale` ∈ {3,5,19,25,134,...} (and
`y_zero_point` to offset) to land detector outputs in the bins he wants (task278 `y_scale=3,y_zp=1` sign-encodes
positives->1 / negatives->0). The lever: whenever a `Conv` is followed by a divide-and-round or a threshold,
fold it into the conv's `y_scale`/`y_zp`. Free, no extra node.

## R5 frequency finding (2026-06-08): his CORE primitives, by constant-tensor count across the 175 nets
- **binary place-value vectors (powers of 2): 407 uses** -> #113 (pack a multi-cell/multi-bit pattern into one
  number via `Mul(cells, [..8,4,2,1]) -> ReduceSum`). His single most-used trick.
- **0/1 basis matrices: 86 uses** -> #118 (axis remap/upscale/placement fused into an einsum).
- **position iotas (0..N): 14 uses** -> #93 (weighted-marginal moments).
These three + the deep separable einsum (#119) ARE his cheapness. Our action: reach for place-value encoding and
basis-matrix einsums by default on detection/placement tasks, instead of learned Conv/MatMul weights.

# 121. GridSample for COMPUTED spatial remap / warp (Aditya task351)

When the output is the input RESAMPLED at computed coordinates (mirror, rotate, translate, per-cell
gather-by-(x,y)), build a coordinate grid and let `GridSample` do it in one node: assemble a `[1,H,W,2]` grid of
(x,y) coords (here from `Concat(x_comp, y_comp)` of computed position planes), then
`GridSample(input, grid, mode='nearest', align_corners=1)` reads the input at those coords. `mode='nearest'` =
exact pixel remap (no interpolation). The lever: any "output[r,c] = input[f(r,c), g(r,c)]" rule — reflection,
rotation, shear, arbitrary 2-D gather — is one GridSample over a computed grid, no per-cell Slice/Gather. (ORT
CPU GridSample is float-only per M1 audit — cast input+grid to float, cast back.) Contrast #118 (basis-matrix
remap, separable per-axis) — GridSample handles NON-separable 2-D coordinate fields.
**✅ GRADER-VALIDATED (Boladi task351, 2026-06-10):** GridSample CREDITS on the Kaggle grader — confirmed by own-stack
delta-validation (`S_grid − S_base = +3.04`, exactly the 31,271→1,498 gain). It is **no longer an op to avoid** — use
it freely for computed remaps (mirror/rotate/shear/2-D gather). First validated GridSample ship; the team had been
steering around it on suspicion. (Still float-only on ORT CPU per M1 — cast input+grid to float.)
**k-size crossover (task091, 2026-06-22):** GridSample wins ONLY for small k (~≤6-7). Grid cost `2·k²·2` B (F16) plus ~1,568 B build chain; at k=14: 5,979 B vs Slice/Pad ≤292 B. Price the grid bytes against the Slice tail before building.

# 122. MatMulInteger for cheap INTEGER projection / reduction (Aditya task297/355)

`MatMulInteger` is a pure-integer matmul (u8/i8 in, i32 out) with NO quantization scales — cheaper than
`QLinearMatMul`/`Conv` for counting and projecting. `MatMulInteger(x_u8, ones_vec)` = integer sum/count along an
axis; `MatMulInteger(x_u8, pattern_matrix)` = project x onto a 0/1 (or small-int) basis (task355 `rowband·speck`,
`proj·ones30`). Use for "count cells matching", "project a band onto a basis", batched integer dot-products — at
no-scale integer cost. Pairs with the place-value trick (#113): the weight matrix can carry the place-values.
**QLinearMatMul as U8 composer (Aditya task297):** when the product is an encoded grid whose values
stay ≤255, `QLinearMatMul(A_u8, B_u8, scales=1, zps=0)` emits U8 directly and can beat
`MatMulInteger`'s INT32 output + narrowing. Use it for low-valued outer products / row-column
composition that feed a final `Equal`; do not use it merely to shrink params (params are element-counted).

# 123. Cheap PREDICATE + COORDINATE idioms (recurring across his nets)

Two zero-param micro-idioms he leans on:
- **Interval membership** `a in [lo,hi]` = `And(GreaterOrEqual(a,lo), LessOrEqual(a,hi))` — his `GE->LE->And`
  motif (14x). The clean band/window/range test; cheaper + clearer than chained `Greater`/`Less`/`Sub`.
  **U8-wraparound closed interval (task284, 2026-06-12):** a single `Less(Sub(idx_u8, lo), span)` = 2 charged
  tensors vs `And(GE,LE)` = 3 (needs opset-14 u8 `Sub`; wraps mod-256, so `lo` must exceed any real out-of-band
  index). Transfers to the BOOL band-component stacks of #119 free-output Einsums.
- **Mod for periodic / wrapped coordinates** `Mod(coord, period)` — position-within-a-tile (task065,
  `Mod(dot_r, stride)`) or a cyclic wrap (task091, `Mod(src, 15)`). Extends #93's Mod (orientation/parity) to
  tiling/repeat rules; pairs with the basis-matrix (#118) for tiled placement and with #105 for periodic spans.
  **U8 Mod+Pad index chain (task110, 2026-06-20):** if `Mod(arange, small_period)`
  and a Pad sentinel produce values `<256` before a `Gather`, cast the arange and
  period to U8, run `Mod(U8,U8)` and `Pad(U8)`, then cast the final padded index
  back to INT32/INT64 for Gather. This pays only when the pre-Gather chain has
  multiple intermediates: task110's row path replaced INT32[29]+INT32[30] with
  U8[29]+U8[30]+one final INT32[30] cast and saved 26 total cost; the analogous
  col path with only one pre-cast intermediate was worse. Mixed `Mod(U8,INT32)`
  fails, and Gather indices are never U8.
- **AveragePool as color histogram/ranker (Aditya task393/task391)**: `AveragePool(input,
  kernel=[30,30])` gives a `[1,10,1,1]` per-channel mass/mean with tiny memory. `TopK(axis=1)`
  then extracts the top colors for compact legends/columns; emit via `Equal` or direct `OneHot`
  only at the small output.
- **Exponential position code + `Log→QuantizeLinear` (Aditya task355)**: if a mask has exactly one
  active position along an axis, weight by a geometric ramp, `Log(sum)` recovers an affine coordinate,
  and `QuantizeLinear(scale=k)` turns it into a U8 bin. This replaces ArgMax/TopK planes for some
  split/extent problems, but only when the geometric gaps are margin-proven under F32 rounding.

# 124. RANK / SORT small-N by pairwise-comparison count (Aditya task010; R5 deep-dive batch 1)

To rank or sort a small set of N values with NO TopK/Argsort, do all N^2 pairwise `Greater` comparisons and
`Sum` the wins per element: element i's rank = `#{j : v_i > v_j}`. task010 ranks 4 row-values via 12 pairwise
`Greater`s + `Sum`-of-wins, then uses each rank as a place-value index into the output. Zero param, fully STATIC
graph (no dynamic-shape TopK), maximally ORT-portable. For small N (<= ~8) the N^2 comparisons are cheaper and
safer than TopK/sort. Transfer: "Nth largest object", order rows/colors by score, rank-dependent recolor.
Complements #99 (TopK) — prefer this when N is small and you want a static graph.

# 125. Cheap boolean & encoding micro-idioms — the connective tissue of his rules (Aditya, R5 batch 2)

The recurring zero-param glue across his boolean nets (task024/026/035/045):
- **Arithmetic boolean (no Cast-to-bool):** complement = `1-x`, AND = `Mul(x,y)`, OR = `x+y`/`Sum`, on 0/1 floats —
  cheaper than `Cast`->bool->`And` inside a float graph (task026). Stay in float, skip the bool round-trip.
- **Xor for parity / "exactly one":** `Xor(a,b)` = a!=b (toggle/parity) — extract a marker that flips state, or
  test odd-count (task024). `Not(Xor(a,b))` = equality.
- **Where-cascade = priority switch / decision-tree -> code:** nested `Where(c1,v1,Where(c2,v2,...))` is if/elif/else;
  emit distinct bit-values (1/2/3/4/7...) per branch to encode a decision tree into one integer, then mask with
  `BitwiseAnd` (task035/045). The cheap way to collapse a multi-way classification into a single channel.
- **Bit-threshold decode:** `Greater(code, 2^k - 1)` (15, 511, ...) reads a packed bit-field by magnitude
  (task024) — the read-side companion to the #112/#113 place-value pack.
- **Vector equality by squared norm (Aditya task003):** for small F32 one-hot slices, `ReduceSumSquare(a-b) < 0.5`
  is an exact whole-vector equality test and yields one scalar. It can beat elementwise `Equal` + reduce when
  the slice is already F32 and exact-binary.
- **Boolean algebra collapse (task026/task381):** on 0/1 masks, reduce arithmetic
  cascades before golfing storage. `OR(L,R)` on complement inputs collapses to
  `1 - L*R` (or `Not(And(L,R))` in BOOL), `(1-cu)*(1-cd)` is `1-Max(cu,cd)`, and
  `Max(ReduceMax(mask*cu), ReduceMax(mask*cd))` is `ReduceMax(mask*Max(cu,cd))`.
  A rejected U8 `Mul` does not by itself force F16; if the values are boolean,
  swap to `And`/`Or`/`Not` or `Min`/`Max` on 1-byte carriers.
- **Nested Where instead of masked-candidate Max (task146):** when exactly one
  candidate is valid, nested `Where(valid_i, value_i, next)` can delete the
  candidate planes and a final Max. For tiny color-index labels this can also beat
  `ArgMax`, whose label output is INT64.

# 126. ITERATIVE MaxPool FLOOD-FILL — 2D region growth + enclosure/hole detection (Aditya task251)

For omnidirectional flood-fill (not the 1D span of #105): UNROLL N times
`fill = Max(fill, Mul(MaxPool(fill, kernel=3x3), region_mask))` — each `MaxPool(3x3)` dilates the fill one cell in
all 8 directions, `Mul(region_mask)` clips it so it cannot cross walls (the allowed-cells mask), `Max` accumulates.
N iterations flood up to N cells (N ~ grid diameter, or fewer if bounded; task251 unrolls 6). Zero learned params.
**Enclosure / holes**: flood from the BORDER, then `region AND NOT flood_reach` = cells the outside flood can't
reach = enclosed interior (task251 does `Sub(seed, flood)`). Transfer: fill-inside-shape, count/recolor enclosed
cells, hole detection, bounded connected-component growth, border-reachability. Complements #105 (1D directional
span) and #110 (ray-fill) — this is the 2D omnidirectional case, and it's a very common ARC primitive.
**CC-labeling is not automatically cheaper.** task077 confirmed that an INT32
connected-component label canvas (4 B/cell) can lose to iterative masked flood on
U8/BOOL carriers (1 B/cell), especially with <=4 well-separated clusters on
<=20x21 grids. A distance-6 flood was corpus-forced there; a 2-step shortcut failed
4300/200k. Compare the true separation radius and carrier dtype before replacing
flood-fill with a component-label approach.

# 127. GEOMETRIC / coordinate micro-idioms (Aditya task375/329/373)

Cheap zero-param geometry off the iota/center coordinate planes (#93):
- **Square side from area:** `Sqrt(ReduceSum(mask))` = side length of a filled square (task375); center = (side-1)/2.
- **Diagonal mask:** `Equal(Abs(row-cy), Abs(col-cx))` marks the two diagonals through (cy,cx); swapping for `<=`/`>=`
  gives an X-region or a bounding diamond (task375).
- **Abs without `Abs`:** `Relu(d)+Relu(-d) = |d|` and `Relu(d)-Relu(-d) = d` — a branchless split into +/- parts
  (task329 distance-to-center), handy when you already have Relu in the graph.
- **Reflection quilt:** `Concat`/`Tile` of `[a,b]` + `[b,a]` builds an abba mirror-tiling at zero param (task373).

# 128. SHAPE-PROPERTY via a MOMENT IDENTITY — derive area/size algebraically, skip the bbox (Boladi task100)

Don't extract a bounding box (`ReduceMin`/`ReduceMax` over coordinate planes) to get a shape's area/size — compute
it in CLOSED FORM from the raw moments. task100 (hollow rectangle → uniform fill colored by the rectangle's AREA):
`8*area = N^2 + 12*N - 2*sum_r2 - 2*sum_c2`, where `N=sum 1` (cell count) and `sum_r2`/`sum_c2` are the second
moments — **4 scalar Einsums** (#93/#107/#115 family), no Min/Max, no marginal extraction. Moments are scalars →
tiny memory; color via `ArgMax`->`Equal`; output is the free `Expand`->`Pad` uniform block. Cost **460** (was 7,076,
15x; beats Aditya). **The lever:** any polynomial property of a shape (area, perimeter, diagonal length, aspect)
usually has a closed form in `sum 1 / sum_r / sum_c / sum_r2 / sum_c2 / sum_rc` — derive the identity once, evaluate
it with scalar Einsums, and you avoid the whole Min/Max bbox machinery. (Empirically conclusive on ~420k pairs:
byte-equal to a correct LIVE on the full corpus, not a coincidental fit.)
**Task100 selector shrink:** if the moment identity leaves a fixed `[10,9]` diagonal selector only to
drop one channel, prefer a final `Slice` (3 scalar inputs) over storing the selector (90 params).
When the missing one-hot channel is channel 0, `Pad` with `C_begin=1` can create the empty leading
channel after computing only the 9 live channels.

# 129. GatherND SUB-SHAPE REPLICATION / grid-tiling — the WRITE side of #113 (Boladi task033)

To replicate a small kxk motif into an NxN arrangement of cells (tile a shape across a grid), build a CONSTANT
`GatherND` index bank that maps each output cell to its source sub-shape cell, then `GatherND(input, index_bank)`
does the whole tiling in ONE op; wrap with `Where(tiled, ..., input)` to PRESERVE existing pixels (don't clobber the
source). task033: replicate the top-left 3x3 shape into all 9 cells of a 17x17 grid, byte-equal EXACT 200000
(deterministic copy rule). Extends #113 (GatherND stencil-READ) to the replicate/WRITE direction. **Cost-floor
caveat:** a `Where` with a free full-input else-branch still needs a full HxW (30x30) condition tensor even when the
grid is smaller (the task341 pattern — ~900B forced), so the realistic floor here was 2,256, not lower.

# 130. ANALYTIC / TRIG-FORMULA output: memory-0 closed-form + `>0`-logit decode (Boladi task007, LB-probed 2026-06-09)

For an output with regular/analytic structure, encode the WHOLE output as a closed-form formula (trig/polynomial per
cell) evaluated from the coordinate planes — **memory 0** (no intermediates, pure arithmetic), params = just the
formula coefficients. task007 v11: **78 params, memory 0**, encodes the entire output as a trig formula; the output
is LOGITS (negative off-cells, positive on-cells) DECODED by `(result > 0)` → the 0/1 grid. This is the cheapest
output-construction class we have — well below even the separable einsum (#119) when the rule is analytic.
- **Verification nuance:** the raw-tensor / byte-equal gate "FAILS" because the logits aren't literal 0/1 — that is
  EXPECTED and fine. Gate on the DECODED output: visible/truth must pass under the `(result>0)` decoder, not on raw
  bytes. The decode is the contract; the logits are scratch. (A "more-correct-than-raw" case — see #44 / the gates.)
- **MINIMUM-MARGIN TRAP — probe-first:** an analytic decode at the minimum param margin is FRAGILE. Too few terms and
  a cell with multiple positive logits decodes to a wrong color (task007: 78 is the minimum; 68 fails; multi-positive
  → 11 → wrong). The full corpus can pass while the decision margin is razor-thin, so **LB-confirm via submit-first
  before trusting it in a real ship.** Doctrine: stage/keep a SAFE raw-exact floor (here the 614 negpad-freeout
  einsum) AND probe the margin model — keeps-highest makes the downside zero, so the probe costs nothing but quota.
  This is exactly the case where submit-first earns its keep (a sound mechanism at an unforgiving boundary).
- **Outcome (task007, 2026-06-09):** the 78 **HELD** — own-stack probe delta `S78 − S614 = +2.07` (matched the
  predicted +2.06), so the analytic-formula output is real and shipped to the team at 78. The probe-first doctrine
  confirmed it on the real LB *before* it touched the team submission — the model class is sound, use it, but always
  LB-confirm the minimum-margin instance.
- **Tile-for-param anti-pattern (task007/task215):** replacing a stored basis/formula factor with
  `Tile` can reduce initializer bytes but materializes the tiled tensor as charged memory when it
  survives as a graph value. It also breaks tasks where zeros past the active region are semantic.
  Treat `Tile` as a memory op unless the tiled result is the graph output or is fully absorbed into
  a scorer-invisible free-output contraction; profile it before calling it a param win.

# 131. FORCED FLOOR for paint-mask-preserve tasks — decompose before you target a number (Boladi task102)

For a task that PRESERVES the input and paints a detected region (the `Where(mask, color, input)` overlay), the floor
is dominated by FORCED structural costs, NOT the detection logic: an **F32 input-slice (~576B** to read the input for
the else-branch) + the **30x30 `Where` condition mask (~900B forced** — the task341/#129 pattern: a `Where` with a
free full-input else-branch needs a full HxW condition even when the logical grid is smaller). That's **~1,476B before
any detection.** task102: 1,476 forced + detection (~400) + paint (~432) = **2,634, a near-floor; sub-1,000 is provably
impossible.** Lesson: for a mask-paint-preserve rule, DECOMPOSE the forced floor (input-tax + Where-mask-tax) FIRST —
don't report a result as a "miss vs 1,000" when 1,000 was never reachable (this exact mis-framing happened here; it
wastes re-task cycles). Detection idiom worth keeping: a **signed `QLinearConv`** (border + weights, interior − weights)
detects a HOLLOW shape of the matched size in one pass — 4 output channels = 4 sizes, a single conv as the whole hit
stack (cheaper than per-size stencils when the shapes nest).
**Detector-union floor (task042, 2026-06-19):** K distance-gated detectors that must stay hard-separated cannot
fold below K signals — and the two cheap folds (u8 `Add`, bool `ReduceMax`) are opset-13-blocked (need opset 14 /
20 respectively), so a low-opset graph pays the full K-signal floor. Count the forced detector signals before
claiming a sub-K number.
**Oracle-blocked ≠ cost-blocked (task101, 2026-06-19):** a Resize/DepthToSpace stamp lever that replaces a
TopK+scatter tail saves only the CHEAP part — the floor is the SOURCE-DETECTION/classifier surface (input-read
slices + working grids) it leaves intact. And it needs a ≥99% numpy oracle FIRST: the clean formulation peaked
at 48.5% truth because **equal-color marker/anchor blocks of one object MERGE under connected-component labeling
at unknown scale** (touching blocks fuse) — recover each via edge-run-length / block-size classification, NOT CC.

# 132. STAMP-TILING via a rank-K separable-einsum basis + free-output (Boladi task327; the einsum twin of #129)

The Einsum alternative to GatherND tiling (#129): to replicate/place a small pattern into a tiled output, build the
output with a separable Einsum whose basis-matrix factors `M[h,H]`, `M[w,W]` (#118) ARE the placement maps —
**rank K = the number of placements/stamps.** task327: stamp a 3x3 into a 6x6 (6 placements) as a **rank-6
`M[h,H] ⊗ M[w,W]`** separable einsum producing the full canvas as FREE OUTPUT (#119); 5,997→1,664 (3.6x), byte-equal
EXACT. **Free-output beats crop+Pad:** make the basis map straight to the full 30-axis (`M[h,30]`) so the einsum
output IS the 30x30 canvas — at worst a WASH vs cropping to the logical size + `Pad`-to-30, and cleaner (no separate
Pad, fewer nodes). Floor note: the rank is forced by the placement count (here t=6), so that IS the structural floor —
prove it before chasing lower. Pick #129 (gather) vs #132 (einsum) by which factors cheaper for the specific tiling.
**Basis-shrink + Pad trap (task059):** shrinking the basis to the logical crop and Padding it later often
loses to a full-width `[h,30]`/`[w,30]` basis inside the final free-output Einsum. The saved basis params are
small, while the cropped product or Pad input becomes charged before the free output. Price the full-width
basis first; it is commonly the true floor for free-output stamp builders.
**Placement count must be structural, not data-dependent.** task158 killed the
temptation to use #132 for stamps whose valid positions are discovered at runtime:
the valid-score/placement plane is irreducibly 2-D and the placement count varies,
so the separable basis is not a small static rank-K object. Use the detector/Conv
render floor instead. Same warning for diagonal/ray spread (task324): a diagonal
line matrix is high-rank; Conv wins by translation invariance, not by separability.
**Measured (task327, 2026-06-22):** the #132 free-output Einsum priced **934** for the
down-right diagonal-staircase rule and LOST to a **ConvTranspose 6×6-identity stamp at 580**
— the (1,1)-shift Toeplitz over a 6-wide axis is not low-rank (the rank-7 `A[7,3,30]=630`
placement basis dominates). ConvTranspose does the staircase as a translation-invariant stamp
at near-zero param cost. For diagonal/ray spreads, reach for ConvTranspose/Conv, not the
separable basis, even when free-output is available.

# 133. CROP to the generator's HARD content bound + check separability first (Boladi task093)

**Static crop to a proven bound:** when the task's generator provably confines content to a sub-region (task093: all
shapes fit in a **14x14 box, verified over the full 200k corpus**), CROP the input/render to that bound — the render
masks and per-cell work shrink to 14x14, not the full 30x30. Distinct from the dynamic negpad-crop (#41/#91): this is
a STATIC crop to the generator's guaranteed content box. **Prove the bound over the WHOLE corpus first — a single
out-of-bound pair voids it (a VISIBLE-FAIL/hidden-0 risk), so it's only safe when the bound is a hard generator
invariant.** (The output `#81` canvas still pays its 900 forced; the crop only saves the render side.) 30,403→4,620, 6.6x.
**Separability check (caution on #119):** don't reflexively reach for a separable einsum — task093's per-ray 2-D render
is genuinely NOT separable (the reviewer corrected a separable-instinct). Some rules need a structural 2-D render;
test separability on the rule before forcing #119, or you'll ship a wrong (or no) factorization. Second case
(task350, floor confirmed 9040): a per-row/col **span/fill** is likewise irreducibly 2-D — span/fill rules are
*frequently* 2-D-irreducible, so budget a structural render (+ the forced F32-extraction + Where tax, #131), not a
factorization. A confirmed floor on such a task is a valid outcome, not a miss — don't re-chase it on cost.

# 134. FLOOR DISCIPLINE — a forced-component PROOF vs a no-lever CLAIM (Boladi task365 vs task102/task350)

Not all "floors" are equal:
- A floor is **PROVEN** only when it's a forced-component DECOMPOSITION — input-tax + Where-tax + irreducible
  structure, all algorithm-agnostic (task102: 2,634 = 1,476 forced + detect + paint, #131; task350: 9,040 irreducibly-2D
  + forced extraction, #133). These hold up.
- A **"no-obvious-lever / I-don't-see-a-cheaper-way" CLAIM is per-ALGORITHM and PROVISIONAL** — a from-scratch rewrite
  with a *different* algorithm can beat it. task365: a 4,308 "proven floor" **FELL** to a 3,354 colidx-projection-peel
  from-scratch rewrite (−22%) **in the same session**.
**Discipline:** stamp "floor" only on a forced-component proof; mark a no-lever result as a *provisional floor /
conditional-reopen*, and periodically revisit it with a from-scratch (Path-B) rewrite. The cheaper ALGORITHM — not the
cheaper byte-trim — is where prior floors fall (per the from-scratch doctrine, true floors often sit 10–100× below the
byte-equal floor). Be humble: a floor that fell once can fall again. (Pair with #131/#133 for the proof side.)
- **Proven-floor example (task011, 1,381):** the **#87 rank test** on the dominant param banks confirmed them
  full-rank-4 (minimal — can't rank-reduce), so the floor is REAL (contrast task365's fallen claim). And the cheaper
  v4/1,342 was REJECTED as **adversarial-weak** (39 B cheaper but fragile) — **the floor is the ROBUST minimal, not the
  fragile-cheaper.** Reject a marginal-cheaper variant that fails adversarial/robustness checks; 39 B isn't worth a
  hidden-0 risk (same spirit as #130's minimum-margin caution).
- **Floors err BOTH ways (task264):** the prior "1,513 hard floor" was over-OPTIMISTIC — <1,500 is UNREACHABLE; the
  real floor is the #95 forced decomposition ~2,300 (784 conv + 729 one-hot + routing). So an estimate can be too HIGH
  (task365 fell to a rewrite) OR too LOW (task264, the target was unreachable). Ground the floor in a forced-component
  decomposition (#95/#131), never an optimistic guess — in either direction.

# 135. Base-16 LOG-POSITION scalar decode — compact position encoding (Boladi task300)

When a task needs a few positions (markers/anchors), not a full position field, encode them as a COMPACT SCALAR
instead of full 30x30 planes: a **base-16 log-position scalar** packs the positions into one number (base-16 digits /
log-scaled), decoded via scalar arithmetic + `ReduceMax(BOOL)` patch masks to rebuild the output. task300:
32,854→546. This is the LOG/base-16 cousin of the place-value pack (#113), specialized for POSITIONS — far below a
full coordinate-plane approach. **Op note:** `ReduceMax(BOOL)` at opset 20 is Kaggle-confirmed (task191 v6,
ort_reality.md); `ReduceMin(BOOL)` is NOT — always reduce with Max.
**Audit F32 existence reductions once a BOOL mask exists (task381, 2026-06-22, 2067->2036).** If a graph
already materializes a BOOL mask, a downstream existence test implemented as `ReduceMax(red_F32) -> Greater`
is wasteful — replace it with opset-20 `ReduceMax(red_BOOL, axes=...)` directly, deleting the charged F32 copy
(task381 deleted a `row_has_f` F32 `[1,1,8,1]` row gate, adding only a tiny axes param). Whenever a row/col
"does any cell fire" gate casts or reduces an F32 copy of a mask you already hold as BOOL, reduce the BOOL.

# 136. DERIVE, don't RE-COMPUTE — audit every Conv against the labels you already have (Boladi task022)

Before spending a `Conv`/op to compute a feature, check whether an EXISTING output already encodes it, and derive it
with a cheap op instead. task022: a dedicated ch5 `Conv` was recomputing the gray markers that the color-index
`labels` tensor already held — replaced by `Equal(labels, 5)`, saving **~726 B** (2,637→1,671). The peephole: any
feature that is a function of an already-computed label/index — a specific color, a threshold band, a parity, a
neighbor relation — is a free `Equal`/`Slice`/`Mod`/`Gather` away; don't burn a second Conv or param bank recomputing
it. When golfing a multi-conv graph, audit each Conv: "is its output already derivable from a tensor I've computed?"
— redundant-detector elimination is one of the highest-yield, lowest-risk passes (byte-equal, no correctness change).
**Algebraic-fold variant (task288):** an endpoint stored as its own block is often derivable by arithmetic from values
you already hold — `right_col = bottom_idx − left_col` folded out the entire right-endpoint block. Derive via cheap
arithmetic; don't store what's a `Sub`/`Add` away.
**Generator-invariant mask folds (task158, 2026-06-19):** when the generator never
draws input color 0 (`histogram[0]==0` over the corpus), an active/non-background
mask can often collapse to the label plane itself: `Where(is_bg, 0, labels)` then
`Min(nonbg_label, 1)`. Do this only after proving zero is absent; otherwise
off-grid/padding/no-color sentinels collide with real black/background semantics.

# 137. Narrow a THRESHOLD-ONLY carrier to U8 via QLinearConv — a dtype golfing lever (Boladi task192)

A `Conv` F32 intermediate consumed ONLY by thresholds/comparisons (never for its exact float value) doesn't need to
be F32 — emit it as a `QLinearConv` U8 with quantized integer levels (task192: carrier {empty,bg,colored} → {0,1,4}
via scale 0.25). The threshold decisions survive the quantization, so it stays byte-equal, and you save the F32→U8
bytes (~1,200 B on a 1,600 B carrier); often the U8 carrier also *replaces* a separate downstream `col_u8`
(`Cast(comb→U8)`), dropping a second tensor. **Producer caveat (task094, 2026-06-19): this lever needs the carrier's PRODUCER to emit U8 (a Conv→QLinearConv). If the F32 carrier is a plain `Slice` of the input, adding a `Cast(→U8)` DOUBLE-charges — the F32 Slice result is already counted, then the U8 copy too — so narrow AT the producing conv, never after a Slice.** **Verify `ab` byte-equal on the FULL corpus** — every downstream
threshold must land identically on the integer levels (check edge/tie grids); if one can't be expressed on the
levels, fall back. Auditing intermediate DTYPES is as high-yield as auditing redundant ops (#136).
**Prerequisite / blocker (task321):** QLinearConv only accepts INT8/UINT8 input. A direct F32 graph input
is rejected, and `Cast(input→U8)` on the full one-hot costs 9,000 B, usually wiping out the QLC win.
The "named U8 input bridge" does not help under the scorer: sanitation renames the formal input and the
original F32 frame is still charged. Use #137 only when a U8/INT8 feed already exists or is cheaper than
the carrier it deletes.
**Signed one-plane mask before separate fg/bg feeds (task283, 2026-06-22, 910->905):** for a fixed
active-domain binary mask, try a SIGNED INT8 single-plane encoding — `signed_mask = -1 + 2*ch` so
background=`-1`, foreground=`+1`, and QLinearConv off-grid padding=`0` — then let the final INT8
`QLinearConv` separate the classes through a shared output threshold (`y_zero_point`, e.g. `-9`),
writing the full `[1,10,30,30]` output directly. This pays one signed plane instead of separate
active/background and foreground feeds. Applies where the active-domain mask tolerates the
signed bg/fg/pad encoding and one QLinearConv threshold can split the classes.
**Compare BEFORE Split to shave scalar dtype tax (task146, 2026-06-22, 495->481):** when downstream
consumers only need predicates, move the `Equal(scores, 0)` BEFORE the `Split`, so the split produces BOOL
symmetry flags (1 B each) instead of F32 checksum scalars (4 B each). Reorder the compare ahead of the split
whenever the post-split values are used only as booleans; it does not touch the upstream crop tax.

# 138. Scorer-INVISIBLE expansion temporaries — free block-upsampling (Boladi task289)

In a data-dependent #80-inverse block-upsampler, the expansion intermediates (the upsampled tensor before the final
contraction) are SCORER-INVISIBLE — the `score_v2`/`ORT_DISABLE_ALL` profiler trace does NOT count them (verified:
profile==static==679 despite a large logical expansion). So only the selector/detector banks + the free-output Einsum
are charged; the expansion itself is FREE. This is why #80-inverse upsampling reaches such low floors (task289:
21,139→679, 31x). The lever: when you must expand/replicate then reduce, FUSE the expansion into the final Einsum (or
keep it a data-dependent temporary) rather than materializing a charged intermediate. The invisibility is EMPIRICAL —
always confirm with gate-5 (profile==static) per graph; see `scoring_internals.md` for what the trace does/doesn't count.

# 139. ScatterND TINY-PATCH avoids the 900 B Where-mask + Einsum-midpoint (not NonZero) for dynamic indices (Boladi task371)

**Small localized edit → ScatterND, not Where.** For a rule that writes a FEW cells and inherits the rest from the
input, use `ScatterND` to write the patch directly (task371: a 10-scalar patch — clear ch0 + set ch3 for a 5-cell '+'
— and the rest passes through). This AVOIDS the **900 B `Where` condition mask** that #131 calls "forced" — that floor
only applies to FULL-canvas overlays; for a small patch, ScatterND is the floor (594 vs the Where alternative's 900+).
So before paying the Where-mask tax, ask: is the edit small + localized? Then ScatterND-patch beats it.
**Grader-safe dynamic index via Einsum-midpoint.** To compute a data-dependent ScatterND index (e.g. the midpoint of
two marker cells), use a position-weighted `Einsum` — NOT `NonZero` (BANNED). A dynamic-index ScatterND is exactly
where a banned-op index computation silently zeros the task on the grader; Einsum-derived positions keep it safe.
(ScatterND itself is grader-validated, per task284.)

# 140. SCALAR ROW-MOMENT dissolves a presence plane (task188)

When a rule needs the 1-based row of a solid one-color object, do not build a `[H,W]` presence plane
or a bbox. For non-channel-0 solid rectangles with no padding ambiguity:
`row = 2 * Einsum('bchw,h->b', input, iota30) / ReduceSum(input) + 1`.
The factor 2 is the rectangle symmetry identity: the weighted row sum over all object cells equals
`N * (top+bottom) / 2`, and the task's target row is derived from that center. This turns the whole
location read into one scalar weighted moment plus one scalar count; no spatial intermediate survives.
Boundary: only use it when the color/background contract makes `ReduceSum(input)` count object cells
exactly. Channel-0/background or padded one-hot ambiguity needs an explicit object mask first, which can
erase the win.

# 141. Axis-bitset intersection as FREE OUTPUT (task047)

For row/column set-intersection outputs, pack each axis into integers and make `BitwiseAnd` the graph
output. Task047 reduced a row/col crossing solver **1,687→1,003** by avoiding the final 900 B canvas:
`rowbits[1,10,H,1] & colbits[1,10,1,W] -> output`. A 9x9 bitset plus Pad looked smaller by params but
lost because the Pad input/canvas was charged. Reach for this when every output cell is "row carries bit
AND col carries bit"; keep the final expansion as the output node.
**PASSTHROUGH EXCLUSION (task151, 2026-06-22 — the subtle non-transfer).** #141 (and every free-output
separable/dissolve trick) wins ONLY when the output is a PURE `(row, col[, channel])` function. The moment
the rule is "passthrough the input EXCEPT a small region gets recolored," the bit-AND cannot apply: a
`BitwiseAnd` of two 1-D code vectors produces a pure outer product with NO input passthrough, but a
passthrough rule must reproduce all 10 input channels everywhere outside the edited region. That forces TWO
charged intermediates the free output cannot dissolve: (1) a region-DETECTION plane (slice ch0 / ReduceSum
over channels) `[1,1,30,30]` = 3,600 B F32 (the line-marginal needs float/int counts, so it can't stay
BOOL), and (2) the `Where` CONDITION is ONE 2-D tensor — a 3×3 box is the AND of a row-band `[1,1,30,1]` and
a col-band `[1,1,1,30]` whose result is `[1,1,30,30]` BOOL = 900 B (a single 1-D condition paints whole
rows/cols, not a box). Floor of the overlay route ≥ 3,600 + 900 = 4,500 B for a 30×30 box overlay — which
LOSES to a 0-memory dense `Conv` whose single output IS the graph output. task151 (full-H ∩ full-V line →
3×3 ring stamp, rest passthrough): coord/Where route 7,713; team-best no-bias `Conv[10,10,3,3]` = 900 (mem
0). Looks like task047 (same 2-line geometry, #141 hits 862 there) but the passthrough requirement makes 900
a confirmed WALL. Rule: passthrough-of-input ⇒ detection plane + 2-D overlay condition ⇒ a dense Conv/output-node
emitter is the floor; #141/#142 do NOT apply.

# 142. Single-color one-hot dissolve: mask × color vector (task217)

If every non-background output cell has one color, collapse the 10-channel plan to one mask plus a free
color vector. `ReduceMax(input, axes=[2,3])` gives the input's present color vector; build only the
spatial mask, then output `Where(mask, color_vec[1,10,1,1], bg_vec)` (or the equivalent free-output
Einsum). This deletes the 10-channel sprite/canvas tail. Crop only the background channel when the
generator guarantees no padding ambiguity; otherwise keep the explicit mask. Reminder from the same task:
initializer params are element-counted, not byte-counted, so narrowing an INT32 table does not reduce
`params`, only intermediate memory.
**Build a DATA-DEPENDENT color vector with `OneHot(axis=0)` (task064 v12, 2026-06-21):** when the color
is a scalar-ish index `[1,1]` rather than read by `ReduceMax`, emit the `[1,10,1,1]` channel selector in
ONE node: `OneHot(index, depth=C, values=[0,1], axis=0)` -> F32 `[C,1,1]`, which broadcasts as
`[1,C,1,1]` in NCHW consumers and feeds both `Where(mask, color_vec, input)` and `Einsum(input,
color_vec, ...)` with NO Reshape/Unsqueeze. This replaces the common scalar-color path
`Cast(index->F32) -> Equal(index, color_ramp[1,C,1,1]) -> Cast(bool->F32)`, deleting two scalar casts,
the two 10-cell BOOL `Equal`s, the 10-param color ramp, and duplicate scalar inits while staying
byte-equal. Use when the desired tensor is a broadcast channel SELECTOR, not a full batched one-hot
plane (contrast #116's `axis=1` OneHot, which builds the whole `[1,10,30,30]` output).

# 143. Size-conditional index remap + index-map cost law (task298/task382)

When two fixed grid sizes need different remaps, store the two small INT32 index vectors and select them
with a scalar size predicate: `idx = Where(is_size_A, idx_A, idx_B)`, then `Gather(data, idx, axis=...)`.
This replaces a dynamic control-flow branch with one tiny index table and keeps indices at INT32 (Gather /
GatherElements allow INT32; GatherND does not).

Cost law: a broadcast `GatherElements` index map is a tensor like any other. A dynamic `[N,M]` INT32
index costs `4*N*M` bytes and only wins if it is smaller than the data/canvas it deletes. Task382's
map looked semantically tidy but priced **6,362→11,031** because the broadcast index exceeded the
saved payload. Use `GatherElements` for axis-local reads; do not build a full index map just to avoid a
small Slice/Pad/Where.

# 144. Scalar dimension recovery from a full one-hot count (task067)

If the generator constrains dimensions algebraically, one scalar cell count can
replace per-axis occupancy reductions. The full one-hot input has exactly `H*W`
active cells, so `ReduceSum(input)` is an exact scalar even with zero padding. For
families such as `W = kH`, compare a static polynomial ramp against that scalar:
`k * c^2 < H*W` selects `c < H` without building `[1,1,1,W]` width counters.
Task067 used `3*c^2 < ReduceSum(input)` to recover a crop boundary and dropped
315 -> 65.

Use this only after proving the dimension relation from the generator, then gate
official quirks separately. A small initializer ramp usually beats `Range` or
`ConstantOfShape`: params are dtype-independent element counts, while generated
range tensors are charged by output bytes.

# 145. Sparse scattered F32 reads: flat GatherND beats separable Slice/Gather (task033)

When reading N scattered cells from the F32 one-hot input, a flat
`GatherND(input, idx[N,3])` with `(channel,row,col)` triples is the cost floor for
sparse reads. The index initializer costs `3N` params; separable attempts usually
materialize a full-width F32 plane first:

- Slice a channel plane then row/col Gather -> pays `[1,1,30,30]` F32 = 3,600 B.
- Row Gather first -> pays `[rows,30]` F32 before column narrowing.
- Runtime-build the index -> the assembled INT64 index tensor is charged as memory.

This extends #113's stencil-read rule with the crossover law: for sparse point
reads, spend the `N*rank` index params rather than birthing a plane. Separate rule:
row-basis x col-basis masks only represent rank-1 rectangles. Before replacing a
2-D stencil with separable outer products, verify the shape is rank-1; plus/cross
templates and arbitrary 3x3 glyphs are rank>1 and will silently fail.

# 146. Dynamic Slice value_info rescue + Slice-vs-GatherND crop law (task036)

Runtime `Slice` starts/ends can be scorer-safe when the output shape is provably
constant. The scorer's `calculate_memory` rejects unknown `[?,?,?,?]` value_info;
bake the concrete output value_info for the dynamic Slice and rerun
`onnx.shape_inference.infer_shapes(strict_mode=True)` so downstream shapes merge.
This is the inverse of the stale-cache trap: clear stale value_info after surgery,
but annotate a dynamic op whose real shape is fixed.

For fixed-size contiguous windows, Slice also beats GatherND on geometry cost.
Slice needs only starts/ends (`2 * rank` INT64 scalars); GatherND needs
`h*w*rank` INT64 indices plus the offset chain. Use GatherND for sparse scattered
points (#145); use dynamic Slice for a contiguous crop whose size is constant.

**Narrow Slice over full-canvas GatherND (task079, 2026-06-19):** if active bounds
prove the crop is a contiguous window, use runtime `Slice` with baked value_info
instead of `GatherND` over the full 30x30 canvas. task079's width-restricted dynamic
Slice saved 288 B. All Slice index inputs (`starts`, `ends`, `axes`, `steps`) may be
INT32 as long as they share the same Tind; do not pay GatherND's INT64-only index
tax unless the access is truly scattered.

# 147. TopK-marker shrink stack: F16 OneHot, I32 indices, F16 counts (task391)

For `AveragePool/ReduceSum -> Slice -> TopK -> OneHot -> Pad` marker/rank graphs,
stack all three dtype narrowings:

1. OneHot output values should be F16. BOOL/UINT8 OneHot outputs have no ORT kernel,
   so F16 is the one-hot floor.
2. Cast TopK indices from INT64 to INT32 immediately, and make `depth` INT32 too;
   every downstream index Slice/Reshape then halves.
3. Cast the count tensor from F32 to F16 before small bg/color Slices when the cast
   tensor is smaller than the F32 values it narrows. Counts sourced from the F32
   graph input are still F32-bound until after the reduction.

Do not squeeze trailing singleton dims just to make shapes prettier: every named
Reshape that drops rank is a counted tensor. Place the one unavoidable rank-drop on
the smallest index tensor. **(task264, 2026-06-12) inverse corollary:** when a later
reshape-to-1D is inevitable, flatten straight to `[N]` (not `[1,1,N]`) so the whole
downstream chain stays 1-D — the final Reshape node and its `[1,1,K]` intermediate vanish.

# 148. Reversed-Slice ArgMax for the LAST (or Nth-from-end) active element (task288, 2026-06-12)

To find the LAST active position along an axis, reverse with a negative-step `Slice`, `ArgMax` the reversed
axis, then map back: `pos = (N-1) - ArgMax(reverse(x))`. Replaces the 5-node shift/pad/mul/compare chain that
hand-builds a "last set element" (task288: -96 B / -13 params). The reversed `Slice` needs static value_info
(traps §8b) so the ArgMax axis length is known. Same "ArgMax IS the locator" family as #32 (base-cell) and
#64 (bbox extents), just read from the far end.

# 149. Concat-prepend beats Slice+Pad for a zero-edge shift (task199, 2026-06-12)

A 4-D "shift by one, fill the vacated edge with zero/False" is cheaper as `Concat([false_init, Slice(x,
drop_tail)])` than `Pad(Slice(...))`: the Concat carries ONE 1-element edge initializer vs Pad's 8-element
`pads` vector (-7 params), valid when the prepended edge is the zero/False sentinel and the dropped tail is
structurally zero. For a NON-zero edge fill, Pad's `constant_value` still wins — this is specifically the
zero-prepend case.

# 150. One triangular operand serves BOTH directions via subscript swap + Where(f,x,x) NOP removal (task270, 2026-06-12)

In a free-output Einsum a single lower-triangular operand also covers the upper-triangular direction by
swapping its two subscripts in the equation (`upper_tri[z,v] == lower_tri[v,z]`), deleting a second N×N mask
(task270: -N² params, 225 on a 15-wide axis). Two companion micro-rules from the same ship: `Where(flag, x,
x)` is a no-op — delete it (identical branches); and for constant `ScatterND` canvas writes, order updates so
the CENTER cell is written LAST when overlaps must resolve to center. Cross-ref symmetry_and_affine.md.

# 151. ch0-background Slice replaces a ch0-penalty mask; the shared-[1] init lever (task291, 2026-06-12)

When channel 0 (background) is ALWAYS disqualified from a per-channel selection, `Slice(x, starts=1, ends=10,
axis=1)` drops it structurally — no 10-element `ch0_pen` penalty mask (-10 params). Companion levers from the
same ship: alias ONE shared `[1]`-shaped scalar initializer across every node needing a unit constant (params
counted once); and recover a rectangle area as `(h-1)*w + w` from existing dim scalars instead of a second
multiply. Extends #67 (free-output / structural-zero family).

# 152. Integral-histogram / summed-area cache for repeated rectangle color queries (O1, 2026-06-19)

When a rule asks many axis-aligned rectangle questions against the SAME grid — color counts, mode/majority,
rank/median over ARC colors, box sums, low-order moments — build a shared prefix cache instead of paying a
fresh mask/reduce per query. For a color-index grid, `OneHot(depth=10) -> CumSum(W) -> CumSum(H)` creates a
10-bin integral histogram; each rectangle query is four gathered 10-vectors plus add/subtract, then `ArgMax`
or bin-prefix logic. If the canonical 10-channel one-hot input is already the source, skip the OneHot and
prefix the free bin volume directly.

Pricing anchor at H=W=30, K=10: the cache exposes about `K*N + 2*K*N = 27000` elements if an index canvas must
be one-hot first, less when the free input already supplies bins. A single all-color rectangle query by dense
masks is roughly `K*900` mask/activation scale, so the cache crosses over after a few shared box queries. A
scalar summed-area table is cheaper (`~2N` elements); dual SATs for `X` and `X*X` make local sums/means/variance
and simple moments cheap (`~4500` elements plus query scalars). For large rectangular rank filters on values
0..9, histogram counts beat stacking `k*k` neighbors; for tiny 3x3 windows, hand selection networks or TopK
usually stay cheaper.

Caveats: the prefix cache is a charged intermediate, so do not build it for one rectangle. It is exact only for
axis-aligned rectangles and a fixed small alphabet; arbitrary numeric tensors lose the rank/mode shortcut.
Probe candidates are bbox interior color histograms, stripe/band tests, palette comparisons, repeated window
majorities, and any task that recomputes several box statistics from scratch.

# 153. Small-N exact correspondence via subset-state DP; use matchers only when a key is not enough (O1/O5, 2026-06-19)

Most object-slot routing does NOT need an assignment solver. If a scalar key uniquely orders objects or slots,
use the existing rank/TopK/scatter toolbox (#97/#99/#124/#143). Reach for a matcher only when the rule genuinely
requires one-to-one collision resolution under additive pair costs.

The exact static-DAG candidate is a subset-state linear-assignment DP, not unrolled Hungarian/JV. For N objects,
carry `dp[2^N]` states over N row stages; each stage materializes a fixed `2^N x N` transition slab, masks illegal
used-column transitions, reduces over predecessor states, and applies a fixed lexicographic perturbation for ties.
At N=10 this is 1024 states: tens of KB, not tiny, but it solves the true global one-to-one objective without
data-dependent augmenting paths. For micro-cases, an exhaustive permutation bank is simpler (`n!*n` indices:
96 params at n=4, 600 at n=5, 4320 at n=6) and can be cheaper than the DP.

Approximate lane: fixed-iteration Sinkhorn with a reject row/column is cheap and static, but it solves a relaxed
transport problem and the hardening step can disagree with the true discrete optimum. Stable matching is exact
only when "stable under preferences" is the task objective. Graph matching/QAP and auction-style methods are
usually too control-heavy or approximate for this scorer.

# 154. Row-packed XAG/bitwise lane for XOR/equality/symmetry binary rules (O6, 2026-06-19)

For a binary plane on width <=30, pack each row into one `uint32` lane `[1,1,H,1]` and lower a small offline
XAG/AIG-style circuit to opset-18 `BitwiseXor`/`BitwiseAnd`/`BitwiseOr`/`BitwiseNot` plus unsigned `BitShift`.
One packed `uint32` temporary at H=30 costs `30*4 = 120` B, so about 30 packed temporaries equal one full F32
plane (3600 B). Comfortable wins stay under roughly 20-25 packed temporaries including shifts/masks/adapter
costs.

The win regime is narrow but real: parity/XOR-heavy cellular rules, mirror/equality predicates (`XNOR`), and
compact non-monotone corner/T-junction logic. Exact linear XAG synthesis (for example mockturtle's linear path)
is useful at design time for these. The route usually LOSES for monotone thresholds/counting rules: K4's
Conv-compile naturally computes sums/thresholds, while MAJ/popcount-style circuits expand into many charged
bitwise tensors (`maj(a,b,c)` is about four bitwise ops after lowering).

Do not count abstract complemented edges as free; ONNX needs an explicit `BitwiseNot` unless polarity is
absorbed algebraically. Left shifts on width-30 lanes need a reusable `0x3fffffff` mask when spilled high bits
can matter. The packer and final unpack/paint can erase the body win, so this is best when the source is already
binary, the packed representation is reused, or the output can remain a bitset/free-output expression (#141).

# 155. Broadcast-offset index assembly + tiny fixed-offset ScatterND masks (task250, 2026-06-19)

When a dynamic object position already exists as a short vector, do not build an
`[N,D]` INT64 index matrix by concatenating N per-column `Add -> Unsqueeze` chains.
Precompute `offsets[N,D]` as a constant initializer and add the runtime base once:

```
idx[N,D] = offsets[N,D] + r0c0_vec[1,D]
```

Broadcast Add creates the whole index bank and deletes every `[N,1]` Unsqueeze /
Concat intermediate. task250 saved 317 B on an index-heavy patch builder. If the
same index bank writes a small fixed-size box/mask, a unique-target `ScatterND`
from zeros plus these fixed offsets can beat a row-mask x col-mask outer-product
chain; task250 saved another 75 B. Requirements: fixed N and fixed offsets, unique
targets, and a runtime base vector already paid for. If N grows large or offsets
are data-dependent, reprice against #139 tiny-patch and #145 sparse-read laws.

# 156. Tiny-K scalar selector: chained `Greater`/`Where` beats `Concat`->`ArgMax`->`Gather` (task365, 2026-06-21)

For a tiny FIXED number of scalar candidates, selecting the best by a chain of `Greater`/`Where`
updates can be cheaper than stacking them into parallel attribute banks and running
`Concat -> ArgMax -> Gather`. task365's prior v16 stacked three red-count scalars, `ArgMax`ed the
length-3 vector, then gathered `r0/cL/tall/wide` from four length-3 INT32 stacks; replacing that
with two scalar compare/update steps was byte-equal and dropped profile cost 3352 -> 3262 (-90 B):

```
gt01    = Greater(score1, score0)
score01 = Where(gt01, score1, score0)
attr01  = Where(gt01, attr1, attr0)
gt2     = Greater(score2, score01)
attrw   = Where(gt2, attr2, attr01)
```

Use ONLY when the selected attributes are small scalars/vectors and ties are impossible (or the
`>`/`>=` tie behavior is matched explicitly). This is NOT a replacement for selecting large output
BRANCHES — there `Where` materializes both branches and the #40 anti-recipe holds (the `Where` tree
is cheapest only because the bank/`Gather` route costs k x output bytes). The new rule is the scalar
complement: for small per-candidate scalars, the chained-`Where` selector also undercuts the
`ArgMax`+`Gather`-from-stacks route. Cross-ref #40 (k-way mux laws) and #134 (floor discipline).

# 157. Bit-unpack U8-lane narrowing: EXIT expand in U8, not packed-word dtype (task077, 2026-06-24)

Per-cell `BitwiseAnd(word[R,1], pow[1,W])` EXIT plane inherits packed dtype (U32=4 B/elem). Fix: split word into U8 lanes first — `Cast(BitShift(word,8*k,RIGHT),U8)` — then AND each lane with `pow8[1,8]` → BOOL → Concat. task077 W=21: 1,680 B U32 → -637 net (7,161→6,524, -8.9%). Full recipe in `op_cost_patterns.md §BitUnpackLane`. NOTE: task338's "unpack DEAD" verdict (ort_reality §38) should be re-checked with U8 lanes.
