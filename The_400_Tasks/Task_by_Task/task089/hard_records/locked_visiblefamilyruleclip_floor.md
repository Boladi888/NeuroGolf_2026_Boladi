live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Reviewer-locked route: replace the descriptor carrier with a direct VisibleFamilyRuleClip predicate and emit through a free-output/preserve-input bridge.

Build used: `prototypes/bo_ng_cdx2_worker3_20260708/task089_locked_preserve_bridge_floor.onnx`.

Exact price gate: `score_network_v2` returned `(0, 0, 9900)`, cost 9900. This is build 1 of the hard cap and stops the route before visible/truth/profile.

Forced charged tensors in the cheapest direct `Where(mask, paint, input)` bridge:
- `mask30_b BOOL [1,1,30,30]` = 900 params/bytes floor for a full-canvas preserve-input condition.
- `paint_full_f32 F32 [1,10,30,30]` = 9000 params in the skeleton. A real dynamic predicate would pay at least the same element count as memory, and likely more. It cannot be a single broadcast color vector: visible data has 89/267 cases with both marker families active, and all 89 use different payload colors.

Why #139 ScatterND does not rescue the locked route: visible output paints up to 20 cells. Preserving the one-hot input by ScatterND must clear ch0 and set the payload channel, so it needs 40 scalar writes. ScatterND indices are forced INT64, so indices alone cost `40 * 4 * 8 = 1280`; F32 updates cost `40 * 4 = 160`; subtotal 1440 before any predicate/index-generation logic. This already violates the 1400 stop line.

Why the Equal/Pad color-index bridge does not rescue it: a generated color-index output still pays the fixed `color30_u8 [1,1,30,30]` 900 B Equal-tail canvas, and reading the active 13x13 input colors costs at least `color13_f [1,1,13,13]` 676 B before predicate/stamp logic. That is 1576 before marker-family rule work.

Failed gate: price gate, not visible. Per reviewer instruction, no second build and no full gates after skeleton cost >=1400.

Remaining hypothesis: a sub-1400 task089 solution would need a non-preserve-input free-output expression that directly emits the final one-hot channels without a full paint tensor, full condition mask, ScatterND one-hot patch index bank, or color-index Equal canvas. The visible marker-copy rule is data-dependent multi-color/multi-target, so no such low-rank expression was identified.
