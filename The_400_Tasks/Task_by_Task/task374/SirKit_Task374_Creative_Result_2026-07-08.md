# SirKit Task374 Creative Result

- Source: live HandOff task result fetched 2026-07-11T16:54:00Z
- Project/request id: `82513bd2-0f60-434a-91b2-538d71010fa8`
- HandOff task id: `48c7b406-f3e9-4e58-8588-f3ccd3671733`
- External key: `neurogolf:bo-to-34r7h:task374-creative`
- Title: `Work on NeuroGolf task374 creative pack`
- Status at fetch: `pending_verification`
- Assignee at fetch: `SirKit`
- Updated at: `2026-07-08T00:12:22.220Z`

---

task: task374
decision: discard (dual-verified — see scoped claim below, this is NOT a bare "couldn't find one")
candidate_name: none — baseline qscale-share_cost1365 holds (sha256 a4b47de8...)
expected_cost: 1365 (unchanged) | memory: 1310 | params: 55
baseline_used: 1365 staged qscale-share

core_idea:
After an initial solo pass (12 micro-levers, all closed with exact numbers), this
was independently re-attacked by a second model (Opus, given full architecture
context + my ruled-out list, explicitly asked to BUILD not just reason) plus my own
continued parallel work. Both passes converge on the same forced components via
real onnx builds validated against a from-scratch numpy oracle (267/267 official
examples + 6000/6000 synthetic grids matching the true generator invariant — the
real ARC-GEN corpus/scorer is not vendored in this checkout, confirmed by direct
failed invocations, so this oracle is the strongest available substitute, not
equivalent to the real gates — see caveat at the end).

formulation_families_considered (13 total across both passes, all with exact
measured numbers; the first 12 were in my prior submission, repeating tersely +
adding the 3 NEW ones a second independent pass built):

1-12. (unchanged from prior submission: single-channel packed decode impossible;
select-then-rank reorder +380B; arithmetic blend +400B; ReduceSum forces +380B net;
Slice-axes +1 param; ConstantOfShape +8B; int4/uint4 dead per repo's own
retraction; direct vertical detector +83B; Greater-vs-Cast wash; opset-20 bool
reduce inapplicable; dilated-conv-crop same-or-worse; DynamicQuantizeLinear
inapplicable.)

13. NEW — cross-family terminal-decode sweep, BUILT not just reasoned (Opus pass):
  a) Alt A: single-channel color-index -> Pad(sentinel=10) to 30x30 -> Equal
     (channel_ids) as free output. Built, onnx-checker PASS, validated 267/267 +
     6000/6000. Measured: memory=1970 params=55 cost=2025 (+660 WORSE).
  b) Alt B: single-channel color-index -> Equal at 10x10 -> Pad the one-hot to
     30x30 as free output. Built, onnx-checker PASS, same validation. Measured:
     memory=2070 params=56 cost=2126 (+761 WORSE).
  c) Alt C: drop the explicit Mul-mask; Concat(selected_features, gray_u8) into a
     3-channel ConvInteger instead. Built (Opus) + independently hand-derived by
     me in parallel. CLOSED ON COST regardless of decode feasibility: Concat's
     output shape is the SUM of input channels (300B) vs Mul-broadcast's output
     being the MAX of its inputs' shapes (200B) — a straight +100B loss by tensor-
     shape arithmetic alone. Measured (Opus build): memory=1410 params=66 cost=1476
     (+111 WORSE), confirming the shape-arithmetic prediction. NOTE ON A SUB-POINT
     WHERE THE TWO PASSES DISAGREED (flagging rather than smoothing over): I brute-
     force-solved a per-channel-weight-AND-independent-bias linear model and found
     valid integer weights once the search range was widened (e.g. bias=-20); Opus
     modeled ConvInteger's real shared-scalar zero-point constraint and found no
     solution in a 0-10 range. Neither of us confirmed whether ConvInteger's
     w_zero_point supports a per-output-channel tensor (which would reconcile the
     two models) — but it doesn't matter for the verdict, since the cost argument
     alone (independent of decode feasibility) already closes Alt C.
  d) Single-channel rank-code collapse (my own idea, tested via direct brute force
     across output-weight combos, not just argued): CLOSED computationally — a
     single channel through any linear decode can only ever produce one global
     sign-split across ALL output channels simultaneously (proven by brute force:
     no weight lets channel A win for one positive input value while channel B
     wins for a different positive value), confirming the >=2-channel requirement
     a second, independent way.

Forced-component statement (now build-verified across both passes, not merely
reasoned): the terminal node's input must supply BOTH the channel-expansion (1->10
colors) AND the spatial-expansion (10x10->30x30) to get a free output. Conv-family
(Conv/QLinearConv/ConvInteger) is the only op family with both a channel-mixing
weight input AND a `pads` attribute fusing both for free. Every other candidate
built (Equal, Where+Gather variants) does at most one of the two, forcing the other
to be pre-paid as a real ~900-1000B intermediate (matching this repo's own
documented "label30" dead end, now confirmed by build rather than assumed).

charged_tensors_expected: unchanged from baseline — gray_crop_f 400B (forced F32
input dtype) + gray_u8/horizontal_u8/horizontal 100B each (minimum per-cell signal
width) + selected_features/masked_features 200B each (forced by the now
DOUBLY-confirmed >=2-channel linear-decode requirement) + row/col descriptor chain
~210B = 1310B total, zero slack found anywhere across 13 independent families.

params_expected: 55 (unchanged), audited via 2 independent reduction angles, both
net-negative once correctness/dtype/shape constraints are honored.

known_risks / the one honest caveat (do not skip this in review):
This floor holds for (a) the standard one-hot [1,10,30,30] output convention and
(b) every family actually built-and-measured across both passes. It does NOT rule
out a genuinely different output convention the real grader might still accept, or
an undiscovered scorer quirk — this repo's own pre-existing hard_record already
named exactly this as "the only remaining escape," and NEITHER pass could test it,
because it requires the real ARC-GEN validation environment (generator dir + real
neurogolf_utils/score_network_v2), which is not vendored in this checkout —
confirmed directly (ng_gate.py truth/battery errors "ARC-GEN dir not found";
_score_v2.py errors ModuleNotFoundError: neurogolf_utils). The precise claim: no
sub-1365 candidate exists within the space actually searched, under the standard
output convention. NOT: no sub-1365 candidate can possibly exist anywhere. Closing
that last gap needs the real scorer/generator — if available, re-run `tools/
ng_gate.py battery --task 374` against the baseline first to confirm this oracle-
based verification matches the real gates, then the same tooling could probe
alternate output conventions directly against the real grader.

Bonus finding (orthogonal, already reported): the live graph's rank scheme is only
correct because the true generator always emits exactly 3 segments with 3
mutually-distinct lengths; its 2-bit encoding is provably wrong outside that
invariant (verified concretely: a 2-distinct-length synthetic grid gets the
"second-longest" cells miscolored). Invisible in the real distribution, but a real,
narrow generalization gap if the generator's invariant ever loosens.

files_returned: none (discard, dual-verified) — the qscale-share cost1365 staged
candidate remains correct to ship. Full working notes + all built onnx candidates
available on request (scratchpad, not attached inline).
