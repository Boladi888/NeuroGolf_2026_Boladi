# task077 v1 v5runJ organic rebuild — library intake (2026-06-06, worker-7)

Ship-ladder: LIVE 19,422 (rawwhere/hull8/tplsafe approximation family) -> **7,235** static
opset-18, truth-FULL 200,000/200,000 + visible 266/266, §DUAL-TRUTH-certified
more-correct-than-LIVE (ab_divs 3,074 == LIVE_misses 3,074 - 0). Staged
`staged_submits/task077_v1_v5runj_organic_cost7235/`. Builder
`The_400_Tasks/Task_by_Task/task077/prototypes/build_task077_v1_v5runj.py`; numpy lab
`prototypes/rule_lab_v5run.py`.

## Promote-worthy general items

1. **QLinearConv ternary-vote gadget (techniques candidate, #-grade)**: with binary U8
   input, kernel w=[1,2,1] (any axis), x_scale=w_scale=1, y_scale=3, zps=0:
   `y = round((a + 2b + c)/3) = b OR (a AND c)` exactly (sums 0..4; 2/3->1, 1/3->0; no
   rounding ties). One op + 3 weight params computes: 1D single-hole fill
   (`mask | (left & right)`), between-fill (`m | (up & down)`), or any center-biased
   majority — replaces ~4 shift/Min/Max planes (~1,680 B) per use. Generalizes: pick
   integer weights + y_scale so the accept-set and reject-set sums are separated by a
   rounding boundary; verify no .5 ties.

2. **Orientation lemma for separated-component bbox/region fills**: when components are
   size-bounded (here <=3 rows) and pairwise separated by >=g in at least one axis, build
   the support mask on the TIGHT axis (window +- (size-1)) and reach along the LOOSE axis
   with steps <= g (g=2 here: MaxPool[1,5]-steps cannot jump >=2-wide mask gaps). The
   mirrored design (loose-axis mask +-6, tight-axis reach) is UNSOUND: the wide support
   window admits third-component interference that bridges gaps. Proof + counterexample
   reasoning in task077.md.

3. **Generator-invariant vs officials-quirk doctrine (traps candidate)**: ARC-GEN
   `generate()` constraint loops only run on the `width is None` branch; official pairs
   are injected via fixed `colors` and can violate every generated invariant. Corollary:
   decode the generator FIRST (it defines hidden), but gate every lab candidate on
   officials too — the officials' violations form a small, enumerable quirk set that
   becomes explicit repair sub-rules (each must then be re-proven harmless on the corpus).

4. **Datapoint for §39/§41/F16 ledgers**: negative-pad Conv crop + all-U8 MaxPool/Min/
   QLinearConv detection + #48 Where-overlay tail again priced exactly (static class,
   macs 0, memory 7,200 + params 35); `polish_candidate.py` no-op on first build.
