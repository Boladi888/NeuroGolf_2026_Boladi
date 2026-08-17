# s[10] Channel Sign Fold Infeasibility

- live bar: 118 bytes, manifest/team-best SHA-256 `5121f6312e9e3bd6ee13b5b4a91bdd98dd1ec249358a67bf65a63fea07ff3f7e`
- reviewer-locked route: start from `Team_Best_Onnx/task001.onnx` and remove the 10-param `s[10]` sign vector by folding its foreground/background role into the existing one-node free-output `Einsum`, with no new length-10/channel selector initializer.
- routed references used: `Onnx_Advanced/INDEX.md` points this graph to `techniques.md`; relevant routes are §67 free-output, §89 `Einsum` final/runtime contract, §109 free-output multi-term separable `Einsum`, and §119 deep/repeated-input `Einsum`. §81 Pad/Equal-tail was not priced because the current 118 graph already has no charged mask/canvas.
- parent graph: one `Einsum`, params only: `Q[30,3]` 90 + `K[2,3,3]` 18 + `s[10]` 10 = 118; profile exact `static=118 profile=118`.
- proof script: `The_400_Tasks/Task_by_Task/task001/prototypes/prove_s_fold_infeasible.py`
- proof command: `python The_400_Tasks/Task_by_Task/task001/prototypes/prove_s_fold_infeasible.py`
- exact result:
  - `valid foreground masks n=2..8: 501`
  - `parent_topology_delete_s: forced_violations=40581 contradictory_pairs=40581 best_possible_min_margin=0.0`
  - `channel_uniform_no_selector: forced_violations=18144 contradictory_pairs=18144 best_possible_min_margin=0.0`
  - `Rows/cols >=9 can be kept nonpositive in the best case by zeroing the spatial basis there; both contradictions above are entirely inside the active 9x9 block, so the outside-grid condition does not rescue the fold.`

## Formulation 1: Same Parent Topology, Delete `s`

If both `s[c]` and `s[d]` are removed and their role is folded only into spatial `Q/K` factors, the output channel `d` is carried only by raw input count `n_d >= 0`. For every valid color-9 foreground mask and every active 9x9 output cell, channels 0 and 9 therefore share one sign. Exactly one of those two channels must be positive at every cell: channel 9 on foreground-product cells, channel 0 otherwise. This forces one sign violation per `(mask, active cell)`: `501 * 81 = 40581`. The best possible minimum signed margin is `0.0`; a positive margin for one of the pair makes the other wrong.

## Formulation 2: More Permissive Channel-Uniform Fold, Still No Channel Selector

Even if the equation is rearranged to let a channel-uniform one-node formula see the current channel mask `D`, no channel-length selector means the same scalar rule `f(D,u,v)` must serve background channel 0 and foreground channel 9. For any subset `D` of size 2..7, `D` can be a foreground mask in one valid example and the background mask in another. On mixed ordered source pairs, the foreground case requires AND=false (`f <= 0`) while the background case requires OR=true (`f > 0`). The exact contradiction count is:

`sum_{m=2..7} C(9,m) * 2*m*(9-m) = 18144`

Again the best possible minimum signed margin is `0.0`; choosing either sign fails the opposite requirement. This includes foreground color 9 and assumes the best-case outside-grid behavior (`rows/cols >=9` zero/nonpositive), so the failure is entirely inside the active 9x9 block.

## Verdict

No ONNX candidate was built. Removing `s[10]` without a replacement channel selector cannot be made correct under the locked free-output `Einsum` family. The next plausible sub-118 route would need a different way to encode the channel-0-vs-foreground sign asymmetry without paying a length-10 channel operand.
