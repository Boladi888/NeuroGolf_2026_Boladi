route: cross-channel grouped single Conv under the 270-param bar

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: keep the params-only one-node/final-output Conv structure, but relax the previous depthwise-only search by allowing smaller cross-channel groups (`group=1,2,5,10`). The hope was that an IG/ch0 cross-channel read could shrink the spatial footprint enough to beat the current `[10,1,3,9]` depthwise kernel.

Exact failed gate:
- Prototype search script: `The_400_Tasks/Task_by_Task/task098/prototypes/task098_search_crossgroup_conv.py`.
- Result: `no hit groups=1; checked=29`; `no hit groups=2; checked=393`; `no hit groups=5; checked=5640`; `no hit groups=10; checked=21831`; final line `no sub-270 cross-group Conv footprint found`.

Why dead: within the scorer-valid one-node Conv family, the cost budget forces area <=2 for `group=1`, <=5 for `group=2`, <=13 for `group=5`, or <=26 for depthwise `group=10`. None linearly separates all visible output bits under those receptive-field/group constraints. The current area-27 depthwise 3x9 kernel remains the smallest found single-Conv final-output representation.

Remaining hypothesis: only a non-Conv final op with an implicit structured local predicate, scorer-supported sparse/structured Conv weights, or a free-output contraction that encodes the same stencil with fewer than 270 dense params can beat the bar.
