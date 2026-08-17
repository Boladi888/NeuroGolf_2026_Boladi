live bar: 472 profile cost, sha256 e0241fe189209d009711a368f3e782f0afab6435aedac5b3199da11233e6c277 (`ng_current_bar.py --task 159 --json`, manifest authority).

reviewer-locked route: remove `Resize_chspatial`/`roi6`, avoid the 900-byte `[1,1,30,30]` U8 Equal-tail canvas, and try #81/#84/#85 final placers from the live graph's existing U8 `features3`/`source8` carrier.

pricing/proof command: `python The_400_Tasks/Task_by_Task/task159/prototypes/price_no_resize_placer.py`.

why no ONNX build was made in this rework: `features3` and `source8` encode only `(sprite color, 3x3 mask)`, not the red marker count that selects side `5/8/11/14`. The corpus contains the same carrier with different red counts: color 9, mask `(0,1,0,1,0,1,1,0,0)`, red count 16 at `arc-gen[8]` and red count 52 at `arc-gen[15]`. Any final placer reading only this carrier must emit the same output for both, but the required output side differs. A visible miss would be guaranteed.

why #81 Pad-last/Equal-tail does not rescue it: decoded one-hot costs are side5=250, side8=640, side11=1210, side14=1960 bytes before Pad-as-output. Since task159 includes 11 and 14 sides, the small-grid inversion threshold (`h*w < 90`) is exceeded. The full U8 value-canvas Equal-tail is 900 bytes and loses to the current 200-byte `source8` carrier plus small ROI family.

why #84/#85 Conv/QLinearConv placer loses: a bias-free output QLinearConv can avoid padded-canvas bias leaks, but a static Conv/QLinearConv from `source8` has no red-count input and therefore cannot select among the four magnifications. Adding a dense output placer is already too expensive: even a 5x5 `10*8` kernel is 2000 params, and a fully flexible 30x30 kernel is 72000 params, both above the 472-byte bar before any gating.

remaining hypothesis: a future win must find a new sub-472 way to inject the red-count selector into the final output op without materializing a full-size canvas or dense dynamic weights; the current no-Resize final-placer route is a priced dead end.
