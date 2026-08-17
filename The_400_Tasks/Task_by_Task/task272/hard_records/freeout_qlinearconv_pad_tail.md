# task272 hard record: free-output U8 QLinearConv Pad tail

Live/team-best bar: cost 378, sha256 `f0302ce51976334b74f18b95e7cb9e455aa91d56d97a77bc47b1ee604000d526`, source `python tools/ng_current_bar.py --task 272 --json` / manifest, owner Vamshi, updated 2026-07-02T02:35:15Z.

Attempted route: reviewer-locked small free-output one-hot tail. Graph: `Slice(red5 F32) -> Cast(U8) -> QLinearConv([ch0=1-red, ch1=isolated red, ch2=non-isolated red]) -> Pad(channel+spatial) == output`. Probe path: `The_400_Tasks/Task_by_Task/task272/prototypes/task272_bo_ng_worker3_freeout_qlinearconv.onnx`, sha256 `e39aa5baf0cd346966fd55fc7f64d1bd34da6a87dfb190f9908ff53a34810950`.

Exact gate results: static score `(0, 200, 50) 250`; real-input profile `static=250 profile=250 (memory 200 + params 50; 267 real inputs)`; visible gate `34/267 (12.734082397003744%) FAIL`; first failing input `[[0,2,2],[0,2,2],[2,0,0]]`. Full-tensor first diff: pair 0, shape 3x3, `(0,0,0,3)` expected `0.0`, actual `1`, mismatching elements `20`.

Why dead: the #67/#81 Pad-last tail prices well because output is free and the small U8 one-hot is only `[1,3,5,5]`, but `ch0 = 1 - red_center` is not active-masked. For 3x3/4x4 inputs, padding cells inside the fixed 5x5 crop have red=0, so the QLinearConv writes background channel 0 there. This is traps §FREE-OUTPUT active-mask/boundary leakage, and one visible miss is a hidden zero.

One remaining hypothesis: an active-safe free-output tail would need QLinearConv to read ch0/IG as well as red. That adds at least another F32 crop/cast path or a multi-channel F32 crop before the U8 QLinearConv, wiping out the sub-378 saving; no more scatter variants should be explored for this locked request.
