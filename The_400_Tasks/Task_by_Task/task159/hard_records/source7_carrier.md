live bar: 472 profile cost, sha256 e0241fe189209d009711a368f3e782f0afab6435aedac5b3199da11233e6c277 (`ng_current_bar.py --task 159 --json`, manifest authority).

attempted route: collapse Aditya's `[1,8,5,5]` U8 `source8` channel carrier to 7 or fewer channels and keep the final channel+spatial `Resize` placer; reproduced by `prototypes/analyze_source_carrier_width.py`.

failed gate: analytical map gate before ONNX build; `python The_400_Tasks/Task_by_Task/task159/prototypes/analyze_source_carrier_width.py` reports `Csrc=7: NO common start` and `Csrc=6: NO common start` for the `tf_crop_and_resize` / `round_prefer_floor` channel-axis mapping across colors `{1,3,4,5,6,7,8,9}`.

why dead: for every common channel ROI start searched, at least one required color cannot map output channel 0 to blank, channel 2 to frame, and channel `c` to the sprite source without also mapping another output channel to one of those active sources. Color 9 is the tight blocker under the parent start; broad start search still finds no 7-channel common-start solution.

one remaining hypothesis: an index-canvas formulation could avoid the 8-channel one-hot carrier, but it must also avoid the usual `[1,1,30,30]` U8 index intermediate (900 bytes), otherwise it loses to the current 200-byte source carrier.
