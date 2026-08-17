# task030 row-bitset/class-LUT Equal-tail dead route

- Live bar used: 1054 profile cost from `python tools/ng_current_bar.py --task 30 --json`; authoritative source `manifest`; live SHA-256 `88cbb94de8c671d33eef5aefee30ea2224755632b487d15dc5f68528598835e8`.
- Reviewer hypothesis tested: from-scratch row-bitset/class-LUT decoder plus one free `Equal` tail, replacing the current dynamic 3x3 Slice -> code grid -> quadratic QLinearConv carrier.
- Prototype-build count for this rework: two total. Build 1 had an invalid `ConstantOfShape` scalar attribute and did not price; build 2 corrected only that ONNX form and is the pricing probe below. No corrected gateable build was attempted because the pricing probe exceeded the live bar.
- Pricing probe artifact: `The_400_Tasks/Task_by_Task/task030/prototypes/worker1_rowbit_lut_equal_pricing_probe.onnx`; SHA-256 `aa0cc5cb5601cf470432a1b556207f0505664507b156a17b459476fc940eeec8`.
- Exact cost gate:

```text
(0, 4500, 30) 4530
task030 worker1_rowbit_lut_equal_pricing_probe.onnx: static=4530 profile=4530 (memory 4500 + params 30; 266 real inputs)
```

- Charged lower-bound tensors:

```text
3,600 INT32 [1,1,30,30] lut_index_i32
  900 UINT8 [1,1,30,30] label_canvas_u8
```

- Why dead: an `Equal` tail needs a per-cell label canvas. A row-bitset/class-LUT decoder can produce that from row class + column only by materializing an ONNX `Gather` index plane, whose INT32 floor is already 3,600 B before any real row-bitset/top/height machinery. The best-case index+label surface is 4,530 cost, which is +3,476 over the 1,054 bar.
- #89 separability check: the row/column membership is not a fixed separable row-vector times column-vector map. Each row's column set is an input-dependent shifted row bitset, so an Einsum-final can express it only after the same per-cell LUT/index or mask materialization.
- Alternate no-index route: materializing class masks or a canonical 10-channel 10x10 intermediate reaches at least 1,000 B before active-height and label-tail costs, triggering the reviewer stop condition as well.
- Remaining hypothesis: a sub-1054 win must avoid the Equal-tail/class-LUT canvas entirely, or discover a final op that decodes input-dependent row bitsets without an INT32 per-cell index plane or canonical one-hot/mask intermediate.
