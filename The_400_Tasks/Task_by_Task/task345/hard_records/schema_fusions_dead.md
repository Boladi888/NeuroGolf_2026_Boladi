# task345 schema fusions dead

- Live/team bar checked: manifest live cost 839, sha `0b9b6fb90e2b41a45b5171cd399ae0d0885bb585abcb0a332a75fe9076c31020`; task-local effective staged bar cost 832, sha `b655d33edd48f78b1e407bbb964620a1c3fbc9ba9a861f9b640cc697c730a1c6`.
- Attempted route: fuse `k = m1 + nb8 + g + g` with one U8 variadic `Sum`, and remove `Cast(k->u32)` by feeding `BitShift(uint32_base, uint8_shift)`.
- Exact failed gate: schema check before build. `onnx.defs.get_schema("Sum", 18)` allows only float/bfloat16 tensors; `onnx.defs.get_schema("BitShift", 18)` uses homogeneous input type `T` for both operands.
- Why dead: v8's width-10 Add chain and `Cast(k->u32)` are schema-forced under the current BitShift class-LUT formulation; building these variants would only score `None`.
- One remaining hypothesis: a different final carrier that avoids `BitShift` entirely could remove `ku/B30`, but the task note's feature-bit-sharing proof says <=8-bit class codes are not available for the 17 column classes.
