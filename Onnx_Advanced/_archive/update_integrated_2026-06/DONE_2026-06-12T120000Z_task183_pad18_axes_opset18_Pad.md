# DONE: task183 opset-18 Pad axes input saves 2 params

**Date**: 2026-06-12  
**Worker**: worker-183  
**Task**: 183

## Finding

ONNX opset 18 added an optional `axes` input to the `Pad` operator (previously axes was attribute-only at opset 11, then removed; re-added as input at opset 18). With `axes=[2,3]`, the `pads` tensor only needs to specify the begin/end values for those two axes (4 elements) rather than all 4 axes (8 elements). This saves `8 - 4 - 2 = 2` net params when the saved pads elements (4) exceed the new axes tensor (2).

**Applicable when**: the Pad is zero-padding only on axes 2 and 3 (H and W), with zero begin-padding on axes 0 and 1. The zero-pad values on axes 0/1 carry no information and can be dropped by specifying axes.

**Result on task183**: params 81→79, cost 841→839. All 5 gates pass. Profile=839.

**ORT compat**: BOOL Pad confirmed working at opset 18 with ORT 1.24.4. Opset 13 BOOL Pad also works. Opset 12 and below break BOOL Pad (returns Type 'tensor(bool)' invalid error).

## Generalizable technique

Any model with a Pad node that pads along fewer axes than the tensor rank can save `2 * (rank - padded_axes_count) - len(axes_tensor)` net params. For a 4-D tensor padded only on axes 2 and 3: saves `2*2 - 2 = 2` params. Bump opset to 18, split the pads initializer, add axes initializer.

## Target §

This should be added or referenced in `techniques.md` under the Pad/opset section (the existing §81 output-tail addenda notes Pad as an output-fold; this is a distinct param-saving trick).
