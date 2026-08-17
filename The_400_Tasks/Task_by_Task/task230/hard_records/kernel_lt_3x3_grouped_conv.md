live_bar: 459
live_sha256: 8ace68fe9a517702bc3b1a0abe6140c455dcff3a4e7e871a2cd21149e401dfe8
route: Replace the centered 3x3 group=2 Conv with 1x1, 2x2, 2x3, or 3x2 grouped Conv kernels while keeping ch0-local marker classification and carried-color identity.
failed_gate: Pre-ONNX visible separability audit on bundled train+test+arc-gen via `python The_400_Tasks/Task_by_Task/task230/prototypes/probe_task230_kernel_sizes.py`.
why_dead: Every sub-3x3/asymmetric pad alignment has at least one ch0 patch label conflict for output channels 0..4, so any such single-Conv representation would visible-fail before coefficient fitting. Only centered 3x3 with pads [1,1,1,1] is conflict-free and affine-separable.
remaining_hypothesis: A strict sub-459 win would need a different legal scorer/accounting lever than Conv kernel shrink or short-bias UB, because the known legal centered 3x3 group=2 family costs 460 and the manifest 459 uses `conv_bias[9]` for M=10.
