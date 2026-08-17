"""ngs — Neural-Guided ONNX program Search for NeuroGolf.

A program-synthesis engine that composes ONNX ops into tiny, exact, low-cost graphs, scored by the
real correctness gates + memory+params cost; a learned policy guides the search. See README.md.

Import-light on purpose: submodules pull their own heavy deps (onnx/onnxruntime/torch/openvino) only
when imported, so e.g. ov_eval.py can run in a bare OpenVINO env that lacks onnxruntime.
"""
