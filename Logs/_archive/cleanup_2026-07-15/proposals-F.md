# Cleanup Proposals - Agent F

## For Agent A: NOTICE / Third-Party Attribution

`tools/onnx_tool_1.0.1_reference/` contains verbatim reference copies from upstream
`onnx-tool` 1.0.1:

- `tools/onnx_tool_1.0.1_reference/onnx_tool_1.0.1_node.py`
- `tools/onnx_tool_1.0.1_reference/onnx_tool_1.0.1_graph.py`

The sibling markdown files document provenance, but the copied `.py` files have no copyright
or license header and the root NOTICE currently only names this repository owner. Suggested
NOTICE text:

```text
This repository includes reference excerpts from onnx-tool 1.0.1.

onnx-tool
Copyright (c) ThanatosShinji and contributors
License: MIT
Upstream: https://github.com/ThanatosShinji/onnx-tool

The reference copies live under tools/onnx_tool_1.0.1_reference/ and are retained for
historical scorer/profiling analysis, not as active project code.
```

## For Owner / Orbit Wars Archive: Arc A770 ML Environment Notes

The replaced top-level `ml/README.md` was not a NeuroGolf `ml/` README. It described an older
Orbit Wars / local-machine ML setup and shadowed `ml/ngs/README.md`, which is the current
NeuroGolf ML documentation. The durable hardware notes likely belong back in the Orbit Wars
repo or in a machine-environment archive.

Useful content to preserve there:

- Intel Arc A770 cannot use CUDA; viable local paths were OpenVINO, PyTorch XPU, and PyTorch
  DirectML.
- Keep heavyweight ML environments separate from a minimal project/runtime venv:
  `.venv-ml` for OpenVINO + PyTorch XPU, `.venv-dml` for DirectML, project `.venv` for runtime.
- Pin `numpy<2` for older ML stacks that still require NumPy 1.x.
- PyTorch XPU wheels are installed from the PyTorch XPU index; do not install IPEX unless its
  wheel exactly matches the installed torch version.
- OpenVINO GPU and torch XPU should run in separate Python processes; loading both GPU runtimes
  into one process was observed to hang on the first operation from the second runtime.
- Use `PYTHONIOENCODING=utf-8` or `sys.stdout.reconfigure(encoding="utf-8")` on Windows if tool
  output may include Unicode that cp1252 cannot encode.
- `ml/verify_ml.py` is a small backend validator for that local hardware setup. It is not
  Orbit-Wars-branded in code, but it validates the same OpenVINO/XPU/DirectML environment family
  and could move with the environment notes if the owner wants `ml/` to contain only NGS.
