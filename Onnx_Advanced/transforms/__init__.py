"""Reusable, tested ONNX graph-surgery transforms for NeuroGolf cost golf. See README.md.

These are the *library* form of the recurring win patterns (p43/p44): importable, output-preserving
(or explicitly gated) functions with a fixture pair + a test each, so a worker reuses a proven
transform by name instead of re-deriving the same graph surgery as a one-off script.

Import a helper directly (from repo root):

    import sys; sys.path.insert(0, "Onnx_Advanced/transforms")
    from share_initializer import share_initializers

`__init__` intentionally does NOT re-export via relative imports (Onnx_Advanced is not a package
across every runner), so helpers stay importable by plain module name after the sys.path insert above.
"""
