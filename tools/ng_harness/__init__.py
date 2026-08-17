"""ng_harness — shared NeuroGolf per-task verification harness.

Modules:
  killjob   — Windows JobObject kill-on-close (orphan safety)
  session   — thread-capped ORT session factory (sanitize + ORT_DISABLE_ALL)
  encode    — grid <-> [1,10,30,30] one-hot (vectorized batch from corpus)
  registry  — task NNN -> uuid / generator / official pairs / parent onnx
  corpus    — cached, deterministic ARC-GEN corpus on the Q: cache drive
  throttle  — cross-process self-throttle (cap concurrent inference to ~80% CPU)
  modes     — the 5 gate modes (visible, ab, truth, synth, adversarial) + battery
  safety    — fast submit-safety checks for models and bundles

CLI entry point: tools/ng_gate.py
Safety entry point: tools/ng_safety.py
Design + decisions: tools/ng_harness/PLAN.md
"""
__all__ = ["killjob", "session", "encode", "registry", "corpus", "throttle", "modes", "safety"]
