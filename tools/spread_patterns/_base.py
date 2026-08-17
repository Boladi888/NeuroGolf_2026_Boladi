"""Contract for spread-knowledge optimization patterns.

A SpreadPattern is a self-contained, math-equivalent rewrite that can
be applied to a task ONNX without changing its output. The driver
applies it to every task in submission.zip and records cost deltas.

Two outcome categories:
  * ``advisory`` — pattern only reports applicability (e.g. crop-to-size
    needs task-specific bounds derived from the JSON). Cost delta is
    estimated, not measured.
  * ``transform`` — pattern actually rewrites the graph and returns the
    new model. Cost delta is measured via score_network_v2.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import onnx


@dataclass
class SpreadResult:
    """Outcome of applying one pattern to one task."""
    pattern_name: str
    verdict: str  # "applied" | "no-op" | "skipped" | "verify-failed" | "error" | "advisory"
    cost_delta: int | None = None  # negative = saves bytes
    new_model: onnx.ModelProto | None = None  # only set when verdict == "applied"
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpreadPattern:
    """One spreadable optimization technique."""
    name: str
    category: str  # "math-equivalent" | "advisory" | "selective"
    description: str  # one-line human-readable summary
    apply: Callable[[onnx.ModelProto, str], SpreadResult]
    """``apply(model, task_id)`` returns a SpreadResult. ``task_id`` is
    a hint like ``"task017"`` so advisory patterns can read the JSON.
    """
    safe_by_construction: bool = True
    """If True, the rewrite is math-equivalent and only needs sanity
    verification. If False (selective passes like FP16), the driver runs
    full byte-exact verification across multiple random inputs.
    """


def iter_patterns() -> Iterable[SpreadPattern]:
    """Yield every PATTERN exposed by tools.spread_patterns.pass_*."""
    import tools.spread_patterns as pkg

    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg or not name.startswith("pass_"):
            continue
        mod = importlib.import_module(f"tools.spread_patterns.{name}")
        pattern = getattr(mod, "PATTERN", None)
        if isinstance(pattern, SpreadPattern):
            yield pattern
