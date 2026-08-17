"""Spreadable-optimization patterns for the spread_scan driver.

Each module under this package exposes a :class:`SpreadPattern` instance
named ``PATTERN`` describing how to attempt a math-equivalent rewrite on
a task's ONNX. The driver loads all of them, runs them on every task in
``submissions/submission.zip``, and records which yield cost savings.

Spread-knowledge philosophy: when a clever shrink lands on one task
(e.g. ``task000``'s ``ones10 -> scalar1`` broadcast trick), that
insight should propagate to every other task whose graph has the same
shape. These pattern modules are the durable embodiment of those
insights.
"""
from __future__ import annotations

from tools.spread_patterns._base import (
    SpreadPattern,
    SpreadResult,
    iter_patterns,
)

__all__ = ["SpreadPattern", "SpreadResult", "iter_patterns"]
