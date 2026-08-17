"""Detector-to-builder parity tests.

These tests catch the silent-fail class where a detector routes a task to a
builder, but the ONNX output does not decode the same way the competition
validator decodes it. In particular, raw output must use exact one-hot cells
inside the active grid; all-zero cells are only valid in canvas padding.
"""

from __future__ import annotations

import re
import os
import sys
import unittest
from pathlib import Path

import onnxruntime

NG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NEUROGOLF_DATA_DIR", NG_ROOT / "data" / "competitions" / "neurogolf-2026"))
UTILS_DIR = DATA_DIR / "neurogolf_utils"

sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / "tools"))
sys.path.insert(0, str(UTILS_DIR))

import build_submission as bs  # noqa: E402
import neurogolf_utils as nu  # noqa: E402
from tools.inspect_tasks import SOLVED_TASKS  # noqa: E402

nu._NEUROGOLF_DIR = str(DATA_DIR) + "/"


KNOWN_STRICT_PARITY_FAILURES = set()


def discover_pattern_pairs() -> list[tuple[str, str]]:
    """Return [(builder_name, detector_name)] for each *_network + _detect_*."""
    source = (NG_ROOT / "tools" / "build_submission.py").read_text(encoding="utf-8")
    def_re = re.compile(r"^def (\w+)\(", re.MULTILINE)
    funcs = {m.group(1) for m in def_re.finditer(source)}
    pairs = []
    for name in sorted(funcs):
        if not name.endswith("_network"):
            continue
        base = name[: -len("_network")]
        det = f"_detect_{base}"
        if det in funcs:
            pairs.append((name, det))
    return pairs


def load_task_examples(task_num: int, arc_gen_limit: int = 30):
    examples = nu.load_examples(task_num)
    return (
        list(examples.get("train", []))
        + list(examples.get("test", []))
        + list(examples.get("arc-gen", [])[:arc_gen_limit])
    )


def strict_verify(session, examples) -> tuple[int, int, str]:
    """Verify via raw output decode, not the looser local threshold helper."""
    right = 0
    wrong = 0
    first_diag = ""
    for idx, example in enumerate(examples):
        benchmark = nu.convert_to_numpy(example)
        if benchmark is None:
            continue
        actual_raw = session.run(["output"], {"input": benchmark["input"]})[0]
        actual = nu.convert_from_numpy(actual_raw)
        if actual == example["output"]:
            right += 1
        else:
            wrong += 1
            if not first_diag:
                first_diag = (
                    f"example {idx}: expected {example['output']!r}, "
                    f"decoded {actual!r}"
                )
    return right, wrong, first_diag


def output_is_strict_binary(out_array) -> tuple[bool, str]:
    """Check exact one-hot-or-padding semantics for a single ONNX output."""
    import numpy as np

    if out_array is None:
        return False, "output is None"
    if out_array.dtype != np.float32:
        return False, f"output dtype is {out_array.dtype}, expected float32"
    if out_array.shape != (1, 10, 30, 30):
        return False, f"output shape is {out_array.shape}, expected (1, 10, 30, 30)"

    is_one = out_array == 1.0
    is_zero = out_array == 0.0
    binary = is_one | is_zero
    if not binary.all():
        non_binary_count = int((~binary).sum())
        sample_vals = out_array[~binary].flat[:5].tolist()
        return False, f"{non_binary_count} non-binary values; samples: {sample_vals}"

    per_cell_ones = is_one.sum(axis=1)
    multi_lit = int((per_cell_ones > 1).sum())
    if multi_lit:
        return False, f"{multi_lit} cells have multiple channels equal to 1.0"
    return True, "ok"


class DetectorBuilderParityTest(unittest.TestCase):
    """Scan matched task/pattern pairs and flag new strict-decode failures."""

    @classmethod
    def setUpClass(cls):
        if not DATA_DIR.exists():
            raise unittest.SkipTest(f"NeuroGolf data not found at {DATA_DIR}")
        cls.pairs = discover_pattern_pairs()

    def test_discovery_finds_builder_detector_pairs(self):
        self.assertGreaterEqual(
            len(self.pairs),
            50,
            f"Expected at least 50 builder/detector pairs; got {len(self.pairs)}.",
        )

    def test_matched_builders_decode_strictly(self):
        failures = {}
        match_count = 0
        pattern_names = set()

        for task_num in range(1, 401):
            examples = load_task_examples(task_num)
            matches = bs._collect_matches(examples)
            for label, network_fn in matches:
                match_count += 1
                pattern_names.add(label)
                model = network_fn()
                session = onnxruntime.InferenceSession(model.SerializeToString())
                right, wrong, diag = strict_verify(session, examples)
                if wrong:
                    failures[(task_num, label)] = f"{right}/{right + wrong}: {diag}"

        unknown = {
            key: diag
            for key, diag in failures.items()
            if key not in KNOWN_STRICT_PARITY_FAILURES
        }
        known_open = sorted(key for key in failures if key in KNOWN_STRICT_PARITY_FAILURES)

        self.assertGreaterEqual(match_count, 100)
        self.assertGreaterEqual(len(pattern_names), 50)
        self.assertEqual(
            unknown,
            {},
            f"New strict-decode failures: {unknown}. "
            f"Known still open: {known_open}",
        )


class HighRiskPatternListTest(unittest.TestCase):
    """Check that the high-risk audit pattern names still map to builders."""

    HIGH_RISK_2026_05_06 = [
        "aligned_marker_fill",
        "color_swap_5_always_stays",
        "cummax_down_column",
        "diagonal_symmetry_fill",
        "dominant_color_keep",
        "frame_interior_fill",
        "grid_pattern_color_to_grid",
        "horizontal_or",
        "horizontal_tile_2x",
        "marker_driven_shape_copy",
        "marker_rectangle_interior",
        "pi_shape_project_to_bottom",
        "rank_vertical_bars",
        "two_by_two_zero_fill",
        "two_marker_cross",
        "vertical_halves",
    ]

    def test_high_risk_patterns_still_exist(self):
        pairs = discover_pattern_pairs()
        present = {builder_name[: -len("_network")] for builder_name, _ in pairs}
        missing = [name for name in self.HIGH_RISK_2026_05_06 if name not in present]
        self.assertEqual(
            missing,
            [],
            "High-risk pattern names drifted; re-run the silent-fail audit.",
        )


class SolvedTasksInventoryTest(unittest.TestCase):
    """Keep mining/source-of-truth tooling aligned with strict-clean matches."""

    @classmethod
    def setUpClass(cls):
        if not DATA_DIR.exists():
            raise unittest.SkipTest(f"NeuroGolf data not found at {DATA_DIR}")

    def test_solved_tasks_matches_strict_clean_detector_inventory(self):
        matched_tasks = set()
        for task_num in range(1, 401):
            examples = load_task_examples(task_num)
            if bs._collect_matches(examples):
                matched_tasks.add(task_num)

        self.assertEqual(
            matched_tasks,
            SOLVED_TASKS,
            "Update tools.inspect_tasks.SOLVED_TASKS when strict-clean detector coverage changes.",
        )


if __name__ == "__main__":
    unittest.main()
