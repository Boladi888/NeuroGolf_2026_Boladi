import argparse
import os
import pathlib
import tempfile
import unittest

from tools import ng_gate


class TestNGGateBudgetPolicy(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "NEUROGOLF_GATE_ALLOW_LONG_BUDGET",
                "NEUROGOLF_GATE_MAX_SECONDS",
                "NEUROGOLF_HEAVY_GATE_N",
            )
        }
        for key in self._old_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _args(self, mode="truth", max_seconds=900.0, no_budget=False, task=15, cand="cand.onnx"):
        return argparse.Namespace(
            mode=mode,
            max_seconds=max_seconds,
            no_budget=no_budget,
            task=task,
            cand=cand,
        )

    def test_rejects_hour_plus_heavy_gate_budget(self):
        err = ng_gate._apply_gate_budget_policy(
            self._args(max_seconds=7200.0),
            8000,
        )

        self.assertIsNotNone(err)
        self.assertIn("requested max_seconds=7200s", err)

    def test_rejects_no_budget_for_heavy_gate(self):
        err = ng_gate._apply_gate_budget_policy(
            self._args(no_budget=True),
            8000,
        )

        self.assertIsNotNone(err)
        self.assertIn("--no-budget is not allowed", err)

    def test_allows_default_heavy_gate_budget(self):
        err = ng_gate._apply_gate_budget_policy(
            self._args(max_seconds=900.0),
            8000,
        )

        self.assertIsNone(err)

    def test_rejects_hour_plus_light_gate_budget(self):
        err = ng_gate._apply_gate_budget_policy(
            self._args(max_seconds=7200.0),
            2000,
        )

        self.assertIsNotNone(err)
        self.assertIn("requested max_seconds=7200s", err)

    def test_allows_default_light_gate_budget(self):
        err = ng_gate._apply_gate_budget_policy(
            self._args(max_seconds=900.0),
            2000,
        )

        self.assertIsNone(err)

    def test_timeout_marker_rejects_same_candidate_before_rerun(self):
        old_timeout_dir = ng_gate._TIMEOUT_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ng_gate._TIMEOUT_DIR = pathlib.Path(tmp) / "timeouts"
                cand = pathlib.Path(tmp) / "cand.onnx"
                cand.write_bytes(b"slow model")
                args = self._args(cand=str(cand))

                ng_gate._write_timeout_marker(args, 8000, "wall budget 900s exceeded")
                err = ng_gate._apply_timeout_marker_policy(args, 2000)

                self.assertIsNotNone(err)
                self.assertIn("prior wall budget timeout", err)
                self.assertIn("gate-timeout/pathological", err)
        finally:
            ng_gate._TIMEOUT_DIR = old_timeout_dir


if __name__ == "__main__":
    unittest.main(verbosity=2)
