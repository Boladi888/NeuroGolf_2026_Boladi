"""Unit tests for the shared ng_harness verification harness.

Style matches the rest of tests/ (resolve NG_ROOT/DATA_DIR via env+default, plain
test functions, skip gracefully when ARC-GEN / candidate onnx aren't present so a
clean checkout still passes). Run: python -m unittest tests.test_ng_harness
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

NG_ROOT = Path(os.environ.get("NEUROGOLF_MAIN_ROOT", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(NG_ROOT))

from tools.ng_harness import killjob, session, encode, registry, corpus, safety, throttle  # noqa: E402

DATA_DIR = Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data/competitions/neurogolf-2026")))

V3 = NG_ROOT / "The_400_Tasks/Task_by_Task/task324/prototypes/task324_compress_v3.onnx"
HAVE_ARCGEN = registry.arc_gen_dir() is not None
HAVE_V3 = V3.exists()
# The Kaggle dataset (official task pairs + the rules the safety scanner reads) is
# not vendored; on a bare checkout the tests that need it skip instead of failing.
HAVE_OFFICIAL = (DATA_DIR / "neurogolf_utils").is_dir()


class TestEncode(unittest.TestCase):
    def test_roundtrip_and_batch(self):
        grid = [[8, 8, 9], [9, 1, 8], [0, 0, 0]]
        t = encode.encode_grid(grid)
        self.assertEqual(t.shape, (1, 10, 30, 30))
        self.assertEqual(int(t.sum()), 9)  # 9 active cells, one-hot
        idx, hw = encode.grid_to_index(grid)
        oh = encode.encode_batch_index(idx[None], np.array([hw]))
        self.assertTrue(np.array_equal(oh, t))  # batch == single

    def test_oversize_is_none(self):
        self.assertIsNone(encode.encode_grid([[0] * 31]))


class TestSession(unittest.TestCase):
    @unittest.skipUnless(HAVE_V3, "candidate onnx not present")
    def test_thread_cap_and_run(self):
        s, name = session.make_session(str(V3))
        self.assertEqual(s.get_session_options().intra_op_num_threads, 1)
        self.assertEqual(s.get_session_options().inter_op_num_threads, 1)
        x = np.zeros((1, 10, 30, 30), np.float32); x[0, 0] = 1.0
        out = session.run_raw(s, name, x)
        self.assertEqual(out.shape, (1, 10, 30, 30))


class TestRegistry(unittest.TestCase):
    @unittest.skipUnless(HAVE_ARCGEN, "ARC-GEN not vendored")
    def test_task_uuid(self):
        self.assertEqual(registry.task_uuid(324), "d07ae81c")

    @unittest.skipUnless(HAVE_OFFICIAL, "competition data not present")
    def test_official_pairs(self):
        pairs = registry.official_pairs(324)
        self.assertTrue(len(pairs) >= 1)
        self.assertIn("input", pairs[0]); self.assertIn("output", pairs[0])

    def test_parent_resolves(self):
        path, src = registry.parent_onnx(324)
        self.assertTrue(path is not None, f"parent unresolved: {src}")


class TestCorpus(unittest.TestCase):
    @unittest.skipUnless(HAVE_ARCGEN, "ARC-GEN not vendored")
    def test_build_and_determinism(self):
        X, Y = corpus.get_tensors(324, 20)
        self.assertEqual(X.shape, (20, 10, 30, 30))
        self.assertEqual(Y.shape, (20, 10, 30, 30))
        X2, _ = corpus.get_tensors(324, 10)   # reload subset
        self.assertTrue(np.array_equal(X[:10], X2))  # deterministic
        info = corpus.cache_info(324)
        self.assertGreaterEqual(info["cached"], 20)


class TestThrottle(unittest.TestCase):
    def test_acquire_release(self):
        base = throttle.status()["live_slots"]
        with throttle.Slot("unittest"):
            self.assertGreaterEqual(throttle.status()["live_slots"], base + 1)
        self.assertEqual(throttle.status()["live_slots"], base)


class TestSafety(unittest.TestCase):
    def _blob(self, model: onnx.ModelProto, name: str = "task001.onnx") -> safety.ModelBlob:
        return safety.ModelBlob(task=1, name=name, blob=model.SerializeToString(), source="memory")

    def _identity_model(self, input_name: str = "input", output_name: str = "output") -> onnx.ModelProto:
        inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 10, 30, 30])
        out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 10, 30, 30])
        node = helper.make_node("Identity", [input_name], [output_name])
        return helper.make_model(helper.make_graph([node], "g", [inp], [out]), opset_imports=[helper.make_opsetid("", 13)])

    @unittest.skipUnless(HAVE_OFFICIAL, "competition data not present")
    def test_safety_hard_fails_official_banned_ops_case_insensitive(self):
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
        out = helper.make_tensor_value_info("output", TensorProto.INT64, [4, None])
        node = helper.make_node("NonZero", ["input"], ["output"])
        model = helper.make_model(helper.make_graph([node], "g", [inp], [out]), opset_imports=[helper.make_opsetid("", 13)])

        row = safety.scan_model(self._blob(model), load=False)

        self.assertEqual(row["verdict"], "FAIL")
        self.assertTrue(any(f["code"] == "FAIL_BANNED_OP" for f in row["failures"]))

    def test_safety_hard_fails_wrong_official_io_names(self):
        row = safety.scan_model(self._blob(self._identity_model("x", "y")), load=False)

        self.assertEqual(row["verdict"], "FAIL")
        codes = {f["code"] for f in row["failures"]}
        self.assertIn("FAIL_OFFICIAL_INPUT_CONTRACT", codes)
        self.assertIn("FAIL_OFFICIAL_OUTPUT_CONTRACT", codes)

    def test_safety_accepts_official_io_names(self):
        row = safety.scan_model(self._blob(self._identity_model()), load=False)

        self.assertEqual(row["verdict"], "PASS")
        self.assertTrue(row["official_io_contract"]["ok"])


class TestKilljob(unittest.TestCase):
    # arm() assigns THIS process to a JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE job. Under a
    # CI runner (itself job-object managed) that took the pytest process down mid-run.
    # tests/test_ng_killjob_orphan.py covers the same machinery safely, in a child.
    @unittest.skipUnless(
        os.environ.get("NEUROGOLF_TEST_KILLJOB") == "1",
        "arms a kill-on-close job against the test process; opt in with NEUROGOLF_TEST_KILLJOB=1",
    )
    def test_arm_status(self):
        killjob.arm()
        st = killjob.status()
        self.assertTrue(st["armed"])
        if os.name == "nt":
            self.assertTrue(st["job_created"])


class TestModesSmoke(unittest.TestCase):
    @unittest.skipUnless(HAVE_V3 and HAVE_ARCGEN and HAVE_OFFICIAL, "need onnx + ARC-GEN + competition data")
    def test_visible_and_truth(self):
        from tools.ng_harness import modes
        vis = modes.mode_visible(324, str(V3))
        self.assertTrue(vis["pass"], f"visible {vis['correct']}/{vis['total']}")
        tru = modes.mode_truth(324, str(V3), n=30)
        self.assertEqual(tru["correct"], tru["total"])  # v3 exact


if __name__ == "__main__":
    unittest.main(verbosity=2)
