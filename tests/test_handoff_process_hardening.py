from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Handoff" / "runner"))
sys.path.insert(0, str(ROOT))

from handoff_runner import Runner  # noqa: E402
from tools.ng_harness import throttle  # noqa: E402


class HandoffProcessHardeningTest(unittest.TestCase):
    def make_runner(self, tmp: Path) -> Runner:
        config = {
            "workflow": {
                "lane": "Codex",
                "lane_id": "codex",
                "batch_id": "99",
                "runner_id": "neurogolf:codex:batch99",
                "worker_attempts_before_review": 1,
            },
            "paths": {
                "private_state": str(tmp / "runner_state.json"),
                "raw_log_dir": str(tmp / "raw"),
                "operation_log": str(tmp / "operations.jsonl"),
                "prompt_dir": str(tmp / "prompts"),
                "runner_prompt_dir": str(tmp / "runner_prompts"),
                "captain_lane_inbox": str(tmp / "captain"),
            },
            "agents": {"worker": "worker1", "reviewer": "reviewer1", "workers": ["worker1"], "reviewers": ["reviewer1"]},
            "codex": {"executable": "codex", "model": "gpt-5.5", "reasoning_effort": "xhigh"},
            "tasks": [
                {
                    "task": "task338",
                    "title": "test task",
                    "handoff_task_id": "00000000-0000-0000-0000-000000000001",
                    "prompt_file": "task338.md",
                }
            ],
        }
        cfg_path = tmp / "runner_config.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        return Runner(types.SimpleNamespace(config=str(cfg_path), init_state=True, dry_run=True, run_codex=False, sync_handoff=False))

    def test_throttle_intent_and_lock_include_handoff_owner(self):
        old_dir = throttle._DIR
        env_keys = [
            "NEUROGOLF_HANDOFF_LANE",
            "NEUROGOLF_HANDOFF_LANE_ID",
            "NEUROGOLF_HANDOFF_BATCH",
            "NEUROGOLF_HANDOFF_TASK",
            "NEUROGOLF_HANDOFF_TASK_NUM",
            "NEUROGOLF_HANDOFF_ROLE",
            "NEUROGOLF_HANDOFF_AGENT",
            "NEUROGOLF_HANDOFF_RUN_ID",
            "NEUROGOLF_HEAVY_GATE_N",
        ]
        old_env = {k: os.environ.get(k) for k in env_keys}
        with tempfile.TemporaryDirectory() as td:
            throttle._DIR = Path(td)
            os.environ.update(
                {
                    "NEUROGOLF_HANDOFF_LANE": "Codex",
                    "NEUROGOLF_HANDOFF_LANE_ID": "codex",
                    "NEUROGOLF_HANDOFF_BATCH": "99",
                    "NEUROGOLF_HANDOFF_TASK": "task338",
                    "NEUROGOLF_HANDOFF_TASK_NUM": "338",
                    "NEUROGOLF_HANDOFF_ROLE": "worker",
                    "NEUROGOLF_HANDOFF_AGENT": "worker1",
                    "NEUROGOLF_HANDOFF_RUN_ID": "run-338",
                    "NEUROGOLF_HEAVY_GATE_N": "8000",
                }
            )
            intent = throttle.write_intent("t338", "ab", 8000, task=338, candidate="candidate.onnx")
            self.assertIsNotNone(intent)
            intent_data = json.loads(Path(intent).read_text(encoding="utf-8"))
            self.assertEqual(intent_data["run_id"], "run-338")
            self.assertEqual(intent_data["task_num"], "338")
            slot = throttle.Slot("t338", kind="slot", check_cpu=False).acquire()
            try:
                sidecars = list(Path(td).glob("slot_*.lock.json"))
                self.assertEqual(len(sidecars), 1)
                lock_data = json.loads(sidecars[0].read_text(encoding="utf-8"))
                self.assertEqual(lock_data["batch"], "99")
                self.assertEqual(lock_data["heavy_n_threshold"], 8000)
            finally:
                slot.release()
                throttle.clear_intent(intent)
        throttle._DIR = old_dir
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_task_gate_cleanup_uses_owner_metadata_and_preserves_other_lanes(self):
        with tempfile.TemporaryDirectory() as td:
            runner = self.make_runner(Path(td))
            runner._process_table = lambda: [
                {"pid": 111, "ppid": 1, "name": "python.exe", "command_line": "python tools/ng_gate.py ab --task 338"},
                {"pid": 222, "ppid": 1, "name": "python.exe", "command_line": "python tools/ng_gate.py ab --task 338"},
                {"pid": 333, "ppid": 1, "name": "python.exe", "command_line": "python tools/ng_gate.py ab --task 213"},
            ]
            runner._gate_owner_by_pid = lambda: {
                111: [{"lane_id": "codex", "batch": "99", "task": "task338", "runner_id": runner.runner_id}],
                222: [{"lane_id": "codex2", "batch": "99", "task": "task338", "runner_id": "other"}],
                333: [{"lane_id": "codex", "batch": "99", "task": "task213", "runner_id": runner.runner_id}],
            }
            killed: list[int] = []
            runner._kill_pid_tree = lambda pid, reason, context=None: killed.append(pid) or {"pid": pid, "alive_after": False}

            report = runner.cleanup_task_gates("task338", "unit_test")

            self.assertEqual(killed, [111])
            self.assertEqual(report["killed"][0]["process"]["pid"], 111)
            registry = runner.process_registry_path.read_text(encoding="utf-8")
            self.assertIn('"kind":"orphan_sweep"', registry)
            self.assertIn('"kind":"cleanup_verified"', registry)

    def test_terminate_tree_kills_fake_sleeping_child_tree(self):
        with tempfile.TemporaryDirectory() as td:
            runner = self.make_runner(Path(td))
            code = textwrap.dedent(
                """
                import subprocess, sys, time
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                print(child.pid, flush=True)
                try:
                    time.sleep(60)
                finally:
                    child.terminate()
                """
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = int(proc.stdout.readline().strip())
            result = runner._terminate_tree(proc, {"task": "task338", "role": "worker", "run_id": "unit"})
            time.sleep(0.5)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

            self.assertFalse(runner._pid_running(proc.pid), result)
            self.assertFalse(runner._pid_running(child_pid), result)
            registry = runner.process_registry_path.read_text(encoding="utf-8")
            self.assertIn('"kind":"terminate_attempt"', registry)


if __name__ == "__main__":
    unittest.main()
