from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from tools import ng_gate


@unittest.skipUnless(os.name == "nt", "ng_gate queue watcher auto-open is Windows-only")
class TestNGGateQueueWatcher(unittest.TestCase):
    def setUp(self):
        self._old_dir = ng_gate.throttle._DIR
        self._old_popen = ng_gate.subprocess.Popen
        self._old_running = ng_gate._queue_watcher_running
        self._old_alive = ng_gate._watcher_pid_alive
        self._old_tk = ng_gate._tkinter_available
        self._old_env = {
            name: os.environ.get(name)
            for name in (
                "NEUROGOLF_GATE_WATCH",
                "NEUROGOLF_GATE_WATCH_TERMINAL",
                "NG_GATE_QUEUE_WATCHER",
            )
        }
        self.tmp = tempfile.TemporaryDirectory()
        ng_gate.throttle._DIR = pathlib.Path(self.tmp.name)
        ng_gate._queue_watcher_running = lambda: False
        ng_gate._watcher_pid_alive = lambda pid: pid == 12345
        ng_gate._tkinter_available = lambda: True
        for name in self._old_env:
            os.environ.pop(name, None)
        self.calls = []

        class Proc:
            pid = 12345

        def fake_popen(*args, **kwargs):
            self.calls.append((args, kwargs))
            return Proc()

        ng_gate.subprocess.Popen = fake_popen

    def tearDown(self):
        ng_gate.throttle._DIR = self._old_dir
        ng_gate.subprocess.Popen = self._old_popen
        ng_gate._queue_watcher_running = self._old_running
        ng_gate._watcher_pid_alive = self._old_alive
        ng_gate._tkinter_available = self._old_tk
        for name, value in self._old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()

    def _args(self, *, watch=False, no_watch=False):
        return SimpleNamespace(watch_queue=watch, no_watch_queue=no_watch)

    def test_default_auto_open_is_singleton(self):
        ng_gate._maybe_open_queue_watch(self._args())
        ng_gate._maybe_open_queue_watch(self._args())

        self.assertEqual(len(self.calls), 1)
        self.assertTrue((ng_gate.throttle._DIR / "queue_watch.pid").exists())

    def test_disable_env_and_no_watch_skip_launch(self):
        os.environ["NEUROGOLF_GATE_WATCH"] = "0"
        ng_gate._maybe_open_queue_watch(self._args())
        ng_gate._maybe_open_queue_watch(self._args(watch=True, no_watch=True))

        self.assertEqual(self.calls, [])

    def test_explicit_watch_overrides_disable_env(self):
        os.environ["NEUROGOLF_GATE_WATCH"] = "0"
        ng_gate._maybe_open_queue_watch(self._args(watch=True))

        self.assertEqual(len(self.calls), 1)

    def test_child_watcher_process_does_not_recurse(self):
        os.environ["NG_GATE_QUEUE_WATCHER"] = "1"
        ng_gate._maybe_open_queue_watch(self._args(watch=True))

        self.assertEqual(self.calls, [])

    def test_existing_launch_lock_skips_launch(self):
        lock = ng_gate.throttle._DIR / "queue_watch.launch.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("99999 now", encoding="utf-8")

        ng_gate._maybe_open_queue_watch(self._args())

        self.assertEqual(self.calls, [])
        self.assertTrue(lock.exists())

    def test_tk_unavailable_uses_terminal_snapshot(self):
        ng_gate._tkinter_available = lambda: False

        ng_gate._maybe_open_queue_watch(self._args())

        self.assertEqual(len(self.calls), 1)
        cmd = self.calls[0][0][0]
        self.assertIn("ng_gate_queue_snapshot.py", str(cmd))
        self.assertIn("--watch", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
