"""Orphan-kill contract for ng_harness.killjob (ported from Iarc_Recorder's
test_orphan_kill.ps1): when the launcher dies HOWEVER it dies, the kill-on-close
JobObject must terminate every bound child. This is the task018-orphan fix.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

NG_ROOT = Path(__file__).resolve().parent.parent

_LAUNCHER = (
    "import sys; sys.path.insert(0, r'{root}')\n"
    "from tools.ng_harness import killjob\n"
    "import subprocess, sys, time\n"
    "killjob.arm()\n"                                    # bind launcher (+children) to kill-job
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "print(c.pid, flush=True)\n"                          # tell the test the child PID
    "time.sleep(120)\n"
).format(root=str(NG_ROOT))


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0); return True
        except Exception:
            return False


class TestOrphanKill(unittest.TestCase):
    # This spawns a launcher that arms a KILL_ON_JOB_CLOSE JobObject and then kills it.
    # On a CI runner - which manages its own step process tree with a job object, and
    # where the launcher's job could not break away - that reaped pytest itself: the run
    # stopped dead at this test with 396 passed / 0 failed and a non-zero exit. Opt in
    # on a machine where taking out a process tree is acceptable.
    @unittest.skipUnless(
        os.environ.get("NEUROGOLF_TEST_KILLJOB") == "1",
        "kills a process tree; opt in with NEUROGOLF_TEST_KILLJOB=1",
    )
    def test_orphan_kill_on_launcher_death(self):
        """Kill ONLY the launcher (not /T); the bound child must die via the job."""
        if os.name != "nt":
            self.skipTest("JobObject is Windows-only")
        launcher = subprocess.Popen([sys.executable, "-c", _LAUNCHER],
                                    stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(launcher.stdout.readline().strip())
            self.assertTrue(_alive(child_pid), "child should be alive after launch")
            # simulate window-close / crash: terminate the launcher process ONLY
            launcher.kill()
            launcher.wait(timeout=10)
            # KILL_ON_JOB_CLOSE should have reaped the child
            deadline = time.time() + 8
            while time.time() < deadline and _alive(child_pid):
                time.sleep(0.25)
            self.assertFalse(_alive(child_pid), f"ORPHAN: child {child_pid} survived launcher death")
        finally:
            if launcher.poll() is None:
                launcher.kill()
            try:
                launcher.stdout.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
