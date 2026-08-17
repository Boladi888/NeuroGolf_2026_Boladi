"""Windows JobObject kill-on-close — orphan safety for the test harness.

Ported from Q:\\AIz\\Intel_Arc_Stuff\\Iarc_Recorder\\start_iarc_recorded_run.py.

Why: agents have left verification sweeps running detached long after the owning
session ended (the task018 incident, 2026-05-31), burning CPU until hunted down.
With JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, the Windows kernel terminates every
process bound to the job the instant the job handle goes away — which happens
when the launcher process exits HOWEVER it exits (window closed, taskkill,
parent shell death, crash). BREAKAWAY_OK lets a watchdog spawn outside the job.

Usage (in the launcher, e.g. ng_gate.py):
    from tools.ng_harness import killjob
    killjob.arm()              # create the job + bind this process (and thus all children)
    ...                        # spawn workers / run inference; everything inherits the job
    # on ANY exit, the kernel kills the whole tree. No orphans.

On non-Windows or any failure, all functions are graceful no-ops (return False/None)
and an atexit fallback explicitly terminates tracked child PIDs.
"""
from __future__ import annotations

import atexit
import os
import sys

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JobObjectExtendedLimitInformation = 9

_JOB_HANDLE = None          # kept alive for process lifetime (kernel closes on exit)
_ARMED = False
_SELF_BOUND = False
_TRACKED_PIDS: set[int] = set()   # fallback termination targets if job binding failed


def _is_windows() -> bool:
    return os.name == "nt"


def _make_job():
    """Create the kill-on-close job object. Returns handle or None."""
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _JBLI(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_void_p),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JELI(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JBLI),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL

        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = _JELI()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        if not k32.SetInformationJobObject(
                handle, _JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(handle)
            return None
        _JOB_HANDLE = handle
        return handle
    except Exception:
        return None


def attach_pid(pid: int) -> bool:
    """Bind a running PID to the kill-on-close job. Best-effort."""
    if pid and pid > 0:
        _TRACKED_PIDS.add(int(pid))
    job = _make_job()
    if job is None or not pid or pid <= 0 or not _is_windows():
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
        if not h:
            return False
        try:
            return bool(k32.AssignProcessToJobObject(job, h))
        finally:
            k32.CloseHandle(h)
    except Exception:
        return False


def arm() -> bool:
    """Create the kill-job and bind THIS process to it, so every child and
    grandchild inherits the job and dies when this launcher exits. Idempotent.
    Returns True if the job is armed (self bound or at least created)."""
    global _ARMED, _SELF_BOUND
    if _ARMED:
        return _SELF_BOUND
    _ARMED = True
    atexit.register(_terminate_tracked)   # belt-and-suspenders fallback
    job = _make_job()
    if job is None:
        return False
    if not _is_windows():
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        _SELF_BOUND = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
        return _SELF_BOUND
    except Exception:
        return False


def status() -> dict:
    return {"windows": _is_windows(), "armed": _ARMED, "job_created": _JOB_HANDLE is not None,
            "self_bound": _SELF_BOUND, "tracked_pids": sorted(_TRACKED_PIDS)}


def _terminate_tracked():
    """atexit fallback: explicitly kill tracked child PIDs if the job didn't bind
    them (e.g. self-bind failed). Normally a no-op because KILL_ON_JOB_CLOSE
    already handled it."""
    if _SELF_BOUND:
        return  # the job owns the whole tree; nothing to do
    if not _TRACKED_PIDS:
        return
    try:
        import psutil
        for pid in list(_TRACKED_PIDS):
            try:
                p = psutil.Process(pid)
                for c in p.children(recursive=True):
                    c.kill()
                p.kill()
            except Exception:
                pass
    except Exception:
        # last resort without psutil
        for pid in list(_TRACKED_PIDS):
            try:
                if _is_windows():
                    os.system(f"taskkill /F /T /PID {pid} >nul 2>&1")
                else:
                    os.kill(pid, 9)
            except Exception:
                pass


if __name__ == "__main__":
    # self-test: arm, print status
    ok = arm()
    print("arm() ->", ok)
    print("status ->", status())
