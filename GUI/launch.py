#!/usr/bin/env python3
"""NeuroGolf Lab windowed launcher with orphan-proof shutdown.

Runs uvicorn IN-PROCESS in the console window that hosts it, so closing the
window terminates the server (standard Windows console semantics). On top of
that, the process binds ITSELF to a Windows JobObject with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE — the same pattern as
Q:/AIz/Intel_Arc_Stuff/Iarc_Recorder — so if anything ever spawns helper
children, the kernel terminates them the instant this process dies, no matter
HOW it dies (window close, taskkill /F, logoff). No orphans.

Usage:
    python launch.py            # serve on HOST/PORT from .env (default 8082)
    python launch.py --port N   # override port
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JobObjectExtendedLimitInformation = 9


def bind_self_to_kill_job() -> bool:
    """Create a kill-on-close JobObject and assign THIS process to it.

    The handle is intentionally leaked: the kernel closes it when this process
    exits, which triggers KILL_ON_JOB_CLOSE on every process in the job
    (children inherit membership). Returns True when bound.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

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

        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return False
        info = _JELI()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_BREAKAWAY_OK
        )
        if not kernel32.SetInformationJobObject(
            handle, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(handle)
            return False
        if not kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(handle)
            return False
        return True
    except Exception:  # noqa: BLE001 - launcher must never die on safety plumbing
        return False


def set_console_title(title: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:  # noqa: BLE001
            pass


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    os.chdir(ROOT)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run NeuroGolf Lab in this console window.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8082")))
    args = parser.parse_args()

    if port_in_use(args.host, args.port):
        print(f"ERROR: {args.host}:{args.port} is already in use.")
        print("Another NeuroGolf Lab (or something else) is listening there.")
        print("Close that window first, or run: python launch.py --port <other>")
        return 1

    bound = bind_self_to_kill_job()
    set_console_title(f"NeuroGolf Lab :{args.port}")
    print(f"NeuroGolf Lab  ->  http://{args.host}:{args.port}")
    print(f"Orphan guard: kill-on-close job {'ACTIVE' if bound else 'unavailable (plain console kill only)'}")
    print("Closing this window stops the server. Ctrl+C also works.")
    print()

    import uvicorn

    uvicorn.run("server:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
