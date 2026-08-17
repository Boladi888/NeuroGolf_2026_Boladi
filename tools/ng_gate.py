#!/usr/bin/env python3
"""ng_gate — the one shared per-task verification harness. Replaces hand-rolled
ab_gate / synth_gate / final_battery / argen_verify scripts.

  py tools/ng_gate.py <mode> --task NNN --cand cand.onnx [opts]

Modes: visible | ab | truth | synth | adversarial | battery | fill-corpus | corpus-info

Orphan-safe: arms a Windows kill-on-close JobObject (killjob.py) so the whole
worker tree dies when this process exits HOWEVER it exits. With --new-window the
run opens in its own visible console that breaks away from the caller's job, so
closing that window kills the run (and the caller exiting does NOT). Self-throttles
to ~80% CPU across all agents, and serializes heavy full-corpus gates to
NEUROGOLF_HEAVY_GATE_K (default 5) so many agents can't melt the box at once
(throttle.py). Corpus is the cached, deterministic ARC-GEN set on the Q: cache
drive (corpus.py).

Design + decisions: tools/ng_harness/PLAN.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NG_ROOT))

from tools.ng_harness import killjob, throttle, corpus, modes  # noqa: E402

_CREATE_NEW_CONSOLE = 0x00000010
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_TRUTHY = {"1", "true", "yes", "on", "watch", "auto"}
_FALSEY = {"0", "false", "no", "off", "none", "never", "disable", "disabled"}
_WATCHER_SCRIPTS = {"ng_gate_queue_window.py", "ng_gate_queue_snapshot.py"}
_HEAVY_GATE_MODES = {"ab", "battery", "synth", "truth"}
_DEFAULT_HEAVY_GATE_N = 8000
_DEFAULT_GATE_MAX_SECONDS = 900.0
_TIMEOUT_DIR = NG_ROOT / "Handoff" / "private" / "ng_gate_timeouts"


def _pid_identity(pid: int) -> float | None:
    psutil = getattr(throttle, "psutil", None)
    if psutil is None:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001
        return None


def _same_pid_alive(pid: int, create_time: float | None) -> bool:
    if pid <= 0:
        return False
    psutil = getattr(throttle, "psutil", None)
    if psutil is None:
        return throttle._pid_alive(pid)
    try:
        proc = psutil.Process(pid)
        if create_time is None:
            return True
        return abs(float(proc.create_time()) - create_time) < 0.01
    except Exception:  # noqa: BLE001
        return False


def _hard_exit_later(
    seconds: float | None,
    reason: str,
    on_timeout=None,
) -> threading.Timer | None:
    if seconds is None or seconds <= 0:
        return None

    def _exit() -> None:
        try:
            if on_timeout is not None:
                try:
                    on_timeout()
                except Exception:  # noqa: BLE001
                    pass
            print(f"[ng_gate] {reason}; exiting to avoid orphaned gate work", file=sys.stderr, flush=True)
        finally:
            os._exit(124)

    timer = threading.Timer(float(seconds), _exit)
    timer.daemon = True
    timer.start()
    return timer


def _arm_parent_watchdog() -> None:
    if os.environ.get("NG_IN_WINDOW") or os.environ.get("NG_GATE_QUEUE_WATCHER"):
        return
    if not _env_flag("NEUROGOLF_GATE_PARENT_WATCH", True):
        return
    parent_pid = os.getppid()
    if parent_pid <= 0:
        return
    parent_create_time = _pid_identity(parent_pid)

    def _watch() -> None:
        while True:
            time.sleep(5.0)
            if not _same_pid_alive(parent_pid, parent_create_time):
                try:
                    print(
                        f"[ng_gate] parent pid {parent_pid} exited; self-terminating",
                        file=sys.stderr,
                        flush=True,
                    )
                finally:
                    os._exit(124)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()


def _fill_worker(args):
    """Module-level worker for parallel fill-corpus (separate process => own RNG
    state => deterministic). Returns (task, info_or_None, err_or_None)."""
    task, n, max_seconds = args
    try:
        from tools.ng_harness import corpus
        corpus.build(task, n, max_seconds=max_seconds)
        return (task, corpus.cache_info(task), None)
    except Exception as e:  # noqa: BLE001
        return (task, None, f"{type(e).__name__}: {e}")


def _set_title(t: str):
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(t)
        except Exception:  # noqa: BLE001
            pass


def _respawn_in_window(argv) -> int:
    """Launch this same command in a fresh visible console that breaks away from
    the caller's job (so the caller can exit without killing it, but closing the
    window kills it via the child's own killjob). Fire-and-forget by default."""
    env = dict(os.environ, NG_IN_WINDOW="1")
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve())] + [a for a in argv if a != "--new-window"]
    # Prefer breakaway (so closing THIS caller doesn't kill the window), but fall
    # back gracefully if the caller's job forbids breakaway, then to no flags.
    attempts = ([_CREATE_NEW_CONSOLE | _CREATE_BREAKAWAY_FROM_JOB, _CREATE_NEW_CONSOLE, 0]
                if os.name == "nt" else [0])
    last = None
    for flags in attempts:
        try:
            proc = subprocess.Popen(cmd, creationflags=flags, env=env, close_fds=True)
            note = "" if (flags & _CREATE_BREAKAWAY_FROM_JOB) else " (no breakaway: tied to this caller's job)"
            print(f"[ng_gate] launched in new window, pid={proc.pid}{note}. Close that window to kill it.")
            return 0
        except OSError as e:  # noqa: PERF203
            last = e
    print(f"[ng_gate] could not open a new window ({last}); rerun without --new-window.")
    return 2


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _candidate_identity(cand: str) -> tuple[str, str]:
    raw = pathlib.Path(cand)
    try:
        path = raw if raw.is_absolute() else (NG_ROOT / raw)
        path = path.resolve()
    except Exception:  # noqa: BLE001
        path = raw
    path_text = str(path)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest, path_text
    except Exception:  # noqa: BLE001
        digest = hashlib.sha256(path_text.lower().encode("utf-8", "replace")).hexdigest()
        return f"path-{digest}", path_text


def _timeout_marker_path(task: int, cand: str) -> pathlib.Path:
    digest, _ = _candidate_identity(cand)
    return _TIMEOUT_DIR / f"task{int(task):03d}_{digest[:24]}.json"


def _read_timeout_marker(args: argparse.Namespace) -> dict | None:
    marker = _timeout_marker_path(args.task, args.cand)
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return {"marker": str(marker), "unreadable": True}


def _write_timeout_marker(args: argparse.Namespace, n: int, reason: str) -> None:
    digest, cand_path = _candidate_identity(args.cand)
    payload = {
        "schema": "neurogolf.ng_gate.timeout_marker.v1",
        "task": int(args.task),
        "mode": args.mode,
        "n": int(n),
        "candidate": args.cand,
        "candidate_path": cand_path,
        "candidate_identity": digest,
        "reason": reason,
        "pid": os.getpid(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _TIMEOUT_DIR.mkdir(parents=True, exist_ok=True)
    marker = _timeout_marker_path(args.task, args.cand)
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(marker)


def _apply_timeout_marker_policy(args: argparse.Namespace, n: int) -> str | None:
    if _env_flag("NEUROGOLF_GATE_ALLOW_LONG_BUDGET", False):
        return None
    marker = _read_timeout_marker(args)
    if not marker:
        return None
    mode = marker.get("mode", "unknown")
    prior_n = marker.get("n", "unknown")
    return (
        "gate_timeout_marker: prior wall budget timeout for "
        f"task{int(args.task):03d} candidate; previous gate={mode} n={prior_n}; "
        "stop this candidate as gate-timeout/pathological"
    )


def _apply_gate_budget_policy(args: argparse.Namespace, n: int) -> str | None:
    """Keep shared HandOff verification from turning into hour-plus CPU burns."""
    if _env_flag("NEUROGOLF_GATE_ALLOW_LONG_BUDGET", False):
        return None

    cap = _env_float("NEUROGOLF_GATE_MAX_SECONDS", _DEFAULT_GATE_MAX_SECONDS)
    if args.no_budget:
        return (
            "gate_budget_violation: --no-budget is not allowed for "
            f"{args.mode} n={n}; max allowed is {cap:g}s"
        )
    if args.max_seconds > cap:
        return (
            f"gate_budget_violation: requested max_seconds={args.max_seconds:g}s "
            f"for {args.mode} n={n}; max allowed is {cap:g}s"
        )
    return None


def _watcher_pid_alive(pid: int) -> bool:
    psutil = getattr(throttle, "psutil", None)
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).lower()
            cwd = str(proc.cwd() or "").lower()
            root = str(NG_ROOT).lower()
            return (root in cmdline or cwd == root) and any(script in cmdline for script in _WATCHER_SCRIPTS)
        except psutil.NoSuchProcess:
            return False
        except Exception:  # noqa: BLE001
            return throttle._pid_alive(pid)
    return throttle._pid_alive(pid)


def _queue_watcher_running() -> bool:
    psutil = getattr(throttle, "psutil", None)
    if psutil is None:
        return False
    try:
        root = str(NG_ROOT).lower()
        for proc in psutil.process_iter(["cmdline", "cwd"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                cwd = str(proc.info.get("cwd") or "").lower()
            except Exception:  # noqa: BLE001
                continue
            if (root in cmdline or cwd == root) and any(script in cmdline for script in _WATCHER_SCRIPTS):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _tkinter_available() -> bool:
    try:
        chk = subprocess.run(
            [sys.executable, "-c", "import tkinter"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return chk.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _take_watch_launch_lock(lock_file: pathlib.Path) -> int | None:
    now = time.time()
    try:
        age = now - lock_file.stat().st_mtime
        if age > 30:
            lock_file.unlink()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        return None
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {now:.0f}".encode("ascii", "replace"))
        return fd
    except FileExistsError:
        return None
    except Exception:  # noqa: BLE001
        return None


def _release_watch_launch_lock(fd: int | None, lock_file: pathlib.Path) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass
    try:
        lock_file.unlink()
    except Exception:  # noqa: BLE001
        pass


def _maybe_open_queue_watch(args) -> None:
    """Open one visible queue watcher for future/current gate runs.

    Default is on for Windows ng_gate runs. Use --no-watch-queue or
    NEUROGOLF_GATE_WATCH=0 to disable. A PID sentinel plus launch lock prevents
    a pile of watcher windows when many agents start gates at once.
    """
    want = bool(getattr(args, "watch_queue", False)) or _env_flag("NEUROGOLF_GATE_WATCH", True)
    if bool(getattr(args, "no_watch_queue", False)):
        want = False
    if not want or os.name != "nt" or os.environ.get("NG_GATE_QUEUE_WATCHER"):
        return
    lock_fd = None
    lock_file = None
    try:
        throttle._DIR.mkdir(parents=True, exist_ok=True)
        pid_file = throttle._DIR / "queue_watch.pid"
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text(encoding="utf-8", errors="replace").split()[0])
            except Exception:
                old_pid = 0
            if old_pid and _watcher_pid_alive(old_pid):
                return
            try:
                pid_file.unlink()
            except Exception:
                pass
        if _queue_watcher_running():
            return
        lock_file = throttle._DIR / "queue_watch.launch.lock"
        lock_fd = _take_watch_launch_lock(lock_file)
        if lock_fd is None:
            return
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text(encoding="utf-8", errors="replace").split()[0])
            except Exception:
                old_pid = 0
            if old_pid and _watcher_pid_alive(old_pid):
                return
        if _queue_watcher_running():
            return
        interval = os.environ.get("NEUROGOLF_GATE_WATCH_INTERVAL", "5")
        watch_exe = sys.executable
        watch_script = NG_ROOT / "tools" / "ng_gate_queue_snapshot.py"
        watch_args = ["--watch", "--interval", str(interval)]
        pythonw = pathlib.Path(sys.executable).with_name("pythonw.exe")
        if (pythonw.exists() and _tkinter_available()
                and os.environ.get("NEUROGOLF_GATE_WATCH_TERMINAL", "").lower() not in _TRUTHY):
            watch_exe = str(pythonw)
            watch_script = NG_ROOT / "tools" / "ng_gate_queue_window.py"
            watch_args = ["--interval", str(interval)]
        cmd = [watch_exe, str(watch_script)] + watch_args
        env = dict(os.environ, NG_GATE_QUEUE_WATCHER="1")
        flags_to_try = [_CREATE_NEW_CONSOLE | _CREATE_BREAKAWAY_FROM_JOB, _CREATE_NEW_CONSOLE, 0]
        last = None
        for flags in flags_to_try:
            try:
                proc = subprocess.Popen(cmd, cwd=str(NG_ROOT), env=env, creationflags=flags, close_fds=True)
                pid_file.write_text(f"{proc.pid} {time.time():.0f}", encoding="utf-8")
                note = "" if (flags & _CREATE_BREAKAWAY_FROM_JOB) else " (no breakaway)"
                print(f"[ng_gate] queue watcher opened, pid={proc.pid}{note}. Close that watcher window to stop it.")
                return
            except OSError as e:
                last = e
        print(f"[ng_gate] queue watcher could not open ({last}). You can run NG_Gate_Queue_Watch.bat manually.")
    except Exception as e:  # noqa: BLE001
        print(f"[ng_gate] queue watcher skipped: {type(e).__name__}: {e}")
    finally:
        if lock_fd is not None and lock_file is not None:
            _release_watch_launch_lock(lock_fd, lock_file)


def _emit(result: dict, as_json: bool, out_path: str | None):
    if out_path:
        pathlib.Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(result)


def _print_human(r: dict):
    m = r.get("mode", "?")
    if r.get("error"):
        print(f"[{m}] ERROR: {r['error']}")
        if r.get("trace"):
            print("        " + " | ".join(r["trace"]))
        return
    if m == "visible":
        print(f"[visible] {r['correct']}/{r['total']} ({r['pct']:.2f}%)  {'PASS' if r['pass'] else 'FAIL'}")
    elif m == "truth":
        print(f"[truth] vs ARC-GEN ground truth: {r['correct']}/{r['total']} ({r['pct']:.3f}%)")
        if "ref_agree_pct" in r:
            print(f"        cand vs ref agreement: {r['ref_agree']}/{r['total']} ({r['ref_agree_pct']:.3f}%)")
    elif m == "ab":
        if r.get("error"):
            print(f"[ab] ERROR: {r['error']}")
        else:
            print(f"[ab] vs {r['ref_source']}: match {r['match']}/{r['total']} | "
                  f"divergences={r['divergences']} | byte_equal={r['byte_equal']} "
                  f"(first_div={r['first_divergence']})")
    elif m == "synth":
        if r.get("error"):
            print(f"[synth] ERROR: {r['error']}")
        else:
            print(f"[synth] vs numpy ref: {r['agree']}/{r['total']} ({r['pct']:.2f}%)")
    elif m == "adversarial":
        print(f"[adversarial] {json.dumps({k:v for k,v in r.items() if k!='mode'}, default=str)[:400]}")
    elif m == "battery":
        print(f"=== battery task{r['task']:03d}: {r['verdict']} ===")
        _print_human(r["visible"]); _print_human(r["truth"]); _print_human(r["ab"])
        if r.get("score"):
            s = r["score"]; print(f"[score] cost={s['cost']} (mem {s['memory']}+par {s['params']})  points={s['points']:.3f}")
    else:
        print(json.dumps(r, indent=2, default=str))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="ng_gate — shared NeuroGolf verification harness")
    p.add_argument("mode", choices=list(modes.DISPATCH) + ["fill-corpus", "corpus-info"])
    p.add_argument("--task", type=int, help="task number 1..400")
    p.add_argument("--tasks", type=str, help="range/list for fill-corpus, e.g. 1-400 or 18,110,324")
    p.add_argument("--cand", type=str, help="candidate onnx")
    p.add_argument("--ref", type=str, default=None, help="reference onnx (ab/truth); default=auto parent")
    p.add_argument("--synth-ref", type=str, default=None, help="numpy ref module (solve(grid)->grid) for synth")
    p.add_argument("--n", type=int, default=2000, help="corpus pairs (default 2000; --full overrides)")
    p.add_argument("--full", action="store_true", help="use the full cap (200k)")
    p.add_argument("--early-exit", action="store_true", help="ab: stop at first divergence")
    p.add_argument("--no-sanitize", action="store_true", help="skip neurogolf_utils.sanitize_model")
    p.add_argument("--no-throttle", action="store_true")
    p.add_argument("--watch-queue", action="store_true", help="open one visible ng_gate queue watcher for this run")
    p.add_argument("--no-watch-queue", action="store_true", help="do not auto-open the visible ng_gate queue watcher")
    p.add_argument("--new-window", action="store_true", help="run in a fresh visible console (kills on close)")
    p.add_argument("--jobs", type=int, default=None, help="fill-corpus: parallel worker processes (default=80%% cores)")
    p.add_argument("--max-seconds", type=float, default=900.0,
                   help="wall budget for active gate work; fill-corpus applies it per task (default 900)")
    p.add_argument("--no-budget", action="store_true",
                   help="disable the wall budget for fill-corpus; per-candidate gates reject this without NEUROGOLF_GATE_ALLOW_LONG_BUDGET=1")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=str, default=None, help="write result JSON here")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    # Relaunch in a visible window if asked (and not already the windowed child).
    if args.new_window and not os.environ.get("NG_IN_WINDOW"):
        return _respawn_in_window(argv)

    _maybe_open_queue_watch(args)

    # Orphan safety: bind this process (and all children) to a kill-on-close job.
    killjob.arm()
    _arm_parent_watchdog()

    n = corpus.MAX_N if args.full else args.n
    in_window = bool(os.environ.get("NG_IN_WINDOW"))

    # ---- corpus utility modes ----
    if args.mode == "corpus-info":
        tasks = _parse_tasks(args)
        for t in tasks:
            print(corpus.cache_info(t))
        return 0
    if args.mode == "fill-corpus":
        import time as _time
        tasks = _parse_tasks(args)
        jobs = max(1, min(args.jobs if args.jobs else throttle._k(), len(tasks)))
        msec = None if args.no_budget else args.max_seconds
        print(f"[fill-corpus] {len(tasks)} tasks, n={n}, jobs={jobs}, max_seconds={msec} "
              f"(parallel; own-process workers = deterministic; checkpointed; slow gens save partial)", flush=True)
        t0 = _time.time(); done = 0; partials = []
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_fill_worker, (t, n, msec)): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        task, info, err = fut.result()
                    except Exception as e:  # noqa: BLE001 -- worker process died (crash/kill)
                        task, info, err = futs[fut], None, f"worker died: {type(e).__name__}: {e}"
                    done += 1
                    if not err and isinstance(info, dict) and info.get("cached", 0) < n:
                        partials.append((task, info.get("cached")))
                    _set_title(f"fill-corpus {done}/{len(tasks)} jobs={jobs} {(_time.time()-t0)/60:.0f}m")
                    print(f"  [{done}/{len(tasks)}] task{task:03d}: {'ERROR ' + err if err else info}", flush=True)
        else:
            for t in tasks:
                done += 1
                _set_title(f"fill-corpus {done}/{len(tasks)} task{t:03d}")
                task, info, err = _fill_worker((t, n, msec))
                if not err and isinstance(info, dict) and info.get("cached", 0) < n:
                    partials.append((t, info.get("cached")))
                print(f"  [{done}/{len(tasks)}] task{t:03d}: {'ERROR ' + err if err else info}", flush=True)
        if partials:
            print(f"[fill-corpus] {len(partials)} task(s) below n (slow generators, partial cached): {partials}", flush=True)
        print(f"[fill-corpus] done {len(tasks)} tasks in {(_time.time()-t0)/60:.1f} min", flush=True)
        if in_window:
            input("\n[fill-corpus done] press Enter to close window...")
        return 0

    # ---- per-candidate modes ----
    if not args.task or not args.cand:
        p.error("--task and --cand are required for this mode")
    timeout_marker_error = _apply_timeout_marker_policy(args, n)
    if timeout_marker_error:
        result = {"mode": args.mode, "task": args.task, "error": timeout_marker_error}
        print(f"[ng_gate] {timeout_marker_error}", file=sys.stderr, flush=True)
        _emit(result, args.json, args.out)
        return 124
    budget_policy_error = _apply_gate_budget_policy(args, n)
    if budget_policy_error:
        result = {"mode": args.mode, "task": args.task, "error": budget_policy_error}
        print(f"[ng_gate] {budget_policy_error}", file=sys.stderr, flush=True)
        _emit(result, args.json, args.out)
        return 124
    _set_title(f"ng_gate {args.mode} task{args.task:03d} n={n}")
    opts = dict(n=n, ref=args.ref, synth_ref=args.synth_ref,
                early_exit=args.early_exit, sanitize=not args.no_sanitize, verbose=args.verbose)
    intent = None
    slot = None
    budget_timer = None
    try:
        if not args.no_throttle:
            intent = throttle.write_intent(f"t{args.task}", args.mode, n, task=args.task, candidate=args.cand)
            slot = throttle.acquire_gate(args.mode, n, f"t{args.task}")
        if not args.no_budget:
            budget_reason = f"wall budget {args.max_seconds:g}s exceeded for {args.mode} task{args.task:03d}"
            budget_timer = _hard_exit_later(
                args.max_seconds,
                budget_reason,
                on_timeout=lambda: _write_timeout_marker(args, n, budget_reason),
            )
        result = modes.DISPATCH[args.mode](args.task, args.cand, **opts)
    except Exception as e:  # noqa: BLE001
        import traceback
        result = {"mode": args.mode, "error": f"{type(e).__name__}: {e}",
                  "trace": traceback.format_exc().splitlines()[-3:]}
    finally:
        if budget_timer:
            budget_timer.cancel()
        if slot:
            slot.release()
        throttle.clear_intent(intent)
    _emit(result, args.json, args.out)
    if in_window:
        input("\n[done] press Enter to close window...")
    # exit code: 1 if a clear failure
    bad = (result.get("error")
           or (args.mode == "visible" and not result.get("pass"))
           or (args.mode == "ab" and not result.get("byte_equal"))
           or (args.mode == "battery" and result.get("verdict") != "PASS"))
    return 1 if bad else 0


def _parse_tasks(args):
    if args.task:
        return [args.task]
    if args.tasks:
        out = []
        for part in args.tasks.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
        return sorted(set(out))
    return list(range(1, 401))


if __name__ == "__main__":
    raise SystemExit(main())
