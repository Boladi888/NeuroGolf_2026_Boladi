#!/usr/bin/env python
"""Hang-safe, resumable ARC-GEN corpus filler for NeuroGolf.

WHY THIS EXISTS
---------------
Some ARC-GEN generators infinite-loop on specific RNG seeds. `corpus.build()`
has no per-`gen()` timeout, so a single bad seed freezes the whole fill (this is
what left task076/324/018 stuck and what froze naive fill probes). This tool
wraps generation in a worker subprocess that writes a heartbeat (current raw
seed) BEFORE each `gen()` call. A watchdog kills + skips any seed whose
heartbeat stalls past --hang-timeout, records it in a durable skip-set, and
resumes from the last checkpoint. With no hangs it behaves like a normal fill,
so it is safe to use for every task.

STATE
-----
Durable (next to the cache, survives a session/job teardown):
  <CACHE>/_fill_state/skip_<t>.json   seeds that hang (reused across runs)
  <CACHE>/_fill_state/fill.log        append-only progress log
  <CACHE>/_fill_state/manager.pid     manager PID (to stop it)
Local/fast (heartbeats only; ephemeral, regenerated):
  <TEMP>/ng_fill_hb/hb_<t>.json

USAGE
-----
  python tools/ng_fill_hang_safe.py --all                 # fill every task < target
  python tools/ng_fill_hang_safe.py --tasks 18,76,157     # fill specific tasks
  python tools/ng_fill_hang_safe.py --status              # print progress, exit
  python tools/ng_fill_hang_safe.py --stop                # signal manager to stop
Re-running is safe + resumable: full tasks are skipped, partial tasks continue
from their checkpoint, and known hang-seeds are remembered. Idempotent.
"""
import argparse, json, os, sys, time, threading, subprocess, queue, zipfile, re, tempfile, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure tools/ importable
import numpy as np
from ng_harness import corpus, registry, encode

STATE = corpus.CACHE_DIR / "_fill_state"
HBDIR = pathlib.Path(tempfile.gettempdir()) / "ng_fill_hb"


def _ensure_dirs():
    STATE.mkdir(parents=True, exist_ok=True)
    HBDIR.mkdir(parents=True, exist_ok=True)


def _skip_path(t):
    return STATE / f"skip_{t}.json"


def _hb_path(t):
    return HBDIR / f"hb_{t}.json"


def load_skip(t):
    try:
        return set(int(x) for x in json.load(open(_skip_path(t))))
    except Exception:
        return set()


def add_skip(t, seed):
    s = load_skip(t)
    s.add(int(seed))
    tmp = STATE / f"skip_{t}.tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(s), f)
    os.replace(tmp, _skip_path(t))
    return s


def fast_count(t):
    """Pairs cached, read from the .npy header (no full decompress/load)."""
    p = corpus.CACHE_DIR / f"task{t:03d}.npz"
    if not p.exists():
        return 0
    try:
        with zipfile.ZipFile(p) as z:
            head = z.open("in_idx.npy").read(128)
        m = re.search(rb"'shape':\s*\((\d+)", head)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(STATE / "fill.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- worker
def run_worker(task, target, ckpt):
    import random
    gen, err = registry.generator(task)
    if err:
        print(f"UNAVAILABLE {err}", flush=True)
        return 2
    d = corpus._load(task)
    in_idx = list(d["in_idx"]) if d is not None else []
    in_sh = list(d["in_shapes"]) if d is not None else []
    out_idx = list(d["out_idx"]) if d is not None else []
    out_sh = list(d["out_shapes"]) if d is not None else []
    n_raw = int(d["n_raw"]) if d is not None else 0
    skip = load_skip(task)
    hbp = _hb_path(task)

    def pack():
        return {"in_idx": np.asarray(in_idx, np.uint8), "in_shapes": np.asarray(in_sh, np.uint16),
                "out_idx": np.asarray(out_idx, np.uint8), "out_shapes": np.asarray(out_sh, np.uint16),
                "n_raw": np.int64(n_raw), "seed_offset": np.int64(corpus.SEED_OFFSET)}

    valid = len(in_idx)
    last = valid
    last_save_t = time.time()
    TIME_CKPT = 300.0  # also bank progress every 5 min — slow generators (<2 pairs/s)
    # may otherwise never reach the pair-based checkpoint before a kill/stop
    while valid < target:
        if n_raw in skip:
            n_raw += 1
            continue
        try:
            with open(hbp, "w") as f:
                json.dump({"n_raw": n_raw, "valid": valid, "t": time.time()}, f)
        except Exception:
            pass
        random.seed(corpus.SEED_OFFSET + n_raw)
        n_raw += 1
        try:
            ex = gen()
        except Exception:
            continue
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            continue
        ii, ish = encode.grid_to_index(ex["input"])
        oi, osh = encode.grid_to_index(ex["output"])
        if ii is None or oi is None:
            continue
        in_idx.append(ii); in_sh.append(ish); out_idx.append(oi); out_sh.append(osh)
        valid += 1
        if valid - last >= ckpt or (valid > last and time.time() - last_save_t >= TIME_CKPT):
            corpus._save(task, pack()); last = valid; last_save_t = time.time()
    corpus._save(task, pack())
    print(f"WORKER_DONE {valid}", flush=True)
    return 0


# ---------------------------------------------------------------- watchdog (one task)
def fill_one(task, target, hang_timeout, ckpt, stop_evt):
    py = sys.executable
    self = os.path.abspath(__file__)
    while not stop_evt.is_set():
        cur = fast_count(task)
        if cur >= target:
            log(f"task{task:03d}: complete ({cur})")
            return
        try:
            os.remove(_hb_path(task))
        except OSError:
            pass
        w = subprocess.Popen([py, self, "--mode", "worker", "--task", str(task),
                              "--target", str(target), "--ckpt", str(ckpt)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last_seed = None
        last_change = time.time()
        while not stop_evt.is_set():
            if w.poll() is not None:
                break
            try:
                hb = json.load(open(_hb_path(task)))
            except Exception:
                hb = None
            now = time.time()
            if hb is not None:
                if hb["n_raw"] != last_seed:
                    last_seed = hb["n_raw"]; last_change = now
                elif now - last_change > hang_timeout:
                    # a long checkpoint write is NOT a hang — the .tmp.npz mtime
                    # advances while _save streams the compressed arrays out
                    tmp_p = corpus.CACHE_DIR / f"task{task:03d}.tmp.npz"
                    try:
                        saving = tmp_p.exists() and (now - tmp_p.stat().st_mtime) < hang_timeout
                    except OSError:
                        saving = False
                    if saving:
                        last_change = now
                    else:
                        w.kill()
                        add_skip(task, last_seed)
                        log(f"task{task:03d}: HANG seed {last_seed} -> skipped (pairs~{hb.get('valid')})")
                        break
            time.sleep(2)
        if stop_evt.is_set():
            w.kill()
            return
        w.wait()
    try:
        os.remove(str(corpus.CACHE_DIR / f"task{task:03d}.tmp.npz"))
    except OSError:
        pass


# ---------------------------------------------------------------- manager
def manager(tasks, target, workers, hang_timeout, ckpt):
    _ensure_dirs()
    # singleton guard: a fresh manager.alive (touched every 60s) means another
    # manager is running -> refuse, so two managers never race the same cache.
    alive = STATE / "manager.alive"
    if alive.exists() and (time.time() - alive.stat().st_mtime) < 180:
        log(f"another manager appears active (manager.alive is fresh); exiting pid={os.getpid()}")
        return
    alive.write_text(str(os.getpid()))
    with open(STATE / "manager.pid", "w") as f:
        f.write(str(os.getpid()))
    todo = [t for t in tasks if fast_count(t) < target]
    log(f"manager start pid={os.getpid()} tasks={tasks} todo={todo} workers={workers} target={target}")
    q = queue.Queue()
    for t in todo:
        q.put(t)
    stop_evt = threading.Event()

    def _stop_watch():
        while not stop_evt.is_set():
            if (STATE / "STOP").exists():
                log("STOP file seen -> stopping")
                stop_evt.set()
                return
            time.sleep(5)

    def worker_thread():
        while not stop_evt.is_set():
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            try:
                fill_one(t, target, hang_timeout, ckpt, stop_evt)
            except Exception as e:
                log(f"task{t:03d}: fill_one error {e}")
            finally:
                q.task_done()

    threading.Thread(target=_stop_watch, daemon=True).start()
    threads = [threading.Thread(target=worker_thread, daemon=True)
               for _ in range(min(workers, max(len(todo), 1)))]
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        time.sleep(60)
        try:
            alive.write_text(str(os.getpid()))  # liveness heartbeat for the singleton guard
        except Exception:
            pass
        snap = {t: fast_count(t) for t in tasks}
        done = sum(1 for t in tasks if snap[t] >= target)
        log(f"progress {done}/{len(tasks)} full | " + " ".join(f"{t}:{snap[t]}" for t in tasks))
    log("manager DONE: " + " ".join(f"{t}:{fast_count(t)}" for t in tasks))
    for f in ("manager.pid", "STOP", "manager.alive"):
        try:
            os.remove(STATE / f)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Hang-safe resumable ARC-GEN corpus filler")
    ap.add_argument("--mode", default="manager", choices=["manager", "worker"])
    ap.add_argument("--task", type=int)
    ap.add_argument("--tasks", type=str, help="comma-separated task numbers")
    ap.add_argument("--all", action="store_true", help="fill every task below target")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="signal a running manager to stop")
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--hang-timeout", type=float, default=120.0,
                    help="seconds of frontier-stall before a seed is declared hung; "
                         "some healthy generators have 20-60s slow-tail seeds (task157), "
                         "so do not set this low")
    ap.add_argument("--ckpt", type=int, default=5000)
    a = ap.parse_args()
    _ensure_dirs()

    if a.stop:
        (STATE / "STOP").write_text("stop")
        print("wrote STOP flag; manager will halt within ~5s")
        return
    if a.status:
        inc = [(t, fast_count(t)) for t in range(1, 401)]
        below = [(t, c) for t, c in inc if c < a.target]
        print(f"{400 - len(below)}/400 full (target {a.target})")
        for t, c in sorted(below, key=lambda x: x[1]):
            sk = len(load_skip(t))
            print(f"  task{t:03d}: {c:>7} pairs" + (f"  ({sk} seeds skipped)" if sk else ""))
        return
    if a.mode == "worker":
        sys.exit(run_worker(a.task, a.target, a.ckpt))

    if a.all:
        tasks = [t for t in range(1, 401) if fast_count(t) < a.target]
    elif a.tasks:
        tasks = [int(x) for x in a.tasks.split(",") if x.strip()]
    elif a.task:
        tasks = [a.task]
    else:
        print("specify --all, --tasks, or --task")
        sys.exit(1)
    manager(tasks, a.target, a.workers, a.hang_timeout, a.ckpt)


if __name__ == "__main__":
    main()
