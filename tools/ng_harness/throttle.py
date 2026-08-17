"""Cross-process self-throttle so many agents don't collectively saturate the CPU.

Model: each ng_gate RUN holds ONE slot for its duration. Sessions are single-threaded
(session.py), so 1 run ~= 1 core; K = floor(TARGET_FRAC * cpu_count) caps concurrent
runs at ~80% of cores. Plus an adaptive guard: if total machine CPU (incl. the user's
own apps) is already above TARGET, new runs wait. Crash-safe: slots are lockfiles keyed
by PID; a dead PID's slot is reclaimable immediately (and the JobObject in killjob.py
kills runaway runs, so their PIDs die and free their slots).

Heavy sub-cap: a full-corpus truth/ab/battery/synth run (8k+ examples, sustained CPU +
RAM) is far heavier than a quick visible/small-n run, yet the uniform slot above treats
them the same — so N agents could all launch full gates at once and melt the box (the
p42 resource-pressure abort). A heavy gate therefore ALSO takes a scarce 'heavy' slot
(cap NEUROGOLF_HEAVY_GATE_K, default 5) via acquire_gate(), so only that many run at
once; cheap gates keep using just the normal ~80%-cores cap. Same crash-safe lockfile
reclaim, so a killed heavy gate frees its heavy slot immediately.

Coordination dir is shared per-user via %TEMP% (all agents run as the same user).
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time

TARGET_FRAC = 0.80
STALE_SECS = 6 * 3600            # a slot older than this with a live PID is still honored;
                                  # this only bounds clock-skew zombies
_DIR = pathlib.Path(os.environ.get("NEUROGOLF_THROTTLE_DIR",
                                   str(pathlib.Path(tempfile.gettempdir()) / "ng_throttle")))

# Corpus-heavy modes: a large-n run of these is the CPU/RAM hog worth serializing.
# (visible is fast; adversarial is small.) A gate counts as heavy when its mode is
# in this set AND its corpus n is >= NEUROGOLF_HEAVY_GATE_N (so the batch-standard
# --n 8000 and larger runs are heavy, while the default --n 2000 stays light and parallel).
HEAVY_MODES = frozenset({"truth", "ab", "battery", "synth"})

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None


def _k() -> int:
    return max(1, int(TARGET_FRAC * (os.cpu_count() or 4)))


def _heavy_k() -> int:
    """Max concurrent heavy gates across all agents (env-overridable; default 5)."""
    try:
        return max(1, int(os.environ.get("NEUROGOLF_HEAVY_GATE_K", "5")))
    except (TypeError, ValueError):
        return 5


def _heavy_n() -> int:
    """Corpus size at/above which a HEAVY_MODES gate counts as heavy (default 8k)."""
    try:
        return max(1, int(os.environ.get("NEUROGOLF_HEAVY_GATE_N", "8000")))
    except (TypeError, ValueError):
        return 8000


def _owner_meta(kind: str, label: str, pid: int, extra: dict | None = None) -> dict:
    """Best-effort owner metadata for read-only watchers and scoped cleanup."""
    keys = {
        "lane": "NEUROGOLF_HANDOFF_LANE",
        "lane_id": "NEUROGOLF_HANDOFF_LANE_ID",
        "batch": "NEUROGOLF_HANDOFF_BATCH",
        "task": "NEUROGOLF_HANDOFF_TASK",
        "task_num": "NEUROGOLF_HANDOFF_TASK_NUM",
        "role": "NEUROGOLF_HANDOFF_ROLE",
        "agent": "NEUROGOLF_HANDOFF_AGENT",
        "runner_id": "NEUROGOLF_HANDOFF_RUNNER_ID",
        "run_id": "NEUROGOLF_HANDOFF_RUN_ID",
        "raw_dir": "NEUROGOLF_HANDOFF_RAW_DIR",
    }
    meta = {name: os.environ.get(env) for name, env in keys.items() if os.environ.get(env)}
    meta.update(
        {
            "schema": "neurogolf.ng_gate.owner.v1",
            "kind": kind,
            "label": label,
            "pid": pid,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "heavy_n_threshold": _heavy_n(),
        }
    )
    if extra:
        meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


def _reclaim_intents() -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for f in _DIR.glob("intent_*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            meta = {}
        try:
            pid = int(meta.get("pid") or f.stem.split("_")[1])
        except Exception:
            pid = None
        try:
            age = now - f.stat().st_mtime
        except Exception:
            age = 0
        if pid is None or not _pid_alive(pid) or age > STALE_SECS:
            try: f.unlink()
            except Exception: pass


def write_intent(label: str, mode: str, n: int, task: int | None = None, candidate: str | None = None) -> pathlib.Path | None:
    """Record a gate process before it waits on throttle slots.

    Slot lockfiles only exist after acquire() succeeds. The intent sidecar lets
    watchers and scoped runner cleanup identify waiting-heavy processes too.
    """
    _DIR.mkdir(parents=True, exist_ok=True)
    _reclaim_intents()
    pid = os.getpid()
    path = _DIR / f"intent_{pid}.json"
    try:
        path.write_text(
            json.dumps(
                _owner_meta(
                    "intent",
                    label,
                    pid,
                    {
                        "mode": mode,
                        "n": int(n),
                        "task_num": str(task) if task is not None else None,
                        "candidate": candidate,
                    },
                ),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path
    except Exception:  # noqa: BLE001
        return None


def clear_intent(path: pathlib.Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except Exception:
        pass


def is_heavy(mode: str, n: int) -> bool:
    """True if this gate should take the scarce heavy slot (full-corpus verification)."""
    try:
        return mode in HEAVY_MODES and int(n) >= _heavy_n()
    except (TypeError, ValueError):
        return mode in HEAVY_MODES


def _pid_alive(pid: int) -> bool:
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return os.name == "nt"  # can't easily check on win without psutil; assume alive


def _reclaim_and_count(kind: str = "slot") -> int:
    """Delete dead/stale slot files of this kind; return count of live ones."""
    _DIR.mkdir(parents=True, exist_ok=True)
    _reclaim_intents()
    live = 0
    now = time.time()
    for f in _DIR.glob(f"{kind}_*.lock"):
        try:
            pid = int(f.stem.split("_")[1])
        except Exception:  # noqa: BLE001
            try: f.unlink()
            except Exception: pass
            try: f.with_suffix(f.suffix + ".json").unlink()
            except Exception: pass
            continue
        try:
            age = now - f.stat().st_mtime
        except Exception:  # noqa: BLE001
            continue
        if not _pid_alive(pid) or age > STALE_SECS:
            try: f.unlink()
            except Exception: pass
            try: f.with_suffix(f.suffix + ".json").unlink()
            except Exception: pass
        else:
            live += 1
    return live


def _cpu_over_target() -> bool:
    if psutil is None:
        return False
    try:
        return psutil.cpu_percent(interval=0.15) > TARGET_FRAC * 100.0
    except Exception:  # noqa: BLE001
        return False


class Slot:
    """Context manager: acquire one throttle slot for the duration of a run.

    kind="slot" (default) is the normal ~80%-cores cap applied to every gate; it
    also honors the machine-wide CPU guard. kind="heavy" is a separate, scarcer
    cap (see acquire_gate) and is a pure counting semaphore — no CPU guard, since
    the normal slot the same gate also holds already enforces it.
    """
    def __init__(self, label: str = "", poll: float = 0.4, kind: str = "slot",
                 cap: int | None = None, check_cpu: bool | None = None):
        self.label = label
        self.poll = poll
        self.kind = kind
        self._cap = cap
        self._check_cpu = (kind == "slot") if check_cpu is None else check_cpu
        self._file = None

    def acquire(self):
        pid = os.getpid()
        # tiny per-pid jitter to de-synchronize concurrent agents
        jitter = (pid % 7) * 0.03
        while True:
            k = self._cap if self._cap is not None else _k()
            live = _reclaim_and_count(self.kind)
            if live < k and not (self._check_cpu and _cpu_over_target()):
                # optimistic claim, then re-check we didn't overshoot the cap
                self._file = _DIR / f"{self.kind}_{pid}_{int(time.time()*1000)%100000}.lock"
                try:
                    self._file.write_text(f"{pid} {self.label} {time.time():.0f}")
                    self._file.with_suffix(self._file.suffix + ".json").write_text(
                        json.dumps(_owner_meta(self.kind, self.label, pid), sort_keys=True),
                        encoding="utf-8",
                    )
                except Exception:  # noqa: BLE001
                    try:
                        if self._file is not None:
                            self._file.with_suffix(self._file.suffix + ".json").unlink()
                    except Exception:
                        pass
                    try:
                        if self._file is not None:
                            self._file.unlink()
                    except Exception:
                        pass
                    self._file = None
                    time.sleep(self.poll + jitter); continue
                if _reclaim_and_count(self.kind) <= k:
                    return self
                # overshot: yield and retry
                self.release()
            time.sleep(self.poll + jitter)

    def release(self):
        if self._file is not None:
            try: self._file.with_suffix(self._file.suffix + ".json").unlink()
            except Exception: pass
            try: self._file.unlink()
            except Exception: pass
            self._file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *a):
        self.release()


class _GateHold:
    """Aggregate handle for the one-or-two slots a gate holds; release() frees all."""
    def __init__(self, slots):
        self._slots = list(slots)

    def release(self):
        for s in reversed(self._slots):
            s.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()


def acquire_gate(mode: str, n: int, label: str = "") -> "_GateHold":
    """Acquire throttle slot(s) for a gate run.

    A heavy corpus gate (full truth/ab/battery/synth) takes the scarce 'heavy'
    slot FIRST so at most _heavy_k() run at once, then the normal slot; a cheap
    gate takes only the normal slot. Acquiring the scarce slot first bounds heavy
    gates without deadlock: light gates hold only the normal slot and always drain
    to completion, freeing it for the one heavy gate that is waiting on it.
    """
    slots = []
    if is_heavy(mode, n):
        slots.append(Slot(f"heavy {label}", kind="heavy", cap=_heavy_k()).acquire())
    slots.append(Slot(label, kind="slot").acquire())
    return _GateHold(slots)


def status() -> dict:
    return {"dir": str(_DIR), "k": _k(), "live_slots": _reclaim_and_count("slot"),
            "heavy_k": _heavy_k(), "live_heavy": _reclaim_and_count("heavy"),
            "cpu_count": os.cpu_count(), "psutil": psutil is not None}
