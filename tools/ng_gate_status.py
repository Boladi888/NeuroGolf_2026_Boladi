"""ng_gate_status -- read-only view of the shared heavy-gate throttle (the de-facto gate 'lease').

The gate-lease the win-rate proposals ask for already EXISTS as `tools/ng_harness/throttle.py`: each
ng_gate run holds a lockfile 'slot', and a full-corpus truth/ab/battery/synth run also takes a scarce
'heavy' slot (cap NEUROGOLF_HEAVY_GATE_K, default 5). What was missing is VISIBILITY -- who holds those
slots, so a reviewer can tell a candidate is blocked by a BUSY heavy gate (return GATE_WAIT) rather than
by a bad candidate. Reaping is automatic (a dead PID's slot is reclaimed on the next acquire), so there
is no separate reap command; this view is strictly read-only and never deletes another agent's lock.

    python tools/ng_gate_status.py [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ng_harness import throttle


def _read_locks(kind: str) -> list[dict]:
    out: list[dict] = []
    d = throttle._DIR
    if not d.exists():
        return out
    now = time.time()
    for f in sorted(d.glob(f"{kind}_*.lock")):
        try:
            parts = f.read_text(encoding="utf-8", errors="replace").split()
        except Exception:
            parts = []
        try:
            pid = int(f.stem.split("_")[1])
        except Exception:
            pid = None
        label = " ".join(parts[1:-1]) if len(parts) >= 3 else (parts[1] if len(parts) == 2 else "")
        try:
            age = round(now - f.stat().st_mtime, 1)
        except Exception:
            age = None
        alive = throttle._pid_alive(pid) if pid is not None else None
        meta = {}
        try:
            meta = json.loads(f.with_suffix(f.suffix + ".json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        out.append(
            {
                "kind": kind,
                "pid": pid,
                "label": label,
                "age_s": age,
                "alive": alive,
                "lock": f.name,
                "owner": meta,
            }
        )
    return out


def _owner_text(slot: dict) -> str:
    owner = slot.get("owner") or {}
    parts = [
        owner.get("lane_id") or owner.get("lane"),
        f"p{owner.get('batch')}" if owner.get("batch") else "",
        owner.get("task"),
        owner.get("role"),
        owner.get("agent"),
    ]
    return " ".join(str(p) for p in parts if p)


def gather() -> dict:
    k, hk = throttle._k(), throttle._heavy_k()
    slots = [s for s in _read_locks("slot") if s["alive"]]
    heavy = [s for s in _read_locks("heavy") if s["alive"]]
    return {
        "throttle_dir": str(throttle._DIR), "cpu_count": os.cpu_count(), "psutil": throttle.psutil is not None,
        "normal_cap": k, "normal_live": len(slots), "normal_free": max(0, k - len(slots)),
        "heavy_cap": hk, "heavy_live": len(heavy), "heavy_free": max(0, hk - len(heavy)),
        "heavy_modes": sorted(throttle.HEAVY_MODES), "heavy_n_threshold": throttle._heavy_n(),
        "slots": slots, "heavy": heavy,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only view of the ng_gate heavy-gate throttle/lease.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    st = gather()
    if args.json:
        print(json.dumps(st, indent=2))
        return 0
    print(f"== ng_gate throttle status ({st['throttle_dir']}) ==")
    print(f"normal slots: {st['normal_live']}/{st['normal_cap']} used ({st['normal_free']} free)   "
          f"heavy slots: {st['heavy_live']}/{st['heavy_cap']} used ({st['heavy_free']} free)")
    print(f"heavy = mode in {st['heavy_modes']} with n >= {st['heavy_n_threshold']}")
    if st["heavy"]:
        print("HEAVY gate(s) holding the scarce slot (full truth/ab/battery/synth):")
        for s in st["heavy"]:
            owner = _owner_text(s)
            owner_suffix = f"  owner={owner}" if owner else ""
            print(f"  pid {s['pid']}  age {s['age_s']}s  {s['label']}{owner_suffix}")
    if st["heavy_free"] == 0 and st["heavy_cap"] > 0:
        print("=> heavy slot BUSY: a candidate waiting only on a full gate is GATE_WAIT, not a failure "
              "-- do NOT DROP it or re-run the worker; re-verify when the slot frees.")
    else:
        print("=> heavy slot FREE: a full truth/ab/battery run can start now.")
    if st["slots"]:
        print(f"normal slots held by {len(st['slots'])} run(s):")
        for s in st["slots"]:
            owner = _owner_text(s)
            owner_suffix = f"  owner={owner}" if owner else ""
            print(f"  pid {s['pid']}  age {s['age_s']}s  {s['label']}{owner_suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
