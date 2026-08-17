import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "Logs"


CHILD = r'''
import json, math, os, pathlib, sys, types
for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
root = pathlib.Path.cwd()
data = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(root / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(data / "neurogolf_utils"))
sys.path.insert(0, str(root / "tools"))
import neurogolf_utils
neurogolf_utils._NEUROGOLF_DIR = f"{data.as_posix()}/"
from _score_v2 import score_network_v2
import onnxruntime

p = pathlib.Path(sys.argv[1])
task = int(sys.argv[2])
base_cost_arg = sys.argv[3]
base_cost = None if base_cost_arg == "NA" else int(base_cost_arg)
base_ok = sys.argv[4] == "1"
min_delta = float(sys.argv[5])
out = {}
try:
    if not neurogolf_utils.check_network(str(p)):
        print(json.dumps({"status": "CHECK_FAIL"}))
        raise SystemExit
    macs, mem, params = score_network_v2(str(p))
    if macs is None or mem is None or params is None:
        print(json.dumps({"status": "SCORE_REJECT", "error": str((macs, mem, params))}))
        raise SystemExit
    cost = int(mem) + int(params)
    points = max(1.0, 25 - math.log(max(1, cost)))
    out.update({"status": "SCORED", "cost": cost, "points": points, "macs": int(macs), "memory": int(mem), "params": int(params)})
    if cost < 1000:
        out["status"] = "SKIP_LOW_COST"
        print(json.dumps(out))
        raise SystemExit
    if base_ok and base_cost is not None:
        base_points = max(1.0, 25 - math.log(max(1, base_cost)))
        delta = points - base_points
        out["delta"] = delta
        if delta <= min_delta:
            out["status"] = "SKIP_NOT_BETTER"
            print(json.dumps(out))
            raise SystemExit
    else:
        out["delta"] = points
    session = onnxruntime.InferenceSession(str(p))
    examples = neurogolf_utils.load_examples(task)
    out["status"] = "OK"
    for set_name in ("train", "test", "arc-gen"):
        pairs = examples.get(set_name, [])
        if not pairs:
            out[set_name + "_right"] = 0
            out[set_name + "_total"] = 0
            continue
        right, wrong, err = neurogolf_utils.verify_subset(session, pairs)
        out[set_name + "_right"] = int(right)
        out[set_name + "_total"] = int(right) + int(wrong)
        if err:
            out["status"] = "ERROR"
            out["error"] = str(err)
        elif wrong:
            out["status"] = "FAIL"
except Exception as exc:
    out.setdefault("error", repr(exc))
    if out.get("status") in (None, "SCORED", "OK"):
        out["status"] = "EXCEPTION"
print(json.dumps(out))
'''


def load_base():
    base = {}
    with (LOGS / "coconut4_costs_2026-05-07.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and row["cost"]:
                cost = int(float(row["cost"]))
                base[int(row["task"])] = cost

    base_ok = {}
    with (LOGS / "coconut4_task_verify_scan_2026-05-07.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base_ok[int(row["task"])] = row["status"] == "OK"
    return base, base_ok


def run_one(label, path, base, base_ok, min_delta, timeout):
    task = int(path.stem[4:7])
    row = {
        "source": label,
        "task": task,
        "path": str(path),
        "base_cost": base.get(task, ""),
        "base_ok": base_ok.get(task, False),
    }
    start = time.time()
    try:
        cp = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD,
                str(path),
                str(task),
                str(row["base_cost"]) if row["base_cost"] != "" else "NA",
                "1" if row["base_ok"] else "0",
                str(min_delta),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if cp.returncode != 0:
            row.update({"status": f"child-exit-{cp.returncode}", "error": (cp.stdout + cp.stderr)[-600:]})
        else:
            row.update(json.loads(cp.stdout.strip().splitlines()[-1]))
    except subprocess.TimeoutExpired:
        row.update({"status": "timeout", "error": "timeout"})
    row["elapsed_s"] = f"{time.time() - start:.3f}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("source_dir")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=70)
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    files = sorted(source_dir.glob("task*.onnx"))
    base, base_ok = load_base()

    print(f"{args.label}: files {len(files)}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, args.label, path, base, base_ok, args.min_delta, args.timeout) for path in files]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if idx % 100 == 0:
                print(f"{args.label}: done {idx}/{len(files)}", flush=True)

    rows.sort(key=lambda row: int(row["task"]))
    out = LOGS / f"broad_{args.label}_against_coconut_2026-05-07.csv"
    fields = [
        "source", "task", "status", "delta", "cost", "points", "base_cost",
        "base_ok", "macs", "memory", "params", "train_right", "train_total",
        "test_right", "test_total", "arc-gen_right", "arc-gen_total",
        "elapsed_s", "path", "error",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    positives = [row for row in rows if row.get("status") == "OK" and float(row.get("delta") or 0) > args.min_delta]
    positives.sort(key=lambda row: float(row.get("delta") or 0), reverse=True)
    print(f"wrote {out} rows {len(rows)} positive {len(positives)}")
    for row in positives[:40]:
        print(
            f"task{int(row['task']):03d} delta={float(row['delta']):.4f} "
            f"cost={row['cost']} base={row['base_cost']} path={row['path']}"
        )


if __name__ == "__main__":
    main()
