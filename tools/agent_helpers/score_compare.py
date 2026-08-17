"""Side-by-side cost comparison of two ONNXes.

Usage:
    python tools/agent_helpers/score_compare.py before.onnx after.onnx
"""
import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools._score_v2 import score_network_v2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_a")
    p.add_argument("model_b")
    args = p.parse_args()

    results = {}
    for label, path in [("A", args.model_a), ("B", args.model_b)]:
        try:
            r = score_network_v2(path)
        except Exception as exc:
            print(f"{label} ({path}): score failed — {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if r is None:
            print(f"{label} ({path}): score_network_v2 returned None")
            return 1
        macs, mem, params = r
        cost = mem + params
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))
        results[label] = dict(macs=macs, memory=mem, params=params, cost=cost, points=points, path=path)

    a, b = results["A"], results["B"]
    print(f"=== Score compare ===")
    print(f"{'metric':<10} {'A':>15} {'B':>15} {'delta':>15}")
    print(f"{'-'*60}")
    for k in ("memory", "params", "cost"):
        d = b[k] - a[k]
        print(f"{k:<10} {a[k]:>15,} {b[k]:>15,} {d:>15,}")
    print(f"{'points':<10} {a['points']:>15.4f} {b['points']:>15.4f} {b['points'] - a['points']:>15.4f}")
    print()
    print(f"A: {a['path']}")
    print(f"B: {b['path']}")
    if b["cost"] < a["cost"]:
        print(f"\nB is CHEAPER by {a['cost'] - b['cost']:,} bytes ({100*(a['cost']-b['cost'])/max(a['cost'],1):.1f}%).")
        # Calibrated LB estimate
        if a["cost"] > 0 and b["cost"] > 0:
            log_e_gain = math.log(a["cost"]) - math.log(b["cost"])
            calibrated_lb = log_e_gain * 0.48
            print(f"Expected LB gain (calibrated): +{calibrated_lb:.4f}")
            print(f"Raw log_e gain (overestimate): +{log_e_gain:.4f}")
    elif b["cost"] > a["cost"]:
        print(f"\nB is MORE EXPENSIVE by {b['cost'] - a['cost']:,} bytes. Don't ship.")
    else:
        print(f"\nA and B have IDENTICAL cost.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
