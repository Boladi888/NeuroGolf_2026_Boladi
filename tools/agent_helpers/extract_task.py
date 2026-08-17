"""Extract taskNNN.onnx from submissions/submission.zip to a temp/visible file.

Usage:
    python tools/agent_helpers/extract_task.py 110
    python tools/agent_helpers/extract_task.py 110 --out /tmp/task110.onnx
    python tools/agent_helpers/extract_task.py 110 --live-zip submissions/_canonical_5874_78_backup.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = REPO_ROOT / "submissions" / "submission.zip"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_id", help="3-digit task ID (e.g. 110) or 'taskNNN'")
    p.add_argument("--out", default=None, help="Output path (default: ./taskNNN.onnx in cwd)")
    p.add_argument("--live-zip", default=str(DEFAULT_ZIP), help=f"Source zip (default: {DEFAULT_ZIP})")
    args = p.parse_args()

    tid = args.task_id if args.task_id.startswith("task") else f"task{int(args.task_id):03d}"
    src = Path(args.live_zip)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path.cwd() / f"{tid}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src, "r") as z:
        try:
            data = z.read(f"{tid}.onnx")
        except KeyError:
            print(f"ERROR: {tid}.onnx not in {src}", file=sys.stderr)
            return 1

    out.write_bytes(data)
    print(f"Extracted {tid}.onnx ({len(data):,} bytes) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
