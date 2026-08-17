"""Build submission zip by swapping task175.onnx with v1_closed_form in base submission.zip."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

WORKTREE = Path(r"<REPO_ROOT>/.claude/worktrees/loving-poincare-5e357f")
BASE_ZIP = WORKTREE / "submissions" / "submission.zip"
PROTO_ONNX = WORKTREE / "The_400_Tasks" / "Task_by_Task" / "task175" / "prototypes" / "task175_v1_closed_form.onnx"
OUT_ZIP = Path(r"<WORKSPACE_ROOT>/.notongit/orchestrator/kaggle") / "loving-poincare-task175-v1.zip"


def main():
    if not PROTO_ONNX.exists():
        print(f"ERROR: prototype onnx missing at {PROTO_ONNX}")
        return 1
    if not BASE_ZIP.exists():
        print(f"ERROR: base submission.zip missing at {BASE_ZIP}")
        return 1

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    new_blob = PROTO_ONNX.read_bytes()
    print(f"v1 blob size: {len(new_blob)} bytes")

    n_entries = 0
    n_swapped = 0
    with zipfile.ZipFile(BASE_ZIP, "r") as zin, zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "task175.onnx":
                old_size = len(data)
                zout.writestr("task175.onnx", new_blob)
                n_swapped += 1
                print(f"  swapped task175.onnx: {old_size} -> {len(new_blob)} bytes")
            else:
                zout.writestr(name, data)
            n_entries += 1
    print(f"\nWrote {OUT_ZIP} ({OUT_ZIP.stat().st_size} bytes) — {n_entries} entries, {n_swapped} swapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
