"""Check that local task data + neurogolf_utils.py matches Kaggle's current versions.

Compares `kaggle competitions files -c neurogolf-2026` output against local
`data/competitions/neurogolf-2026/`. Warns if anything is stale.

Usage:
    python tools/agent_helpers/check_data_freshness.py
    python tools/agent_helpers/check_data_freshness.py --quiet   # only print on stale
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "competitions" / "neurogolf-2026"


def kaggle_file_list():
    """Run `kaggle competitions files -c neurogolf-2026` and parse the output."""
    files = []
    next_token = None
    while True:
        cmd = ["kaggle", "competitions", "files", "-c", "neurogolf-2026"]
        if next_token:
            cmd += ["--page-token", next_token]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception as exc:
            print(f"ERROR: kaggle CLI call failed: {exc}", file=sys.stderr)
            return None

        out = r.stdout
        # Parse lines like:
        # neurogolf_utils/neurogolf_utils.py       21599  2026-05-14 21:00:09.350000
        for line in out.splitlines():
            m = re.match(r"^(\S+)\s+(\d+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if m:
                files.append({
                    "name": m.group(1),
                    "size": int(m.group(2)),
                    "creation_date": m.group(3),
                })
        # Next page token
        nm = re.search(r"Next Page Token = (\S+)", out)
        if nm:
            next_token = nm.group(1)
        else:
            break
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print("Fetching Kaggle file list...")
    kaggle_files = kaggle_file_list()
    if kaggle_files is None:
        return 1

    if not args.quiet:
        print(f"Kaggle reports {len(kaggle_files)} files.")

    stale = []
    missing = []
    same_size = 0
    for kf in kaggle_files:
        local = DATA_DIR / kf["name"]
        if not local.exists():
            missing.append(kf["name"])
            continue
        local_size = local.stat().st_size
        if local_size != kf["size"]:
            stale.append({
                "name": kf["name"],
                "local_size": local_size,
                "kaggle_size": kf["size"],
                "kaggle_date": kf["creation_date"],
            })
        else:
            same_size += 1

    print(f"\n=== Data freshness ===")
    print(f"Files matching size: {same_size}")
    print(f"Files stale (size differs): {len(stale)}")
    print(f"Files missing locally: {len(missing)}")

    if stale:
        print(f"\n=== Stale (re-pull recommended) ===")
        for s in stale[:20]:
            print(f"  {s['name']:<40} local={s['local_size']:>8} kaggle={s['kaggle_size']:>8} kaggle_date={s['kaggle_date']}")
        if len(stale) > 20:
            print(f"  ... ({len(stale) - 20} more)")
        print(f"\nRe-pull all stale files:")
        print(f"  kaggle competitions download -c neurogolf-2026 -p data/competitions/neurogolf-2026/ --force")
        print(f"  (then unzip into place)")

    if missing:
        print(f"\n=== Missing locally ===")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... ({len(missing) - 20} more)")

    return 0 if not (stale or missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
