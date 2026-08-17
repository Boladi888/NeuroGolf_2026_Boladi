#!/usr/bin/env python3
"""Robust Kaggle submission poller for the Submit Captain — parse the CSV BY COLUMN.

WHY THIS EXISTS: the old ship one-liners piped the kaggle submissions CSV through
`grep "taskNNN" | grep -oE 'SubmissionStatus.[A-Z]+,[0-9.]+'`. Submit descriptions are
full of commas and numbers (costs, ratios, gate counts like 200265, 441/441), so the
grep matched a STRAY number inside a still-PENDING row and reported a phantom score
(2026-06-08: a phantom -74 "regression" that nearly got reconciled). The publicScore is
a SPECIFIC CSV COLUMN, not "the first number after SubmissionStatus." Always parse it.

Usage:
    python Agenters/Submit_Captain/poll_kaggle.py "task295 bcastadd"
    python Agenters/Submit_Captain/poll_kaggle.py "task295 bcastadd" --prev 6711.28

  - The positional arg is a substring of the submit -m description (unique enough to
    identify the row — include the task id + a distinctive word).
  - --prev (optional): the last confirmed LB. If given, the poller prints the delta and
    SHOUTS if |delta| > 5 (a single-task swap can't move the LB that much => read error
    or corrupted zip; verify before reconciling).

Exit code 0 on complete/error, 2 on timeout, 3 on Kaggle CLI failure. Prints one line: STATUS=... publicScore=...
Run it with run_in_background:true and read the output file when notified.
"""
from __future__ import annotations
import argparse, csv, io, subprocess, sys, time

COMPETITION = "neurogolf-2026"


def fetch_rows() -> list[dict]:
    # kaggle CSV occasionally contains non-utf8 bytes -> decode with errors='replace'
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
        capture_output=True,
    )
    if r.returncode != 0:                                            # auth/network/CLI failure: fail fast, don't loop as NOT_FOUND/TIMEOUT
        sys.stderr.write((r.stderr or b"").decode("utf-8", "replace")[-800:] + "\n")
        sys.exit(3)
    text = r.stdout.decode("utf-8", "replace")
    lines = text.splitlines()                                       # kaggle v2.0.2 prints a version-warning on STDOUT before
    for i, ln in enumerate(lines):                                  # the CSV header; skip any preamble so DictReader sees the
        if ln.startswith("fileName,"): text = "\n".join(lines[i:]); break   # real header (no-op when the warning is absent)
    return list(csv.DictReader(io.StringIO(text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("desc_substring", help="unique substring of the submit -m description")
    ap.add_argument("--prev", type=float, default=None, help="last confirmed LB (for the >5 sanity guard)")
    ap.add_argument("--tries", type=int, default=120)
    ap.add_argument("--interval", type=int, default=45)
    args = ap.parse_args()

    for _ in range(args.tries):
        rows = fetch_rows()
        row = next((r for r in rows if args.desc_substring in (r.get("description") or "")), None)
        if row is None:
            # NOT in the list at all => the submit never happened (e.g. bad staged path).
            # Keep waiting a couple cycles in case of API lag, but this usually means re-fire.
            print("NOT_FOUND (submit may have failed — verify the staged path and re-fire)", flush=True)
            time.sleep(args.interval)
            continue
        status = row.get("status") or ""
        if "complete" in status.lower() or "error" in status.lower():
            ps = row.get("publicScore")
            line = f"STATUS={status} publicScore={ps}"
            if args.prev is not None and ps:
                try:
                    delta = float(ps) - args.prev
                    flag = "  <<< |delta|>5 = READ ERROR or CORRUPTED ZIP, VERIFY before reconciling" if abs(delta) > 5 else ""
                    line += f" delta={delta:+.2f}{flag}"
                except ValueError:
                    pass
            print(line, flush=True)
            return 0
        time.sleep(args.interval)
    print("TIMEOUT still pending", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
