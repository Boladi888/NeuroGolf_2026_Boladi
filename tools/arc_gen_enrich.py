"""Enrich a NeuroGolf task's arc-gen subset with synthetic pairs from google/ARC-GEN.

Several solved tasks have very sparse arc-gen sampling (e.g. task104: only 4 pairs)
combined with low LUT key-space coverage (D3, D4 in `Research/audits/audit-2026-05-02-deep.md`).
The bundled arc-gen subset is a sample of `google/ARC-GEN`'s output for the task's
generator; we can pull more pairs to (a) shrink the silent-fail surface and (b)
push the LUT coverage above the 5% threshold flagged by `lut_density_check.py`.

Mapping: the NeuroGolf task001..task400 numbering corresponds 1:1 to ARC-AGI
training tasks sorted alphabetically by UUID. Verified: all 400 NG task001..task400
match the sorted-by-UUID ARC-AGI training task on train[0].

Setup (one-time):
    cd /tmp
    git clone --recurse-submodules https://github.com/google/ARC-GEN.git
    # (Or set ARC_GEN_DIR env var to a custom path.)

Usage:
    python tools/arc_gen_enrich.py --task 104 --num 500
        # Generates 500 new pairs for NG task104 via ARC-GEN, appends them to
        # data/competitions/neurogolf-2026/task104.json's `arc-gen` list.
        # Original data/.../task104.json is backed up to *.json.pre-enrich.

    python tools/arc_gen_enrich.py --tasks 56,104,262 --num 500
        # Bulk enrich multiple tasks.

    python tools/arc_gen_enrich.py --task 104 --num 500 --dry-run
        # Show what would be generated, don't write.

    python tools/arc_gen_enrich.py --task 104 --restore
        # Restore from .pre-enrich backup.

After enriching:
    .venv/Scripts/python.exe build_submission.py --hybrid
        # Rebuild submissions/ — the LUT for enriched tasks now sees more keys,
        # potentially capturing more of the hidden private set's input space.

Exit code:
    0  enrichment succeeded for all requested tasks
    1  at least one task failed (task missing from ARC-GEN, generator error, etc.)
    2  invocation / setup error (ARC-GEN not found, etc.)
"""

import argparse
import json
import os
import pathlib
import random
import shutil
import sys

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
NG_DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))


def find_arc_gen_dir():
    """Locate the ARC-GEN clone. Authoritative in-project copy first, then env/temp."""
    env = os.environ.get("ARC_GEN_DIR")
    if env:
        p = pathlib.Path(env)
        if (p / "arc_gen.py").exists() and (p / "external" / "ARC-AGI" / "data" / "training").exists():
            return p
    candidates = [
        NG_ROOT / "external" / "ARC-GEN",   # authoritative: vendored inside the repo (gitignored)
        pathlib.Path(os.path.expandvars(r"%TEMP%\ARC-GEN")),
        pathlib.Path("/tmp/ARC-GEN"),
        NG_ROOT.parent / "ARC-GEN",
    ]
    for p in candidates:
        if (p / "arc_gen.py").exists() and (p / "external" / "ARC-AGI" / "data" / "training").exists():
            return p
    return None


def task_num_to_uuid(task_num, arc_gen_dir):
    """Map NG task001..task400 to the corresponding ARC-AGI sorted-by-UUID task."""
    arc_dir = arc_gen_dir / "external" / "ARC-AGI" / "data" / "training"
    uuids = sorted(p.stem for p in arc_dir.iterdir() if p.suffix == ".json")
    if not (1 <= task_num <= len(uuids)):
        return None
    return uuids[task_num - 1]


def generate_pairs(task_num, num_pairs, arc_gen_dir, seed_offset=2026):
    """Use ARC-GEN's generator for the task to produce `num_pairs` new examples.

    Uses ARC-GEN's `task_list.task_list()` lookup directly (rather than spawning
    arc_gen.py as a subprocess), to keep things in-process and capture the
    structured output dicts directly.
    """
    sys.path.insert(0, str(arc_gen_dir))
    try:
        # ARC-GEN itself imports from `tasks.task_<UUID>` so the import has to
        # happen with arc_gen_dir on sys.path.
        import task_list as arc_task_list  # noqa: E402
    except ImportError as e:
        return None, f"could not import ARC-GEN's task_list: {e}"

    uuid = task_num_to_uuid(task_num, arc_gen_dir)
    if uuid is None:
        return None, f"task_num {task_num} out of range 1..400"
    task_dict = arc_task_list.task_list()
    info = task_dict.get(uuid)
    if info is None:
        return None, f"ARC-GEN has no generator for {uuid}"
    generator, _ = info

    pairs = []
    for example_id in range(num_pairs):
        random.seed(seed_offset + example_id)
        try:
            ex = generator()
        except Exception as e:  # noqa: BLE001
            return None, f"generator raised on example_id={example_id}: {e}"
        # ARC-GEN's generator() returns a single {"input": ..., "output": ...} dict
        # per call (verified empirically). Take it directly as a new arc-gen pair.
        if "input" not in ex or "output" not in ex:
            return None, f"generator returned unexpected shape on example_id={example_id}: keys={list(ex)}"
        pairs.append({"input": ex["input"], "output": ex["output"]})

    return pairs, None


def enrich_one(task_num, num_pairs, arc_gen_dir, dry_run=False, verbose=True):
    """Generate `num_pairs` new pairs for task_num and append to its task JSON.

    Returns (n_appended, error_msg). On success, error_msg is None.
    """
    target = NG_DATA_DIR / f"task{task_num:03d}.json"
    if not target.exists():
        return 0, f"missing {target}"

    pairs, err = generate_pairs(task_num, num_pairs, arc_gen_dir)
    if err is not None:
        return 0, err

    if dry_run:
        if verbose:
            print(f"  task{task_num:03d}: would append {len(pairs)} pairs (dry-run; no write)")
        return len(pairs), None

    backup = target.with_suffix(target.suffix + ".pre-enrich")
    if not backup.exists():
        shutil.copy2(target, backup)

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    existing_arcgen = data.get("arc-gen", [])
    data["arc-gen"] = existing_arcgen + pairs

    # Write atomically: write to .tmp, fsync, rename.
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(target)

    if verbose:
        print(f"  task{task_num:03d}: appended {len(pairs)} pairs "
              f"(was {len(existing_arcgen)}, now {len(data['arc-gen'])}; backup at {backup.name})")

    return len(pairs), None


def restore_one(task_num, verbose=True):
    """Restore task_num from its .pre-enrich backup."""
    target = NG_DATA_DIR / f"task{task_num:03d}.json"
    backup = target.with_suffix(target.suffix + ".pre-enrich")
    if not backup.exists():
        if verbose:
            print(f"  task{task_num:03d}: no backup at {backup.name}")
        return False
    shutil.copy2(backup, target)
    if verbose:
        print(f"  task{task_num:03d}: restored from {backup.name}")
    return True


def _parse_tasks(arg_task, arg_tasks):
    if arg_task is not None:
        return [arg_task]
    if arg_tasks:
        return sorted({int(t) for t in arg_tasks.split(",") if t.strip()})
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", type=int, default=None, help="Single NG task id")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated NG task ids")
    parser.add_argument("--num", type=int, default=500, help="Pairs to generate per task (default 500)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, don't write")
    parser.add_argument("--restore", action="store_true", help="Restore from .pre-enrich backups")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = _parse_tasks(args.task, args.tasks)
    if not tasks:
        print("Specify --task N or --tasks N,M,...", file=sys.stderr)
        sys.exit(2)

    if args.restore:
        for tn in tasks:
            restore_one(tn, verbose=not args.quiet)
        return

    arc_gen_dir = find_arc_gen_dir()
    if arc_gen_dir is None:
        print("ARC-GEN clone not found. Set ARC_GEN_DIR env var or clone:", file=sys.stderr)
        print("  cd /tmp && git clone --recurse-submodules https://github.com/google/ARC-GEN.git", file=sys.stderr)
        sys.exit(2)
    if not args.quiet:
        print(f"Using ARC-GEN at: {arc_gen_dir}")

    failures = 0
    for tn in tasks:
        n, err = enrich_one(tn, args.num, arc_gen_dir, dry_run=args.dry_run,
                            verbose=not args.quiet)
        if err is not None:
            failures += 1
            if not args.quiet:
                print(f"  task{tn:03d}: FAIL - {err}")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
