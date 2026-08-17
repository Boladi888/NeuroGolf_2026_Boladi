#!/usr/bin/env python3
"""Build a historical isolated-probe queue from archived 2026-05 ledgers."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETIRED_LEDGER_DIR = ROOT / "The_400_Tasks" / "_archive" / "retired_top_level_ledgers_2026-06-21"

CURRENT_BASE = "yuzu9"
CURRENT_SCORE = 5573.72
KNOWN_NEGATIVE_TASKS = {
    "025": "apple7 = strawberry6 + task025 scored 5525.01",
    "044": "jackfruit8 = imbe8 + task044 scored 5540.76 vs imbe8 5555.02",
    "145": "xigua6/yuzu6 = strawberry6 + task145 organic scored 5535.04",
    "177": "xigua9 = tangerine9 + task177 kernel-output-konbu17-lb5126 scored 5560.09",
    "182": "ugli9 = tangerine9 + task182 afr1ste-6225-51 scored public-flat 5573.44",
    "224": "watermelon9 = tangerine9 + task224 afr1ste-5653-86 scored 5559.61",
    "366": "acerola10 = yuzu9 + compressed task366 lookup scored public-flat 5573.72; true organic rule still worth targeting",
    "370": "zucchini6 = strawberry6 + task370 scored 5525.77",
    "382": "xigua7 = watermelon7 + task382 scored 5540.94 vs watermelon7 5553.24",
    "387": "yuzu7 = xigua7 + task387 scored 5528.33 vs xigua7 5540.94",
    "089": "old public-negative guardrail",
    "241": "old public-negative guardrail",
    "380": "old public-negative guardrail",
}
KNOWN_PENDING_TASKS = {}
KNOWN_POSITIVE_TASKS = {
    "017": "yuzu9 = tangerine9 + organic periodic-hole-fill scored 5573.72",
    "003": "lemon9 = lime9 + organic periodic-row-extension scored 5566.41",
    "052": "nectarine9 = mango9 + organic uniform-row scored 5567.34",
    "014": "elderberry9 = cherry9 + organic fast least-color bbox crop scored 5559.49; require task014_synthetic_guard for future cheaper replacements",
    "020": "mango9 = lemon9 + organic rotational-symmetry-completion scored 5567.07",
    "021": "honeydew9 = grape9 + organic grid-division-count scored 5563.08",
    "027": "fig9 = elderberry9 + organic rotation-symmetry-fill scored 5560.82",
    "029": "imbe9 = honeydew9 + organic marker-rectangle-interior scored 5564.11",
    "229": "quince9 = papaya9 + organic dominant-color-keep scored 5568.02",
    "304": "orange9 = nectarine9 + organic outer-product-marker-tile scored 5567.58",
    "315": "papaya9 = orange9 + organic outer-product-marker-tile scored 5567.81",
    "126": "jackfruit9 = imbe9 + organic pi-project-bottom scored 5564.94",
    "249": "lime9 = jackfruit9 + corrected organic horizontal-tile-2x scored 5565.72; kiwi9 was pipeline-bad from wrong scratch payload",
    "251": "grape9 = fig9 + organic enclosed-region-fill scored 5562.00",
    "076": "orange5 positive member inherited by strawberry6 line",
    "133": "orange5 positive member inherited by strawberry6 line",
    "153": "raspberry6 organic replacement inherited by strawberry6 line; cheap afr1ste-5044-56 cost replacement was hidden-bad in date9, require task153_synthetic_guard",
    "015": "strawberry7 positive ladder member inherited by cherry9",
    "024": "quince7 positive ladder member inherited by cherry9",
    "037": "lemon8/papaya8 positive member inherited by cherry9",
    "041": "kiwi8/papaya8 positive member inherited by cherry9",
    "095": "orange8/strawberry8 positive member inherited by cherry9",
    "102": "ugli8/zucchini8/cherry9 positive member",
    "105": "strawberry9 = raspberry9 + organic gridline-adjacent-fill scored 5568.77",
    "129": "raspberry9 = quince9 + organic const-dominant-color scored 5568.68",
    "141": "elderberry7 = strawberry6 + task141 scored 5538.24",
    "150": "imbe8 positive ladder member inherited by cherry9",
    "155": "nectarine8/raspberry8 positive member inherited by cherry9",
    "176": "watermelon8/apple9/cherry9 positive member",
    "190": "banana7 = strawberry6 + task190 scored 5538.36",
    "197": "grape8 positive ladder member inherited by cherry9",
    "199": "raspberry7 positive ladder member inherited by cherry9",
    "220": "tangerine7 positive ladder member inherited by cherry9",
    "221": "yuzu8/cherry9 positive member",
    "230": "watermelon7 positive ladder member inherited by cherry9",
    "246": "fig7 = strawberry6 + task246 scored 5538.08",
    "261": "date7 = strawberry6 + task261 scored 5538.29",
    "282": "mango8/quince8 positive member inherited by cherry9",
    "290": "ugli7 positive ladder member inherited by cherry9",
    "303": "xigua8/banana9/cherry9 positive member",
    "305": "cherry7 = strawberry6 + task305 scored 5538.36",
    "310": "honeydew7 positive ladder member inherited by cherry9",
    "312": "honeydew8 positive ladder member inherited by cherry9",
    "317": "tangerine8 positive ladder member inherited by cherry9",
    "350": "fig8 positive ladder member inherited by cherry9",
    "352": "vanilla7 positive ladder member inherited by cherry9",
    "381": "vanilla8/zucchini8/cherry9 positive member",
    "353": "strawberry6 = raspberry6 + task353 scored 5536.58",
    "386": "orange5 positive member inherited by strawberry6 line",
    "388": "orange5 positive member inherited by strawberry6 line",
    "389": "orange5 positive member inherited by strawberry6 line",
    "394": "tangerine9 = strawberry9 + afr1ste-5044-56 cost displacement scored 5573.44",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -999.0


def _int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 999_999_999


def build_rows() -> list[dict[str, str]]:
    candidate_rows = _read_csv(RETIRED_LEDGER_DIR / "submission_candidate_tasks.csv")
    state_rows = {row["task"]: row for row in _read_csv(ROOT / "The_400_Tasks" / "task_state.csv")}

    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        if not row.get("delta") or not row.get("payload_path"):
            continue
        rows_by_task[row["task"]].append(row)

    ranked: list[dict[str, str]] = []
    for task, task_rows in rows_by_task.items():
        if task in KNOWN_NEGATIVE_TASKS or task in KNOWN_PENDING_TASKS or task in KNOWN_POSITIVE_TASKS:
            continue
        best = sorted(task_rows, key=lambda row: (_float(row["delta"]), -_int(row["cost"])), reverse=True)[0]
        memberships = sorted({row["candidate"] for row in task_rows if row.get("candidate")})
        state = state_rows.get(task, {})
        why = "Next best remaining local delta after known negatives and consumed positives are excluded."
        priority_bias = 0
        ranked.append(
            {
                "task": task,
                "base_zip": CURRENT_BASE,
                "source": best["source"],
                "payload_path": best["payload_path"],
                "expected_local_delta": best["delta"],
                "cost": best["cost"],
                "candidate_memberships": ";".join(memberships),
                "state_status": state.get("best_known_status", ""),
                "state_reference": state.get("reference", ""),
                "known_negative_exclusion": "; ".join(f"task{k}: {v}" for k, v in sorted(KNOWN_NEGATIVE_TASKS.items())),
                "why": why,
                "sort_key": f"{priority_bias + _float(best['delta']):08.4f}:{-_int(best['cost']):010d}:{task}",
            }
        )
    ranked.sort(key=lambda row: row["sort_key"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = str(idx)
    return ranked


def build_md(date: str, rows: list[dict[str, str]]) -> str:
    top = rows[:5]
    lines = [
        f"# Next Isolated Probe Recommendations - {date}",
        "",
        f"Base: `{CURRENT_BASE}` ({CURRENT_SCORE:.2f}), after the 12:23 UTC yuzu9 task017 organic cost-displacement batch.",
        "",
        "Known exclusions used for this queue:",
        "",
        "- `task017`: organic periodic-hole-fill cost displacement is public-positive in yuzu9.",
        "- `task003`: organic periodic-row-extension cost displacement is public-positive in lemon9.",
        "- `task052`: organic uniform-row cost displacement is public-positive in nectarine9.",
        "- `task304`: organic outer-product-marker-tile cost displacement is public-positive in orange9.",
        "- `task315`: organic outer-product-marker-tile cost displacement is public-positive in papaya9.",
        "- `task014`: organic fast least-color bbox crop is public-positive in elderberry9; cheaper task014 replacements must pass `tools/_archive/task014_synthetic_guard.py`.",
        "- `task020`: organic rotational-symmetry-completion cost displacement is public-positive in mango9.",
        "- `task021`: organic grid-division-count cost displacement is public-positive in honeydew9.",
        "- `task027`: organic rotation-symmetry-fill cost displacement is public-positive in fig9.",
        "- `task029`: organic marker-rectangle-interior cost displacement is public-positive in imbe9.",
        "- `task229`: organic dominant-color-keep cost displacement is public-positive in quince9.",
        "- `task126`: organic pi-project-bottom cost displacement is public-positive in jackfruit9.",
        "- `task249`: corrected organic horizontal-tile-2x cost displacement is public-positive in lime9; `kiwi9` is excluded as a pipeline-bad scratch-payload package.",
        "- `task251`: organic enclosed-region-fill cost displacement is public-positive in grape9.",
        "- `task190`, `task305`, `task261`, `task141`, `task246`, and `task353`: positive strawberry6-lineage members inherited by elderberry9.",
        "- `task310`, `task024`, `task199`, `task015`, `task220`, `task290`, `task352`, `task230`, `task350`, `task197`, `task312`, `task150`, `task041`, `task037`, `task282`, `task155`, `task095`, `task317`, `task102`, `task381`, `task176`, `task303`, `task221`: positive ladder members consumed through cherry9.",
        "- `task076`, `task133`, `task386`, `task388`, `task389`: orange5 positive members inherited by the cherry9 line.",
        "- `task153`: organic positive member inherited by the cherry9 line; cheap task153 replacements are excluded unless they pass `tools/_archive/task153_synthetic_guard.py`.",
        "- `task145`: public-negative on strawberry6 in xigua6/yuzu6 despite strict local organic correctness.",
        "- `task177`: public-negative on tangerine9 in xigua9 (`5560.09` vs `5573.44`) despite full local verification.",
        "- `task370`: public-negative on strawberry6 in zucchini6.",
        "- `task025`: public-negative on strawberry6 in apple7.",
        "- `task382`: public-negative on the watermelon7 ladder in xigua7 (`5540.94` vs `5553.24`).",
        "- `task387`: public-negative in the xigua7/yuzu7 path (`5528.33` vs `5540.94`).",
        "- `task044`: public-negative on the imbe8 ladder in jackfruit8 (`5540.76` vs `5555.02`).",
        "- `task089`, `task241`, `task380`: older avoid/retest guardrails remain active.",
        "- `task105`: organic gridline-adjacent-fill cost displacement is public-positive in strawberry9.",
        "- `task394`: afr1ste-5044-56 cost displacement is public-positive in tangerine9.",
        "- `task129`: organic const-dominant-color risk candidate is public-positive in raspberry9 despite sub-1000 cost.",
        "",
        "## Recommended Next 5",
        "",
        "| Rank | Base zip | Task | Source ONNX | Expected upside | Cost | Why |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    for row in top:
        lines.append(
            f"| {row['rank']} | `{row['base_zip']}` | {row['task']} | `{row['payload_path']}` | "
            f"+{float(row['expected_local_delta']):.4f} local | {row['cost']} | {row['why']} |"
        )
    if not top:
        lines.append(
            f"| - | `{CURRENT_BASE}` | - | - | - | - | "
            "No unconsumed tracked candidate-task rows remain after current positives and negatives are excluded. |"
        )

    by_task = {row["task"]: row for row in rows}
    lines += [
        "",
        "## Named-Task Rationale",
        "",
    ]
    for task in ("190", "305", "261", "141", "246", "310", "102", "381"):
        if task in by_task:
            row = by_task[task]
            lines.append(
                f"- task{task}: rank {row['rank']}, +{float(row['expected_local_delta']):.4f} local, "
                f"source `{row['source']}`. {row['why']}"
            )
        elif task in KNOWN_NEGATIVE_TASKS:
            lines.append(f"- task{task}: excluded because {KNOWN_NEGATIVE_TASKS[task]}.")
        elif task in KNOWN_POSITIVE_TASKS:
            lines.append(f"- task{task}: excluded because {KNOWN_POSITIVE_TASKS[task]}.")
    lines += [
        "",
        "## Packaging Rule",
        "",
        "- This queue is historical evidence from the retired 2026-05 blend-ladder flow, not the current upload workflow.",
        "- For current results, update the OnnxlyFans `Team_Submission` manifest / current leaderboard state and `The_400_Tasks/task_state.csv` for per-task outcomes. Do not edit the archived candidate CSVs unless deliberately replaying the old queue.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-05-08")
    parser.add_argument("--out-prefix", default="Logs/next_isolated_probe_recommendations_2026-05-08")
    args = parser.parse_args()

    rows = build_rows()
    out = ROOT / args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "task",
        "base_zip",
        "source",
        "payload_path",
        "expected_local_delta",
        "cost",
        "candidate_memberships",
        "state_status",
        "state_reference",
        "known_negative_exclusion",
        "why",
        "sort_key",
    ]
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".md").write_text(build_md(args.date, rows).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out.with_suffix('.csv').relative_to(ROOT)}")
    print(f"Wrote {out.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
