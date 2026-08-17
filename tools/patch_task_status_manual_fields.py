"""Patch manual fields in NeuroGolf_Task_Status.xlsx from session ledger.

The `enrich_task_state.py --refresh-costs` tool re-scores costs from the
current submission.zip but does NOT touch manual fields. This script patches
them based on `The_400_Tasks/current_strategy.md` ship history + `traps.md
§27 cherry13b campaign` results + filesystem scan.

Also fixes a bug in enrich_task_state.py: --refresh-costs updates
current_cost/memory/params but NOT current_points. This script recomputes
current_points = max(1, 25 - ln(current_cost)) for all rows so they match
the refreshed costs.

Run after `enrich_task_state.py --refresh-costs` + `merge_task_status.py`.

Patches:
  - cherry13b_tested: SESSION-ACTIVE tasks per session ledger;
    BULK: all other tasks marked confirmed_positive per §27 47-submit campaign
  - structural_attempt + structural_kaggle_delta: session-ledger ship history
  - unlock_potential: REALIZED for shipped unlocks, DEAD for failed-architecture
    families, LOW elsewhere (enrich's derived value preserved if not patched)
  - prototype_path: filesystem scan of Task_by_Task/taskNNN/prototypes/
  - assigned_to: 4 currently-active workers
  - last_attempted_at: ISO date for 26 session-active tasks

Usage:
    python tools/patch_task_status_manual_fields.py [--xlsx PATH]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import openpyxl


REPO = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx"


# ---------------------------------------------------------------------------
# §27 ZERO-WEIGHT FAMILY — per traps.md §27 cherry13b campaign 2026-05-25T21:40Z
# All 401 tasks were probed. These 13 are the confirmed zero-weight family;
# every other task is confirmed_positive.
ZERO_WEIGHT_FAMILY = {
    "task018": "confirmed_zero_2026-05-11",
    "task023": "confirmed_zero_2026-05-25",
    "task066": "confirmed_zero_2026-05-23",
    "task076": "confirmed_zero_2026-05-23",
    "task133": "confirmed_zero_2026-05-25",
    "task157": "confirmed_zero_2026-05-25",
    "task182": "confirmed_zero_2026-05-23",
    "task219": "confirmed_zero_2026-05-25",
    "task255": "confirmed_zero_2026-05-25",
    "task277": "confirmed_zero_2026-05-23",
    "task319": "confirmed_zero_2026-05-25",
    "task324": "confirmed_zero_2026-05-25",
    "task366": "confirmed_zero_2026-05-11",
}

# Session-summary notes for active/shipped tasks (2026-05-27).
# Appended after any existing notes content (with a separator).
SESSION_NOTES: dict[str, str] = {
    "task018": "2026-05-27: structural rewrite v45 unlock +13.20 LB; v51/v54 Path A bit-equal-chain +0.39 LB. Cumulative +13.59 LB. Architecture: flood+D4+GridSample (multi-template object copy). Worker pushing toward sub-80k tier.",
    "task023": "2026-05-27: dynamic_v5 unlock +14.38 LB (deterministic 2x2-square + length-3-bar exact cover decomposition). top3_qtiny_h4 Path A +2.31; singlescatter Path A +0.08. Cumulative +16.77 LB. Sub-3k structural restructure (3823) regressed -16.58 — reverted. Worker retired.",
    "task066": "Shipped structural unlock at +15.62 LB. Cherry13b zero-weight (2026-05-23).",
    "task076": "Shipped structural unlock at +15.00 LB. Cherry13b zero-weight (2026-05-23).",
    "task133": "2026-05-25: v16 structural QLinearConv UINT8 replacement of zero-weight hash-matcher canonical. Cost 161,071 -> 677,366 (INCREASED but unlocked +11.57 LB hidden weight).",
    "task157": "2026-05-27: 3 architectures failed (structural_v1 combo-table+23,250 synth + combo_vis 49k + anchor_shift_b3 25k) -> 0 LB each. H2 verdict: combo-table-with-exact-cover and anchor-signature families both wrong. Pivot needed to 3rd-tier (contextual / in-context symbol / multi-stage compositional). Worker retired.",
    "task182": "2026-05-25: v17 structural frame-detection + template-matching unlock +14.62 LB. v47 Path A +0.70 LB (30-iteration compression chain, byte-equal A/B 0/1000). Cumulative +15.32 LB.",
    "task219": "2026-05-26: v5 structural rewrite of 4810-node canonical unlock +15.94 LB. Resolved §27f canonical mystery (canonical was at hidden floor). v5_7107/v5_7175 Path A attempts +0/-0.03 (local cost discrepancy unresolved). Cumulative +15.94 LB.",
    "task233": "2026-05-27: H-Global cost-min variant 'b' UNLOCK +13.20 LB EXACT after 5 prior failed attempts (4 structural_slots + count-max). cost_min_v8 Path A +0.33 LB (7-stage compression chain, 1269/1269 bit-equal vs SHIPPED PARENT). Cumulative +13.53 LB. Trap-escape playbook fully validated.",
    "task255": "2026-05-27: v12 structural corridor-fill unlock +14.90 LB (trap escaped from fullsig_pairpack 11k/9k lookup-family 0 LB). v17 Path A +0.08 LB; v71 soft_core_cast Path A +0.18 LB. Cumulative +15.16 LB. Worker on r2-predicate sub-20k frontier.",
    "task277": "Shipped structural unlock at +15.27 LB.",
    "task319": "2026-05-27: v34_polished structural unlock +15.27 LB EXACT (Conv FLOAT color-index projection + canvas pipeline). Worker progression v28 26,360 -> v34 16,716 -36.6%. Critical visible-blind bug fix (Conv color-index padding-vs-color-0 ambiguity, §14f trap). Post-ship analysis confirmed v34 = structural floor; worker retired.",
    "task324": "2026-05-25: v5b structural Slice-then-Cast + plaid/majority simplifications unlock +13.90 LB. v5 Einsum variant BLOCKED by intra-version A/B (179/500 col-unequal divergence — §27b Einsum-vs-ReduceMax tiebreak trap).",
    "task366": "2026-05-27: v72 structural K=3 colors + library + lattice unlock +13.50 LB (trap escaped from v21 sparse-stamp lookup 0 LB). v108 Path A +0.30 LB. Cumulative +13.80 LB. v198 sub-50k attempt -13.81 LB regression (A/B vs INTERMEDIATE not vs shipped parent — methodology trap); reverted to v108. Worker on sub-50k frontier preserving full-component-fit gate.",
    "task025": "Path A shipped +1.03 LB (2026-05-25).",
    "task051": "Path A shipped v28+v34 +0.92 LB (2026-05-24).",
    "task077": "Path A shipped safe+rawwhere+tplsafe19422 +1.61 LB (2026-05-24).",
    "task096": "Shipped +15.83 LB (2026-05-24).",
    "task101": "Shipped +15.19 LB (2026-05-24).",
    "task110": "2026-05-25T19:32Z: v40 redemption +0.95 LB EXACT after v25/v28 -12.80 regressions. New §14e canonical-extract A/B baseline rule established. 3000-grid cumulative A/B = 0/3000 byte-equal vs canonical v9.",
    "task118": "Shipped +15.51 LB (2026-05-24).",
    "task158": "Shipped +14.56 LB (2026-05-24).",
    "task175": "Path A shipped v17 +0.32 LB (2026-05-24).",
    "task285": "2026-05-26: keyfirst20 Path A +0.10 LB EXACT calibration (§27d-supplement task-shaped algorithm A/B gate refinement; 998/1000 random-grid diff but task-shape OOD). 6x6/7x7 wildcards confirmed hidden L-inf <= 4 (-0.09/-0.16 LB exact cost-penalty match).",
    "task363": "Shipped v2 +14.31 LB (2026-05-24).",
    "task367": "Shipped v2 +14.24 LB (2026-05-24).",
}

# Per current_strategy.md as of 2026-05-27T20:22Z
# fields: (structural_attempt, structural_kaggle_delta, unlock_potential, last_attempted_at)
SHIP_HISTORY: dict[str, tuple[str, float | str, str, str]] = {
    "task018": ("shipped_unlock_v54_path_a_x2", +13.59, "REALIZED", "2026-05-27"),
    "task023": ("shipped_unlock_singlescatter_path_a_x2", +16.77, "REALIZED", "2026-05-27"),
    "task066": ("shipped_unlock", +15.62, "REALIZED", "2026-05-23"),
    "task076": ("shipped_unlock", +15.00, "REALIZED", "2026-05-23"),
    "task133": ("shipped_unlock_v16", +11.57, "REALIZED", "2026-05-25"),
    "task157": ("3_architectures_dead_H2_verdict", 0.0, "DEAD", "2026-05-27"),
    "task182": ("shipped_unlock_v17_v47_path_a", +15.32, "REALIZED", "2026-05-25"),
    "task219": ("shipped_unlock_v5_path_a", +15.94, "REALIZED", "2026-05-26"),
    "task233": ("shipped_unlock_h_global_cost_min_v8_path_a", +13.53, "REALIZED", "2026-05-27"),
    "task255": ("shipped_unlock_v12_v17_v71_path_a", +15.16, "REALIZED", "2026-05-27"),
    "task277": ("shipped_unlock", +15.27, "REALIZED", "2026-05-23"),
    "task319": ("shipped_unlock_v34_polished_floor_reached", +15.27, "REALIZED", "2026-05-27"),
    "task324": ("shipped_unlock_v5b", +13.90, "REALIZED", "2026-05-25"),
    "task366": ("shipped_unlock_v72_v108_path_a", +13.80, "REALIZED", "2026-05-27"),

    # Positive-weight ship history this session
    "task025": ("shipped_path_a", +1.03, "REALIZED", "2026-05-25"),
    "task051": ("shipped_path_a_x2", +0.92, "REALIZED", "2026-05-24"),
    "task077": ("shipped_path_a_x3", +1.61, "REALIZED", "2026-05-24"),
    "task096": ("shipped", +15.83, "REALIZED", "2026-05-24"),
    "task101": ("shipped", +15.19, "REALIZED", "2026-05-24"),
    "task110": ("v40_redemption_shipped", +0.95, "REALIZED", "2026-05-25"),
    "task118": ("shipped", +15.51, "REALIZED", "2026-05-24"),
    "task158": ("shipped", +14.56, "REALIZED", "2026-05-24"),
    "task175": ("shipped_path_a", +0.32, "REALIZED", "2026-05-24"),
    "task285": ("shipped_keyfirst20_path_a", +15.22, "REALIZED", "2026-05-26"),
    "task363": ("shipped_v2", +14.31, "REALIZED", "2026-05-24"),
    "task367": ("shipped_v2", +14.24, "REALIZED", "2026-05-24"),
}

# Currently-active worker assignments (2026-05-27T20:22Z)
ACTIVE_WORKERS = {
    "task018": "worker-task018-compression",
    "task233": "worker-task233-h_global-compression",
    "task255": "worker-task255-r2_predicate",
    "task366": "worker-task366-sub50k-fcc_gate",
}


def find_prototype_path(task_id: str) -> str:
    """Scan filesystem for prototype artifacts, return ;-joined path string."""
    proto_dir = REPO / "The_400_Tasks" / "Task_by_Task" / task_id / "prototypes"
    if not proto_dir.exists():
        return ""
    files = []
    for ext in ("*.py", "*.onnx"):
        files.extend(sorted(proto_dir.glob(ext)))
    # Cap at ~5 entries to keep cells readable
    files = files[:5]
    return ";".join(f"The_400_Tasks/Task_by_Task/{task_id}/prototypes/{f.name}" for f in files)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    args = p.parse_args()

    wb = openpyxl.load_workbook(args.xlsx)
    ws = wb["Tasks"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]

    def col(name: str) -> int:
        return headers.index(name) + 1  # 1-indexed

    c_task = col("task")
    c_cherry = col("cherry13b_tested")
    c_struct = col("structural_attempt")
    c_kag = col("structural_kaggle_delta")
    c_unlock = col("unlock_potential")
    c_proto = col("prototype_path")
    c_assigned = col("assigned_to")
    c_last_attempted = col("last_attempted_at")
    c_notes = col("notes")
    c_cost = col("current_cost")
    c_pts = col("current_points")

    patched_ship = 0
    patched_cherry_bulk = 0
    patched_proto = 0
    patched_assigned = 0
    patched_last = 0
    patched_notes = 0
    patched_pts = 0

    for row in ws.iter_rows(min_row=2):
        task = row[c_task - 1].value
        if not task:
            continue

        # current_points: recompute from current_cost (enrich's --refresh-costs
        # doesn't touch this column, leaving it stale). Formula matches
        # neurogolf_utils: max(1.0, 25 - ln(cost)).
        cost = row[c_cost - 1].value
        if cost is not None and cost > 0:
            new_pts = max(1.0, 25.0 - math.log(cost))
            old_pts = row[c_pts - 1].value
            if old_pts is None or abs(float(old_pts) - new_pts) > 0.001:
                row[c_pts - 1].value = round(new_pts, 4)
                patched_pts += 1

        # SHIP_HISTORY: patches structural_attempt, kaggle_delta,
        # unlock_potential, last_attempted_at (for the 26 session-active tasks)
        if task in SHIP_HISTORY:
            struct, kag, unlock, last = SHIP_HISTORY[task]
            row[c_struct - 1].value = struct
            row[c_kag - 1].value = kag
            row[c_unlock - 1].value = unlock
            row[c_last_attempted - 1].value = last
            patched_ship += 1
            patched_last += 1

        # cherry13b_tested: per §27 campaign, ALL 401 tasks were probed.
        # 13 are zero-weight family (use detailed value), rest are
        # confirmed_positive_2026-05-25 (campaign completion date).
        if task in ZERO_WEIGHT_FAMILY:
            row[c_cherry - 1].value = ZERO_WEIGHT_FAMILY[task]
        else:
            # Only overwrite "not_yet" with bulk confirmed_positive — preserve
            # any other detailed cherry13b notes that may exist.
            current_cherry = row[c_cherry - 1].value
            if current_cherry == "not_yet" or current_cherry is None or current_cherry == "":
                row[c_cherry - 1].value = "confirmed_positive_2026-05-25"
                patched_cherry_bulk += 1

        # prototype_path: derive from filesystem if not already set
        current_proto = row[c_proto - 1].value
        if not current_proto:
            new_proto = find_prototype_path(task)
            if new_proto:
                row[c_proto - 1].value = new_proto
                patched_proto += 1

        # assigned_to: only patch the 4 active workers (don't overwrite stale
        # assigned_to values for tasks the user may want to reassign)
        if task in ACTIVE_WORKERS:
            row[c_assigned - 1].value = ACTIVE_WORKERS[task]
            patched_assigned += 1

        # notes: append session summary (preserve existing content via separator)
        if task in SESSION_NOTES:
            session_note = SESSION_NOTES[task]
            existing = row[c_notes - 1].value
            if existing and isinstance(existing, str) and session_note not in existing:
                row[c_notes - 1].value = f"{existing} | {session_note}"
            else:
                row[c_notes - 1].value = session_note
            patched_notes += 1

    wb.save(args.xlsx)
    print(f"Patched ship_history:     {patched_ship}/{len(SHIP_HISTORY)} tasks")
    print(f"Patched cherry13b bulk:   {patched_cherry_bulk} tasks (was 'not_yet'/empty)")
    print(f"Patched prototype_path:   {patched_proto} tasks (was empty)")
    print(f"Patched assigned_to:      {patched_assigned}/{len(ACTIVE_WORKERS)} active workers")
    print(f"Patched last_attempted_at: {patched_last} session-active tasks")
    print(f"Patched notes:            {patched_notes} session-active tasks")
    print(f"Refreshed current_points: {patched_pts} tasks (recomputed from current_cost)")
    print()
    print(f"Saved: {args.xlsx}")
    print("Run `python tools/merge_task_status.py` to refresh CSV mirror.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
