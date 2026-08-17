#!/usr/bin/env python3
"""Token-bucket refill for ``.hermes/board/submit_credit_state.json``.

The original board design (per Bovath) was: submit-captain gets 1 submit credit
every ``submit_rate_limit_minutes`` (default 10), stacking up to the daily cap.
The submit-captain LLM agent deducts credits on each Kaggle submit but the
refill side was never implemented — ``available`` just counts down toward 0.

This cron-driven script computes refills deterministically:

  elapsed = now - max(last_refill_at, last_submit_at, created_at)
  refill_tokens = floor(elapsed_minutes / submit_rate_limit_minutes)
  available = min(daily_cap - daily_used, available + refill_tokens)

It is idempotent: running twice in the same window produces no extra refill
because ``last_refill_at`` gets advanced by exactly the number of windows
consumed.

Daily reset: Kaggle's 100-submit cap resets at **UTC midnight** (= 8pm EDT
during DST). This script clears ``daily_used`` to 0 whenever the current
UTC date is past the previously-recorded ``last_daily_reset_at`` date. The
submit-captain agent increments ``daily_used`` on each Kaggle upload; this
reset just zeroes the counter at day boundary so the cap doesn't drift.

Usage:
  python tools/refill_submit_credits.py            # dry-run
  python tools/refill_submit_credits.py --apply    # mutate
  python tools/refill_submit_credits.py --apply --json   # for cron --no-agent
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CREDIT_STATE = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def compute_daily_reset(state: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Detect a UTC-midnight day boundary crossing and zero ``daily_used``.

    Returns a dict with the new ``daily_used`` value and a flag describing
    whether a reset is needed. Caller applies the reset if ``reset`` is
    True. The decision is purely date-based: if the UTC date of ``now`` is
    later than the UTC date of ``last_daily_reset_at`` (or ``created_at``
    as fallback), the counter has rolled over.
    """
    credits = state.get("credits") or {}
    try:
        prior_used = int(credits.get("daily_used") or 0)
    except (TypeError, ValueError):
        prior_used = 0
    anchor_ts = (
        parse_ts(credits.get("last_daily_reset_at"))
        or parse_ts(state.get("created_at"))
    )
    if anchor_ts is None:
        # No anchor — assume today's the first day; don't reset yet, but
        # record now so future ticks have a baseline.
        return {
            "reset": True,
            "new_daily_used": prior_used,
            "prior_daily_used": prior_used,
            "new_last_daily_reset_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "reason": "no last_daily_reset_at recorded; anchoring now without zeroing daily_used",
        }
    if now.date() > anchor_ts.date():
        return {
            "reset": True,
            "new_daily_used": 0,
            "prior_daily_used": prior_used,
            "new_last_daily_reset_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "reason": f"UTC date rolled from {anchor_ts.date().isoformat()} to {now.date().isoformat()}",
        }
    return {
        "reset": False,
        "new_daily_used": prior_used,
        "prior_daily_used": prior_used,
        "new_last_daily_reset_at": anchor_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": f"still within UTC day {now.date().isoformat()}",
    }


def compute_refill(state: dict[str, Any], now: datetime) -> dict[str, Any]:
    credits = state.get("credits") or {}
    policy = state.get("policy") or {}
    try:
        rate_minutes = int(policy.get("submit_rate_limit_minutes") or 10)
    except (TypeError, ValueError):
        rate_minutes = 10
    try:
        daily_cap = int(policy.get("daily_submit_cap") or 100)
    except (TypeError, ValueError):
        daily_cap = 100
    # Pre-refill, apply the UTC-midnight daily reset if appropriate. This is
    # decided in compute_daily_reset() and surfaced in the plan so the
    # apply path can write it out.
    reset_plan = compute_daily_reset(state, now)
    if reset_plan["reset"]:
        daily_used = reset_plan["new_daily_used"]
    else:
        try:
            daily_used = int(credits.get("daily_used") or 0)
        except (TypeError, ValueError):
            daily_used = 0
    try:
        available = int(credits.get("available") or 0)
    except (TypeError, ValueError):
        available = 0
    try:
        reserved = int(credits.get("reserved") or 0)
    except (TypeError, ValueError):
        reserved = 0

    # Anchor: prefer last_refill_at; fall back to last_submit_at; then created_at.
    # Clock safety: if anchor is somehow in the future (drift / manual edit),
    # snap it to `now` and refill 0 tokens this tick.
    anchor = (
        parse_ts(credits.get("last_refill_at"))
        or parse_ts(credits.get("last_submit_at"))
        or parse_ts(state.get("created_at"))
        or now
    )
    if anchor > now:
        anchor = now
    anchor = anchor.replace(microsecond=0)
    elapsed_min = (now - anchor).total_seconds() / 60.0
    new_tokens_raw = int(elapsed_min // rate_minutes)
    # Don't exceed the daily-remaining headroom.
    daily_headroom = max(0, daily_cap - daily_used - available - reserved)
    new_tokens = max(0, min(new_tokens_raw, daily_headroom))
    new_available = available + new_tokens
    # Advance last_refill_at by exactly the windows we actually granted, so
    # remainder time carries forward to the next cron tick (no off-by-one
    # accumulation, no drift).
    from datetime import timedelta
    if new_tokens > 0:
        new_anchor = anchor + timedelta(minutes=new_tokens * rate_minutes)
    else:
        new_anchor = parse_ts(credits.get("last_refill_at")) or anchor
        new_anchor = new_anchor.replace(microsecond=0)
    return {
        "anchor_before": anchor.isoformat().replace("+00:00", "Z"),
        "elapsed_minutes": round(elapsed_min, 1),
        "rate_minutes": rate_minutes,
        "windows_elapsed": new_tokens_raw,
        "tokens_added": new_tokens,
        "available_before": available,
        "available_after": new_available,
        "daily_used": daily_used,
        "daily_cap": daily_cap,
        "daily_headroom_before_refill": daily_headroom + new_tokens,
        "new_last_refill_at": new_anchor.isoformat().replace("+00:00", "Z"),
        "daily_reset": reset_plan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Mutate the credit-state file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text (for cron --no-agent).")
    parser.add_argument("--state", type=Path, default=CREDIT_STATE)
    args = parser.parse_args(argv)
    if not args.state.exists():
        print(f"ERROR: credit state not found at {args.state}", file=sys.stderr)
        return 2
    state = json.loads(args.state.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    plan = compute_refill(state, now)
    reset = plan["daily_reset"]
    needs_write = args.apply and (
        plan["tokens_added"] > 0
        or reset["reset"]
    )
    if needs_write:
        credits = state.setdefault("credits", {})
        if plan["tokens_added"] > 0:
            credits["available"] = plan["available_after"]
            credits["last_refill_at"] = plan["new_last_refill_at"]
        if reset["reset"]:
            credits["daily_used"] = reset["new_daily_used"]
            credits["last_daily_reset_at"] = reset["new_last_daily_reset_at"]
        state["updated_at"] = now.isoformat().replace("+00:00", "Z")
        notes = state.setdefault("notes", [])
        if plan["tokens_added"] > 0:
            notes.append(
                f"{state['updated_at']} refill_submit_credits.py added {plan['tokens_added']} token(s) "
                f"(available {plan['available_before']} -> {plan['available_after']}; "
                f"{plan['windows_elapsed']} window(s) elapsed since {plan['anchor_before']})"
            )
        if reset["reset"] and reset["new_daily_used"] == 0:
            notes.append(
                f"{state['updated_at']} refill_submit_credits.py UTC-midnight daily reset: "
                f"daily_used {reset['prior_daily_used']} -> 0 ({reset['reason']})"
            )
        elif reset["reset"]:
            notes.append(
                f"{state['updated_at']} refill_submit_credits.py anchored last_daily_reset_at "
                f"({reset['reason']})"
            )
        # Keep notes log bounded.
        if len(notes) > 200:
            state["notes"] = notes[-200:]
        tmp = args.state.with_suffix(args.state.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.state)
    if args.json:
        out = {"schema": "neurogolf.refill_submit_credits.v1", "applied": args.apply, **plan}
        json.dump(out, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"elapsed={plan['elapsed_minutes']:.1f}min  windows={plan['windows_elapsed']}  tokens_added={plan['tokens_added']}")
        print(f"available: {plan['available_before']} -> {plan['available_after']}  (daily {plan['daily_used']}/{plan['daily_cap']})")
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
