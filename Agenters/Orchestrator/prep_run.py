#!/usr/bin/env python3
"""prep_run.py - deterministic Part-A scaffolder for one Orchestrator run.

Collapses the mechanical Supervisor setup (proposals codex_run_prep_helper.md + supervisor_setup_token_reduction.md
P3) into one command: list task IDs from the prompt folders, clear the allowlist against sibling runs, pick a unique
run id, copy the ledgers, snapshot the manifest -> team_best.csv, build the EV table, and render the launch packet +
lane-correct launch scripts from runs/_template/*.template. It does NOT replace Supervisor judgement - it removes the
re-derivation and large command output, then leaves you to review the generated packet (EV table, model policy, probe
boundary) before launch.

Lane shape is enforced, not assumed: Codex launch pins gpt-5.5 with xhigh/extra-high effort, requests the priority/fast tier,
and uses danger-full-access; Claude launch pins --model and forces subscription auth. The generated files are validated
for these before the script reports success.

Usage:
  python Agenters/Orchestrator/prep_run.py --run-id beta_p7g_codex_cli --lane Codex \
      --prompt-folder "The_400_Tasks/Task Prompts/2" --prompt-folder "The_400_Tasks/Task Prompts/3" \
      --worker-concurrency 3 --exclude task042 [--bump] [--dry-run]
Codex runs refuse `--worker-concurrency 1`/`2` unless `--allow-low-codex-concurrency` is passed deliberately.
Stdlib-only; paths derive from __file__.
"""
import argparse
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # Agenters/Orchestrator -> repo root
RUNS = HERE / "runs"
TPL = RUNS / "_template"
sys.path.insert(0, str(HERE))
from run_audit import norm, read_csv         # noqa: E402
from runs_status import iter_runs, state_of, newest_mtime  # noqa: E402

TASK_RE = re.compile(r"[Tt]ask0*(\d+)")
DEFAULT_MANIFEST = REPO.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Backend" / "manifest.csv"
LANE_INBOX = {"Claude": "Updates_Claude", "Codex": "Updates_Codex"}


def stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def tasks_in_folder(folder):
    out = {}
    for p in sorted(folder.glob("*.md")):
        m = TASK_RE.search(p.name)
        if m:
            out[f"task{int(m.group(1)):03d}"] = p.name
    return out


def resolve_run_id(want, bump):
    if not (RUNS / want).exists():
        return want, "requested run id was free."
    n = 2
    while (RUNS / f"{want}_{n}").exists():
        n += 1
    nxt = f"{want}_{n}"
    if not bump:
        raise SystemExit(f"ERROR: run id '{want}' already exists. Re-run with --bump to use '{nxt}', "
                         f"or pass a different --run-id.")
    return nxt, f"requested '{want}' existed -> bumped to '{nxt}'."


def collision_scan(allow):
    """Return (active_hits, prior_art, prior_status_map). active_hits => hard stop."""
    active, prior = [], []
    pstatus = {}
    for run in iter_runs():
        st = state_of(run, 2 * 3600, _NOW)
        rep = read_csv(run / "final_report.csv") or []
        cla = read_csv(run / "task_claims.csv") or []
        seen = set()
        for src, rows in (("report", rep), ("claims", cla)):
            for r in rows:
                t = norm(r.get("task"))
                if not t or t not in allow or (t, run.name) in seen:
                    continue
                seen.add((t, run.name))
                s = (r.get("status") or "").strip()
                if st == "ACTIVE":
                    active.append((t, run.name, s))
                else:
                    prior.append((t, run.name, st, s))
                    if src == "report" and s:
                        prev = pstatus.get(t)
                        mt = newest_mtime(run)
                        if prev is None or mt > prev[0]:
                            pstatus[t] = (mt, f"{s} ({run.name})")
    return active, prior, {t: v[1] for t, v in pstatus.items()}


def ev_table(allow, task_file, team_best, pstatus):
    by_cost = {norm(r.get("task")): (r.get("cost") or "?") for r in team_best} if team_best else {}
    lines = ["| task | team_best | prompt | prior status (most recent terminal elsewhere) |",
             "|---|---:|---|---|"]
    for t in allow:
        lines.append(f"| {t} | {by_cost.get(t, '?')} | `{task_file.get(t, '?')}` | {pstatus.get(t, '-')} |")
    return "\n".join(lines)


def lane_blocks(lane, orch_model):
    inbox = LANE_INBOX[lane]
    if lane == "Claude":
        method = "headless via `claude -p`"
        model = (f"MODEL DOCTRINE (owner 2026-07-10, 'Opus fast + Fable on hard'): default hard/novel workers + their "
                 f"held reviewers to Opus fast mode - `claude-opus-4-8` + headless fast mode "
                 f"(`--settings '{{\"fastMode\":true}}'`; Opus-only, ~2.5x, identical quality, +cost). Escalate known "
                 f"hard-floor / hardest-novel tasks to Fable 5 (`--model fable` / `claude-fable-5`, the heaviest "
                 f"long-autonomous model); `claude-sonnet-4-6` remains the cheap default for mechanical work. 'Fable "
                 f"Ultracode fast' is NOT a real profile (fast mode is Opus-only, not Fable); if the fast-mode "
                 f"`--settings` passthrough isn't wired in the launch path yet, use plain `claude-opus-4-8`. A prompt "
                 f"carrying its own `model:` hint overrides. Janitor (if needed) = `claude-haiku-4-5-20251001`. "
                 f"Your orchestrator process is pinned to `{orch_model}` by the launcher - that governs only "
                 f"coordination, not the subagents you spawn.")
    else:
        method = "via `codex -a never exec`"
        model = ("Codex workers/reviewers MUST run on `gpt-5.5` with "
                 "`model_reasoning_effort=\"xhigh\"`; also pass `service_tier=\"priority\"` to request the "
                 "fast tier on CLIs that expose it. If `codex debug models` or the first launch rejects that model "
                 "or tier, STOP and record the exact CLI error in `issues.md`; do not silently fall back to the "
                 "default model.")
    probe = (f"You write reviewer-approved SHIP packets (JSON, per `PACKET_SCHEMA.md`) to "
             f"`Agenters/Submit_Captain/{inbox}/<UTC>-taskNNN.json`, labeled `SAFE_BATCH_QUEUE` / "
             f"`RISK_SINGLE_PROBE` / `SAFE_FALLBACK_FOR_SPREAD`. You NEVER touch `Staged_Boladi`, `submission.zip`, "
             f"shared ledgers, or `Onnx_Advanced/`.\n"
             f"- **DEFAULT: do NOT run own-stack probes.** The home `submission.zip` snapshot/restore is NOT "
             f"concurrency-safe; if any other process (e.g. a Part-B captain) may touch the home zip, hand off "
             f"instead. Hand each reviewer-approved candidate off as a held packet, but close the run row with "
             f"`final_report.captain_status=external_owner`, `expected_lb_delta` set, `staged_path` pointing at "
             f"the source ONNX, and `blocker=external_owner=<CAPTAIN> owns probe+stage`. Bare `held` is only "
             f"an in-flight value and must not survive run close for a positive delta.\n"
             f"- **HiddenBad tasks MUST be `RISK_SINGLE_PROBE`** (mandatory own-stack probe before staging), never "
             f"`SAFE_BATCH_QUEUE`.\n"
             f"- **Supervisor: confirm who owns probe+stage this wave and replace `<CAPTAIN>` above.**")
    return inbox, method, model, probe


def fill(template, mapping):
    out = template
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def main():
    global _NOW
    import time
    _NOW = time.time()
    ap = argparse.ArgumentParser(description="Deterministic Part-A scaffolder for one Orchestrator run.")
    ap.add_argument("--run-id", required=True, help="desired run id stem, e.g. beta_p7g_codex_cli")
    ap.add_argument("--lane", required=True, choices=["Claude", "Codex"])
    ap.add_argument("--prompt-folder", action="append", default=[], required=True,
                    help="path to a Task Prompts folder (repeatable)")
    ap.add_argument("--worker-concurrency", type=int, default=3)
    ap.add_argument("--allow-low-codex-concurrency", action="store_true",
                    help="permit Codex worker concurrency below 3 for an explicitly authorized constrained batch")
    ap.add_argument("--exclude", action="append", default=[], help="task id(s) to exclude (repeatable or comma-list)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest.csv to snapshot into team_best.csv")
    ap.add_argument("--launch-mode", default="cli-orchestrator-native-workers")
    ap.add_argument("--orch-model", default="claude-sonnet-4-6", help="Claude lane: orchestrator coordination model")
    ap.add_argument("--supervisor", default="", help="label for the packet header, e.g. ALPHA / BETA")
    ap.add_argument("--bump", action="store_true", help="auto-bump the run id if it already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + rendered packet; write nothing")
    args = ap.parse_args()
    if args.lane == "Codex" and args.worker_concurrency < 3 and not args.allow_low_codex_concurrency:
        raise SystemExit("ERROR: Codex runs must allow 3 concurrent workers. Use --worker-concurrency 3 "
                         "or pass --allow-low-codex-concurrency only for an explicit constrained-batch override.")

    # 1. task IDs from prompt folders
    folders, task_file, allow = [], {}, set()
    for raw in args.prompt_folder:
        f = (REPO / raw) if not Path(raw).is_absolute() else Path(raw)
        if not f.is_dir():
            raise SystemExit(f"ERROR: prompt folder not found: {raw}")
        folders.append(raw.replace("\\", "/"))
        found = tasks_in_folder(f)
        if not found:
            print(f"WARN: no Task IDs parsed from {raw}")
        for t, fname in found.items():
            task_file[t] = fname
            allow.add(t)
    excludes = {norm(x) for chunk in args.exclude for x in chunk.split(",") if x.strip()}
    allow = sorted(allow - excludes)
    if not allow:
        raise SystemExit("ERROR: empty allowlist after exclusions.")

    # 2. unique run id
    run_id, idnote = resolve_run_id(args.run_id, args.bump)
    run_dir = RUNS / run_id

    # 3. collision scan
    active, prior, pstatus = collision_scan(set(allow))
    if active:
        for t, name, s in active:
            print(f"HARD STOP: {t} is claimed by ACTIVE run {name} (status={s!r})")
        raise SystemExit(f"ERROR: {len(active)} task(s) collide with an ACTIVE run - resolve before scaffolding.")

    # 4. manifest snapshot (read for EV table; copied into run on write)
    man = Path(args.manifest)
    team_best = read_csv(man) if man.exists() else None
    if team_best is None:
        print(f"WARN: manifest not found at {man} - team_best.csv will be missing; EV costs shown as '?'. "
              f"Add a fresh snapshot before launch.")

    # 5. render
    inbox, method, model_policy, probe = lane_blocks(args.lane, args.orch_model)
    sup_label = (args.supervisor.strip() + " Supervisor").strip() if args.supervisor.strip() else "Supervisor"
    mode_note = ("you are a top-level headless process; spawn NATIVE worker/reviewer subagents. If native subagent "
                 "spawn fails, STOP and record the exact blocker in `issues.md` - do not silently fall back.")
    mapping = {
        "RUN_ID": run_id, "LANE": args.lane, "LAUNCH_METHOD": method, "SUPERVISOR": sup_label,
        "LAUNCH_MODE": args.launch_mode, "LAUNCH_MODE_NOTE": mode_note, "LANE_INBOX": inbox,
        "TASK_COUNT": len(allow), "TASK_ALLOWLIST": ", ".join(f"`{t}`" for t in allow),
        "PROMPT_FOLDERS": "; ".join(f"`{x}/`" for x in folders), "WORKER_CONCURRENCY": args.worker_concurrency,
        "EV_TABLE": ev_table(allow, task_file, team_best, pstatus), "MODEL_POLICY": model_policy,
        "PROBE_BOUNDARY": probe,
    }
    packet = fill((TPL / "launch_packet.template.md").read_text(encoding="utf-8"), mapping)
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", packet)
    if leftover:
        raise SystemExit(f"ERROR: unfilled placeholders in packet: {sorted(set(leftover))}")

    if args.lane == "Claude":
        launch_files = {"launch.sh": fill((TPL / "launch.sh.template").read_text(encoding="utf-8"),
                                           {"RUN_ID": run_id, "ORCH_MODEL": args.orch_model})}
        launch_cmd = f"bash Agenters/Orchestrator/runs/{run_id}/launch.sh"
    else:
        rmap = {"RUN_ID": run_id, "REPO": str(REPO)}
        launch_files = {"launch.ps1": fill((TPL / "launch.ps1.template").read_text(encoding="utf-8"), rmap),
                        "start_orchestrator.ps1": fill((TPL / "start_orchestrator.ps1.template").read_text(encoding="utf-8"), rmap)}
        launch_cmd = f"powershell -NoProfile -ExecutionPolicy Bypass -File Agenters\\Orchestrator\\runs\\{run_id}\\start_orchestrator.ps1"

    # 6. lane-shape guards on generated launch files (strip full-line comments so we guard the command, not the notes)
    def code_of(body):
        return "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    guard_errs = []
    for fname, body in launch_files.items():       # negative guards apply to EVERY launch file (incl. the wrapper)
        code = code_of(body)
        low = code.lower()
        if args.lane == "Codex":
            if fname == "launch.ps1":
                if not re.search(r"(^|\s)-m\s+gpt-5\.5(\s|$)", code):
                    guard_errs.append(f"{fname}: Codex launch must pin -m gpt-5.5")
                if 'model_reasoning_effort="xhigh"' not in code:
                    guard_errs.append(f"{fname}: Codex launch must set model_reasoning_effort=\"xhigh\"")
                if 'service_tier="priority"' not in code:
                    guard_errs.append(f"{fname}: Codex launch must request service_tier=\"priority\"")
            if "claude" in low:
                guard_errs.append(f"{fname}: Codex launch references claude")
        elif "codex" in low:
            guard_errs.append(f"{fname}: Claude launch references codex")
    primary = "launch.ps1" if args.lane == "Codex" else "launch.sh"   # positive shape applies to the primary file
    plow = code_of(launch_files[primary]).lower()
    if args.lane == "Codex" and ("codex" not in plow or "danger-full-access" not in plow):
        guard_errs.append(f"{primary}: missing codex / danger-full-access invocation")
    if args.lane == "Claude" and not all(k in plow for k in ("claude -p", "env -u anthropic_api_key", "--model")):
        guard_errs.append(f"{primary}: missing claude -p / subscription auth / --model")
    if guard_errs:
        raise SystemExit("ERROR: generated launch file failed lane-shape guard:\n  " + "\n  ".join(guard_errs))

    collision_md = render_collision(run_id, folders, allow, excludes, idnote, prior, args.lane)
    mode_md = render_mode(run_id, args, folders, idnote, inbox)

    if args.dry_run:
        print(f"\n==== DRY RUN: would create {run_dir} ====")
        print(f"run_id: {run_id}  ({idnote})")
        print(f"lane: {args.lane}  | tasks ({len(allow)}): {', '.join(allow)}")
        print(f"launch: {launch_cmd}")
        print(f"audit:  python Agenters/Orchestrator/run_audit.py Agenters/Orchestrator/runs/{run_id}")
        print("\n---- launch_packet.md (rendered) ----\n" + packet)
        return 0

    # 7. write
    run_dir.mkdir(parents=True, exist_ok=False)
    for led in ("task_claims.csv", "reviewer_leases.csv", "final_report.csv"):
        shutil.copy2(TPL / led, run_dir / led)
    if team_best is not None:
        shutil.copy2(man, run_dir / "team_best.csv")
    (run_dir / "launch_packet.md").write_text(packet, encoding="utf-8")
    for fname, body in launch_files.items():
        (run_dir / fname).write_text(body, encoding="utf-8")
    (run_dir / "collision_check.md").write_text(collision_md, encoding="utf-8")
    (run_dir / "launch_mode.md").write_text(mode_md, encoding="utf-8")

    print(f"== prep_run: scaffolded {run_dir} ==")
    print(f"  run_id     : {run_id}  ({idnote})")
    print(f"  lane       : {args.lane}   worker_concurrency: {args.worker_concurrency}")
    print(f"  allowlist  : {len(allow)} tasks -> {', '.join(allow)}")
    print(f"  excluded   : {', '.join(sorted(excludes)) or '(none)'}")
    print(f"  team_best  : {'snapshot copied' if team_best is not None else 'MISSING - add before launch'}")
    print(f"  prior-art  : {len(prior)} terminal sibling row(s) (see collision_check.md)")
    print(f"  files      : launch_packet.md, {', '.join(launch_files)}, collision_check.md, launch_mode.md, 3 ledgers"
          + (", team_best.csv" if team_best is not None else ""))
    print(f"\n  REVIEW the packet (EV table / model policy / probe boundary), then:")
    print(f"    launch : {launch_cmd}")
    print(f"    audit  : python Agenters/Orchestrator/run_audit.py Agenters/Orchestrator/runs/{run_id}")
    return 0


def render_collision(run_id, folders, allow, excludes, idnote, prior, lane):
    L = [f"Collision scan for {run_id}", "",
         "Prompt folders supplied:"]
    L += [f"- `{f}`" for f in folders]
    L += ["", f"Run id: {idnote}", "",
          f"Allowlist ({len(allow)}, after excluding {', '.join(sorted(excludes)) or 'nothing'}): {', '.join(allow)}",
          "", "Historical / prior-art ledger rows (terminal in non-active runs - do not duplicate verified work):"]
    if prior:
        for t, name, st, s in sorted(prior):
            L.append(f"- {t}: `{name}` [{st}] status={s or '(blank)'}")
    else:
        L.append("- none")
    L += ["", "No ACTIVE-run collisions were found (prep_run hard-stops otherwise).",
          "Re-check `Agenters/Submit_Captain/_probe` and the lane inbox immediately before any probe - "
          "`submission.zip` and the Kaggle account are shared single resources.",
          f"Generated by prep_run.py at {stamp()}."]
    return "\n".join(L) + "\n"


def render_mode(run_id, args, folders, idnote, inbox):
    sandbox = "n/a (claude -p --dangerously-skip-permissions)" if args.lane == "Claude" else \
              "use -s danger-full-access for spawned Codex CLI processes on this Windows host"
    model = f"pin --model; orchestrator={args.orch_model}; workers/reviewers per packet" if args.lane == "Claude" \
            else "pin gpt-5.5; effort=xhigh; service_tier=priority"
    return "\n".join([
        f"orchestrator_id: {run_id}",
        f"launch_mode: {args.launch_mode}",
        f"lane: {args.lane}",
        f"prompt_folders: {'; '.join(folders)}",
        f"worker_concurrency: {args.worker_concurrency}",
        f"model_policy: {model}",
        f"sandbox_policy: {sandbox}",
        f"downstream_policy: Submit Captain is the only stager/integrator; this lane writes packets to {inbox} "
        f"and probes only if it can guarantee exclusive home-zip access",
        f"run_id_note: {idnote}",
        f"generated_by: prep_run.py {stamp()}",
    ]) + "\n"


if __name__ == "__main__":
    sys.exit(main())
