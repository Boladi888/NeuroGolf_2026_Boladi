# Agenters process proposals

Root policy: active proposal drafts live here. Once promoted into live role docs, rejected, or kept only for
historical context, move the record to `Implemented/` or `Handoff_archive/` with its status line updated.

**PROMOTED 2026-07-03 (implemented + archived):** all five 2026-07-03 ONNX win-rate / HandOff-cleanup drafts (`codex_p43`, `codex_p44`, `codex_external_partner`, `neurogolf_onnx_win_actions_handoff_cleanup`, `isolated_probe_false_confirm_stack_only_gate`) were implemented this pass and moved to `Implemented/`. Full Landed / Adapted / Deferred disposition + the HandOffz-skill junction + the prompt wiring: [`Implemented/onnx_win_actions_implemented_2026-07-03.md`](Implemented/onnx_win_actions_implemented_2026-07-03.md). The two draft blurbs below are kept as historical context (now superseded).

**(superseded -- promoted above) OPEN (2026-07-03):** [`neurogolf_onnx_win_actions_handoff_cleanup_2026-07-03.md`](neurogolf_onnx_win_actions_handoff_cleanup_2026-07-03.md) — action-only ONNX win-rate fixes from the HandOff cleanup/lifecycle pass: Python goal clearing, pre-board collision pruning, stack-overlay-only staging, review rescue, gate leases, external intake, heat-scored task selection, compact packets, downstream outcome tracking, and runner closeout as the next-batch source of truth.

**(superseded -- promoted above) PARTIALLY LANDED (2026-07-03):** [`isolated_probe_false_confirm_stack_only_gate_2026-07-03.md`](isolated_probe_false_confirm_stack_only_gate_2026-07-03.md) — the 0-base `probe-isolated` verdict (`confirmed_public`) gives **false CONFIRMs**: 9 isolated-CONFIRMED candidates this session (p42 ALPHA) hidden-failed the full-submission overlay at −17…−36 pts, and `reconcile-banked` re-surfaced known failures as "leaked gold." Regression of the 2026-06-24 gold-gate proposal (byte-binding landed; the batch-attribution + isolated-vs-stack gaps it named were left open). **Landed in `captain_ctl.py`:** the `probe-stack` per-task stack-overlay gate + `probe_kind` tagging (only `probe_kind==stack` stages), isolated→`isolated_smoke_ok`, and `reconcile-banked` hidden-fail suppression; docs `Submit_Captain/README.md`+`SPAWN_PROMPT.md` updated. **Still open:** the `Prompts/*Supervisor*.txt` sweep (code enforces the gate regardless), the zero-submit local pre-filter, overlay-harness verification, and the `_shipped/` audit. **Do not archive yet.**

Current state (2026-07-02 late): the three post-run ONNX win-rate critique drafts (p39/p42/p43) were reviewed
and partly implemented, then promoted to `Implemented/`. Full disposition:
[`Implemented/onnx_win_rate_actions_implemented_2026-07-02.md`](Implemented/onnx_win_rate_actions_implemented_2026-07-02.md).
- **Landed:** generous role-aware worker/reviewer timeouts (the 45-min cap was reaping *productive* workers mid-gate — now 4 h worker / 3 h reviewer via `_timeout_for`, in both runners + the 3 base and 3 live configs); unambiguous packet cost fields (`live_cost`/`candidate_profile_cost`/`cost_delta`/`points_delta`, `expected_delta` deprecated) with runner-stamped `batch_id`/`runner_id` + derived `cost_delta`; gate ladder + visible-zero fast-fail + orphan-safe **foreground-only** gates + dead-route `hard_records/` in both lanes' worker/reviewer/rework prompts and `Onnx_Advanced/WORKER_PROTOCOL.md`.
- **Rejected:** hard 15-min / ban-search cutoffs (contradict the owner's generous-timeout directive); hard-failing packets that omit the new fields (would route every in-flight reviewer packet to `needs_packet_fix` mid-batch — kept additive instead).
- **Deferred:** `run_gate_guarded` Job-Object wrapper (`_terminate_tree` + foreground rule cover the 80%), batch collision-pruning / stale board-card hygiene, and the larger captain-preflight / heat-score / queue-split / budget-cap features.
The old 2026-06-30 HandOff audit drafts remain archived/promoted.
The live HandOff docs now cover several platform requests (additive team members, roster merge defaults,
external-key goal/task create-or-resolve, stateless MCP, and tombstoned `410` resources). This repo now
integrates the pieces the local runner can own: prompt-folder config prep via `Handoff/runner/prepare_handoff_batch.py`,
stable `external_key` fields, stale-id resolve/repoint on `404`/`410`, strict packet schema validation, opt-in
board/local drift checks, `-ExitWhenDrained`, corrected `-RunCodex`/`-RunClaude` launch commands, and active-config
monitor commands. The remaining platform/server/UI ideas are archived as historical context unless the owner
explicitly reopens them; they are not blockers for handing supervisors the `Prompts/Handoff/` launch prompts.

2026-07-02 p37/p38 token-efficiency pass (promoted to `Implemented/`):
- [`Implemented/claude_p37_process_token_efficiency_2026-07-02.md`](Implemented/claude_p37_process_token_efficiency_2026-07-02.md) and [`Implemented/codex_p38_process_token_efficiency_2026-07-02.md`](Implemented/codex_p38_process_token_efficiency_2026-07-02.md) — firsthand p37 (Claude) + p38 (Codex) run reviews. **Landed:** `captain_ctl.py reconcile-banked` (staged-but-not-banked leak audit — first run flagged ~26 tasks / ~+4.8 pts of historical un-merged staged gold, verify before re-merge), `Handoff/scripts/Stop-HandoffBatch.ps1` (surgical safe-abort), and strict-gain-vs-LIVE-manifest + dup/tie + narrow-rework hardening in both reviewer prompts. **Specced between-waves** (launch-critical runner code): per-task budget cap, scout gate, batch pruning, compact packets, do-not-repeat block — in `Handoff/runner/README.md`. **Rejected:** min-gain probe skip.

Archived HandOff records:
- [`Handoff_archive/HANDOFF_AUDIT_2026-06-30.md`](Handoff_archive/HANDOFF_AUDIT_2026-06-30.md): archived 2026-07-02 after local runner/prompt integrations landed; kept as historical evidence for active-claim leases, presence/self-healing, CAS/idempotency, and board truth concerns.
- [`Handoff_archive/codex2_handoff_parallel_lane_audit_2026-06-30.md`](Handoff_archive/codex2_handoff_parallel_lane_audit_2026-06-30.md): archived 2026-07-02 after local lane/config integrations landed; remaining platform/UI ideas are historical context unless reopened.
- [`Handoff_archive/claude_handoff_platform_audit_2026-06-30.md`](Handoff_archive/claude_handoff_platform_audit_2026-06-30.md): archived 2026-07-02 after local stale-id, launch, monitor, and packet-validation fixes landed; remaining platform deletion/backfill/MCP ideas are historical context unless reopened.
- [`Handoff_archive/claude_handoff_audit_actionable_2026-06-29.md`](Handoff_archive/claude_handoff_audit_actionable_2026-06-29.md): runner/deployment defects — kept as historical context for SKILL-1…10 (the friend's workflow package) and prior local fixes.
- [`Handoff_archive/claude_handoff_cld_lane_findings_2026-06-28.md`](Handoff_archive/claude_handoff_cld_lane_findings_2026-06-28.md): cld-lane bring-up — historical context for A1–A10 HandOff broker/protocol feedback.

Archived 2026-06-30:
- [`Implemented/claude_staged_gold_gate_hidden_fail_leak_2026-06-24.md`](Implemented/claude_staged_gold_gate_hidden_fail_leak_2026-06-24.md) — **RESOLVED.** The P0 byte-bound probe gate (`probe_sha` vs onnx sha + `confirmed_public` requirement) + ORT load-gate + `reconcile --apply` quarantine + honest default all landed in `captain_ctl.py` on 2026-06-25 (verified by code-read this session). The P0c `dont_probe` retry-gate was **owner-rejected** 2026-06-20 (no task is ever permanently blocked).

Prior pass (2026-06-21): **no open proposals.** The four that were open were implemented this pass and moved to `Implemented/`:

- `2026-06-20_prep_run_template_followups.md` — **F1** (`CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` in `launch.sh.template`, stops the 600s mid-wave death) + **F2** (`launch_packet.template.md` quotes the exact Submit Captain packet keys, and `captain_ctl.py` normalizes `candidate_path`/`static_cost`/`captain_lane` aliases).
- `2026-06-19_supervisor_run_pulse_helper.md` — `runs_status.py pulse <run_id> [--audit-if-terminal]`.
- `2026-06-19_submit_captain_probe_status_command.md` — `captain_ctl.py probe-status [--json]`.
- `2026-06-19_compact_supervisor_launch_request.md` — short-form launch in `Prompts/README.md`.

Also landed this pass (the Claude orchestrator-assessment follow-ups, owner-authorized 2026-06-21):

- **Captain ledger writeback** — `captain_ctl.py stage/reject` stamp the originating run's `final_report.csv` `captain_status` (the L1 drift fix); documented in `PACKET_SCHEMA.md`.
- **run_audit reproducibility** — paths derive from `__file__` (cwd-independent; the C12 bug), and `external_owner` cross-lane rows downgrade staged-path-missing to a WARN.
- **The 6 stale runs are reconciled + closed** (`beta_p5_cli`, `beta_p7_codex55_cli`, `beta_p7_codex_cli_recovery1`, `claude_p2`, `claude_p2_cli`, `claude_p4_cli`) — ground-truth verified vs `Staged_Boladi/_shipped` + manifest; all pass `run_audit` and have `RUN_CLOSED.md`. **This replaces the prior "still fail run_audit / owner bookkeeping (L1/L3/L4)" note** — that drift is resolved, and the writeback fix above stops it recurring.

Open follow-ups (parked, not blocking a launch):

- Consolidate the four `Prompts/*Supervisor*.txt` into one parameterized `{lane}×{backend}` template (drift risk to launch-critical files; do between waves).
- Drain the Librarian inbox (61 lessons: 5 Claude + 56 Codex) — a dedicated Librarian session (`librarian_ctl.py digest`); §-assignment + dedup is knowledge work, not a mechanical sweep.
- Optional `prep_run.py --request <file>` to record the short-form launch fields into the run folder.

Promoted records:

- `Implemented/claude_staged_gold_gate_hidden_fail_leak_2026-06-24.md` (probe-sha byte-binding + ORT load-gate + reconcile quarantine + honest default; P0c retry-gate owner-rejected) — archived 2026-06-30
- `Implemented/2026-06-20_prep_run_template_followups.md` (F1 ceiling + F2 packet keys/aliases)
- `Implemented/2026-06-19_supervisor_run_pulse_helper.md` (runs_status pulse)
- `Implemented/2026-06-19_submit_captain_probe_status_command.md` (captain_ctl probe-status)
- `Implemented/2026-06-19_compact_supervisor_launch_request.md` (Prompts/README short-form)
- `Implemented/2026-06-13T034453Z-wall-status-run-audit-vocab.md`
- `Implemented/2026-06-19_beta_supervisor_prompt_parameterization.md`
- `Implemented/2026-06-19_captain_ctl_kaggle_rows_warning_line.md`
- `Implemented/2026-06-19_codex_run_prep_helper.md`
- `Implemented/2026-06-19_supervisor_setup_token_reduction.md`
- `Implemented/agent_type_consistency_2026-06-11.md`
- `Implemented/captain_ctl_promotion_notes.md`
- `Implemented/claude_audit_agenters_full_2026-06-12.md`
- `Implemented/claude_audit_agenters_system_2026-06-10.md`
- `Implemented/claude_audit_agenters_system_2026-06-11.md`
- `Implemented/claude_audit_codex_orchestration_2026-06-10.md`
- `Implemented/claude_p2_run_kinks.md`
- `Implemented/claude_p4_cli_test_findings_2026-06-11.md`
- `Implemented/cli_default_allowlist_transcripts_2026-06-10.md`
- `Implemented/cli_orchestrator_manifest_path_unreachable_2026-06-11.md`
- `Implemented/codex_audit_agenters_system_2026-06-12.md`
- `Implemented/codex_p1_native_supervisor_followups_2026-06-11.md`
- `Implemented/dual_lane_coordination.md`
- `Implemented/librarian_not_running_onnxadv_stale_2026-06-11.md`
- `Implemented/orchestration_cli_worker_review_plan.md`
- `Implemented/profile_4input_undercount_and_snapshot_merge_gap_2026-06-11.md`
- `Implemented/spread_doublecount_probe_batch_2026-06-11.md`
- `Implemented/supervisor_live_run_hardening.md`
- `Implemented/supervisor_parent_spawn_model.md`
- `Implemented/token_saving_coordination.md`

Live homes created or updated from those records:

- `Agenters/README.md`
- `Agenters/Supervisor/README.md`
- `Agenters/Supervisor/SPAWN_PROMPT.md`
- `Agenters/Orchestrator/README.md`
- `Agenters/Orchestrator/SPAWN_PROMPT.md`
- `Agenters/Orchestrator/TROUBLESHOOTING.md`
- `Agenters/Orchestrator/WORKER_BRIEF.md`
- `Agenters/Orchestrator/runs/README.md`
- `Agenters/Tough_Reviewer/README.md`
- `Agenters/Submit_Captain/README.md`
- `Agenters/Submit_Captain/SPAWN_PROMPT.md`
- `Agenters/Submit_Captain/PACKET_SCHEMA.md`
- `Agenters/Submit_Captain/captain_ctl.py`
- `Agenters/Submit_Captain/_probe/README.md`
- `Agenters/doc_lint.py`
- `.gitignore`

New tooling from the 2026-06-19 setup-ergonomics landing:

- `Agenters/Orchestrator/runs_status.py` (new — liveness + collision signal)
- `Agenters/Orchestrator/prep_run.py` (new — Part-A scaffolder)
- `Agenters/Orchestrator/run_audit.py` (updated — `wall`/`floor` DROP backstop)
- `Agenters/Orchestrator/runs/_template/launch_packet.template.md` + `launch.sh.template` + `launch.ps1.template` + `start_orchestrator.ps1.template` (new)
- `Agenters/Orchestrator/runs/README.md` (updated — tooling + status-vocab steering)
- `Agenters/Submit_Captain/poll_kaggle.py` (updated — `fileName,` preamble skip)
- `Prompts/ALPHA Supervisor Claude-.txt`, `Prompts/ALPHA Supervisor Codex-.txt`, `Prompts/BETA Supervisor Claude-.txt`, `Prompts/BETA Supervisor Codex-.txt` (updated — prep_run fast-path + precedence clause; BETA-Codex re-parameterized)

Moved out of proposal root:

- `Agenters/Submit_Captain/dont_probe.example.json`
- `Agenters/_archive/proposal_reference_scripts/captain_ctl.py`
- `Agenters/_archive/proposal_reference_scripts/librarian_ctl.py`

Removed:

- `Agenters/Token_Use.csv` (empty placeholder)
