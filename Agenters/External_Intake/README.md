# External Intake

Status: live path (created 2026-07-03 from `Agenters/_proposals/codex_external_partner_onnx_win_actions_2026-07-03.md` P0 "Add A Repo-Less Builder Intake Path" + `neurogolf_onnx_win_actions_handoff_cleanup_2026-07-03.md` P0 "Use External/Partner Agents Only Through Local Intake")

## What this path is

Web-only external / partner agents — Claude-app, ChatGPT web, or paid HandOff
partners — cannot clone this repo, run `ng_gate.py`, import the local ONNX
helpers, or profile real score. Their useful output arrives as prose plus fenced
code in a chat page or a HandOff deliverable. Without a structured bridge that
output stays as text and has to be reinterpreted by a local worker, and paid
work can look "done" on the board while producing nothing a local agent can build.

**External Intake turns a repo-less external packet into a task-local
reproducible build attempt (or a hard record) in one command**, gated on the
current bar. It is the *only* sanctioned way external/partner work enters the
task-local build and gate folders.

The path is:

```
external chat / HandOff deliverable
        │  (copy the packet down, do NOT paste code into shared folders by hand)
        ▼
tools/external_packet_intake.py --task N --packet PATH [--agent ID] [--out DIR]
        │
        ▼
The_400_Tasks/Task_by_Task/taskNNN/external_intake/<agent>/<timestamp>/
   ├── build_next.ps1     (runs the extracted builder code)
   ├── first_gate.ps1     (first local validation command)
   └── review_summary.md  (contract fields + current-bar check + decision)
```

## The three packet classes

Every external result must resolve to exactly one class (see
`intake_contract.md` for the required fields of each):

- **`ready`** — an ONNX file or a builder script plus a complete return
  contract. Intake extracts the builder code into `build_next.ps1` and the
  candidate is taken through local gates.
- **`probe`** — an implementation-ready graph plan with the exact local command
  to run first. Intake writes the plan and `first_gate.ps1`; a local worker
  builds it without guessing.
- **`discard`** — searched formulation families, the exact failure reason, and
  one next route worth trying. Intake records it so the failure becomes a
  hard record and does not get rediscovered.

A packet that is "analysis only" — no graph ops, no charged tensors, no
validation command — is not one of these classes and is rejected by the tool
(nonzero exit). See the rejection labels in `intake_contract.md`.

## The gating rule: local reviewer runs intake BEFORE verifying a paid task

**A local reviewer MUST run intake on an external packet before marking the
paid HandOff task `verified`.** Verification releases payment (see
`Handoff/skills/handoff-workflows/references/budgets-and-payments.md`), so it is
the payout trigger, not a status convenience. HandOff `status` alone never
justifies payment.

Order of operations for a paid external task:

1. Copy the packet down locally (never paste chat code straight into shared
   folders — intake is the record).
2. Run `python tools/external_packet_intake.py --task N --packet PATH --agent ID`.
   If it rejects the packet (missing graph-op / charged-tensor fields), the task
   is `reject_unusable` and payment is **not** released.
3. Refresh the live bar with `python tools/ng_current_bar.py --task N --json`
   (the generated `review_summary.md` prints this command). If the packet's
   claimed improvement is **not strictly below** the live bar, classify
   `reject_stale_bar` and do not spend full gates.
4. For a surviving `ready` packet, take the candidate through at least smoke
   gates (`build_next.ps1` → `first_gate.ps1`); for `probe`, confirm the local
   probe reproduces. Only then does a paid-task accept label apply
   (`accept_paid_ready` / `accept_paid_probe`).
5. Record the outcome in `Agenters/_proposals/external_partner_spend_ledger.csv`.

The submit captain ignores external task status entirely until a packet is
locally copied into task-local intake storage and a reviewer note exists.
External workers never submit, never touch team best, and never stage — they
return candidate artifacts only; the submit captain decides staging after local
gates.

## How this connects to the HandOff broker's external-worker flow

The broker side of this — building the input packet, creating the paid task
under an outside-partner goal, and the payment chain — is documented in the
HandOffz skill (in-repo via junction):

- `Handoff/skills/handoff-workflows/references/external-worker-packets.md`
  — assembling the input packet, the return contract embedded in the task body,
  artifacts-vs-deliverables, and the read-only post-create verification.
- `Handoff/skills/handoff-workflows/references/budgets-and-payments.md`
  — the account → project → goal → task allocation chain and the rule that
  **verification releases payment**. Keep `HANDOFF_TESTNET_ONLY=1` unless the
  owner explicitly authorizes real payment.

The broker delivers the packet **out**; External Intake brings the result
**back in** and gates it before any `verified` (and therefore any payout) is
allowed. The two halves share the same return contract: what the broker asks the
worker to return in `external-worker-packets.md` is exactly what
`intake_contract.md` requires the packet to contain to be acceptable.

## Files in this cluster

- `README.md` — this file.
- `intake_contract.md` — required return-contract fields + rejection labels.
- `tools/external_packet_intake.py` — the intake CLI.
- `Agenters/_proposals/external_partner_spend_ledger.csv` — payment-to-outcome ledger.

See also: `Prompts/WebPrompts/_workflow/SELF_CONTAINED_WEBPROMPT_PACK_WORKFLOW.md`
(the outbound pack workflow, including the machine-readable pack files every
external pack must carry) and `The_400_Tasks/Task_by_Task/taskNNN/hard_records/`
(where external negative routes are recorded).
