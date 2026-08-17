# External Packet Intake Contract

Status: live (created 2026-07-03)

This is the return contract an external / partner packet MUST satisfy to be
**acceptable** for local intake. It is the inbound mirror of the return contract
the HandOff broker embeds in the task body
(`Handoff/skills/handoff-workflows/references/external-worker-packets.md`) and of
the web-pack return contract
(`Prompts/WebPrompts/.../04_validation/VALIDATION_AND_RETURN_CONTRACT.md`).

`tools/external_packet_intake.py` parses these fields from the packet (markdown
or JSON). A packet missing a **required** field is rejected with a nonzero exit
and a clear message; it never silently produces a build folder.

## Packet format

The packet is a single file (markdown or JSON) plus optional referenced
attachments (an ONNX file, a builder `.py`). Fields may be given as:

- markdown `key: value` lines (e.g. `graph_ops: Mul, Min, Add`), or
- a fenced ```json block or a `.json` packet with the same keys.

Builder code goes in a fenced code block (```python / ```powershell). The tool
extracts the first builder block into `build_next.ps1` and records any
referenced attachment paths.

Field-name matching is lenient (case-insensitive, `_`/space interchangeable, and
common aliases are accepted) so a packet written against the web-pack contract
validates without renaming. Accepted aliases are listed per field below.

## Required fields (packet is REJECTED without these)

| Field | Aliases accepted | Meaning |
|---|---|---|
| `graph_ops` | `chosen_graph_ops`, `graph ops`, `ops`, `op_list` | The ONNX op types the graph uses (e.g. `Mul, Min, Add, ReduceSum`). A packet with no graph ops is analysis-only → rejected. |
| `charged_tensors` | `charged_tensors_expected`, `expected_charged_tensors`, `charged tensors` | Which tensors are charged and the count. The cost model turns on this; absent it, the packet is not scoreable → rejected. |

These two are the hard gate named in both source proposals: **reject missing
graph-op and charged-tensor fields.** The tool exits nonzero if either is
absent or empty.

## Required-for-acceptance fields (packet builds, but flagged incomplete without these)

The tool records these in `review_summary.md` and marks the packet incomplete
(the reviewer should not issue a paid-accept label if any are missing), but they
do not by themselves cause a hard reject:

| Field | Aliases accepted | Meaning |
|---|---|---|
| `expected_params` | `params_expected`, `expected params`, `param_count` | Expected parameter count of the candidate. |
| `expected_memory` | `expected memory`, `mem`, `memory` | Expected memory footprint. |
| `current_bar_used` | `baseline_used`, `current_bar`, `bar_used`, `baseline` | The bar the worker targeted (task best / staged rung). Cross-checked against the live bar via `ng_current_bar.py` at review time. |
| `validation_command` | `first_local_command_to_run`, `first_local_command`, `validation`, `validate_cmd` | The exact local command to run first. Becomes `first_gate.ps1`. |
| `decision` | `packet_class`, `class`, `verdict` | Exactly one of `ready` / `probe` / `discard`. |

## Decision → what intake expects in the packet

- **`ready`**: `graph_ops` + `charged_tensors` + a builder code block **or** a
  referenced ONNX attachment + `validation_command`. The candidate is taken
  through local gates.
- **`probe`**: `graph_ops` + `charged_tensors` + enough construction detail that
  a local worker can build it without guessing + `first_local_command_to_run`.
- **`discard`**: the formulation families searched, the exact blocker, the
  forced charged tensors, and one next route still worth trying. Recorded as a
  hard-record-ready row (see `taskNNN/hard_records/`). A discard that names no
  routes precisely enough for a hard record has `no_packet_value`.

## Rejection labels

Intake and the reviewer use these labels. The first four are the packet-quality
rejects; they map to the paid-verification labels in
`Handoff/skills/handoff-workflows/references/budgets-and-payments.md` and the
proposal's payment-to-verification policy. **A reject label means the paid task
is NOT marked `verified` and payment is NOT released.**

- **`reject_unusable`** — missing required contract fields (no graph ops, no
  charged tensors), no route detail, or otherwise not buildable. This is the
  label the intake tool's hard reject corresponds to.
- **`reject_duplicate`** — the route matches an existing route or an existing
  `taskNNN/hard_records/index.jsonl` row. No new value.
- **`reject_stale_bar`** — the packet's claimed improvement is **not strictly
  below** the live bar from `tools/ng_current_bar.py --task N --json` at review
  time. Do not spend full gates; requeue the route locally if it might still
  produce a lower variant.
- **`no_packet_value`** — a `discard` / "already optimal" / "no gain" report
  that does not eliminate named route families or create a forced-component
  proof. A no-gain report is only paid if it creates hard records that prevent
  future duplicate work.

## Accept labels (for reference; issued by the reviewer, not the tool)

- `accept_paid_ready` — ONNX or builder passed local intake and at least smoke gates.
- `accept_paid_probe` — the implementation-ready plan created a local reproducible probe.

## Fixed guardrails every external packet is held to

The worker was told (via the outbound pack, see
`SELF_CONTAINED_WEBPROMPT_PACK_WORKFLOW.md`): **Do not submit externally / Do
not modify team best / Return candidate artifacts only / Submit captain decides
staging after local gates.** Intake enforces this by construction: it only
produces candidate artifacts and a review summary in a task-local folder; it
never submits, stages, or touches team best.
