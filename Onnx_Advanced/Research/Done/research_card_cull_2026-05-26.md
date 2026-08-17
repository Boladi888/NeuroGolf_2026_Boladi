# Research Card Cull — 2026-05-26

> **Action**: Five Todo cards (B1, B2, C2, C3, D2) moved to `Todo/Declined/`. Three new cards (G1, G2, G3) added targeting genuine knowledge gaps for our current state. Total active: 1 (C1) + 4 (F1-F4) + 3 (G1-G3) = 8.

## Why these 5 were declined

All five cover techniques Opus 4.7 already knows from training data and that we've already used in production NeuroGolf models. They would yield card-verification reports (like B1 attempts #1 and #2) rather than novel research.

| Card | Topic | Why declined | Where the knowledge already lives |
|---|---|---|---|
| **B1** | Deep ONNX quantization beyond QLinearConv | Standard ONNX quant set (QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul, ConvInteger, MatMulInteger, DynamicQuantizeLinear) is small and well-documented. ORT contrib ops (QLinearAdd/Mul/Concat etc.) are listed in `ort_reality.md`. Two prior attempts (`b1_quantized_ops_verification_attempt{1,2}_2026-05-25.md`) confirmed the card produces verification reports, not novel research. | `Onnx_Advanced/techniques.md` §quantization patterns; `ort_reality.md` |
| **B2** | Bit-packing booleans into integer words | Standard technique: pack 8 bools → INT8 via shifted `BitwiseOr`, unpack via `BitwiseAnd(1<<k)+Cast(BOOL)`. We've used packed UINT8/UINT16 masks in task255 row-packing, task133 stamp packing, task324 cell packing. No new patterns to discover. | `mod_and_bitwise.md`; per-task prototypes (task255 row-packing, task133 mask packing) |
| **C2** | Connected-component labeling without Loop | Label-propagation via MaxPool sweeps + Hoshen-Kopelman variants are standard. Already used in task324 plaid majority, task133 component labeling, task366 component matching. | per-task prototypes (task324/task133/task366); shipped models in `submissions/submission.zip` |
| **C3** | Pathfinding / wavefront propagation without Loop | MaxPool-based BFS frontier expansion + distance-transform via iterated dilations are standard. Already used in task066 L-path routing, task219 reach propagation, task285 keyfirst reach (3-step 5×5 MaxPool covering L∞≤6). | `Onnx_Advanced/techniques.md` §wavefront; task066/task219/task285 prototypes |
| **D2** | ARC-AGI 2024 Kaggle winners' techniques | 2024 winners (BARC, Greenblatt+Khan, MindsAI, Hodel's DSL) are well-documented and already absorbed into `arc_rule_taxonomy.md` + `arc_winners_techniques.md`. Replaced by G1 focused on POST-Jan-2026 ARC research (genuinely outside Opus 4.7 training cutoff). | `Onnx_Advanced/arc_rule_taxonomy.md`; replaced by G1 |

## Why the 3 new G-cards address actual gaps

| Card | Gap it fills | Why Opus 4.7 likely doesn't know |
|---|---|---|
| **G1** | ARC Prize 2025 + post-Jan-2026 ARC research | Opus 4.7 training cutoff is Jan 2026 — anything published after is outside training data. ARC Prize 2025 winners + late-2025 academic ARC research may contain novel rule-decoding techniques applicable to task157/task255/task319/task366. |
| **G2** | Discriminating-input synthesis for self-consistency-trap avoidance | The self-consistency trap (documented in `traps.md` §27g-supplement 2026-05-26) cost us 3 zero-LB ships today (task366 v21, task255 11k, task255 9k). A3 (synth oracle strategies) covers the general framework but not the SPECIFIC methodology of generating inputs that minimally distinguish candidate rule families. Recent semantic-differential-synthesis research + property-based testing literature has techniques Opus 4.7 may not have deep coverage of. |
| **G3** | ARC-AGI-2 vs ARC-AGI-1 novel rule families | ARC-AGI-2 (2024-2026) reportedly contains harder, more compositional puzzles than 2019 ARC-AGI-1. The specific NEW rule families (vs 2019) that appear in ARC-AGI-2 may inform what we should search for in our remaining trap-active tasks. Opus 4.7 has partial ARC-AGI-2 knowledge but a focused taxonomy of "what's new in v2" would help. |

## Active Todo set after cull

- **C1** (cellular automata) — ACTIVE, in progress
- **F1** (radius-encoded weighted cover) — pending; relevant to task255 corridor-fill
- **F2** (row-signature rank reduction) — pending; relevant to row-based ARC tasks
- **F3** (multistage compositional rules) — pending; relevant to task366 multi-stage object copy
- **F4** (multi-QLinearConv direct detection) — pending; relevant to hash-MLP replacements
- **G1** (ARC Prize 2025+) — NEW
- **G2** (discriminating-input synthesis) — NEW
- **G3** (ARC-AGI-2 novel rule families) — NEW

Total: 8 active cards. README.md card table updated.
