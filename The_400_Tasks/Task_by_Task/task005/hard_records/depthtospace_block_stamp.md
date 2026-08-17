# task005 dead route — DepthToSpace block-stamp (2026-07-11)

**Worker:** bo_ng_cld_worker1 (Claude lane).
**Live bar:** cost **2852** (profile, `ng_current_bar --task 5`; manifest-authoritative,
owner Aditya, refreshed 2026-06-22). **Live sha256:**
`d38cdf047af75e26d3bcf919b8b9fe8e70314233612430b87de422317f003d8b`
(= `Team_Best_Onnx/task005.onnx`). Byte-equal micro-win **2843** already staged
(`staged_candidates/task005_deadnode_strip_cost2843`, reviewer2). No
`Team_Submission/Staged_Boladi/task005.json` sidecar present.

## Attempted route
Prompt `Prompt_...Task005_DepthToSpaceBlockStamp` (STRUCTURAL card, floor_est ~1200):
pack the 3×3 motif into channels and use **DepthToSpace** as a cheap expansion step,
replacing the LIVE conv stamp chain (QLinearConv anchor⊗motif=9×9 stamp; QLinearConv
stamp⊗ray-kernel=21×21 paint). This is the one formulation worker2's 2026-07-08 WALL
did NOT cover (it covered direct-30×30 conv and validation-strip). Three DepthToSpace
sub-formulations tried on paper + priced empirically
(`prototypes/_task005_depthtospace_probe.py`).

## Exact failed gate — fast PRICING probe (both prompt kill conditions fire)
Ran the probe the prompt mandates *before* any build. Both stop conditions trigger:

**1) Block size does not divide the target geometry.** The rule stamps a 3×3 motif at
**stride 4**; the forced paint is **21×21** (worker2 99-border proof). DepthToSpace needs
an integer `blocksize b` with `H·b == 21`. The stride-4 period `b=4` gives `21 % 4 != 0`
→ can only emit 20 or 24, never 21. The `b` values that DO divide 21 (`{3,7,21}`) are not
the stride-4 period.

**2) Packed motif == expanded canvas (byte-count identity).** DepthToSpace is a pure
rearrange: `[N, C·b², H, W] → [N, C, H·b, W·b]` has **identical element count**, hence
identical bytes. Empirically for every `b∈{3,7,21}`: packed input `= 441 B ==` expanded
output `441 B`. So emitting the forced 441 B (21×21 U8) paint via DepthToSpace forces a
441 B packed input **in addition** to the 441 paint. LIVE builds the same forced 441 paint
from **188 B** of small scaffolding (81 stamp + 49 ray-kernel + 49 anchor + 9 motif).
**DepthToSpace adds ≥253 B over LIVE for the expansion step alone.** No gate (visible/
truth/ab/battery) run — a candidate that provably costs *more* cannot beat the bar; running
gates would be pure burn (protocol §gate-hygiene).

## Why it's dead (structural, not a tuning miss)
DepthToSpace only wins in golf when the **expanded** form would otherwise need a large
**charged literal template** and the **packed** form is a cheap constant/broadcast. LIVE has
**no literal template** — it already builds the 441 paint from cheap conv scaffolding. The
prompt's premise ("repeated blocks should ride reshape/interleave rather than a charged
literal output template") is **stale**: Aditya's LIVE graph is already reshape/conv-based.
The byte-count identity dooms the *entire* DepthToSpace family at once (motif-build,
stamp-expand, paint-tile all fail identically): a rearrange never compresses, and the 21×21
paint is a forced floor tensor, so there is nothing DepthToSpace can shrink. Library has no
ORT `DepthToSpace` row and no search hits; BoBrain `consistency task005` confirms 2852.

## One remaining hypothesis (unchanged from worker2, NOT DepthToSpace)
The only non-dead lever is **correctness, not compression**: audit whether LIVE's ~1190
truth-misses (truth 99.405%) are inherent tie-break ambiguity or a fixable moment/hash
anchor gap. A byte-*different* but more-correct successor (`cand_misses < LIVE_misses`,
DUAL-TRUTH, cost ≤ 2852) could ship. Out of scope for a DepthToSpace/compression card.

**VERDICT: DROP / REJECT_LOCAL for the DepthToSpace route. Do not re-explore
DepthToSpace for task005 — the byte-count identity dooms the whole family.**

> **UPDATE 2026-07-11 (worker1, same session later):** the "WALL at 2852" was
> subsequently **broken to byte-equal 2727** — but via a NON-DepthToSpace lever
> (`TopK` collapse of the source/second colour ArgMax-penalty machinery; see
> `topk_selection_win.md`). The DepthToSpace conclusion above is UNCHANGED and
> still dead. The new floor is ≈2727.
