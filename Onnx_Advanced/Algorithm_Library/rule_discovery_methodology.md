# rule_discovery_methodology.md — how to figure out what a task computes

The hardest part of a radical rewrite is NOT writing ONNX. It is figuring out what the task computes. Once you know the rule, the ONNX is mechanical.

## Where the data lives

```
data/competitions/neurogolf-2026/taskNNN.json
```

Each `taskNNN.json` has:
- `train` — labeled pairs the public scorer evaluates against.
- `test` — additional labeled pairs.
- `arc-gen` — ARC-GEN-100K bundled subset (extends train + test for visible validation).

PLUS there is a **private held-out suite** that you cannot inspect, used for the Private Leaderboard. Your algorithm must generalize.

## The procedure

### Step 1: read every pair, not just one

A common worker mistake is to study only the first train pair and infer a rule. ARC tasks regularly have edge cases (rotation invariance, color permutation, neighborhood connectivity) that only appear in the second or third pair.

```python
import json
with open(f"data/competitions/neurogolf-2026/task{nnn}.json") as f:
    data = json.load(f)
print(f"train: {len(data['train'])}, test: {len(data['test'])}, arc-gen: {len(data['arc-gen'])}")
for i, pair in enumerate(data['train'] + data['test']):
    inp, out = pair['input'], pair['output']
    print(f"Pair {i}: input {len(inp)}x{len(inp[0])} → output {len(out)}x{len(out[0])}")
```

### Step 2: print pairs visually

A simple ASCII representation makes patterns visible. Build a small script:

```python
def show(grid, title=""):
    print(title)
    for row in grid:
        print(" ".join(str(c) if c > 0 else "." for c in row))

for i, pair in enumerate(data['train'][:3]):
    show(pair['input'], f"--- input {i} ---")
    show(pair['output'], f"--- output {i} ---")
```

Look for:
- Color changes (color permutation? specific recoloring?)
- Shape changes (same shape, different size, completely new shape?)
- Cell additions (fill regions, draw lines?)
- Cell removals (erase certain colors?)
- Position changes (translation, rotation, mirroring?)

### Step 3: write down the rule in plain English

Force yourself to articulate the rule in a sentence:

- "For each cell of color c in the input, output color π(c)." → color permutation (algorithm_patterns.md §1).
- "Find the bounding box of the non-bg cells, recolor inside to color X, leave outside as bg." → bounding-box fill (§2).
- "Each row is periodic with period P; some cells are bg (missing); fill them using the period." → periodic-fill (§4).
- "If a cell is the only color-N cell with no same-color neighbor, recolor to color M." → isolated-cell-recolor (§7b).

If you can't articulate the rule clearly, you don't understand it yet. Go back to Step 1.

### Step 4: write a numpy reference

Before any ONNX work, prove you can compute the right answer in numpy:

```python
def task_nnn_reference(input_grid):
    """Implements task NNN's rule. Returns expected output grid."""
    # ... your algorithm here
    return output_grid

# Verify
correct = 0
for pair in data['train'] + data['test']:
    pred = task_nnn_reference(pair['input'])
    if pred == pair['output']:
        correct += 1
print(f"numpy ref: {correct}/{len(data['train']) + len(data['test'])}")
```

You should hit 100% on visible+arc-gen. If you don't, your rule is wrong.

### Step 5: cross-check against LIVE on adversarial inputs

**100% visible correctness ≠ rule captured.** task008's numpy ref passed 266/266 visible but only 3/86 adversarial against LIVE. Visible distribution can be narrower than hidden distribution.

Run your numpy reference against LIVE (the current canonical ONNX from `submissions/submission.zip`) on synthetic adversarial inputs:

```python
import onnxruntime as ort
import zipfile
import numpy as np

# Extract LIVE
with zipfile.ZipFile("submissions/submission.zip") as z:
    z.extract(f"task{nnn}.onnx", "/tmp/")
live_sess = ort.InferenceSession(f"/tmp/task{nnn}.onnx", providers=["CPUExecutionProvider"])

def synth_input(seed):
    """Generate a plausible NeuroGolf input — random grid sizes, colors."""
    rng = np.random.RandomState(seed)
    H = rng.randint(3, 21)
    W = rng.randint(3, 21)
    return [[rng.randint(0, 10) for _ in range(W)] for _ in range(H)]

# Adversarial sweep
agree, disagree = 0, 0
for seed in range(100):
    g = synth_input(seed)
    inp = encode_grid(g)  # convert to [1,10,30,30] F32 one-hot
    pred_ref = task_nnn_reference(g)
    pred_live = decode_output(live_sess.run(["output"], {"input": inp})[0])
    if pred_ref == pred_live:
        agree += 1
    else:
        disagree += 1
        if disagree <= 3:
            print(f"DIVERGE seed={seed}: ref differs from LIVE")
print(f"Adversarial agreement with LIVE: {agree}/{agree+disagree}")
```

**Interpret adversarial agreement carefully — the 95%/30%/0% bands are advisory only**:
- **≥ 95% agreement**: your rule is well-aligned with LIVE — proceed to ONNX implementation.
- **30-95% agreement**: divergences may be either rule bugs OR off-contract perturbations (e.g., `add_isolated_cell` on a task with "exactly 2 blocks" contract). Inspect the failing cases; if they violate the documented task contract, advisory only. Multiple 2026-05-23 ships landed clean at 50-90% adversarial (task174 57.2%, task136 78.9% add_iso, task062 100%).
- **< 30% agreement**: not automatically a fail — depends on WHAT is failing. Task387 shipped CLEAN at 0/2000 because the adversarial was generic off-contract; task170 shipped CLEAN at 0/100 because the generic adversarial violated the documented motif/cursor contract. The signal that matters is whether the decoded rule captures LIVE on the actual hidden-test slice, NOT the adversarial percentage on synthetic perturbations.
- **< 1% adversarial + 266/266 verify + lookup-style architecture**: memorization signature → expect zero-public-weight (10 confirmed family members per `traps.md §14e`) or -13 LB regression. Cherry13b MANDATORY before staging.

**Only true hard gate**: `verify_against_live --live-zip submission.zip` 100%. Adversarial is INFORMATIVE, not gating.

### Step 6: implement ONNX, verify byte-exact

Now write the ONNX (use `algorithm_patterns.md` templates). Verify byte-exact against LIVE using the actual gate:

```
.venv/Scripts/python tools/verify_against_live.py --proto candidate.onnx --tid NNN --live-zip submissions/submission.zip
```

Required: `OK Prototype matches LIVE` AND `Prototype vs LIVE: N/N` with no failures.

## Common rule-discovery traps

### Trap 1: stopping at one pair
You think the rule is "color permutation π" but the second pair has a different π. The actual rule might be "color permutation depending on the count of color X in the input" or "the most-common-color always maps to 1". Look at ≥ 3 pairs before locking in.

### Trap 2: confusing connectivity (4-conn vs 8-conn)
For any "neighbor" or "isolated cell" rule, both 4-conn and 8-conn are plausible. Verify the edge case of "cell with diagonal-only same-color neighbor" — that determines which.

task272 lesson (2026-05-02): initial 8-conn detector failed because train[0]'s isolated cell had a diagonal 2-neighbor; the rule was actually 4-conn.

### Trap 3: approximating a non-rectangular phenomenon as rectangular
Connected-component analysis is structurally hard in ONNX (Unique is banned). Tempting to approximate as "bounding rectangle = the room". This is wrong for L-shaped rooms.

task145 lesson (2026-05-12): rectangular-partition CC approximation passed 267/267 visible but lost -1.18 LB on Kaggle. **Don't approximate CC as bounding-box** unless you've verified on adversarial inputs that the approximation holds.

### Trap 4: hidden bidirectional asymmetry
Tasks with periodic or symmetric patterns often have one direction (horizontal vs vertical) that matters more than the other. Joint scoring favors whichever has more pair-count. Solution: score per-direction independently, pick per-direction winner.

task110 lesson (2026-05-13): joint V+H ArgMax was wrong; separate V and H ArgMax was correct because test pair had asymmetric periods.

### Trap 5: rule that depends on numerical precision
If the rule involves a threshold like "color appears > 5 times", make sure your counting is exact. F16 / F32 round-trips can quantize counts and produce different answers near thresholds. For numerical robustness use INT32 counts via `ReduceSum(Cast(mask, INT32), ...)`.

### Trap 6: "Agree with LIVE" can hide a candidate-better case
If your candidate disagrees with LIVE, ALWAYS compute `cand_correct` and `live_correct` separately on the synthetic adversarial set. Low agree-with-LIVE can mean LIVE is wrong (your candidate is better) or both are wrong on different inputs.

task110 v4 lesson (2026-05-12): adversarial "agree with LIVE" was 8.6% on tile_corruption — but this was because LIVE was wrong, not us.

### Trap 7: "drawn DIRECTLY from an input crop" — read the output-construction BRANCH, not the draw order
When a generator writes BOTH `input` and `output` inside one loop with a `this-goes-to-output-vs-input` conditional (`bside != foreside` and similar), trace EXACTLY which writes land in `input` vs `output` per branch. A field that appears in `output[...] = X` is NOT necessarily present in any input panel — so a #118 crop + #109/#119 free-output Einsum is only valid if every output cell's value AND extent is materially in that crop. **Cheap falsification before building any free-output candidate:** over ~400 `generate()` samples, compare the bbox of the output's marker cells to the bbox of the forecolor region — if forecolor extent exceeds the marker bbox in a large fraction (task366: 380/400), the extent is NOT recoverable from markers and the crop-Einsum is information-insufficient. A 20-line numpy probe that kills a multi-thousand-line build. Guards #133 (crop-to-content-bound) and the #68 floor-falsification discipline. (task366, 2026-06-19.)

### Trap 8: "that plane looks redundant" — prove it with signature collisions
Before deleting a feature plane, compare the winner signature against all non-winner
signatures over a large generator corpus at increasing feature richness. If any
collision remains, the feature set is not sufficient for an exact rule and the
plane is load-bearing. task079 found 19/200k collisions after dropping the column
profile, proving the column plane was forced. This is cheaper than building a
candidate that will later fail hidden truth: the signature-collision test turns
"probably redundant" into a falsifiable necessity proof.

## When visible verification IS sufficient

For some classes of task:
- **Mathematically identical rewrites** (e.g., F16 of the exact LIVE graph, where {0.0, 1.0} → F16 is exact): visible verify is sufficient. Cross-task generalization is guaranteed by the math.
- **onnxsim and other onnx_opt passes** of the exact LIVE graph: same.
- **`bool_andfold`, `mul_mask_to_where`, `slice_cse`** etc. from the bool-first family: math-equivalent on integer inputs, visible verify sufficient.

For these, the existing 100%-visible verify gate is enough. The hidden-test risk is zero.

## When visible verification is NOT sufficient

- **Bool reformulation of a float algorithm** where the mask values weren't provably 0/1. Risk of silent-fail trap.
- **Approximation of a non-trivial algorithm** (CC, true periodicity, etc.).
- **Numpy reference with assumed preconditions** (e.g., "input always has anchor + mover" — but hidden distribution may have inputs that lack this).
- **From-scratch ONNX of a complex rule** where you derived the rule from train pairs only.

For these, generate 20-50 synthetic boundary inputs and verify byte-exact against LIVE BEFORE staging. Coverage matters more than count: cover small grids (3×3, 5×5), edge-aligned objects, color-permutation invariance, diagonal-only neighbors, etc.

## Practical script template

```python
# Save this as task<NNN>_rule_discovery.py in The_400_Tasks/Task_by_Task/taskNNN/prototypes/
"""
Rule discovery for task<NNN>.
Hypothesis: <one-sentence rule>.
"""
import json
import numpy as np

with open("data/competitions/neurogolf-2026/taskNNN.json") as f:
    data = json.load(f)

def show(grid, title=""):
    print(title)
    for row in grid:
        print(" ".join(str(c) if c > 0 else "." for c in row))

def hypothesis(input_grid):
    """My current hypothesis of the rule."""
    # ... compute output ...
    return output_grid

# Verify on visible+arc-gen
correct = 0
total = 0
for pair in data['train'] + data['test'] + data.get('arc-gen', []):
    pred = hypothesis(pair['input'])
    if pred == pair['output']:
        correct += 1
    else:
        # On first failure, show the diverging case
        if correct + (total - correct) < 5:
            show(pair['input'], 'INPUT')
            show(pair['output'], 'EXPECTED')
            show(pred, 'PRED')
    total += 1
print(f"Visible: {correct}/{total}")
```

When `correct == total`, you have a candidate rule. Then proceed to Step 5 (cross-check against LIVE) before any ONNX work.
