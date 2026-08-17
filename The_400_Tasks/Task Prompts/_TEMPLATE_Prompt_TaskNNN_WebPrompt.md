# WebPrompt template — self-contained "paste into a browser LLM, get the .onnx back" (2026-06-19)

**What this is.** A *different* prompt style from the in-repo worker prompts. These are
**fully self-contained**: you paste the body below into an external code-running chat
(ChatGPT / Gemini / Claude with a code tool) and it builds, self-verifies, and hands back
`taskNNN.onnx`. The model has **no repo access**, so this template contains **zero** repo
references — no `ng_gate`, no Q: drive, no `Onnx_Advanced/*`, no file paths, no `_score_v2`.
Everything the model needs (rule, examples, I/O contract, scoring, verify harness) is inline. Use the repo library DB only while authoring this template; do not paste DB commands into the browser prompt body.

**To author a new one (do this per task):**
1. Map task → ARC-GEN uuid: `taskNNN` is the **N-th** `from tasks import task_XXXXXXXX` line in
   `external/ARC-GEN/tasks/../task_list.py` (1-indexed: task001 = first import). The rule lives in
   `external/ARC-GEN/tasks/task_<uuid>.py`; `generate()` returns `{"input":grid,"output":grid}`.
2. **Inspect `GUI/previews/taskNNN_preview.png`** (author-side; all 400 exist) to read the key visual
   operation — size change, object-select, mirror/rotate, repetition, count, marker colors. Generate a
   handful of instances (`random.seed(s)` for reproducibility), pick **2–3 of the smallest** that EXPOSE
   that operation (include one edge case for count/overlap/bbox/anchor ambiguity) as worked examples.
   Keep them **byte-exact** — they are the model's test oracle.
3. Run the library DB lookup author-side before filling RULE/HINTS:
   `python Onnx_Advanced/library_db/lib_query.py status`, then `context NNN`, plus `search "<rule/op/cost keywords>" --limit 8` or `op <Op> <DType>` when relevant. If stale, rebuild or read cited Markdown.
4. Fill the `{{...}}` slots below: `{{NNN}}` (3-digit), `{{RULE}}` (verified plain-English
   transform), `{{OUTPUT_SIZE}}`, `{{EXAMPLES}}` (paste the SAME list into BOTH the section-2 block
   and the harness `# {{EXAMPLES}}` marker), `{{HINTS}}` (one line). `{{UUID}}` is just for your notes.
5. **Do not edit** sections 3–6 per task — that's the grader's fixed contract + the verify harness.
6. Reality check: the examples make it *likely* to work, but they can't guarantee the hidden set.
   The harder the rule, the more it leans on `{{RULE}}` being correct and general. Prefer rules
   that are verified against the generator, not guessed.

Authoring policy still applies (`_Before writing prompt.md`); register the batch in `current_strategy.md`.

---------------------------------- PASTE EVERYTHING BELOW ----------------------------------

# NeuroGolf Task {{NNN}} -- build the smallest correct ONNX  (paste-and-run)

You are an expert in ONNX graph construction and ARC-style grid puzzles. Working
entirely in this chat (use your code tool), build ONE ONNX model that performs the
transformation described below, make it **as small as possible** under the cost
metric, verify it on the provided examples, and then **give me the finished
`task{{NNN}}.onnx` file to download.** Do the whole thing yourself end to end; only
ask me something if the spec below is genuinely contradictory. **Before you build, name two
candidate implementations and pick the one with lower estimated `memory + params`; avoid
carrying full 10-channel 30x30 float32 tensors unless the rule truly requires them.**

---

## 1. The transformation (the rule)

{{RULE}}

**Output grid size:** {{OUTPUT_SIZE}}

## 2. Worked examples (ground truth -- real instances of the rule)

Each grid is written as rows of single digits; digit = color id `0`-`9` (`0` is the
background). A grid is exactly its meaningful region (height = #rows, width = #cols);
on the 30x30 canvas everything outside that region is empty/off-grid. The same
`EXAMPLES` list is consumed by the verify harness in section 6.

```python
{{EXAMPLES}}
```

> **How to read the examples:** {{HINTS}}

## 3. The grader's hard I/O contract (match it EXACTLY or the task scores 0)

- **Opset** `ai.onnx` (domain `""`) version **13**; keep IR version <= 10. No custom domains.
- Exactly **one input**, name `input`, dtype **FLOAT (float32)**, static shape **[1, 10, 30, 30]**.
- Exactly **one output**, name `output`, static shape **[1, 10, 30, 30]** (any numeric dtype is fine -- the grader thresholds `> 0`; `bool`/`float16` are cheaper).
- **Every tensor dimension must be a concrete integer.** Any dynamic/symbolic dim makes the model score 0.
- **Input encoding:** one-hot over 10 color channels. For an `h x w` grid, cell `(r,c)` sets channel `= color` to `1.0`; all other channels `0`. Cells **outside** `h x w` (`r>=h` or `c>=w`) are **all-zero across every channel** (off-grid). So the sum over the 10 channels is `1` on-grid and `0` off-grid -- that is how you recover the active region and its height & width.
- **Output decoding (how you're scored):** for each cell the predicted color is the channel whose value is `> 0`, and **exactly one** channel may be positive. The grader decodes your `output` to a grid and does `np.array_equal` against the expected grid over the true output region. Make off-grid output cells all-zero.
- **Banned ops (any one -> score 0):** `Loop, Scan, NonZero, Unique, Compress, Script, Function`, any op carrying a **subgraph** attribute (so no `If`), any **custom function**, and any op whose name contains `Sequence` (e.g. `ReverseSequence`, `SplitToSequence`).
- File must be **< 1.44 MiB**. Initializer names must not be `input` or `output`.

## 4. Scoring objective -- smaller is the whole point

`cost = memory + params`, where
- **params** = total **element count** of all **initializers** (`num_elements`, dtype-blind), and
- **memory** = total bytes of all **intermediate** tensors (everything in `value_info` after shape inference; the named `input`/`output` are excluded).

`points = max(1.0, 25.0 - ln(max(1.0, cost)))`. So **correct + tiny ~ 25 pts**; a single
shape-only op (Identity/Transpose/Slice/Gather/Reshape/Concat) with no real initializers
can reach `cost ~ 0`. Levers, by impact: fewer initializer **elements** via shape/factor
compression (changing F32 init -> F16 does **not** reduce params); **narrow intermediate
dtypes** (`bool`/`uint8`/`int8`/`float16` instead of float32); smaller
intermediate shapes (crop to the active region; don't carry full `[1,10,30,30]` float32
tensors you don't need); fewer nodes. **Correctness first, then shrink** -- a wrong model
scores 0 no matter how small.

## 5. Generalize -- do not memorize

The hidden test feeds **fresh random instances of this same rule** at arbitrary sizes up
to 30x30. Build the GENERAL transformation; do **not** hardcode, hash, or look up the
example grids (that scores 0 on the hidden set). Operate on the `[1,10,30,30]` canvas and
keep the active grid anchored at the top-left -- e.g. don't flip the whole 30-wide canvas,
which would shove the grid to the far edge and break the decoder.

## 6. Verify before you hand it over

Build with the `onnx` Python package (`pip install onnx onnxruntime numpy` if needed).
Then run this harness as-is. It checks the examples, enforces the section-3 constraints,
and prints the cost. Only return the file once it prints `ALL EXAMPLES PASS`.

```python
# ============================ VERIFY HARNESS ============================
# Run this AFTER you build task{{NNN}}.onnx. Do not return the file until
# every example prints OK and the asserts pass. (Build code is your job;
# this only checks correctness + the grader's hard constraints + cost.)
import numpy as np, onnx, onnxruntime as ort
from onnx import shape_inference, TensorProto, helper

MODEL = "task{{NNN}}.onnx"

{{EXAMPLES}}

BANNED = {"LOOP","SCAN","NONZERO","UNIQUE","COMPRESS","SCRIPT","FUNCTION"}

def encode(rows):                      # color grid -> [1,10,30,30] one-hot f32
    t = np.zeros((1,10,30,30), np.float32)
    for r,row in enumerate(rows):
        for c,ch in enumerate(row):
            t[0, int(ch), r, c] = 1.0  # off-grid cells stay all-zero
    return t

def decode(out, h, w):                 # [1,10,30,30] -> h*w grid via (>0), exactly-one
    o = np.asarray(out)[0]; g=[]
    for r in range(h):
        row=""
        for c in range(w):
            idx = np.where(o[:, r, c] > 0)[0]
            row += str(int(idx[0])) if len(idx)==1 else "?"
        g.append(row)
    return g

def itemsize(dt): return np.dtype(helper.tensor_dtype_to_np_dtype(dt)).itemsize

def check(path):
    m = onnx.load(path)
    bad = sorted({n.op_type for n in m.graph.node if n.op_type.upper() in BANNED
                  or "SEQUENCE" in n.op_type.upper()})
    assert not bad, "banned op(s): %s" % bad
    for n in m.graph.node:
        for a in n.attribute:
            assert a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS), \
                "subgraph attr (If/Loop body) in %s -> scores 0" % n.op_type
    assert not m.functions, "custom functions present -> scores 0"
    gi={i.name:i for i in m.graph.input}; go={o.name:o for o in m.graph.output}
    assert set(gi)=={"input"},  "input must be exactly ['input'], got %s" % list(gi)
    assert set(go)=={"output"}, "output must be exactly ['output'], got %s" % list(go)
    shp=lambda vi:[d.dim_value for d in vi.type.tensor_type.shape.dim]
    assert shp(gi["input"])==[1,10,30,30] and gi["input"].type.tensor_type.elem_type==TensorProto.FLOAT, "input must be FLOAT[1,10,30,30]"
    assert shp(go["output"])==[1,10,30,30], "output must be [1,10,30,30]"
    inf = shape_inference.infer_shapes(m)
    for vi in inf.graph.value_info:
        for d in vi.type.tensor_type.shape.dim:
            assert d.HasField("dim_value"), "dynamic/symbolic dim in '%s' -> scores 0 (use static shapes)" % vi.name
    params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)  # params are element counts, dtype-blind
    memory = 0
    for vi in inf.graph.value_info:
        n=1
        for d in vi.type.tensor_type.shape.dim: n*=d.dim_value
        memory += n*itemsize(vi.type.tensor_type.elem_type)
    return params, memory

def verify(path=MODEL):
    p,mem = check(path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ok=True
    for i,(inp,exp) in enumerate(EXAMPLES):
        got = decode(sess.run(["output"], {"input": encode(inp)})[0], len(exp), len(exp[0]))
        if got != list(exp):
            ok=False; print("example %d: MISMATCH"%i); print("  got:", got); print("  exp:", list(exp))
        else:
            print("example %d: OK  (%dx%d)"%(i,len(exp),len(exp[0])))
    cost=p+mem
    print("cost = params %d + memory %d = %d   (points ~ %.2f/25)" % (p,mem,cost,max(1.0,25-np.log(max(1,cost)))))
    assert ok, "examples failed -> keep fixing the GENERAL graph (do not hardcode the examples)"
    print("ALL EXAMPLES PASS - now try to shrink cost, then return the file.")

verify()
# ========================================================================
```

## 7. Deliverable

The single file **`task{{NNN}}.onnx`**, after the harness passes. Briefly note the final
`cost` and the main ops you used. If you cannot get all examples to pass, say so and show
the closest attempt plus where it diverges -- do not hand over a model you could not verify.
