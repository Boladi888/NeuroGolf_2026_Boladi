# Pending library update — moment-correlation monochrome-window detection (task143 v1, 2026-06-06)

**Candidate home: techniques.md (new § near #44/#46) + a task_techniques_index row for task143.**

## The pattern

Any "find an exact TRANSLATE of a runtime template among same-colored cells" rule (task143 family;
task206 is a cousin) does NOT need LIVE-style per-channel grouped-Conv correlation (8× plane
multiplicity). Work on the single color-index plane and detect "this window is one solid color"
with TWO moment correlations against the runtime 0/1 template kernel Tw (nT = ReduceSum(Tw)):

```
cV  = Conv(idx,  Tw)        # sum of window colors at the template taps
cV2 = Conv(idx², Tw)        # sum of squared colors
mono(d) ⟺ cV(d)² == cV2(d)·nT  ∧  cV(d) > 0      # variance-zero identity (Cauchy-Schwarz)
```

A window passes iff all template-tap cells hold the same nonzero color — mixed-color windows
that merely COVER the taps are rejected (occupancy-only `corrB==nT` diverged on 49,247/200k
task143 pairs — measured, don't retry).

## F16 exactness wall

The products are F16-exact only while `(9·nT)² < 2048` ⇒ **safe to nT=5, breaks at nT=6**.
Sweep the corpus nT histogram FIRST (task143: nT∈{3,4} over 200k). If nT≥6 occurs, compute the
two products in F32 (+1,200B) — everything else stays F16.

## Companion tricks that made the rewrite 15,965 → 4,160 (all gates green, ab byte-equal 200k)

- **Negpad 1×1 Conv fuses one-hot→index projection + 30→10 crop** in one op, output [1,1,10,10]
  (§41, grader-confirmed). 400B instead of 4,000B Slice-then-Cast.
- **Origin/self-match kill folded into the existing `>0` Greater**: threshold plane init
  (0s + 1e4 at the self-match position, 100 params) — beats Slice+Pad+Sub (412) and
  bool-mask+And (200) because the Greater node was already needed.
- **Stamp-back instead of per-color masks**: claimed windows → cells via Conv with rot180(Tw)
  (rot180 = one negative-step Slice, #44). Multi-match-safe by construction.
- **Per-cell global-count filter via tiny Gather table**: counts = Einsum('bchw->c', input)
  (free-input read, §42) → Equal vs nT → Gather by Cast(idx→INT32). The INT32 index plane
  (400B) is the spec floor (sub-INT32 gather indices = trap).
- **Write the recolor straight onto the free input/output with Where(PadBOOL(mask), e5-col,
  input)** — no one-hot rebuild; kills the entire input_inner/output_inner family (LIVE paid
  6,000B there). BOOL Pad needs opset ≥13.

Full worked example: `The_400_Tasks/Task_by_Task/task143.md` + `task143/prototypes/build_task143_v1.py`.
