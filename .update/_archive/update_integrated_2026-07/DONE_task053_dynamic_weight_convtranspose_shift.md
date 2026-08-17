# Transferable insight — dynamic-weight ConvTranspose beats Gather for grid-shift tasks

**Source**: task053 batch 2026-07-07 (`bo_ng_cld_worker1`). Audited LIVE 14 (Vamshi, 2026-07-02);
this is a **floor confirmation** (14 exhaustively proven optimal), not a new win. It documents a
technique that had already silently beaten a "proven cost-30 floor" — a cautionary + reusable pattern.

**Pattern (the technique)**: A whole-grid **row (or col) shift** under the ch0=1 one-hot background
encoding does NOT need a length-30 `Gather` index (cost 30). Feed the **input grid in as the
ConvTranspose WEIGHT** and a tiny `impulse` as the data:
```
ConvTranspose(X=impulse[1,1,L,1], W=input[1,C,H,W], B=bias[C])   # opset 10
# closed form:  out[co,oh,ow] = Σ_ih impulse[ih]·input[co, oh+pad_top−ih, ow] + bias[co]
```
The 30×30 row-mixing matrix `T` (identical across all C channels) becomes a **banded Toeplitz**
expressed by an `L`-tap impulse → `L` params instead of materialising `T` (900) or a Gather index (30).
Total cost = `L (impulse) + C (bias)`, memory 0 (the node output IS the graph `output`).

**Why it also fixes the ch0-background translation trap**: a plain `Pad`/dwConv zero-fills the vacated
band and can't synthesise the `ch0=1` background row (the classic shift-task divergence). The impulse
can be tuned so overlapping taps *source* a guaranteed-background input row into the vacated band, and
the per-channel `bias` slides the `>0` decode threshold (the scorer only checks `out>0`, not values).

**Floor recipe / forced-component argument (task053, C=10)**:
- **Impulse ≥ 4 taps**: must both reach the guaranteed-bg row (`pad_top ≥ 2`) AND carry the shift
  (`tap at ih=pad_top+1 ≤ L−1` ⇒ `pad_top ≤ L−2`) ⇒ `L ≥ 4`.
- **Bias mandatory, length C**: the shift leaks the all-bg source row into the padding band
  (pre-bias `+d > 0`); only a per-channel additive constant kills it, and ConvTranspose bias length
  = `C_out` by op contract (a shorter bias = §8h sub-declared-bias UB trap). ⇒ `cost = L + C`.
- Prove it exhaustively when the generator palette is tiny: task053's whole truth set is only **70
  grids** (color∈{1,2}, 2–3 pixels in 6 cells). Enumerate all of them; bias feasibility is analytic
  (`max_{off}O < min_{on}O` per channel). No sub-`(L+C)` config → hard floor.

**CAUTION (why this matters)**: task053 carried a confident "cost 30 is optimal" floor proof for a
month; it only enumerated single-Gather/Slice/Pad/dwConv and **missed the dynamic-weight ConvTranspose**.
When writing a shift/translation floor proof, the Toeplitz-via-ConvTranspose (input-as-weight) family
MUST be in the enumeration, or the proof is incomplete.

**Applies to**: any whole-grid translation/shift/roll task with the ch0=1 background encoding where the
current best is a length-`H` `Gather` row-index. Cost drops from `H` to `taps + C`. Check the shift
family: tasks tagged Translation with a single-Gather/Slice shipped graph.
