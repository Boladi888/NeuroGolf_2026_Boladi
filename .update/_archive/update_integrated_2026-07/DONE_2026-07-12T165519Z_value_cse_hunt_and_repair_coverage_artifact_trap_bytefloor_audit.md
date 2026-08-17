# Transferable insight — a value-level CSE hunt extends the byte-equal floor audit, but repair-coverage artifacts create false folds; test remask candidates with CONSTRUCTED triggers

**Source**: task002 (`bo_ng_cld_worker1`, 2026-07-12). Audited LIVE **9413** (Boladi, sha `8c8340bc…`,
Kaggle-confirmed). **Result: WALL CONFIRMED at 9413** (no fold). Complements the existing byte-equal
floor-audit lesson (`neurogolf.lesson.20260702.task002.byte_equal_floor_audit`) and the task005
per-tensor-audit lesson — this adds the *dynamic* (value-level) half of the audit and its main trap.

## Technique — value-level CSE hunt (catches folds structural CSE misses)
The structural audit (dup-initializers / dead-nodes / identical-op+inputs+attrs CSE / identity nodes)
only catches folds where two nodes are *syntactically* the same. It misses two masks that are
*algebraically* always-equal via different op chains. To catch those: expose every intermediate as a
graph output (use `shape_inference` elem types — UNDEFINED/type-0 outputs make ORT reject the model),
run the graph over many **varied** inputs (random/sparse/dense + structured: box, serpentine), and
flag tensor pairs whose bytes are identical on **every** input. A real fold ⇒ share one, drop the
other (−80B/uint32 mask). Script pattern: `scratchpad/value_cse_hunt.py`.

## THE TRAP — memorized visible-repair seeds ⇒ constant-zero masks ⇒ spurious "always-equal" pairs
Structural-LIVE graphs that patch visible misses with **memorized repair seeds** (fire only when a
specific wall-row signature matches an exact 20-bit value) are NOT exercised by random inputs — the
repair conds are all False, so every `repair_selected*`/`repair_bits`/`repair_open` mask is
constant-**zero**, and any downstream `reach_repaired = reach_5 | repair_open` reads as
`reach_5 | 0 == reach_5`. The naive hunt reports these as folds. **Filter them:** flag any tensor that
is byte-**constant across all inputs** as degenerate (an input-coverage artifact, not a fold), and
size-filter to the masks that actually cost (≥80B here). After filtering, task002's 67 raw "equal"
pairs collapsed to **zero real folds**.

## Removable-remask candidates need CONSTRUCTED triggers, not random inputs
The hunt still surfaced one plausible micro-fold: *is the clip `repair_open = repair_bits & open_bits`
a removable remask?* (mirrors the 9545→9413 `reach_0` fold). Random inputs "confirm" it (repair never
fires) — but that is exactly the coverage blind spot. **Build the adversarial input by hand:** set the
cond-checked row to a trigger value AND force a **seed cell to be a WALL** (the repair seed coords sit
in *other* rows than the cond row, so the cond cannot constrain them). On task002: row1 == 84218 +
cell (7,15)=wall → LIVE (with clip) keeps (7,15) color-3; clip-removed variant flips it to color-4.
The `& open_bits` clip is **load-bearing** (stops wall cells being marked enclosed via
`open XOR reach`) ⇒ dropping it is non-byte-equal ⇒ would fail `ab`. Disproven in-process by
counterexample — **no gate burned** (`scratchpad/test_repair_open_fold.py`).

## Priced dead-route (for "MorphologyPool"/local-morphology cards on connectivity tasks)
A flood-fill / enclosed-region task tagged "MorphologyPool / dilation / connected component" is a
GLOBAL 4-connectivity problem, not local morphology. A **bit-packed uint32 row flood is already the
cost-optimal morphological dilation** (BitShift = horiz dilation, Gather = vert dilation, `& open` =
geodesic mask) at **80B/mask = 5 cells/byte**. Spatial `MaxPool` is the SAME operation but ORT rejects
BOOL/INT32 (probe it), so cheapest is uint8 = **400B/mask (20×20) / 900B (30×30)** = 5–11× worse, and
it needs MORE iterations (no ray acceleration; serpentine BFS depth up to ~170). Denser uint64 packing
(56B/mask) is net-negative once row-boundary-mask + cross-lane shift ops are counted; uint16 has no ORT
kernel, so 80B/20-row-mask is the ORT density floor. `lib_query "MorphologyPool"` returns 0 hits — the
card name is a corpus-keyword mis-tag.

## Rule of thumb
A byte-equal floor audit is TWO passes: **static** (dup inits / dead / structural CSE / identity) AND
**dynamic** (value-level always-equal masks over varied inputs). On graphs with memorized visible-repair
seeds, filter constant-across-inputs tensors as artifacts, and test any surviving remask candidate with
a **hand-constructed trigger** (cond-row match + seed-cell-as-wall), never random inputs. When both
passes come back clean and the representation is at the ORT density floor (uint32 row-pack for ≤32-wide
grids), the WALL is real — record it and stop.
