# task196 — `or_cast_collapse` + `flip_where` (mutex predicates + binary-mask trick)

| Field | Value |
|---|---|
| Task | task196 |
| Patterns | `or_cast_collapse` (rule 3) + `flip_where` (rule 6) |
| Cost reduction | -1,575 (combined) |

This task demonstrated TWO independent bool-first rewrites simultaneously.

---

## Rewrite 1: `or_cast_collapse` — `neq2` predicate

### Before

```
Add(Cast(Greater(x, 25), F32), Cast(Less(x, 15), F32))   →   neq2 [F32]
```

The downstream consumer needs a 0/1 mask: "x is NOT in [15, 25]". Original implementation built it as the SUM of two predicates: `gt25 + lt15`.

### After

```
Cast(Or(Greater(x, 25), Less(x, 15)), F32)               →   neq2 [F32]
```

### Why it works (and why mutex matters)

`Greater(x, 25)` and `Less(x, 15)` are **mutually exclusive** — no value of x satisfies both. When the predicates are mutex, `int(a) + int(b)` equals `int(a OR b)` exactly: 0 (neither true), 1 (exactly one true), never 2 (impossible).

If the predicates were NOT mutex (e.g. `Greater(x, 5) + Less(x, 25)` — overlapping ranges), the sum could be 2 where Or is 1, and the rewrite would change the math.

The detector verifies mutex by parsing the threshold values:
- `Greater(x, hi) + Less(x, lo)` with `hi >= lo` → provably mutex.
- `GreaterOrEqual(x, hi) + Less(x, lo)` with `hi >= lo` → mutex.
- `Greater(x, hi) + LessOrEqual(x, lo)` with `hi > lo` → mutex.
- `GreaterOrEqual(x, hi) + LessOrEqual(x, lo)` with `hi > lo` → mutex.

In all other cases the detector skips.

### Savings

- One Cast node eliminated (Or output is bool, Cast once at the boundary).
- One F32 intermediate eliminated.

### Generalizes to

`techniques.md` §1 rule 3 (`or_cast_collapse`). Detector: `pass_or_cast_collapse.py`.

---

## Rewrite 2: `flip_where` — binary-mask Sub-Mul-Where

### Before

```
not_bad = Where(cond, col, 0)                              [F32]
bad_6   = Sub(1, not_bad)                                  [F32]
result  = Mul(col, bad_6)                                  [F32]
```

For binary `col` (0 or 1), this computes: "col where NOT cond, else 0".

### After

```
result = Where(cond, 0, col)
```

### Why it works (truth table)

For binary `col` ∈ {0, 1}:

| col | cond | Where(cond, col, 0) | Sub(1, _) | × col | = LHS | Where(cond, 0, col) = RHS |
|---|---|---|---|---|---|---|
| 0 | F | 0 | 1 | 0 | 0 | 0 |
| 0 | T | 0 | 1 | 0 | 0 | 0 |
| 1 | F | 0 | 1 | 1 | 1 | 1 |
| 1 | T | 1 | 0 | 0 | 0 | 0 |

LHS = RHS in all 4 cases. Identity holds.

### Why it requires binary col

If `col = 2` and `cond = True`: LHS = `Mul(2, Sub(1, Where(True, 2, 0))) = Mul(2, Sub(1, 2)) = Mul(2, -1) = -2`. RHS = `Where(True, 0, 2) = 0`. NOT equal. The math fails for non-binary col.

Detector verifies `col` is binary by tracing back to a `Cast(bool, F32)` source.

### Savings

Drops Mul + Sub + the inner Where's intermediate. One Where stays. Per task196: 3 intermediate F32 tensors eliminated.

### Generalizes to

`techniques.md` §1 rule 6 (`flip_where`). Detector: `pass_flip_where.py`.

---

## Combined effect on task196

Both rewrites applied independently, total -1,575 bytes. Each is fully automated via the named detector. Run `tools/spread_scan.py --tasks task196` to reproduce.

## Reproducibility

Pattern detectors are fully automated. Both rules apply when their structural conditions match — no manual tuning. The `or_cast_collapse` is particularly value-bound (depends on the literal threshold values being mutex-compatible), so it fires on a narrower set of graphs than `bool_andfold`.
