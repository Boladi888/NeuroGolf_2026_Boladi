# einsum.md — STUB (Phase-2; original in `_archive/einsum.md`)

Canonical Einsum knowledge: `techniques.md` **#46** (adopt-list incl. `bchh->bch` diagonal reads), **§42** (free-input marginals, grader-validated), `Research`-grade detail in `_archive/`. ORT CPU Einsum dtypes: float/double/int32/int64 only; 1-2 operands fused, >=3 pairwise. **task191 F16 multi-operand Einsum is NOT byte-equal (0.042 drift flips ArgMax) — use F32 for exactness-critical Einsum.**
