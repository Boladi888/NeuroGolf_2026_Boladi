# task246 v_w3 affine basis floor

- Worker: bo_ng_cdx2_worker1
- Date: 2026-07-14
- Parent/live: `Team_Best_Onnx/task246.onnx`
- Parent SHA-256: `bb5c96491cff6df00f6a4fa1b57f85024ea3634989b36b3381f698d34ad5656c`
- Live/team-best cost: `182` from `python tools/ng_current_bar.py --task 246 --json` authoritative source `manifest`, owner `Sho`.

## Attempted route

Reviewer-locked route: eliminate or shorten Sho's `v_w3 [30,2]` affine coordinate basis inside the one-Einsum formulation. Prior sparse-initializer and rank-2 `v_w8/v_w9` routes were not retried.

One numpy/pricing probe:

```powershell
python The_400_Tasks/Task_by_Task/task246/prototypes/probe_vw3_affine_floor.py
```

Key exact result lines:

```text
v_w3_shape=(30, 2) v_w3_params=60 rank=2
pair_shape=(2, 30, 30) pair_I0_rank=1 pair_I1_rank=2
pair_I1_best_rank1_residual_s2=259.66324345197575
below_182_skeletons=[]
onnx_attempts_used=0
```

The tensor checked was:

```python
pair = np.einsum("jp,Ipq,wq->Ijw", v_w3, v_w5, v_w3)
```

`pair[1]` is the non-constant affine difference slice. Its rank is 2, so a one-dimensional coordinate basis (`[30,1]` with a shrunk core) cannot express it: each such bilinear slice has rank at most 1.

## Pricing certificate

Let all params other than `v_w3` cost `182 - 60 = 122`.

- Keep dense `v_w3`: 60 basis params, total `182`; tie, not stageable.
- Factor `v_w3 = A[30,r] @ B[r,2]`, `r=1`: 32 priced params if counted naively, but impossible because `v_w3` and the required difference slice are rank 2.
- Factor `v_w3 = A[30,r] @ B[r,2]`, `r=2`: 64 basis params, total `122 + 64 = 186`; priced skeleton is already above live.
- Split ones and coordinate vectors: `30 + 30 = 60` params, total `182`; tie, not stageable.
- Generate coordinate as F32 length-30 tensor plus a required second length-30 axis vector: at least `120 + 30` charged bytes/params, total at least `272`; generated tensor memory pushes above live.
- Best-case generated U8/BOOL length-30 tensor plus required second length-30 axis vector: at least `30 + 30`, total at least `182`; tie before any casts/ops, not stageable.
- Shorten coordinate axis then re-expand: a `[1,10,20,20]` intermediate alone costs at least 4000 bytes as BOOL or 16000 bytes as F32, so total is at least `4122`/`16122`; far above live.

This uses the `Onnx_Advanced/INDEX.md` guidance requested by the reviewer:

- #67 free-output does not help because the saving must occur before the uncharged output.
- #89 scorer-invisible Einsum supports keeping work inside one output Einsum; splitting or generating helpers creates charged intermediates.
- #82 singleton coordinate reads still require length-30 coordinate axes when rebuilding a plane.
- `cost_floor_reference.md` generated-tensor warning makes Range/ConstantOfShape coordinate generation charged as memory.

## Dead reason

Every direct replacement/factorization of the `v_w3 [30,2]` affine basis is either non-expressive, ties the 60-byte/param basis cost, or exceeds it before gates. Because no priced skeleton was strictly below 182, no ONNX candidate was emitted and no visible/truth gates were run.

## One remaining hypothesis

A sub-182 win would need a different one-Einsum algebra that avoids the affine difference-coordinate basis entirely, not a direct `v_w3` shortening/factorization.
