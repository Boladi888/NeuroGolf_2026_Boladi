# task058

Family: Gather / lookup-table

## Graph Summary
- 9 nodes: Gather(x2), ReduceSum, Equal, Cast, ArgMax, Reshape, Sub, Concat
- Input: (1,10,30,30), Output: (1,10,30,30)
- Key initializer: v3=(16,1,30,30) float32 lookup table (57600 bytes), v4=(1,1,30,30) all-zeros (3600 bytes)
- Algorithm: Extract ch0, ReduceSum to get count, Equal against 16 thresholds, ArgMax to find matching band, Gather corresponding mask, Sub to produce diff, Concat 10 channels

## Cost Baseline
- cost=26218, params=15318, memory=10900, file_size=61862 bytes

## Attempts

### 2026-05-19: trim20 — STAGED (winner)
- Trimmed v3 from (16,1,30,30) to (16,1,20,20) since active data only in top-left 20x20 (max active row/col = 19)
- Added Pad(10,10 on bottom+right) after Gather to restore 30x30 shape before Sub
- Result: cost=19826 (delta=-6392), file size=30003 (saves 31859 bytes), verify=22/22
- Staged: task058_trim20_t_785a9888_on_5806_87
