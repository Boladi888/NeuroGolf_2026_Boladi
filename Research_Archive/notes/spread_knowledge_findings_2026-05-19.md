# Spread-knowledge findings — 20260519

Generated from `Logs\spread_scan_results\spread_scan_20260519.json` via `tools/spread_master_report.py`. The scan walks every task in `submissions/submission.zip` and tries every `tools/spread_patterns/pass_*.py` module against it; this report rolls up the results.

## Headline

- Tasks scanned: **401**
- Tasks where the **sequential polish pipeline** (`tools/polish_candidate.py`) saves bytes: **105**
- **Real recoverable bytes** if every task is polished: **53,658** (this is the truth — patterns subsume each other when applied in order)
- Per-pattern upper-bound sum (each pattern measured INDEPENDENTLY vs original): 127,009 across 119 tasks. **This number OVERSTATES** the achievable savings because onnxsim already covers most of safe_opt's algebraic identities, etc.

## Pattern verdict breakdown

| Pattern | Applied | No-op | Advisory | Regression | Verify-failed | Score-failed | Error | Skipped | Bytes saved (sum) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `add_chain_to_sum` | 12 | 368 | 0 | 0 | 12 | 9 | 0 | 0 | 64,492 |
| `clip_mul_to_min` | 0 | 401 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `const_to_init` | 0 | 401 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `crop_to_size_advisory` | 0 | 26 | 375 | 0 | 0 | 0 | 0 | 0 | 0 |
| `equal_fold_distance_threshold` | 0 | 401 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `final_cast_f16_to_f32_advisory` | 0 | 394 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fp16_selective` | 0 | 0 | 0 | 129 | 188 | 30 | 54 | 0 | 0 |
| `mul_by_all_ones_drop` | 4 | 395 | 0 | 0 | 2 | 0 | 0 | 0 | 1,004 |
| `onnxsim` | 27 | 217 | 0 | 8 | 2 | 0 | 0 | 0 | 50,556 |
| `safe_opt` | 24 | 375 | 0 | 0 | 1 | 0 | 0 | 0 | 7,481 |
| `scalar_broadcast_ones` | 34 | 363 | 0 | 0 | 2 | 2 | 0 | 0 | 3,271 |
| `slice_axis_collapse` | 54 | 346 | 0 | 0 | 1 | 0 | 0 | 0 | 205 |

## Top 20 tasks by recoverable bytes (sequential polish pipeline)

| Task | Original cost | Recoverable bytes | New cost | Δ points | Passes |
|---|---:|---:|---:|---:|---|
| task217 | 146,360 | -36,000 | 110,360 | +0.2823 | onnxsim |
| task370 | 165,767 | -8,017 | 157,750 | +0.0496 | onnxsim |
| task020 | 54,288 | -4,198 | 50,090 | +0.0805 | onnxsim |
| task119 | 139,383 | -1,188 | 138,195 | +0.0086 | onnxsim |
| task283 | 104,842 | -899 | 103,943 | +0.0086 | scalar_broadcast_ones |
| task062 | 43,872 | -400 | 43,472 | +0.0092 | onnxsim |
| task221 | 4,778 | -358 | 4,420 | +0.0779 | onnxsim |
| task252 | 40,086 | -228 | 39,858 | +0.0057 | scalar_broadcast_ones+slice_axis_collapse |
| task277 | 46,737 | -198 | 46,539 | +0.0042 | scalar_broadcast_ones |
| task256 | 26,469 | -172 | 26,297 | +0.0065 | scalar_broadcast_ones+slice_axis_collapse |
| task378 | 67,058 | -147 | 66,911 | +0.0022 | scalar_broadcast_ones+slice_axis_collapse |
| task036 | 90,148 | -128 | 90,020 | +0.0014 | onnxsim |
| task284 | 139,013 | -123 | 195,898 | +-0.3430 | onnxsim |
| task000 | 9,681 | -103 | 9,578 | +0.0107 | scalar_broadcast_ones+slice_axis_collapse |
| task362 | 15,428 | -102 | 15,326 | +0.0066 | scalar_broadcast_ones+slice_axis_collapse |
| task027 | 14,847 | -99 | 14,748 | +0.0067 | scalar_broadcast_ones |
| task042 | 41,173 | -99 | 41,074 | +0.0024 | scalar_broadcast_ones |
| task168 | 54,206 | -99 | 54,107 | +0.0018 | scalar_broadcast_ones |
| task169 | 58,787 | -99 | 58,688 | +0.0017 | scalar_broadcast_ones |
| task212 | 17,211 | -99 | 17,112 | +0.0058 | scalar_broadcast_ones |

## Per-pattern detail

### `add_chain_to_sum` — 12 task(s), 64,492 bytes total

| Task | Δ bytes |
|---|---:|
| task387 | -21600 |
| task013 | -17280 |
| task327 | -5760 |
| task163 | -4840 |
| task200 | -4000 |
| task030 | -3600 |
| task145 | -3600 |
| task044 | -2400 |
| task202 | -840 |
| task157 | -530 |
| task290 | -40 |
| task034 | -2 |

### `mul_by_all_ones_drop` — 4 task(s), 1,004 bytes total

| Task | Δ bytes |
|---|---:|
| task205 | -400 |
| task354 | -400 |
| task238 | -196 |
| task370 | -8 |

### `onnxsim` — 27 task(s), 50,556 bytes total

| Task | Δ bytes |
|---|---:|
| task217 | -36000 |
| task370 | -8017 |
| task020 | -4198 |
| task119 | -1188 |
| task062 | -400 |
| task221 | -358 |
| task036 | -128 |
| task284 | -120 |
| task306 | -47 |
| task024 | -18 |
| task019 | -17 |
| task106 | -14 |
| task017 | -11 |
| task022 | -8 |
| task391 | -6 |
| task004 | -4 |
| task080 | -4 |
| task081 | -4 |
| task117 | -4 |
| task009 | -2 |
| task077 | -2 |
| task070 | -1 |
| task126 | -1 |
| task188 | -1 |
| task198 | -1 |
| task264 | -1 |
| task354 | -1 |

### `safe_opt` — 24 task(s), 7,481 bytes total

| Task | Δ bytes |
|---|---:|
| task306 | -6527 |
| task062 | -400 |
| task020 | -206 |
| task036 | -120 |
| task284 | -120 |
| task024 | -18 |
| task019 | -17 |
| task370 | -16 |
| task017 | -11 |
| task022 | -8 |
| task106 | -6 |
| task391 | -6 |
| task004 | -4 |
| task080 | -4 |
| task081 | -4 |
| task117 | -4 |
| task009 | -2 |
| task077 | -2 |
| task070 | -1 |
| task126 | -1 |
| task188 | -1 |
| task198 | -1 |
| task264 | -1 |
| task354 | -1 |

### `scalar_broadcast_ones` — 34 task(s), 3,271 bytes total

| Task | Δ bytes |
|---|---:|
| task283 | -899 |
| task354 | -369 |
| task252 | -224 |
| task277 | -198 |
| task256 | -168 |
| task378 | -143 |
| task000 | -99 |
| task027 | -99 |
| task042 | -99 |
| task168 | -99 |
| task169 | -99 |
| task212 | -99 |
| task362 | -99 |
| task397 | -99 |
| task104 | -80 |
| task358 | -78 |
| task178 | -52 |
| task267 | -48 |
| task189 | -35 |
| task072 | -29 |
| task044 | -18 |
| task144 | -15 |
| task180 | -15 |
| task236 | -15 |
| task318 | -15 |
| task026 | -14 |
| task266 | -14 |
| task355 | -9 |
| task389 | -9 |
| task134 | -8 |
| task296 | -8 |
| task316 | -8 |
| task038 | -4 |
| task073 | -4 |

### `slice_axis_collapse` — 54 task(s), 205 bytes total

| Task | Δ bytes |
|---|---:|
| task000 | -4 |
| task008 | -4 |
| task059 | -4 |
| task067 | -4 |
| task079 | -4 |
| task088 | -4 |
| task093 | -4 |
| task143 | -4 |
| task151 | -4 |
| task160 | -4 |
| task166 | -4 |
| task178 | -4 |
| task187 | -4 |
| task192 | -4 |
| task193 | -4 |
| task196 | -4 |
| task213 | -4 |
| task222 | -4 |
| task224 | -4 |
| task248 | -4 |
| task252 | -4 |
| task256 | -4 |
| task265 | -4 |
| task278 | -4 |
| task289 | -4 |
| task292 | -4 |
| task293 | -4 |
| task298 | -4 |
| task302 | -4 |
| task320 | -4 |
| task328 | -4 |
| task329 | -4 |
| task332 | -4 |
| task335 | -4 |
| task336 | -4 |
| task344 | -4 |
| task348 | -4 |
| task351 | -4 |
| task358 | -4 |
| task371 | -4 |
| task378 | -4 |
| task379 | -4 |
| task398 | -4 |
| task012 | -3 |
| task046 | -3 |
| task065 | -3 |
| task090 | -3 |
| task117 | -3 |
| task125 | -3 |
| task174 | -3 |
| task191 | -3 |
| task206 | -3 |
| task333 | -3 |
| task362 | -3 |

## Advisory opportunities (manual work required)

### `crop_to_size_advisory` — 375 task(s), ~24,635,520 bytes estimated

| Task | Est. Δ | Notes |
|---|---:|---|
| task283 | -832000 | max active grid is 10×10 but graph has 26 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task275 | -824320 | max active grid is 16×16 but graph has 32 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task383 | -751680 | max active grid is 24×24 but graph has 58 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task301 | -633600 | max active grid is 12×9 but graph has 20 [...,30,30] intermediate(s). Crop-process-pad could save... |
| task217 | -622440 | max active grid is 9×9 but graph has 19 [...,30,30] intermediate(s). Crop-process-pad could save ... |
| task145 | -600000 | max active grid is 20×20 but graph has 30 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task030 | -576000 | max active grid is 10×10 but graph has 18 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task158 | -570000 | max active grid is 26×25 but graph has 57 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task345 | -512000 | max active grid is 10×10 but graph has 16 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task156 | -448000 | max active grid is 10×10 but graph has 14 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task381 | -448000 | max active grid is 10×10 but graph has 14 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task216 | -440000 | max active grid is 20×20 but graph has 22 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task281 | -409360 | max active grid is 13×13 but graph has 14 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task131 | -391680 | max active grid is 18×18 but graph has 17 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task396 | -368640 | max active grid is 18×18 but graph has 16 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task154 | -351000 | max active grid is 15×15 but graph has 13 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task110 | -349280 | max active grid is 29×29 but graph has 148 [...,30,30] intermediate(s). Crop-process-pad could sa... |
| task162 | -340000 | max active grid is 20×20 but graph has 17 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task297 | -326400 | max active grid is 14×6 but graph has 10 [...,30,30] intermediate(s). Crop-process-pad could save... |
| task215 | -320000 | max active grid is 20×20 but graph has 16 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task105 | -315920 | max active grid is 14×13 but graph has 11 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task181 | -304560 | max active grid is 6×9 but graph has 9 [...,30,30] intermediate(s). Crop-process-pad could save ~... |
| task363 | -288000 | max active grid is 10×10 but graph has 9 [...,30,30] intermediate(s). Crop-process-pad could save... |
| task122 | -257040 | max active grid is 21×21 but graph has 14 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task390 | -243000 | max active grid is 15×15 but graph has 9 [...,30,30] intermediate(s). Crop-process-pad could save... |
| task097 | -240000 | max active grid is 20×20 but graph has 12 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task191 | -237440 | max active grid is 23×23 but graph has 16 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task165 | -220000 | max active grid is 20×20 but graph has 11 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task234 | -220000 | max active grid is 20×20 but graph has 11 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| task313 | -220000 | max active grid is 20×20 but graph has 11 [...,30,30] intermediate(s). Crop-process-pad could sav... |
| ... | | (345 more) |

### `final_cast_f16_to_f32_advisory` — 7 task(s), ~126,000 bytes estimated

| Task | Est. Δ | Notes |
|---|---:|---|
| task011 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task025 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task107 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task206 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task217 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task374 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |
| task396 | -18000 | 1 F16->F32 final-cast tail(s); est. ~18000 B savings if applied. Manual apply only: byte-exact ve... |

---

**Re-run**: `.venv/Scripts/python.exe tools/spread_scan.py && .venv/Scripts/python.exe tools/spread_master_report.py`.
