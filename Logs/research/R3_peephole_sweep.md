# R3 Peephole Sweep

Generated: `2026-06-08T02:31:24Z`

## Verdict

- Tasks scanned: **219**
- Static candidates found: **5**
- Full-AB/profile cuts: **5**
- Staged for captain review: **5**
- Total staged credited cost reduction: **16**
- Total staged point gain estimate: **0.0058**

The v1 pass set is deliberately conservative: `const_to_init`, `zero_pad_into_conv`, `add_chain_to_sum`, guarded `drop_cast_before_pad_bool`, and `opset_bump_pad_uint8`. Equal-tail re-synthesis, groups halving, no-Mul decode, and separable Einsum are left unstaged for v2 because they need task-structural guards beyond local ONNX shape/type checks.

## Rewrite Payoff

| Rewrite | Staged tasks containing rewrite | Credited saved on those tasks |
|---|---:|---:|
| `add_chain_to_sum` | 5 | 16 |

## Task Table

| task | rewrite(s) applied | old_cost | new_cost | byte_equal | profile_gap | staged | notes |
|---|---|---:|---:|---|---|---|---|
| task001 | `-` | 1149 |  | N | not_run | N | no pattern matched |
| task002 | `-` | 9545 |  | N | not_run | N | no pattern matched |
| task005 | `-` | 16738 |  | N | not_run | N | no pattern matched |
| task006 | `-` | 560 |  | N | not_run | N | no pattern matched |
| task008 | `-` | 8332 |  | N | not_run | N | no pattern matched |
| task013 | `-` | 2774 |  | N | not_run | N | no pattern matched |
| task017 | `-` | 11582 |  | N | not_run | N | no pattern matched |
| task018 | `-` | 19175 |  | N | not_run | N | no pattern matched |
| task019 | `-` | 21779 |  | N | not_run | N | no pattern matched |
| task020 | `-` | 2396 |  | N | not_run | N | no pattern matched |
| task023 | `-` | 4532 |  | N | not_run | N | no pattern matched |
| task024 | `-` | 7463 |  | N | not_run | N | no pattern matched |
| task025 | `-` | 23495 |  | N | not_run | N | no pattern matched |
| task029 | `-` | 14545 |  | N | not_run | N | no pattern matched |
| task030 | `-` | 1884 |  | N | not_run | N | no pattern matched |
| task031 | `-` | 1654 |  | N | not_run | N | no pattern matched |
| task032 | `-` | 915 |  | N | not_run | N | no pattern matched |
| task034 | `-` | 13273 |  | N | not_run | N | no pattern matched |
| task035 | `add_chain_to_sum` | 11984 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task038 | `-` | 426 |  | N | not_run | N | no pattern matched |
| task039 | `-` | 710 |  | N | not_run | N | no pattern matched |
| task041 | `-` | 2552 |  | N | not_run | N | no pattern matched |
| task042 | `-` | 2943 |  | N | not_run | N | no pattern matched |
| task044 | `-` | 27232 |  | N | not_run | N | no pattern matched |
| task046 | `-` | 25984 |  | N | not_run | N | no pattern matched |
| task047 | `-` | 1687 |  | N | not_run | N | no pattern matched |
| task048 | `-` | 699 |  | N | not_run | N | no pattern matched |
| task050 | `-` | 9475 |  | N | not_run | N | no pattern matched |
| task052 | `-` | 902 |  | N | not_run | N | no pattern matched |
| task054 | `-` | 26043 |  | N | not_run | N | no pattern matched |
| task055 | `-` | 1141 |  | N | not_run | N | no pattern matched |
| task059 | `-` | 15351 |  | N | not_run | N | no pattern matched |
| task061 | `-` | 1668 |  | N | not_run | N | no pattern matched |
| task062 | `-` | 6757 |  | N | not_run | N | no pattern matched |
| task063 | `-` | 17147 |  | N | not_run | N | no pattern matched |
| task064 | `-` | 14706 |  | N | not_run | N | no pattern matched |
| task066 | `-` | 11769 |  | N | not_run | N | no pattern matched |
| task069 | `-` | 10000 |  | N | not_run | N | no pattern matched |
| task074 | `-` | 21634 |  | N | not_run | N | no pattern matched |
| task075 | `-` | 2386 |  | N | not_run | N | no pattern matched |
| task076 | `add_chain_to_sum` | 22143 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task077 | `-` | 7235 |  | N | not_run | N | no pattern matched |
| task078 | `-` | 842 |  | N | not_run | N | no pattern matched |
| task079 | `-` | 1382 |  | N | not_run | N | no pattern matched |
| task080 | `-` | 5664 |  | N | not_run | N | no pattern matched |
| task081 | `-` | 1819 |  | N | not_run | N | no pattern matched |
| task083 | `-` | 681 |  | N | not_run | N | no pattern matched |
| task086 | `add_chain_to_sum` | 4021 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task088 | `-` | 14797 |  | N | not_run | N | no pattern matched |
| task089 | `add_chain_to_sum` | 9348 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task090 | `-` | 8474 |  | N | not_run | N | no pattern matched |
| task092 | `-` | 4352 |  | N | not_run | N | no pattern matched |
| task094 | `-` | 3806 |  | N | not_run | N | no pattern matched |
| task095 | `-` | 900 |  | N | not_run | N | no pattern matched |
| task096 | `-` | 9545 |  | N | not_run | N | no pattern matched |
| task101 | `-` | 18236 |  | N | not_run | N | no pattern matched |
| task104 | `-` | 1384 |  | N | not_run | N | no pattern matched |
| task107 | `-` | 8339 |  | N | not_run | N | no pattern matched |
| task110 | `-` | 22059 |  | N | not_run | N | no pattern matched |
| task112 | `-` | 14792 |  | N | not_run | N | no pattern matched |
| task117 | `-` | 5865 |  | N | not_run | N | no pattern matched |
| task118 | `-` | 13260 |  | N | not_run | N | no pattern matched |
| task119 | `-` | 3390 |  | N | not_run | N | no pattern matched |
| task121 | `-` | 2141 |  | N | not_run | N | no pattern matched |
| task123 | `-` | 1993 |  | N | not_run | N | no pattern matched |
| task124 | `-` | 2149 |  | N | not_run | N | no pattern matched |
| task125 | `-` | 3330 |  | N | not_run | N | no pattern matched |
| task129 | `-` | 451 |  | N | not_run | N | no pattern matched |
| task130 | `-` | 5148 |  | N | not_run | N | no pattern matched |
| task131 | `add_chain_to_sum` | 5251 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task133 | `add_chain_to_sum` | 22293 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task136 | `-` | 1670 |  | N | not_run | N | no pattern matched |
| task137 | `-` | 2516 |  | N | not_run | N | no pattern matched |
| task138 | `-` | 13204 |  | N | not_run | N | no pattern matched |
| task141 | `-` | 981 |  | N | not_run | N | no pattern matched |
| task142 | `-` | 456 |  | N | not_run | N | no pattern matched |
| task143 | `-` | 4160 |  | N | not_run | N | no pattern matched |
| task145 | `-` | 16223 |  | N | not_run | N | no pattern matched |
| task148 | `-` | 4461 |  | N | not_run | N | no pattern matched |
| task152 | `-` | 1520 |  | N | not_run | N | no pattern matched |
| task153 | `-` | 3991 |  | N | not_run | N | no pattern matched |
| task155 | `-` | 158 |  | N | not_run | N | no pattern matched |
| task157 | `-` | 21938 |  | N | not_run | N | no pattern matched |
| task158 | `-` | 34217 |  | N | not_run | N | no pattern matched |
| task159 | `-` | 7207 |  | N | not_run | N | no pattern matched |
| task162 | `-` | 8553 |  | N | not_run | N | no pattern matched |
| task163 | `-` | 23679 |  | N | not_run | N | no pattern matched |
| task165 | `-` | 8920 |  | N | not_run | N | no pattern matched |
| task166 | `-` | 1900 |  | N | not_run | N | no pattern matched |
| task167 | `-` | 795 |  | N | not_run | N | no pattern matched |
| task168 | `-` | 4673 |  | N | not_run | N | no pattern matched |
| task169 | `-` | 16106 |  | N | not_run | N | no pattern matched |
| task170 | `add_chain_to_sum` | 2967 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task173 | `-` | 35163 |  | N | not_run | N | no pattern matched |
| task174 | `-` | 5763 |  | N | not_run | N | no pattern matched |
| task175 | `-` | 2345 |  | N | not_run | N | no pattern matched |
| task178 | `-` | 729 |  | N | not_run | N | no pattern matched |
| task180 | `-` | 210 |  | N | not_run | N | no pattern matched |
| task181 | `-` | 850 |  | N | not_run | N | no pattern matched |
| task182 | `-` | 13881 |  | N | not_run | N | no pattern matched |
| task183 | `-` | 15138 |  | N | not_run | N | no pattern matched |
| task185 | `add_chain_to_sum` | 1100 | 1098 | Y | 0 (0.0%) | Y | staged profile cut 1100 -> 1098 (staged_submits/task185_peephole_cost1098) |
| task187 | `-` | 11979 |  | N | not_run | N | no pattern matched |
| task188 | `-` | 558 |  | N | not_run | N | no pattern matched |
| task190 | `add_chain_to_sum` | 2907 | 2905 | Y | 0 (0.0%) | Y | staged profile cut 2907 -> 2905 (staged_submits/task190_peephole_cost2905) |
| task191 | `-` | 24807 |  | N | not_run | N | no pattern matched |
| task194 | `-` | 1092 |  | N | not_run | N | no pattern matched |
| task195 | `-` | 1878 |  | N | not_run | N | no pattern matched |
| task196 | `-` | 4998 |  | N | not_run | N | no pattern matched |
| task198 | `-` | 9324 |  | N | not_run | N | no pattern matched |
| task199 | `-` | 3179 |  | N | not_run | N | no pattern matched |
| task201 | `-` | 4718 |  | N | not_run | N | no pattern matched |
| task204 | `-` | 7682 |  | N | not_run | N | no pattern matched |
| task205 | `add_chain_to_sum` | 4193 | 4189 | Y | 0 (0.0%) | Y | staged profile cut 4193 -> 4189 (staged_submits/task205_peephole_cost4189) |
| task206 | `-` | 13571 |  | N | not_run | N | no pattern matched |
| task207 | `-` | 200 |  | N | not_run | N | no pattern matched |
| task208 | `add_chain_to_sum` | 3983 | 3979 | Y | 0 (0.0%) | Y | staged profile cut 3983 -> 3979 (staged_submits/task208_peephole_cost3979) |
| task209 | `-` | 5251 |  | N | not_run | N | no pattern matched |
| task211 | `-` | 1520 |  | N | not_run | N | no pattern matched |
| task214 | `-` | 1394 |  | N | not_run | N | no pattern matched |
| task215 | `-` | 110 |  | N | not_run | N | no pattern matched |
| task216 | `-` | 13995 |  | N | not_run | N | no pattern matched |
| task219 | `add_chain_to_sum` | 7175 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task222 | `-` | 8308 |  | N | not_run | N | no pattern matched |
| task224 | `-` | 1780 |  | N | not_run | N | no pattern matched |
| task225 | `-` | 1614 |  | N | not_run | N | no pattern matched |
| task228 | `-` |  |  | N | not_run | N | old score failed: score_network_v2 returned None |
| task229 | `-` | 1227 |  | N | not_run | N | no pattern matched |
| task233 | `-` | 25658 |  | N | not_run | N | no pattern matched |
| task234 | `-` | 8961 |  | N | not_run | N | no pattern matched |
| task235 | `-` | 131 |  | N | not_run | N | no pattern matched |
| task238 | `add_chain_to_sum` | 15085 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task239 | `-` | 1177 |  | N | not_run | N | no pattern matched |
| task240 | `-` | 1706 |  | N | not_run | N | no pattern matched |
| task243 | `-` | 21328 |  | N | not_run | N | no pattern matched |
| task246 | `-` | 1275 |  | N | not_run | N | no pattern matched |
| task247 | `-` | 543 |  | N | not_run | N | no pattern matched |
| task253 | `-` | 568 |  | N | not_run | N | no pattern matched |
| task255 | `-` | 13983 |  | N | not_run | N | no pattern matched |
| task256 | `-` | 1334 |  | N | not_run | N | no pattern matched |
| task260 | `-` | 1522 |  | N | not_run | N | no pattern matched |
| task264 | `-` | 34606 |  | N | not_run | N | no pattern matched |
| task265 | `-` | 4713 |  | N | not_run | N | no pattern matched |
| task266 | `-` | 319 |  | N | not_run | N | no pattern matched |
| task267 | `-` | 826 |  | N | not_run | N | no pattern matched |
| task268 | `-` | 3515 |  | N | not_run | N | no pattern matched |
| task269 | `-` | 3707 |  | N | not_run | N | no pattern matched |
| task270 | `-` | 2978 |  | N | not_run | N | no pattern matched |
| task272 | `-` | 460 |  | N | not_run | N | no pattern matched |
| task273 | `-` | 6036 |  | N | not_run | N | no pattern matched |
| task278 | `-` | 20348 |  | N | not_run | N | no pattern matched |
| task279 | `-` | 14067 |  | N | not_run | N | no pattern matched |
| task280 | `add_chain_to_sum` | 6716 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task284 | `-` | 25855 |  | N | not_run | N | no pattern matched |
| task285 | `-` | 17654 |  | N | not_run | N | no pattern matched |
| task286 | `-` | 53464 |  | N | not_run | N | no pattern matched |
| task287 | `-` | 3016 |  | N | not_run | N | no pattern matched |
| task289 | `-` | 21139 |  | N | not_run | N | no pattern matched |
| task290 | `-` | 5900 |  | N | not_run | N | no pattern matched |
| task291 | `-` | 1756 |  | N | not_run | N | no pattern matched |
| task292 | `-` | 1310 |  | N | not_run | N | no pattern matched |
| task293 | `-` | 838 |  | N | not_run | N | no pattern matched |
| task299 | `-` | 1022 |  | N | not_run | N | no pattern matched |
| task304 | `-` | 369 |  | N | not_run | N | no pattern matched |
| task307 | `-` | 40 |  | N | not_run | N | no pattern matched |
| task308 | `add_chain_to_sum` | 2990 | 2986 | Y | 0 (0.0%) | Y | staged profile cut 2990 -> 2986 (staged_submits/task308_peephole_cost2986) |
| task310 | `-` | 20112 |  | N | not_run | N | no pattern matched |
| task312 | `-` | 600 |  | N | not_run | N | no pattern matched |
| task314 | `-` | 2889 |  | N | not_run | N | no pattern matched |
| task319 | `-` | 16716 |  | N | not_run | N | no pattern matched |
| task321 | `-` | 434 |  | N | not_run | N | no pattern matched |
| task324 | `-` | 18545 |  | N | not_run | N | no pattern matched |
| task325 | `-` | 5642 |  | N | not_run | N | no pattern matched |
| task327 | `-` | 5997 |  | N | not_run | N | no pattern matched |
| task328 | `-` | 8367 |  | N | not_run | N | no pattern matched |
| task330 | `-` | 4562 |  | N | not_run | N | no pattern matched |
| task332 | `-` | 8590 |  | N | not_run | N | no pattern matched |
| task335 | `-` | 1346 |  | N | not_run | N | no pattern matched |
| task338 | `-` | 8469 |  | N | not_run | N | no pattern matched |
| task340 | `-` | 8612 |  | N | not_run | N | no pattern matched |
| task341 | `-` | 11565 |  | N | not_run | N | no pattern matched |
| task342 | `-` | 12836 |  | N | not_run | N | no pattern matched |
| task343 | `-` | 12872 |  | N | not_run | N | no pattern matched |
| task345 | `add_chain_to_sum` | 887 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task346 | `-` | 523 |  | N | not_run | N | no pattern matched |
| task347 | `-` | 560 |  | N | not_run | N | no pattern matched |
| task348 | `-` | 1693 |  | N | not_run | N | no pattern matched |
| task349 | `-` | 27955 |  | N | not_run | N | no pattern matched |
| task353 | `-` | 2781 |  | N | not_run | N | no pattern matched |
| task354 | `-` | 2497 |  | N | not_run | N | no pattern matched |
| task357 | `-` | 375 |  | N | not_run | N | no pattern matched |
| task358 | `-` | 7434 |  | N | not_run | N | no pattern matched |
| task359 | `-` | 4158 |  | N | not_run | N | no pattern matched |
| task360 | `-` | 1167 |  | N | not_run | N | no pattern matched |
| task362 | `-` | 11536 |  | N | not_run | N | no pattern matched |
| task363 | `-` | 3796 |  | N | not_run | N | no pattern matched |
| task364 | `-` | 14162 |  | N | not_run | N | no pattern matched |
| task365 | `-` | 29489 |  | N | not_run | N | no pattern matched |
| task366 | `-` | 52001 |  | N | not_run | N | no pattern matched |
| task367 | `-` | 19465 |  | N | not_run | N | no pattern matched |
| task368 | `-` | 6294 |  | N | not_run | N | no pattern matched |
| task369 | `-` | 5339 |  | N | not_run | N | no pattern matched |
| task370 | `-` | 8868 |  | N | not_run | N | no pattern matched |
| task371 | `-` | 17340 |  | N | not_run | N | no pattern matched |
| task374 | `-` | 2831 |  | N | not_run | N | no pattern matched |
| task377 | `-` | 9887 |  | N | not_run | N | no pattern matched |
| task378 | `-` | 3176 |  | N | not_run | N | no pattern matched |
| task379 | `-` | 8457 |  | N | not_run | N | no pattern matched |
| task382 | `add_chain_to_sum` | 13298 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task383 | `-` | 13112 |  | N | not_run | N | no pattern matched |
| task386 | `-` | 1150 |  | N | not_run | N | no pattern matched |
| task387 | `-` | 3842 |  | N | not_run | N | no pattern matched |
| task388 | `-` | 1998 |  | N | not_run | N | no pattern matched |
| task389 | `-` | 1296 |  | N | not_run | N | no pattern matched |
| task390 | `-` | 22766 |  | N | not_run | N | no pattern matched |
| task392 | `-` | 13546 |  | N | not_run | N | no pattern matched |
| task395 | `-` | 523 |  | N | not_run | N | no pattern matched |
| task397 | `add_chain_to_sum` | 2990 |  | N | not_run | N | candidate score failed: score_network_v2 returned None |
| task398 | `-` | 2507 |  | N | not_run | N | no pattern matched |
