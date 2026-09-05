# Cohort alignment audit

Source: `/Users/Abhishek.Sharma/Downloads/stream-bsg-repo 2/.claude/worktrees/t13-synb2b-fraud-stream/synb2b-fraud-stream/results/results.json`
Run commit: `50f7e8f73800ed85027164cefbc6d7f183d3a34d` (dirty=True)
Run UTC: 2026-09-05T19:23:54Z
Seeds declared: 20

Every cell is the aggregate `mean` recorded in the results file, over the seed replicates, unless stated otherwise. A field the file does not carry is printed as `not recorded`; nothing is imputed.

## synb2b

### Cohort matching and propensity calibration

| variant | cohort gap (days) | pd MAE | pd Brier | pd clip share |
| --- | --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | not recorded | not recorded |
| `M1` (CONF-FIXED) | not recorded | not recorded | not recorded | not recorded |
| `M2` (COST-THRESH) | not recorded | not recorded | not recorded | not recorded |
| `M3` (ACI-BUDGET) | not recorded | not recorded | not recorded | not recorded |
| `M4` (ACI-BUDGET-CRC) | not recorded | not recorded | not recorded | not recorded |
| `M5` (ACI-BUDGET-CRC-IAP) | 39.6779 | 0.157884 | 0.0461784 | 0.0262148 |
| `M5_joint_slow` (M5_joint_slow) | 39.7657 | 0.284917 | 0.118402 | 0.091969 |
| `M5_joint_static` (M5_joint_static) | 39.7471 | 0.28355 | 0.117483 | 0.0916694 |
| `M5_legacy` (M5_legacy) | 87.0191 | 0.156609 | 0.0454522 | 0.0282348 |
| `M5_oracle` (M5 with oracle propensity) | not recorded | not recorded | not recorded | not recorded |
| `M5_strictmatch` (M5_strictmatch) | 0.357134 | 0.259831 | 0.106702 | 0.371711 |

### Weight concentration

| variant | ESS median | ESS p05 | trim share |
| --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | not recorded |
| `M1` (CONF-FIXED) | not recorded | not recorded | not recorded |
| `M2` (COST-THRESH) | 930.185 | 194.673 | 0 |
| `M3` (ACI-BUDGET) | 940.441 | 189.562 | 0 |
| `M4` (ACI-BUDGET-CRC) | 939.838 | 194.074 | 0 |
| `M5` (ACI-BUDGET-CRC-IAP) | 37010.4 | 5480.51 | 0.153266 |
| `M5_joint_slow` (M5_joint_slow) | 32698 | 1913.33 | 0.230096 |
| `M5_joint_static` (M5_joint_static) | 33041.5 | 1913.52 | 0.227723 |
| `M5_legacy` (M5_legacy) | 37108 | 5480.51 | 0.150869 |
| `M5_oracle` (M5 with oracle propensity) | 37413.9 | 5101.09 | 0.159321 |
| `M5_strictmatch` (M5_strictmatch) | 11657.6 | 817.545 | 0.488052 |

### Estimator bias and flag quality

| variant | bias ratio (warm) | flag balanced acc | n (replicates) |
| --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | 0 |
| `M1` (CONF-FIXED) | not recorded | not recorded | 0 |
| `M2` (COST-THRESH) | 0.116264 [0.111781, 0.120741] | 0.558888 | 20 |
| `M3` (ACI-BUDGET) | 0.0888445 [0.0861751, 0.0916876] | 0.85454 | 20 |
| `M4` (ACI-BUDGET-CRC) | 0.0533294 [0.0521687, 0.0544956] | 0.73114 | 20 |
| `M5` (ACI-BUDGET-CRC-IAP) | 0.828629 [0.806601, 0.851032] | 0.911317 | 20 |
| `M5_joint_slow` (M5_joint_slow) | 1.14747 [1.03264, 1.25919] | 0.665846 | 20 |
| `M5_joint_static` (M5_joint_static) | 1.16837 [1.05525, 1.27751] | 0.664852 | 20 |
| `M5_legacy` (M5_legacy) | 0.877319 [0.85351, 0.902674] | 0.897973 | 20 |
| `M5_oracle` (M5 with oracle propensity) | 1.08622 [1.0676, 1.10723] | 0.771245 | 20 |
| `M5_strictmatch` (M5_strictmatch) | 0.492688 [0.435983, 0.555215] | 0.720116 | 20 |

The interval on the bias ratio is the per-variant interval stored in the results file. It describes that variant's own mean; it is not an interval on any between-variant difference.

## ulb

### Cohort matching and propensity calibration

| variant | cohort gap (days) | pd MAE | pd Brier | pd clip share |
| --- | --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | not recorded | not recorded |
| `M1` (CONF-FIXED) | not recorded | not recorded | not recorded | not recorded |
| `M2` (COST-THRESH) | not recorded | not recorded | not recorded | not recorded |
| `M3` (ACI-BUDGET) | not recorded | not recorded | not recorded | not recorded |
| `M4` (ACI-BUDGET-CRC) | not recorded | not recorded | not recorded | not recorded |
| `M5` (ACI-BUDGET-CRC-IAP) | 1.9754 | 0.281535 | 0.120953 | 0.336758 |
| `M5_joint_slow` (M5_joint_slow) | 1.98363 | 0.300317 | 0.116695 | 0.159765 |
| `M5_joint_static` (M5_joint_static) | 1.98231 | 0.298453 | 0.115815 | 0.161101 |
| `M5_legacy` (M5_legacy) | 90.2518 | 0.249782 | 0.0916985 | 0.183156 |
| `M5_oracle` (M5 with oracle propensity) | not recorded | not recorded | not recorded | not recorded |
| `M5_strictmatch` (M5_strictmatch) | 0.534865 | 0.290455 | 0.125759 | 0.361434 |

### Weight concentration

| variant | ESS median | ESS p05 | trim share |
| --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | not recorded |
| `M1` (CONF-FIXED) | not recorded | not recorded | not recorded |
| `M2` (COST-THRESH) | 20090 | 325.534 | 0 |
| `M3` (ACI-BUDGET) | 18206.8 | 325.634 | 0 |
| `M4` (ACI-BUDGET-CRC) | 17904.8 | 324.134 | 0 |
| `M5` (ACI-BUDGET-CRC-IAP) | 99349.1 | 9518.22 | 0.135762 |
| `M5_joint_slow` (M5_joint_slow) | 76383.7 | 4979.41 | 0.288358 |
| `M5_joint_static` (M5_joint_static) | 76493.5 | 4978.94 | 0.289296 |
| `M5_legacy` (M5_legacy) | 88678.2 | 9518.22 | 0.158383 |
| `M5_oracle` (M5 with oracle propensity) | 98841.7 | 9596.62 | 0.146706 |
| `M5_strictmatch` (M5_strictmatch) | 99333.4 | 6585.74 | 0.151177 |

### Estimator bias and flag quality

| variant | bias ratio (warm) | flag balanced acc | n (replicates) |
| --- | --- | --- | --- |
| `M0` (STATIC) | not recorded | not recorded | 0 |
| `M1` (CONF-FIXED) | not recorded | not recorded | 0 |
| `M2` (COST-THRESH) | 0.362889 [0.343795, 0.383319] | 0.629869 | 20 |
| `M3` (ACI-BUDGET) | 0.430022 [0.408999, 0.452923] | 0.887529 | 20 |
| `M4` (ACI-BUDGET-CRC) | 0.432078 [0.411764, 0.453433] | 0.760966 | 20 |
| `M5` (ACI-BUDGET-CRC-IAP) | 1.26148 [1.17227, 1.34732] | 0.661064 | 20 |
| `M5_joint_slow` (M5_joint_slow) | 0.584207 [0.510145, 0.677205] | 0.877153 | 20 |
| `M5_joint_static` (M5_joint_static) | 0.59397 [0.516893, 0.68777] | 0.87847 | 20 |
| `M5_legacy` (M5_legacy) | 1.38539 [1.29621, 1.47087] | 0.541595 | 20 |
| `M5_oracle` (M5 with oracle propensity) | 1.1614 [1.09817, 1.23022] | 0.677694 | 20 |
| `M5_strictmatch` (M5_strictmatch) | 1.10952 [0.992282, 1.2243] | 0.648977 | 20 |

The interval on the bias ratio is the per-variant interval stored in the results file. It describes that variant's own mean; it is not an interval on any between-variant difference.

## Diagnostics this results file does not contain

Each row was probed against the union of keys across all 22 (stream, variant) records (124 distinct keys). Anything marked absent needs new instrumentation upstream before it can be audited.

| diagnostic | status | keys searched |
| --- | --- | --- |
| per-seed (replicate-level) metric values | present at `/synb2b/M0/block_rate/per_seed`, `/synb2b/M0/budget_rmse/per_seed`, `/synb2b/M0/concept_dollars_allowed/per_seed`, `/synb2b/M0/concept_for/per_seed`, `/synb2b/M0/cost_per_1k/per_seed` | any numeric array of length 20 under a variant record |
| per-update cohort fallback counts | **absent** | `n_fallback`, `n_cohort_fallback`, `cohort_fallback`, `cohort_fallback_count`, `fallback_share`, `fallback_rate` |
| per-update refit counts | **absent** | `n_refit`, `n_refits`, `pd_refits`, `refit_count` |
| thin / zero-weight / abstain update counts | **absent** | `n_thin`, `n_zero`, `n_abstain`, `thin_share`, `abstain_share` |
| q / r cohort row counts per update | **absent** | `n_q_rows`, `n_r_rows`, `q_rows`, `r_rows`, `n_q`, `n_r`, `q_row_count`, `r_row_count` |
| cohort gap distribution beyond the mean | **absent** | `cohort_gap_days_median`, `cohort_gap_days_p90`, `cohort_gap_days_max`, `cohort_gap_days_sd`, `cohort_gap_days_p05` |
| interval construction method / bootstrap replicate count | **absent** | `ci_method`, `bootstrap`, `n_boot`, `n_bootstrap` |
| seed attribution for the stored `composition` / `series` traces | **absent** | any key containing `seed` inside those trace objects |

In short, the following still need instrumentation:

- per-update cohort fallback counts
- per-update refit counts
- thin / zero-weight / abstain update counts
- q / r cohort row counts per update
- cohort gap distribution beyond the mean
- interval construction method / bootstrap replicate count
- seed attribution for the composition / series traces
