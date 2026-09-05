# Estimator validation ladder

Generated 2026-09-05 09:51 by `tools/validation_ladder.py`. Stream `synb2b`, method `M5`, **5 seeds** per stage (11, 23, 37, 53, 71), 40 runs in 121 s.

## The question

Does the M5 risk estimator recover the true release-region risk when every complication is switched off? Each rung below adds exactly one factor to the rung above it, so the first rung whose calibration ratio stops covering 1.0 names the factor responsible.

- **true risk** = `estimator_bias.mean_true_warm`, the decay-weighted true fraud rate among allow-path items below `tau_lo` in the calibration window. Computed with labels nobody actually has; never fed back into the policy.
- **estimated risk** = `estimator_bias.mean_estimate_warm`, the value `e_t` the policy actually acted on.
- **calibration ratio** = `estimator_bias.bias_ratio_warm` = true / estimate. 1.0 is perfect. Above 1.0 the estimator is optimistic (it under-reads the risk it is releasing).
- Cold-start updates - those inside the first `MATURITY_DAYS * delay_scale_mult` days, when the ledger has reported nothing - are excluded by the `_warm` suffix. At rungs S1 and S2 that exclusion window is ~0 days, so essentially every update is scored; from S3 on it is 180 days.
- **95% CI** is a t-interval over seeds on the ratio. A rung **passes** when it covers 1.0.

## Headline: the simplest rung

**S1 FAILS.** Calibration ratio 0.9883 (sd 0.0077, 95% CI [0.9787, 0.9979], n=5).

**The strict check does not pass.** The seed spread is tight enough (sd 0.0077 over 5 seeds) that a 1.17% offset is resolvable, so the interval sits clear of 1.0. The direction is conservative: the estimator over-reads the risk it is releasing, so the policy holds back more traffic than it needs to.

Most of that offset is a **definition mismatch, not an estimator defect** - see the localisation section below. `bias_ratio_warm` scores `e_t` against `e_true`, which is computed over the **ledger channel only**, while `e_t` itself is computed over **every channel** below `tau_lo`. Because `tau_lo` moves, that includes items routed to the review band under an earlier, higher threshold - the high-scoring tail of today's allow region. Scored against the all-channel truth on the same window, the S1 ratio moves to 0.9987 (95% CI [0.9978, 0.9996]).

So the genuine S1 estimator residual is about 0.13% - still resolvable at this seed count, so strictly speaking the estimator core is not exactly unbiased either. For scale, the last rung on this ladder (S8) misses by 16.2%. The core is close to right; essentially all of the operational miscalibration is introduced further up the ladder.

Per-seed ratios at S1: `11` 0.9935, `23` 0.9874, `37` 0.9940, `53` 0.9753, `71` 0.9914.

Because the ladder's own baseline is already outside the interval, the usual read-off - *the first rung to break names the culprit* - has to be made on effect size rather than on the pass/fail bit. Rung-to-rung shifts, against both the headline ledger-only yardstick and the estimator's actual estimand:

- `S2` delay_scale_mult 1e-4 -> 1.0: headline 0.9883 -> 0.9753 (shift -0.0130); vs estimand 0.9987 -> 0.9884 (shift -0.0103)
- `S3` disclose_mult 100.0 -> 1.0: headline 0.9753 -> 0.9236 (shift -0.0516); vs estimand 0.9884 -> 0.9372 (shift -0.0511)
- `S4` pi_floor 1e-9 -> default: headline 0.9236 -> 1.0218 (shift 0.0981); vs estimand 0.9372 -> 0.9813 (shift 0.0440)
- `S5` rho 1.0 -> default: headline 1.0218 -> 1.0384 (shift 0.0166); vs estimand 0.9813 -> 0.9858 (shift 0.0045)
- `S6` oracle_pd True -> False: headline 1.0384 -> 0.8732 (shift -0.1651); vs estimand 0.9858 -> 0.8446 (shift -0.1412) - headline CI excludes 1.0
- `S7` drift_severity 0.0 -> 0.5: headline 0.8732 -> 0.8490 (shift -0.0242); vs estimand 0.8446 -> 0.8119 (shift -0.0327) - headline CI excludes 1.0
- `S8` drift_severity 0.5 -> 1.0: headline 0.8490 -> 0.8378 (shift -0.0112); vs estimand 0.8119 -> 0.7606 (shift -0.0513) - headline CI excludes 1.0

**The largest single step against the estimand is S6** (`oracle_pd True -> False`), which moves the ratio by -0.1412. The next largest is S8 (`drift_severity 0.5 -> 1.0`, -0.0513).

## Stage table: calibration

| rung | factor switched on | true risk | est risk | ratio (true/est) | 95% CI | abs err | RMSE | rel err | covers 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | (baseline) | 0.009175 | 0.009284 | 0.9883 | [0.9787, 0.9979] | 0.000109 | 0.000128 | 0.0119 | **NO** |
| S2 | delay_scale_mult 1e-4 -> 1.0 | 0.008942 | 0.009208 | 0.9753 | [0.8912, 1.0594] | 0.000497 | 0.000635 | 0.0559 | yes |
| S3 | disclose_mult 100.0 -> 1.0 | 0.008932 | 0.009783 | 0.9236 | [0.7932, 1.0541] | 0.000991 | 0.001334 | 0.1107 | yes |
| S4 | pi_floor 1e-9 -> default | 0.008922 | 0.008743 | 1.0218 | [0.9864, 1.0571] | 0.000206 | 0.000276 | 0.0237 | yes |
| S5 | rho 1.0 -> default | 0.009225 | 0.008897 | 1.0384 | [0.9992, 1.0775] | 0.000327 | 0.000400 | 0.0362 | yes |
| S6 | oracle_pd True -> False | 0.009140 | 0.010572 | 0.8732 | [0.7642, 0.9823] | 0.001432 | 0.001712 | 0.1542 | **NO** |
| S7 | drift_severity 0.0 -> 0.5 | 0.010435 | 0.012366 | 0.8490 | [0.7538, 0.9442] | 0.001931 | 0.002163 | 0.1855 | **NO** |
| S8 | drift_severity 0.5 -> 1.0 | 0.019924 | 0.023899 | 0.8378 | [0.7586, 0.9170] | 0.003975 | 0.004329 | 0.1994 | **NO** |

`abs err` is the mean over seeds of |true - estimate|; `RMSE` is the root-mean-square of the same difference across seeds; `rel err` is |true - estimate| / true.

## Localisation: which truth is the estimate being scored against?

Every column is the same estimate `e_t` divided into a different definition of the truth, over the same warm updates, straight from the per-update composition series. It separates *the estimator is wrong* from *the estimator and the yardstick are measuring different populations*.

| rung | vs ledger-only (`e_true`, the headline) | vs all-channel boxcar (`r_trail`) | vs + decay (`r_decay`) | vs + decay and trimming (`r_overlap`, the actual estimand) | warm updates |
|---|---|---|---|---|---|
| S1 | 0.9883 [0.9787, 0.9979] * | 0.9987 [0.9978, 0.9996] * | 0.9987 [0.9978, 0.9996] * | 0.9987 [0.9978, 0.9996] * | 320 |
| S2 | 0.9759 [0.8927, 1.0591] | 0.9884 [0.9047, 1.0720] | 0.9884 [0.9047, 1.0720] | 0.9884 [0.9047, 1.0720] | 143 |
| S3 | 0.9252 [0.7960, 1.0545] | 0.9372 [0.8055, 1.0689] | 0.9372 [0.8055, 1.0689] | 0.9372 [0.8056, 1.0689] | 143 |
| S4 | 1.0231 [0.9882, 1.0581] | 1.0371 [0.9986, 1.0756] | 1.0371 [0.9986, 1.0756] | 0.9813 [0.9447, 1.0178] | 143 |
| S5 | 1.0400 [1.0016, 1.0785] * | 1.0183 [0.9772, 1.0594] | 1.0529 [1.0110, 1.0949] * | 0.9858 [0.9464, 1.0251] | 143 |
| S6 | 0.8739 [0.7633, 0.9844] * | 0.8579 [0.7453, 0.9704] * | 0.8874 [0.7700, 1.0047] | 0.8446 [0.7301, 0.9591] * | 143 |
| S7 | 0.8495 [0.7532, 0.9458] * | 0.8270 [0.7326, 0.9215] * | 0.8588 [0.7581, 0.9595] * | 0.8119 [0.7161, 0.9077] * | 143 |
| S8 | 0.8365 [0.7556, 0.9174] * | 0.7887 [0.7117, 0.8658] * | 0.8387 [0.7567, 0.9207] * | 0.7606 [0.6853, 0.8360] * | 143 |

`*` marks a 95% t-interval that excludes 1.0. The first column reproduces `bias_ratio_warm` up to the slightly wider update set used here (`_summarise` additionally drops updates too close to the end of the stream to fill their realisation horizon).

Reading across a row: the gap between column 1 and column 2 is the **channel-definition** effect - `e_true` looks only at ledger rows, `e_t` at every channel below `tau_lo`, and because `tau_lo` moves, the latter includes items that were sent to the review band under an earlier, higher threshold. The gap between columns 2 and 3 is the **decay** estimand shift, and between 3 and 4 the **trimming** estimand shift. Whatever remains in column 4 is the estimator's own error against the quantity it actually targets.

## Stage table: weights, propensities and flag

| rung | ESS median | trim share | p_d clip share | p_d MAE | top-1% weight share | est-gap rel | flag prec | flag rec | oracle infeas rate | warm updates |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 39816 | 0.0001 | n/a | n/a | 0.0099 | 0.0001 | n/a | 0.000 | 0.003 | 319 |
| S2 | 41786 | 0.0000 | n/a | n/a | 0.0100 | 0.0001 | 0.000 | 0.000 | 0.003 | 142 |
| S3 | 41795 | 0.0000 | n/a | n/a | 0.0100 | 0.0001 | 0.000 | 0.000 | 0.003 | 142 |
| S4 | 37512 | 0.1683 | n/a | n/a | 0.0100 | 0.0709 | 0.000 | 0.000 | 0.003 | 142 |
| S5 | 37046 | 0.1683 | n/a | n/a | 0.0118 | 0.0613 | 0.000 | 0.000 | 0.009 | 142 |
| S6 | 37276 | 0.1522 | 0.0355 | 0.1622 | 0.0118 | 0.0394 | 0.042 | 0.100 | 0.009 | 142 |
| S7 | 37263 | 0.1524 | 0.0310 | 0.1573 | 0.0118 | 0.0434 | 0.308 | 0.449 | 0.026 | 142 |
| S8 | 36926 | 0.1531 | 0.0270 | 0.1586 | 0.0118 | 0.0832 | 0.885 | 0.897 | 0.351 | 142 |

There is no inverse-propensity weight-clipping share to report: `PolicyConfig.weight_clip` is defined but referenced nowhere in `src/t13`, so weights are governed by trimming alone. The clip column is the share of *learned disclosure propensities* pinned to the [0.05, 0.97] bounds of `_predict_pd`, which is undefined wherever p_d is the oracle.

## Stage table: realised risk on the allow path

| rung | FOR overall | FOR / alpha | quiet | covariate | concept | prior | infeasible rate |
|---|---|---|---|---|---|---|---|
| S1 | 0.013799 | 1.113 | 0.016341 | 0.007180 | 0.008921 | 0.014370 | 0.0000 |
| S2 | 0.013776 | 1.111 | 0.016303 | 0.007180 | 0.008926 | 0.014374 | 0.0044 |
| S3 | 0.013757 | 1.109 | 0.016276 | 0.007182 | 0.008926 | 0.014350 | 0.0100 |
| S4 | 0.013697 | 1.105 | 0.016188 | 0.007180 | 0.008921 | 0.014370 | 0.0200 |
| S5 | 0.013697 | 1.105 | 0.016187 | 0.007180 | 0.008921 | 0.014370 | 0.0206 |
| S6 | 0.013298 | 1.072 | 0.015552 | 0.007154 | 0.008921 | 0.014370 | 0.0369 |
| S7 | 0.013650 | 1.101 | 0.014344 | 0.007957 | 0.011992 | 0.018643 | 0.0550 |
| S8 | 0.020494 | 1.653 | 0.012511 | 0.008975 | 0.036822 | 0.052117 | 0.3575 |

The by-regime columns are the per-event read that the missing per-event drift knob cannot give: at S8 they score calibration separately inside the covariate, concept and prior windows.

## Exact knob settings per rung

| rung | name | knobs |
|---|---|---|
| S1 | simplest: oracle p_d, no delay, no trimming, no decay, no drift | `drift_severity=0, delay_scale_mult=0.0001, disclose_mult=100.0, oracle_pd=True, pi_floor=1e-09, rho=1.0` |
| S2 | + arrival delay | `drift_severity=0, delay_scale_mult=1.0, disclose_mult=100.0, oracle_pd=True, pi_floor=1e-09, rho=1.0` |
| S3 | + booked-clean corruption | `drift_severity=0, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=True, pi_floor=1e-09, rho=1.0` |
| S4 | + propensity trimming | `drift_severity=0, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=True, rho=1.0` |
| S5 | + calibration decay | `drift_severity=0, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=True` |
| S6 | + learned p_d | `drift_severity=0, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=False` |
| S7 | + drift, half severity | `drift_severity=0.5, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=False` |
| S8 | full published configuration | `drift_severity=1, delay_scale_mult=1.0, disclose_mult=1.0, oracle_pd=False` |

Everything not listed is the published default from `config.policy_for(stream)` and the `run_policy` signature defaults (`cohort_aligned=True`, `cohort_fallback=True`, `joint_mode=''`, `epsilon=0.12`, `alpha`, `budget`, `min_support`, `window_max_n`, `window_max_days`, `update_every` all at defaults).

## Factors that could not be switched fully off

Stated here rather than left implicit, because a rung that claims to remove a factor and does not is worse than one that admits it.

- **booked-clean corruption** (`disclose_mult`). An allow-path item that is not disclosed is written into the calibration set as obs = 0, i.e. as confirmed-clean, whether or not it was fraud. There is no knob that stops that write. disclose_mult=100 raises every disclosure probability to the hard-coded 0.999 ceiling, so ~0.1% of allow-path frauds are still booked clean at rung S1. The mechanism is attenuated, not removed.
- **zero arrival delay** (`delay_scale_mult`). delay_scale_mult=0.0 is not admissible: it makes the review-channel Gamma scale zero, scipy returns NaN, and every adjudicated row is then trimmed away without warning. The ladder uses 0.0001 instead - delays of order seconds on a 561-day stream - so a residual, negligible arrival lag remains.
- **zero propensity trimming** (`pi_floor`). pi_floor=0.0 admits rows with an exactly-zero propensity and the Horvitz-Thompson numerator evaluates 0.0/0.0, taking the entire risk curve to NaN. The ladder uses 1e-09, which trims only the zero-propensity rows (which carry no label anyway).
- **per-event drift isolation** (`drift_severity`). prepare_stream exposes a single scalar drift_severity that scales all three events together; there is no knob that enables the covariate, concept or prior event on its own. Rungs S7/S8 therefore move severity as a whole, and the per-event read is supplied instead by the by_regime decomposition, which scores calibration separately inside each event window.
- **drift severity 0 is not 'no concept drift'** (`drift_severity`). At severity 0 the covariate shift and the prior surge are genuinely off (magnitude 0 means exp(0)=1 and lambda=0). The concept event is not: beta_path sets beta to ev.magnitude inside the concept window, so severity 0 flips beta from LGP_BETA_BASE=0.9 to 0.0 there rather than leaving it at the base value. A residual conditional-law change survives inside that window at every rung.
- **inverse-propensity weight clipping** (`weight_clip`). PolicyConfig.weight_clip exists but is referenced nowhere in src/t13. Weights are controlled by propensity trimming only, so no weight-clipping share can be reported. The clipping column below is the p_d clip share (the share of learned disclosure propensities pinned to the [0.05, 0.97] bounds of _predict_pd), which is NaN wherever p_d is the oracle.

## Per-seed detail

| rung | seed | true | est | ratio | abs err | ESS median | trim share | FOR | numpy warnings |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 11 | 0.009157 | 0.009217 | 0.9935 | 0.000060 | 39823 | 0.0001 | 0.013714 | 0 |
| S1 | 23 | 0.009020 | 0.009135 | 0.9874 | 0.000115 | 39810 | 0.0001 | 0.013327 | 0 |
| S1 | 37 | 0.009299 | 0.009355 | 0.9940 | 0.000056 | 39825 | 0.0001 | 0.014313 | 0 |
| S1 | 53 | 0.009297 | 0.009532 | 0.9753 | 0.000235 | 39821 | 0.0001 | 0.014262 | 0 |
| S1 | 71 | 0.009103 | 0.009182 | 0.9914 | 0.000079 | 39800 | 0.0001 | 0.013380 | 0 |
| S2 | 11 | 0.008872 | 0.009867 | 0.8991 | 0.000995 | 41847 | 0.0000 | 0.013719 | 0 |
| S2 | 23 | 0.008679 | 0.008267 | 1.0499 | 0.000412 | 41742 | 0.0000 | 0.013240 | 0 |
| S2 | 37 | 0.009427 | 0.009423 | 1.0004 | 0.000004 | 41852 | 0.0000 | 0.014329 | 0 |
| S2 | 53 | 0.009044 | 0.009957 | 0.9083 | 0.000913 | 41870 | 0.0000 | 0.014321 | 0 |
| S2 | 71 | 0.008688 | 0.008528 | 1.0187 | 0.000160 | 41618 | 0.0000 | 0.013269 | 0 |
| S3 | 11 | 0.008853 | 0.011368 | 0.7788 | 0.002514 | 41847 | 0.0000 | 0.013709 | 0 |
| S3 | 23 | 0.008684 | 0.008743 | 0.9932 | 0.000059 | 41700 | 0.0000 | 0.013249 | 0 |
| S3 | 37 | 0.009408 | 0.009981 | 0.9426 | 0.000573 | 41942 | 0.0000 | 0.014297 | 0 |
| S3 | 53 | 0.009044 | 0.010500 | 0.8613 | 0.001456 | 41870 | 0.0000 | 0.014321 | 0 |
| S3 | 71 | 0.008673 | 0.008321 | 1.0423 | 0.000352 | 41618 | 0.0000 | 0.013207 | 0 |
| S4 | 11 | 0.008887 | 0.008739 | 1.0168 | 0.000147 | 37767 | 0.1678 | 0.013741 | 0 |
| S4 | 23 | 0.008610 | 0.008308 | 1.0363 | 0.000302 | 37273 | 0.1696 | 0.013038 | 0 |
| S4 | 37 | 0.009408 | 0.009425 | 0.9982 | 0.000017 | 37799 | 0.1660 | 0.014291 | 0 |
| S4 | 53 | 0.009039 | 0.009089 | 0.9945 | 0.000050 | 37630 | 0.1677 | 0.014250 | 0 |
| S4 | 71 | 0.008668 | 0.008154 | 1.0630 | 0.000514 | 37092 | 0.1705 | 0.013168 | 0 |
| S5 | 11 | 0.009186 | 0.008866 | 1.0360 | 0.000320 | 37297 | 0.1677 | 0.013742 | 0 |
| S5 | 23 | 0.008896 | 0.008402 | 1.0587 | 0.000493 | 36780 | 0.1696 | 0.013033 | 0 |
| S5 | 37 | 0.009742 | 0.009671 | 1.0074 | 0.000071 | 37338 | 0.1660 | 0.014291 | 0 |
| S5 | 53 | 0.009334 | 0.009247 | 1.0094 | 0.000087 | 37174 | 0.1677 | 0.014250 | 0 |
| S5 | 71 | 0.008966 | 0.008300 | 1.0803 | 0.000666 | 36644 | 0.1704 | 0.013168 | 0 |
| S6 | 11 | 0.009086 | 0.011422 | 0.7955 | 0.002336 | 37576 | 0.1525 | 0.012965 | 0 |
| S6 | 23 | 0.008761 | 0.009192 | 0.9531 | 0.000431 | 36752 | 0.1531 | 0.012891 | 0 |
| S6 | 37 | 0.009677 | 0.011668 | 0.8293 | 0.001991 | 37645 | 0.1494 | 0.013827 | 0 |
| S6 | 53 | 0.009296 | 0.011539 | 0.8056 | 0.002243 | 37645 | 0.1528 | 0.013729 | 0 |
| S6 | 71 | 0.008881 | 0.009039 | 0.9825 | 0.000158 | 36765 | 0.1533 | 0.013080 | 0 |
| S7 | 11 | 0.010235 | 0.013447 | 0.7611 | 0.003212 | 37567 | 0.1529 | 0.012821 | 0 |
| S7 | 23 | 0.010442 | 0.011426 | 0.9139 | 0.000983 | 36708 | 0.1529 | 0.013553 | 0 |
| S7 | 37 | 0.010818 | 0.012995 | 0.8325 | 0.002177 | 37640 | 0.1493 | 0.014248 | 0 |
| S7 | 53 | 0.010316 | 0.012955 | 0.7963 | 0.002639 | 37642 | 0.1532 | 0.013443 | 0 |
| S7 | 71 | 0.010362 | 0.011008 | 0.9413 | 0.000646 | 36761 | 0.1538 | 0.014187 | 0 |
| S8 | 11 | 0.020028 | 0.027073 | 0.7398 | 0.007045 | 37211 | 0.1542 | 0.020564 | 0 |
| S8 | 23 | 0.020588 | 0.024005 | 0.8576 | 0.003417 | 36385 | 0.1533 | 0.020775 | 0 |
| S8 | 37 | 0.019803 | 0.021710 | 0.9122 | 0.001907 | 37150 | 0.1496 | 0.020529 | 0 |
| S8 | 53 | 0.019652 | 0.023947 | 0.8206 | 0.004295 | 37146 | 0.1536 | 0.020496 | 0 |
| S8 | 71 | 0.019549 | 0.022760 | 0.8589 | 0.003211 | 36737 | 0.1546 | 0.020104 | 0 |

A non-zero numpy-warning count means the run hit a divide-by-zero or invalid operation somewhere in the estimator and the numbers on that line should not be trusted.

