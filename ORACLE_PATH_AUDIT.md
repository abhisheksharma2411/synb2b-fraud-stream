# Oracle-`p_d` path audit

**Question.** Does the `oracle_pd=True` experiment path bypass all *learned* propensity
logic, while still sharing every other component with the deployable path?

**Answer.** Yes, as of the working tree audited below. The oracle path no longer fits
`q-hat`/`r-hat`, no longer reports a cohort gap, and is otherwise bit-identical to the
deployable path in its decisions. Before the fix it fitted two logistic models it never
consulted and published a ~39.5-day cohort gap that described nothing it used.

## Provenance of this audit

| item | value |
| --- | --- |
| git HEAD | `50f7e8f73800ed85027164cefbc6d7f183d3a34d` |
| working tree | dirty — `src/t13/simulate.py`, `src/t13/config.py`, `src/t13/experiment.py` modified; `src/t13/joint.py` untracked |
| `src/t13/simulate.py` md5 | `6c17f357f269d0b7359c4221c060f1b8` (1024 lines) |
| audited (UTC) | 2026-09-05T16:41:48Z |

Line numbers below refer to that md5. `src/t13/*.py` is under concurrent edit by another
process; re-verify the hash before relying on a line number.

---

## 1. Does the oracle path fit `q-hat` / `r-hat`?

**No — verified.** The nuisance-refit block is guarded:

```python
# src/t13/simulate.py:419
if disc_refit_countdown <= 0 and not oracle_pd:
```

The `and not oracle_pd` clause **is present** in the working tree. It is **not** in
`HEAD`; `git show HEAD:src/t13/simulate.py` has the unguarded form at its line 412:

```python
if disc_refit_countdown <= 0:
```

`git log -S "and not oracle_pd" -- src/t13/simulate.py` returns only `2e58373`, and the
diff of that commit shows the string was introduced there on a *different* line (the
`pd_mae` diagnostic guard, now line 576). The refit guard is therefore an **uncommitted
working-tree change**, added in the same edit that introduced cohort alignment,
`cohort_gap_days` and the `joint_mode` hook. It is real but not yet in history.

Everything inside that guard is skipped on the oracle path: the `mature` / `truth`
cohort construction, the `cohort_aligned` / `cohort_fallback` branching, the
`_fit_disclosure` call that trains `m_obs` (target `q`) and `m_r` (target `r`), the
`JointTimeModel` fit when `joint_mode` is set, and the `pd_fallback` ratio update.

Minor dead state, noted not fixed: `disc_refit_countdown -= 1` (line 488) sits *outside*
the guard, so on the oracle path the counter decrements every update and runs negative
forever. Harmless — the `and not oracle_pd` short-circuits before it is read.

## 2. Does the oracle path still use the ratio and the clipping?

Distinguish two things the word "ratio" can mean here.

**The `q/r` nuisance ratio: bypassed.** `_predict_pd` (lines 235-241), which computes
`clip(m_obs.predict / max(m_r.predict, 1e-3), 0.05, 0.97)`, is only reached in the final
`else` at line 497. The oracle takes the first branch:

```python
# src/t13/simulate.py:490-491
if oracle_pd:
    pd_hat = p_disc_eff[idx_all]
```

`p_disc_eff = np.minimum(dr.p_disclose * disclose_mult, 0.999)` (line 297) is the
simulator's own disclosure probability. Nothing estimated enters.

**The Horvitz–Thompson ratio: shared.** Lines 500-507 build `pi` from `pd_hat` and the
delay CDF identically for both paths; lines 517-518, 529 form `num`/`den` and the estimate
`risk = calib.risk_curve(num, den)` identically. The oracle is corrected by exactly the
same `w * O * Y-tilde / pi` construction — only the `pd_hat` input differs.

**Clipping: partially shared, and asymmetric.**

| clip | deployable | oracle |
| --- | --- | --- |
| `pd_hat` to `[0.05, 0.97]` (inside `_predict_pd`) | applied | **not applied** |
| `p_disc_eff` ceiling at `0.999` (line 297) | applied | applied |
| `pi = np.clip(pi, pol.pi_floor, 1.0)` (line 515) | applied | applied |
| trimming `keep = pi >= pol.pi_floor` (line 514) | applied | applied |

The missing `[0.05, 0.97]` clip on the oracle is a genuine asymmetry, but a small one on
this generator: `config.P_DISCLOSE_FLOOR = 0.03`, `P_DISCLOSE_CEIL = 0.97`, and on
synb2b seed 11 (`small=True`) the true `p_disclose` runs `[0.04771, 0.97000]` with
`4e-05` of rows below `0.05` and none above `0.97`. Any such row is trimmed anyway,
since `pi = p_d * F_G <= p_d < pi_floor = 0.05`. So the asymmetry is real in code and
negligible in effect here; on a stream with a lower disclosure floor it would not be.

## 3. What does `cohort_gap_days` report for the oracle?

**Nothing — correctly N/A.** `_cohort_gap` is initialised to `float("nan")` (line 279)
and is only assigned inside the guarded refit block (lines 452-453). On the oracle path
the block never runs, so every window emits `NaN`. `_diagnostics` filters non-finite
values before aggregating (`v = v[np.isfinite(v)]`, line 728) and emits the
`cohort_gap_days_mean` key only when something survives. On the oracle path the key is
therefore **absent from the summary entirely**, which is the honest encoding of "this
path has no cohorts to be misaligned".

## 4. What the oracle shares with the deployable path

All of it, except the source of `pd_hat`. Verified by reading, and confirmed by the
identical `for_overall` in §5.

| component | code | shared? |
| --- | --- | --- |
| risk estimator (HT numerator/denominator, `risk_curve`) | 517-518, 529 | yes |
| policy / threshold placement (`_place_thresholds`, `crc_threshold`, `invert_cdf`) | 118-139, 614-618 | yes |
| delay law (`_gamma_cdf`, both channels) | 500-507 | yes |
| propensity trimming (`keep = pi >= pol.pi_floor`) | 514-516 | yes |
| ACI controller (`alpha_t += gamma_aci * (alpha - e_t)`) | 614-615 | yes |
| exploration draw, queue slots, overflow-to-midpoint | 262-270, 300-320 | yes |
| decay weights `rho^(t - t_i)` | 415 (IAP), 523 (non-IAP) | yes |
| feasibility oracles, monotonicity diagnostic, flag scoring | 364-399 | yes |
| `pd_hat` source | 490-499 | **no** — the only divergence |

`pd_mae` / `pd_brier` / `pd_clip_share` are `NaN` on the oracle (guard at line 576,
`if uses_iap and not oracle_pd`). That is correct: the oracle has no propensity error to
report, and reporting `0.0` would have been worse than reporting nothing.

## 5. Read-only confirmation run

```
PYTHONPATH=src ../.venv/bin/python -c "
from t13 import config as C
from t13.simulate import prepare_stream, run_policy
ctx = prepare_stream('synb2b', seed=11, small=True)   # n=25000, n_train=5000
pol = C.policy_for('synb2b')
r = run_policy(ctx, 'M5', pol, oracle_pd=True)
..."
```

synb2b, seed 11, `small=True`, method M5:

| field | oracle (`oracle_pd=True`) | deployable (`oracle_pd=False`) |
| --- | --- | --- |
| `nuis_refits` | **0** | 4 |
| `cohort_gap_days_mean` | **absent (`None`)** | 2.8196 |
| `nuis_fallback_share` | `None` | 0.5 |
| `nuis_thin_share` | `None` | 0.5 |
| `nuis_r_rows_median` | `None` | 80.5 |
| `nuis_q_rows_median` | `None` | 1534.0 |
| `pd_mae_mean` | `None` | 0.16106 |
| `trim_share_mean` | 0.097496 | 0.093232 |
| `ess_median` | 10389.49 | 9719.79 |
| `for_overall` | 0.0243336 | 0.0237882 |

`nuis_refits = 0` with `cohort_gap_days_mean` absent is the direct confirmation that the
oracle path performs no nuisance work.

### The fix is reporting-only, not behavioural

To separate "the oracle stopped fitting" from "the oracle changed its answers", the
pre-fix code was reconstructed outside the repo (the tree copied to a scratch directory
with `data/` symlinked and the single guard clause reverted by `sed`; nothing in the
repo was written):

| | pre-fix oracle | post-fix oracle |
| --- | --- | --- |
| `nuis_refits` | 4 | 0 |
| `cohort_gap_days_mean` | 3.1155 | absent |
| `for_overall` | `0.024333597255212457` | `0.024333597255212457` |
| `trim_share_mean` | `0.09749579271490312` | `0.09749579271490312` |
| `update_us_p50` | 4.114 | 3.839 |
| `update_us_p99` | 8.740 | 6.758 |

`for_overall` and `trim_share_mean` agree to every digit, which is what the code
predicts: pre-fix, `m_obs`/`m_r`/`pd_fallback` were written and then unconditionally
overridden at line 490, and `_fit_disclosure` draws from a local
`np.random.default_rng(9176 + ctx.seed)`, so it consumes no shared RNG state. The fix
removes wasted work and a false diagnostic; it changes no published decision. The p99
update cost drops ~23% because 4 of 80 updates were doing a fit whose result was
discarded — and that cost sat *inside* the timed region (`t0` is advanced past the
feasibility-oracle and diagnostic blocks at lines 364/399 and 550/595, but not past the
refit block), so the pre-fix oracle's reported per-update latency was inflated by work
it never used.

## 6. The historical defect, stated plainly

Before the guard was added, `oracle_pd=True` ran the entire nuisance pipeline:

1. It built the `mature` (matured ledger) and `truth` (adjudicated) cohorts, applied the
   cohort-alignment band and the thin-support fallback, and counted the fallbacks.
2. It fitted `m_obs` and `m_r` — two ridge logistic regressions — and computed
   `pd_fallback`.
3. It then discarded all of it one line later, because `if oracle_pd: pd_hat =
   p_disc_eff[idx_all]`.
4. It recorded `_cohort_gap` from those unused cohorts and published it.

The published number is in the committed results file. `results/results.json` (run at
commit `50f7e8f`, `git_dirty: true`, `2026-09-05T09:08:53Z`, 20 seeds) records for
synb2b:

| variant | `cohort_gap_days_mean` |
| --- | --- |
| `M5` (aligned) | 39.6779 |
| `M5_misaligned` (legacy) | 87.0191 |
| `M5_oracle` | **39.5384** |

`COHORT_ALIGNMENT_AUDIT.md` line 23 and `PAIRED_VARIANT_DIFFERENCES.md` line 59 both
carry that 39.5384, and the paired table presents it as a −47.48-day *improvement* of
the oracle over the legacy estimator. That reading was meaningless: the oracle has no
`q`/`r` cohorts in its estimator, so it cannot have a cohort gap, and the 39.5384 was
the gap between two fits that were computed and thrown away. It looked like evidence
that the oracle was well aligned; it was an artefact of dead code.

**Not verified:** whether the current `results/results.json` will change once regenerated
under the fix. Based on §5 the *decision* metrics should be unchanged and only the
`cohort_gap_days_mean` / `nuis_*` fields for `M5_oracle` should disappear, but that
prediction was checked on `small=True`, seed 11, one method — not on the published
20-seed full-stream grid. Re-running `method_comparison` on the published seed list and
diffing `results/results.json` would settle it.
