# synb2b-fraud-stream

**Budget-constrained conformal risk control for fraud decisioning under drift and
endogenous label arrival.**

Companion code, data-extension and results for the T13 paper. The paper itself is a
single self-contained LaTeX file at [`paper/main.tex`](paper/main.tex) — no
`.bib`, no external figures, no `\input`; it compiles on Overleaf with `pdflatex`
alone and carries a one-line toggle for a double-blind build.

---

## What the problem is

A fraud model emits a score. An operations team has to take an *action* — allow, hold,
or send to a human queue — under a fixed analyst headcount, while the data moves under
it. This repository wraps any scorer in a selective-decision layer that controls a
chosen risk on the auto-decided traffic, respects a hard review budget, and is
auditable while it runs.

The part that turns out to matter is that **the label-arrival process is endogenous**.
The policy decides which labels it will ever receive:

| action | who labels it | when | is the label correct? |
|---|---|---|---|
| review | an analyst | hours | yes |
| allow | a dispute, chargeback or reconciliation exception, *if one surfaces* | 30–180 days | **no** — if nothing surfaces the ledger books it clean, fraud or not |
| hold / block | nobody, ever | never | there is no label |

Two consequences, and they pull in opposite directions:

- Clean allow-path items are booked at the 180-day close. Frauds surface in about
  sixty days. So the *arrived* pool over any recent window is made almost entirely of
  bad news, and a naive estimator reads the auto-allow path as far more fraudulent
  than it is. Measured on SynB2B-Fraud, in steady state, not as a cold start: the
  uncorrected estimator overstates by a factor of 19.3 (M4 bias ratio 0.0518).
- The frauds that never surface are recorded as good. That censoring pushes the other
  way, and it does not average out.

Fixing one and not the other is worse than fixing neither, which is the paper's least
comfortable result.

---

## SynB2B-Fraud-Stream: the data extension

This repository adds two annotation layers on top of the published
[SynB2B-Fraud](https://doi.org/10.5281/zenodo.21668281) benchmark. Both are *additive*
— none of the benchmark's 1,500 topology labels is altered or removed.

### Layer A — drift

Three events, three different mechanisms, at 28%, 55% and 80% of the evaluation
stream. Exact transaction indices and days are written to
[`results/drift_events.json`](results/drift_events.json).

| event | mechanism | what moves |
|---|---|---|
| covariate | the log invoice-amount law shifts up by 0.55, ramped over 3 days, and stays there | `P(x)`; `P(y \| x)` is untouched |
| concept | the supplier-novelty coefficient flips sign for 60 days | `P(y \| x)`: new fraud moves off young suppliers onto established relationships |
| prior | a base-rate surge, 4-day ramp then a 9-day half-life decay | `P(y)` at fixed shape |

Drift acts through a second fraud channel layered on the benchmark:

    y_i = b_i OR Bernoulli(p_i),
    logit p_i = a0 + gamma * zeta_i + beta(t) * nu_i + lambda(t)

`b_i` is the generator's own topology label, `zeta_i` a standardised log invoice
amount, `nu_i` a **rolling-rank** supplier-novelty score in [-1, 1] (absolute supplier
age is not stationary in this benchmark — see `DECISIONS.md` D-08). `a0`,
`beta_flipped` and `lambda_peak` are *solved*, per stream, against stated targets by
`tools/calibrate_drift.py` and versioned in
[`src/t13/drift_params.json`](src/t13/drift_params.json).

Measured effect on the frozen scorer's AUC-PR over the ten days after each event:
covariate **-79.88%**, concept **-68.05%**, prior **+32.33%**. The prior event *raises*
AUC-PR, because a larger positive class makes average precision easier. That is
reported, and asserted in the test suite, rather than tidied away.

### Layer B — endogenous label arrival

| action | label value | arrival time |
|---|---|---|
| review (band or exploration) | true `y` | decision + Gamma(2, 0.4) days |
| allow, disclosed | 1 | decision + min(Gamma(2, 30), 180) days |
| allow, not disclosed | **0**, whatever the truth was | decision + 180 days (ledger close) |
| hold / block | — | never |

Disclosure probability at mean invoice size, modulated by
`logit p += 0.62 * z(log amount)` and clipped to [0.03, 0.97]:

| topology | `p_disclose` | why |
|---|---|---|
| T5 wire redirection | 0.86 | the counterparty complains and the wire recall fails loudly |
| T4 shell-supplier ring | 0.71 | the ring unwinds under audit |
| T1 vendor injection | 0.58 | surfaces at supplier re-onboarding |
| T2 invoice cycling | 0.34 | often absorbed as a duplicate-payment write-off |
| T3 payment-term manipulation | 0.19 | almost never surfaces as fraud at all |
| drift-induced | 0.42 | — |

ULB has no topology annotation; there, disclosure is driven by amount alone
(intercept 0.50, coefficient 0.75).

**The invariant.** At decision time *t*, the calibration set may contain only rows
whose label-arrival time is at most *t*. It is enforced at every update by
`assert_calibration_causal`, and `tests/test_calibration_invariants.py` deliberately
violates it to confirm the guard fires.

---

## Methods

All six share one interface: score stream in, action stream out.

| | name | what it does |
|---|---|---|
| M0 | STATIC | thresholds fixed on the initial split, never updated |
| M1 | CONF-FIXED | split-conformal quantile computed once, with the `(n+1)/n` correction |
| M2 | COST-THRESH | cost-sensitive thresholds, refit on the same lagged window as M3–M5 |
| M3 | ACI-BUDGET | Gibbs–Candès recursion on the allow mass, plus a budget projection |
| M4 | ACI-BUDGET-CRC | M3 with a conformal-risk-control threshold search |
| **M5** | **ACI-BUDGET-CRC-IAP** | **M4 plus inverse-arrival-propensity reweighting and an ε exploration allocation** |

`M5_oracle` is M5 handed the true disclosure propensity — an upper bound on what the
correction can buy if the propensity model were free of error.

---

## Reproducing

```bash
make venv                 # .venv from the pinned requirements.txt
make test                 # pytest
make reproduce-small      # reduced-size end-to-end pass, target under 4 minutes
make reproduce-full       # the full grid; writes results/results.json + SUMMARY.md
make paper-data           # regenerate paper_data.tex and splice it into the paper
make audit                # the prose gates
make paper                # pdflatex both anonymisation settings, report page counts
```

The raw data is never committed. `make reproduce-*` fetches SynB2B-Fraud from Zenodo
and the ULB portfolio from OpenML (`data_id=1597`) on first run.

The full grid runs six methods plus the oracle variant on two streams over five seeds,
then nine one-factor-at-a-time ablations, with 10,000-resample percentile bootstrap
intervals on every headline number. A rerun reproduces `results.json` exactly, and
there is a test for that.

---

## Layout

```
src/t13/        config, data + causal features, drift, arrival, scorer,
                calibration primitives, the stream harness, the experiment grid
tests/          calibration invariants, causality, drift, bit-reproducibility
tools/          drift calibration, paper data, prose audit, paper build, probes
results/        results.json, SUMMARY.md, drift_events.json, paper_data.tex
paper/          main.tex — the deliverable
DECISIONS.md    every assumption, deviation and blocked item
WRITING_AUDIT.md output of the prose gates
```

---

## Licences

Code is Apache-2.0 (`LICENSE`). The derived data layers are CC BY 4.0, crediting
SynB2B-Fraud — see [`DATA_LICENSE`](DATA_LICENSE).

## Citation

```bibtex
@misc{synb2b_fraud_stream,
  author = {Sharma, Abhishek},
  title  = {synb2b-fraud-stream: budget-constrained conformal risk control for
            fraud decisioning under drift and endogenous label arrival},
  year   = {2026},
  url    = {https://github.com/abhisheksharma2411/synb2b-fraud-stream}
}
```

The underlying benchmark:

```bibtex
@dataset{synb2b_fraud,
  author    = {Sharma, Abhishek},
  title     = {SynB2B-Fraud: A Synthetic Dataset for Fraud Detection in
               B2B Payment Networks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21668281}
}
```
