# CURRENT_IMPLEMENTATION_MAP.md

What the code actually does, with file and line references, written to settle three
P0 allegations about the M5 estimator. Every claim below was read out of the source,
not inferred from the paper.

## 1. Is `p_d(x)` identified, or is a single regression on ledger rows being mislabelled?

**Allegation.** "If it trains on `Z` versus all closed release rows, it estimates
`q(x) = r(x)p_d(x)`, not `p_d(x)`. Do not continue calling such a model a
disclosure-propensity model."

**Finding: does not apply to this implementation.** `simulate.py:198-203` fits *two*
models and `simulate.py:234-239` takes their ratio:

```
m_obs = DisclosureModel().fit(zeta[idx_ledger], scores[idx_ledger], obs[idx_ledger])
m_r   = DisclosureModel().fit(zeta[idx_truth],  scores[idx_truth],  y[idx_truth])
...
_predict_pd:  clip( m_obs.predict(...) / max(m_r.predict(...), 1e-3), 0.05, 0.97 )
```

- `m_obs` regresses the ledger label `obs = Y*D` on release rows, so its target is
  exactly `q(x) = r(x)p_d(x)`.
- `m_r` regresses the **true** `y` on `idx_truth`, and `simulate.py:370` defines
  `truth = idx[(ch_l == CH_EXPLORE) | (ch_l == CH_REVIEW)]` — rows an analyst
  adjudicated. On those rows `y` is observed in deployment, not latent.
- The ratio with a floor `r_min = 1e-3` and clipping to `[0.05, 0.97]` is precisely
  **Option B1 (two-model factorization)** that the revision prompt asks to be built.

It is already built. What was wrong is the *paper*, which described a single logistic
fit and so read exactly like the flawed design being alleged. Fixed in Sec. IV-B.

## 2. Is the risk-estimator denominator outcome-selected?

**Allegation.** "The current estimator uses only records admitted to the label-arrival
set in both numerator and denominator. This can make the denominator outcome
selected."

**Finding: does not apply to this implementation.** `simulate.py:337-339` builds two
different sets:

```
idx_all = np.nonzero(in_win)[0] + lo     # every decision taken, not blocked
has_label = arrival[idx_all] <= d
idx = idx_all[has_label]                 # the arrived subset
```

and `simulate.py:359-363, 400-404` build the estimator on `idx_all`:

```
w_all  = decay_weights(age_a, rho);  bins_all = bin_of(grid, scores[idx_all])
ob_a   = np.where(has_label, obs[idx_all], 0.0)
keep   = pi >= pi_floor;  wk = w_all * keep
den    = accumulate(bins_all, wk)            # ALL decisions in the window
num    = accumulate(bins_all, wk * ob_a / pi)
```

The in-code comment at `simulate.py:352-358` states the intent: *"The denominator
counts every decision taken in the window, labelled or not, because the policy knows
perfectly well what it decided."* `idx` (the arrived set) is used only for the
`min_calib_items` gate at `simulate.py:350`.

So the denominator is **not** restricted to arrived labels and is not
outcome-selected. The paper's displayed equation, added in the previous revision, put
both sums over `C_t` (the arrived set) and therefore mis-stated the implementation.
That is a real defect **in the paper**, and the allegation correctly identifies the
formula even though it mis-diagnoses the code. Fixed in Sec. IV-B.

## 3. Does any simulator-only latent variable leak into the deployable M5?

**Finding: no.** Two paths use latent truth and both are legitimate or clearly gated:

- `m_r` trains on `y[truth]` where `truth` is review-band plus exploration rows
  (`simulate.py:370`). An analyst adjudication reveals `y` in deployment, so this is
  observable, not a leak. This is exactly the channel Proposition 2 says identifies
  `r` on the release support.
- `oracle_pd` (`simulate.py:382-383`) substitutes the true `p_disc_eff`. It is
  reachable only through the `M5_oracle` condition and is reported as a
  non-deployable diagnostic.

Fraud topology, which actually drives `p_d` in the generator, is **not** given to the
model: `DisclosureModel._design` (`arrival.py:86`) uses `[1, zeta, score, score^2]`
only. The deployable estimate is therefore a marginal approximation over unobserved
topology, which is stated in the paper as the residual the oracle brackets.

## 4. What the allegation gets right

**Trimming does restrict the estimand.** `simulate.py:400-402`: `keep = pi >= pi_floor`
multiplies `w_all`, so low-propensity rows leave numerator *and* denominator. The
estimator therefore targets risk on the overlap population `{pi_i >= pi_0}`, not on
all released traffic, and the geometric decay makes it exponentially weighted rather
than a trailing-window mean. Both were already stated in the paper and in Threats
after the previous revision. Quantifying the gap (trimmed share by topology and
regime, ESS, weight percentiles) remains **REQUIRES-RUN**; it is not currently logged.

## 5. Consequence for the revision plan

Two of the three P0 items in the revision prompt call for repairing the estimator and
rerunning every experiment. The estimator does not need repair, so **no number in the
paper changes as a result of items 1 and 2**, and rerunning would produce identical
results. The correct fix is to the paper's description of its own method, which is
what was done. Items marked REQUIRES-RUN in `REVISION_AUDIT.md` are unchanged and are
still not claimed in the paper.
