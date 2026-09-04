# DECISIONS.md

Every assumption, deviation and blocked item for the T13 paper and the
SynB2B-Fraud-Stream extension. Written as the work went, not reconstructed after.

---

## A. Data

**D-01. The benchmark's own statistics were re-derived from the CSV, not copied.**
`synb2b_fraud.csv` (SHA-256 `e41045ce0429151af6b3486a389b681b32b4890a39cd7b98df8e2a02628d8aaf`,
12,223,327 bytes) holds 100,000 rows, 1,500 fraudulent (1.500%), 1,250 buyers, 3,198
suppliers, exactly 300 rows per topology, timestamps 2024-01-01 00:58:19 to 2025-07-15
09:14:34 — a span of 561.34 days, 18.44 months. Median invoice 3,492.66; maximum
107,654.11; the fraudulent rows carry 11,384,649.69 in total invoice value. Every one
of these matches the published `synb2b_fraud_stats.json`. The paper quotes the
re-derived figures.

**D-02. Fraud is not uniformly distributed in time in SynB2B-Fraud, and this
constrains the experiment.** The generator places legitimate activity from
`uniform(0, 0.7 x span)` forward but places fraud in the 20%-98% band of the span. The
first 20% of rows by count therefore carries only 77 fraudulent rows (0.385%), against
1.78% over the remaining 80%. So the frozen-scorer training split is *not*
representative of the stream it will score, by a factor of about 4.6. Trimming the
fraud-free head does not fix it (measured: trimming to the first injected fraud leaves
the split at 0.406%). This is a property of the published benchmark, not a bug, and
the paper states it. Its consequence is that a risk target calibrated on the training
split would be meaningless, which is why D-07 sets the target as a stipulated
operating constant instead.

**D-03. OpenML `data_id=1597` has no `Time` column.** The brief asks for the ULB
stream to be sorted by `Time` and never shuffled. That column is not present in this
distribution: the frame ships `V1`..`V28`, `Amount` and `Class` only. The rows arrive
in the source file's original temporal order, so the row index *is* the time index and
the loader preserves it exactly — nothing is sorted, nothing is shuffled. Asserted in
`src/t13/data.py::load_ulb`.

**D-04. The ULB stream is replayed on an 18-month calendar.** The published portfolio
covers 48 hours. A 180-day reconciliation window is meaningless against 48 hours of
decisions, so the row order is preserved and mapped linearly onto the same 561.34-day
span as SynB2B-Fraud. This is a change of clock, not of order. The novelty variable
that the concept-shift event acts on has no counterpart in an anonymised PCA feature
set; `-V14` stands in, chosen because V14 is the component most strongly associated
with the positive class in this portfolio. Both choices are stated in the paper.

**D-05. IEEE-CIS was skipped.** `~/.kaggle/kaggle.json` does not exist on the build
machine and `gh`/Kaggle credentials were not available. Recorded in results as
`ieee_cis: "skipped, no credentials"`. Not a blocker: the brief says to skip silently.

---

## B. The drift layer (Layer A)

**D-06. Drift is an additive second fraud channel, not a relabelling.** The published
generator emits fraud as explicit topology firings, so there is no conditional law to
perturb. One is supplied:

    y_i = b_i OR Bernoulli(p_i),  logit p_i = a0 + gamma*zeta_i + beta(t)*nu_i + lambda(t)

`b_i` is the generator's own label and it is never un-set, so SynB2B-Fraud's 1,500
published labels survive intact and the extension is strictly additive. An earlier
version made the label a function of `b_i` inside the logit, which let the drift
process *demote* genuine topology frauds and pushed the realised rate from 1.5% to
6.3%. That was wrong and was discarded.

**D-07. Drift magnitudes are solved, not chosen by eye.** `tools/calibrate_drift.py`
solves `a0`, `beta_flipped` and `lambda_peak` per stream against stated targets — the
quiet regime gains at most 5% on top of its own base rate, the concept window reaches
2.2x the quiet rate, the prior window 3.4x — and writes `src/t13/drift_params.json`.
Solving per stream matters: constants fitted on a 2.1% B2B book made 73% of the ULB
stream's fraud drift-induced when transplanted.

**D-08. Supplier novelty is a rolling rank, not an absolute age.** Absolute supplier
age is not stationary in SynB2B-Fraud — buyers hold fixed supplier rosters, so the
share of rows with a supplier younger than 30 days falls from 36.1% in the training
split to 1.14% in the last third of the stream, and every fixed threshold on age
degenerates. `nu` is instead the causal rank of `-log1p(age)` among the previous 5,000
transactions, mapped to [-1, 1]: stationary by construction, and also the quantity an
analyst actually reasons about. This is a deviation from a literal reading of the
brief ("the sign of the supplier-novelty effect") and it preserves that effect's
meaning rather than its arithmetic.

**D-09. The prior-shift event raises AUC-PR rather than lowering it.** Measured:
covariate -79.7% relative AUC-PR, concept -69.1%, prior *+40.4%*. Enlarging the
positive class makes average precision easier. This is reported as a finding rather
than smoothed away, and the test suite asserts the sign explicitly
(`test_injected_drift_degrades_the_scorer_where_it_was_injected`).

---

## C. The arrival layer (Layer B)

**D-10. Undisclosed frauds are booked clean, not left unlabelled.** The brief says an
allowed item that never surfaces is "permanently censored". Modelling that as a
missing row would understate the problem. In an accounts-payable ledger the item is
not missing — at the close of the reconciliation window it is recorded as *good*. So
the allow path emits a label at exactly 180 days with value 0 whenever nothing
surfaced, including when the row was in fact fraudulent. That is the censoring, and it
is what the inverse-propensity correction has to undo.

**D-11. The arrival laws are treated as known, the disclosure probability as
estimated.** The review SLA and the reconciliation horizon are operational facts a
payments organisation sets and can state. `p_disclose` is not: it depends on the fraud
topology, which is exactly what is unknown at decision time. It is estimated as a
ratio of two fitted logistic models (`src/t13/arrival.py::DisclosureModel`) — see
Proposition 2(a) below for why that ratio needs exploration to be identified.

**D-12. Propensity trimming, not weight clipping.** Items whose arrival propensity is
below `pi_floor = 0.05` are dropped from both numerator and denominator rather than
having their weights capped. A decision taken nine days ago has had almost no chance
to report; its inverse propensity is enormous and its contribution is nearly all
variance. Trimming states the cost honestly: the estimator only looks at decisions old
enough to have had a fair chance of coming back, and that is where its lag comes from.

---

## D. Policy and operating point

**D-13. "Block" means a hard hold, not a decline, and the allowance is 8.00%.** In
accounts payable the high-score action is to hold the payment pending dual-control
release, not to decline it. An earlier 1.25% allowance made the whole feasible band
1.5 percentage points of the score CDF wide, over which the measured risk curve moves
from 0.01381 to 0.01465 — a range in which no method can differ from any other for
reasons that have nothing to do with the methods. 8.00% is defensible for a hold and
gives a band worth studying.

**D-14. The analyst queue is standing capacity and is always spent.** `tau_hi` is
placed so the band carries exactly `b(1-eps)` of traffic; `tau_lo` is the only free
threshold. Without this the policy can choose to review nothing, which is a deadlock:
no reviews means no adjudications, and the ledger says nothing for six months. That
deadlock was observed and is what motivated the constraint.

**D-15. Risk targets are stipulated per stream: alpha = 0.0124 (SynB2B),
alpha = 0.00093 (ULB).** Following D-02 they cannot be calibrated from the training
split. They are set so the target is attainable in the quiet regime with the stated
capacity and strictly unattainable during two of the three drift events — the regime
in which the paper's question has any content. Measured quiet-regime risk floors are
0.01381 (SynB2B) and 0.000233 (ULB) at the tightest admissible allow mass; the concept
and prior windows sit at 0.03866 and 0.05194 respectively. The targets are constants
of the experiment, stated in the paper, and ablated over 0.60x to 1.80x.

**D-16. Cost constants differ by stream, and both are stipulated.** SynB2B: 0.72 of
invoice value unrecovered plus USD 180 fixed per confirmed loss; 0.004 late-payment
penalty plus USD 95 fixed per held legitimate payment; USD 14.30 per analyst review
(a fully-loaded rate against a mean handling time). ULB: 0.91, USD 11.50, 0.0, USD
4.60, USD 2.10. Costing a EUR 88 card authorisation like a five-figure invoice hold
made a 6.3% hold rate read as a 9x cost blow-out for arithmetic reasons unrelated to
the method. Neither set is measured from anything; both are assumptions, they are
stated as such in Threats to Validity, and the paper's claim is risk control, not cost
optimality — M2 minimises cost by construction and it does win on cost.

**D-17. The monotone projection of the risk curve was removed.** Forcing
`R(tau)` non-decreasing with a running maximum let the low-score head of the grid —
where ten items and one accidental positive read as a risk of 0.1 — pin the entire
curve. Replaced by a support gate: a threshold is a candidate only if at least
`min_support = 200` units of calibration weight sit at or below it. Certifying a 1%
risk from a dozen observations is not conservatism, it is arithmetic that happens to
come out small.

**D-18. M0 and M1 coincide exactly on both streams.** Following D-02, the training
split's risk curve lies entirely below the target, so both the static rule and the
split-conformal quantile place `tau_lo` at the same grid point and the finite-sample
`(n+1)/n` correction has nothing to bite on. This is reported rather than papered
over: a conformal calibration computed once on an unrepresentative split is not
different from an eyeballed static threshold, which is itself worth saying.

---

## E. The propositions

**D-19. Proposition 1 holds as stated, under three assumptions.**

Let `F` be the score law, `r(s) = P(y=1 | S=s)`, `R(tau) = E[y | S <= tau]`, `b` the
review budget, `beta` the hold allowance, `eps` the exploration share. Assume
(A1) `F` continuous and strictly increasing on the operating range; (A2) `r`
non-decreasing; (A3) the queue is fully used, so the band carries `b(1-eps)`.

Under (A3) the hold mass is `1 - F(tau_lo) - b(1-eps)`, so `<= beta` forces
`F(tau_lo) >= 1 - beta - b(1-eps) =: q`. Under (A2), `R` is non-decreasing: for
`t1 < t2`, `R(t2)` is a convex combination of `R(t1)` and `E[y | t1 < S <= t2]`, and
the latter is at least `r(t1) >= R(t1)`. Hence

> **(alpha, b) is jointly attainable if and only if `R(F^{-1}(q)) <= alpha`**,
> equivalently `alpha >= alpha_min(b) := R(F^{-1}(1 - beta - b(1-eps)))`,
> and `alpha_min` is non-increasing in `b`.

If (A2) fails the "only if" survives but the "if" does not — the optimal allow set
need no longer be an interval in the score. The implementation raises the flag on the
*estimated* `R`, so the guarantee is only as good as the estimator, which is the
subject of Proposition 2. Measured `alpha_min` per regime is in `results.json`.

**D-20. Proposition 2(a) — identification — holds, in a weaker form than first
conjectured.** On the allow path only `Z = Y * D` is observed, `D ~ Bern(p_d(X))`
independent of `Y` given `X`. Then `P(Z = 1 | X = x) = r(x) p_d(x)`. For any `c(x)`
with `0 < c(x) <= 1/p_d(x)` and `c(x) r(x) <= 1`, the pair `(c r, p_d / c)` induces an
identical observable law, so only the *product* is identified from the ledger channel.
The review band identifies `r` on `(tau_lo, tau_hi]`, which is disjoint from the allow
region and therefore constrains `r` there only through a functional-form assumption.
An exploration sample drawn from the allow region with known probability identifies
`r` there directly, and `p_d = P(Z=1|X) / r(X)` follows.

The honest statement is therefore *not* "without exploration the risk is
unidentified". It is: **at eps = 0 the risk on the auto-allow path is identified only
through an extrapolation that no data in the allow region can test.** The paper says
that, not the stronger thing.

**D-21. Proposition 2(b) — the rate bound — is VACUOUS at the paper's operating
point, and the paper says so.** Bernstein on the exploration sample, with `R ≈ alpha`,
gives `|R_hat - R| <= eta * alpha` with probability `1 - delta` once

    n_exp  >=  2 (1 - alpha) log(2/delta) / (eta^2 alpha),
    and since n_exp ≈ eps * b * L,   eps >= 2 (1-alpha) log(2/delta) / (eta^2 alpha b L).

At `alpha = 0.0124`, `b = 0.0200`, `L = 45000`, `delta = 0.05`: `n_exp >= 588` and
`eps >= 0.653` for `eta = 1` (estimate within one target-width). For `eta = 0.5` the
requirement is `n_exp >= 2350`, i.e. `eps >= 2.61` — **greater than one, so no
admissible exploration rate certifies the target at that precision.** The default
configuration supplies roughly 108 explored transactions per calibration window
against the 588 the bound asks for.

The empirical rate is reported instead, and it is far kinder than the bound: the
estimator's bias ratio improves monotonically from 0.871 at `eps = 0` toward 1 as
`eps` rises, and by `eps = 0.40` it reaches 0.937 — while the bound at that point is
still nowhere near satisfied. Conservative bounds are conservative. Neither
proposition was revised to fit a measurement; the measurement is reported next to the
bound and the gap is named.

---

## F. Citations

**D-22. STREAM-BSG is omitted entirely.** The Zenodo dataset record names a companion,
*"STREAM-BSG: A Streaming Graph Architecture for Real-Time Fraud Detection in B2B
Payment Networks," IEEE InC4 2026*, and the linked GitHub repository states it was
accepted (Paper ID 3092, InC4 2026, Bengaluru, 7-8 August 2026, "to appear in IEEE
Xplore"). That is a self-report. IEEE Xplore returns HTTP 418 to automated search and
could not be queried; two web searches (title, and author plus venue) returned no
indexed record. Per the brief, an unconfirmed venue is omitted rather than cited as
"to appear", and the GitHub repository is not cited as a paper. The streaming-graph
context is described generically instead. **If the Xplore record appears before
submission, add it as a normal conference citation.**

**D-23. Papadopoulos et al. on inductive conformal prediction is cited from its ECML
2002 proceedings entry, not from arXiv.** No arXiv record exists for it. Verified
against the Springer DOI.

**D-24. Every arXiv citation was verified against the arXiv API before use.** Titles,
full author lists and submission dates are in `refcheck/arxiv_verified.json` (not
committed to the public repo; regenerate with `refcheck_arxiv.py`). Two entries in the
brief's table needed correcting from the verified metadata: arXiv:2607.27143 is by
Singh, Srikantha and Lakhanpal, and arXiv:2512.12844 is by Xu, Guo and Wei
(submitted 14 December 2025, revised 27 April 2026). Nothing that failed verification
was cited.

---

## G. BLOCKED

**B-01. The public GitHub repository could not be created or pushed.**
`gh auth status` reports *"You are not logged into any GitHub hosts."* There is no
`~/.config/gh/hosts.yml`, and no `GH_TOKEN` or `GITHUB_TOKEN` in the environment. This
session is non-interactive, so `gh auth login` cannot complete its device flow. Tried:
`gh auth status`, checking both token environment variables, checking the gh config
path. The repository therefore exists only locally, with its full commit history
intact, at `synb2b-fraud-stream/`. To publish it:

    gh auth login                      # interactive, one time
    gh repo create abhisheksharma2411/synb2b-fraud-stream --public --source=. --push

Nothing else in the deliverable depends on this. The paper's repository URL is written
as the intended public location.

**B-02. IEEE-CIS was not run.** See D-05. No credentials; skipped as the brief allows.

**B-03. IEEE Xplore could not be queried programmatically.** It answers automated
requests with HTTP 418. This is what forced D-22 to fall back on web search, and is
why STREAM-BSG is omitted rather than confirmed either way.

---

## H. Housekeeping

**D-25. No secret was found anywhere in the working tree.** `.gitignore` excludes
`*.pem`, `*.key`, `.env*`, `kaggle.json`, `credentials.json` and `secrets.*` as a
standing guard, along with `*.pdf` and the raw data directory.

**D-26. The PDF is not a deliverable.** It is compiled locally under both
anonymisation settings to check that the single file builds with `pdflatex` alone and
fits the page limit, then excluded by `.gitignore`. The `.tex` is the artefact.

**D-27. The reduced-size pass strides the stream rather than truncating it.**
`--small` takes every fourth row (SynB2B) or every fifth (ULB). Truncating to a head
slice squeezed eighteen months of calendar into four and landed the three drift
windows on top of each other, which made `make reproduce-small` a smoke test of
something other than the real configuration.
