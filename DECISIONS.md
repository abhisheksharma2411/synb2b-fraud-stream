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
covariate -79.88% relative AUC-PR, concept -68.05%, prior *+32.33%*. Enlarging the
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

If (A2) fails the "if" survives and the "only if" does not. This entry originally
said the reverse, and the paper repeated it until an external review caught it.
Sufficiency is immediate: `tau_lo = F^-1(q)` is admissible whenever its risk clears
`alpha`. Necessity fails because a non-monotone `r` lets a larger threshold carry
lower cumulative risk. Counterexample, uniform scores, `r = 0.9` on `[0, 0.5]` and 0
above, `q = alpha = 0.5`: the condition reads `R(0.5) = 0.90 > 0.5` and yet
`tau_lo = 1.0` is admissible with `R = 0.45 <= alpha`. The implementation raises the flag on the
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

    n_exp  >=  2 [(1 - alpha) + eta/3] log(2/delta) / (eta^2 alpha),
    and since n_exp ≈ eps * b * L,
    eps >= 2 [(1-alpha) + eta/3] log(2/delta) / (eta^2 alpha b L).

At `alpha = 0.0124`, `b = 0.0200`, `L = 45000`, `delta = 0.05`: `n_exp >= 786` and
`eps >= 0.873` for `eta = 1` (estimate within one target-width). For `eta = 0.5` the
requirement is `n_exp >= 2748`, i.e. `eps >= 3.05` — **three times the whole budget, so
no admissible exploration rate certifies the target at that precision.** The default
configuration is designed to supply `eps*b*L = 108` explored rows per calibration
window and measures 102, against the 786 the bound asks for.

An earlier draft of this entry dropped Bernstein's range term and carried the
variance term alone, `2(1-alpha)log(2/delta)/(eta^2 alpha)`, giving 588 and 2350. That
understates the requirement. The corrected figures are larger, so the conclusion that
the bound is vacuous is strengthened, not weakened, by the fix. See D-36.

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
full author lists and submission dates are in `refcheck/arxiv_verified.json`, which is
committed and regenerable with `make refcheck`. See D-28: when this entry was first
written neither the file nor the script existed, and both were rebuilt. Two entries in the
brief's table needed correcting from the verified metadata: arXiv:2607.27143 is by
Singh, Srikantha and Lakhanpal, and arXiv:2512.12844 is by Xu, Guo and Wei
(submitted 14 December 2025, revised 27 April 2026). Nothing that failed verification
was cited.

---

## G. BLOCKED (B-01 since resolved)

**B-01. RESOLVED. The public repository exists and carries the full history.**

<https://github.com/abhisheksharma2411/synb2b-fraud-stream>, public, default branch
`main`, all 15 commits, local and remote `HEAD` identical at `bebe8fa`.

It stayed blocked through the build because `gh` had no session, no `~/.config/gh` and
no `GH_TOKEN`, and the device flow needs an interactive terminal. The only credential
on the machine was a keychain entry under the username `x-access-token`, a GitHub App
installation token scoped to repositories the app already held, so it could not create
a new one, and a plain `git push` returned *"Repository not found"* because a push
cannot bring a repository into existence. The author supplied a personal access token
with `repo` scope, which cleared it.

The token was passed through the environment and consumed by an inline credential
helper, so it is not in `.git/config`, not in the remote URL and not in any tracked
file; `git grep` over the tree confirms that. It was disclosed in a chat transcript
during the exchange and should be rotated on that basis alone.

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

---

## I. The paper, and what building it turned up

**D-28. Reference verification is now reproducible, because it was not before.**
D-24 claimed the arXiv metadata lived in `refcheck/arxiv_verified.json`, regenerable
with `refcheck_arxiv.py`. Neither existed anywhere in the tree or in any commit, so the
claim rested on nothing a reader could check. `tools/refcheck_arxiv.py` now verifies
all 26 bibliography entries against the arXiv Atom API, the Crossref REST API and the
Zenodo REST API, comparing normalised titles and first-author surnames, and writes
`refcheck/arxiv_verified.json`. Every entry returned OK. Two corrections that D-24
recorded were confirmed against live metadata: arXiv:2607.27143 is Singh, Srikantha
and Lakhanpal, and arXiv:2512.12844 is Xu, Guo and Wei. One title in the brief was
wrong and is cited from the API instead: arXiv:2606.14909 is *"Audited Conformal
Prediction **for Classification** under Unknown Distribution Shift"*. Vovk et al. is
the one entry with no machine-readable record and is marked `MANUAL`.

**D-29. The anonymous-build leak check tested the wrong thing.** It scanned the raw
source for the author name, the repository URL and an acknowledgements heading. A
single file carrying both variants behind `\ifanon` always contains all three, so that
check could never pass by construction. It now resolves the conditional first and
scans what the anonymous build actually renders, extracted from the PDF with
`pdftotext`. The anonymous PDF contains no author, no affiliation, no URL and no
acknowledgement; "A. Sharma" survives only in the two bibliography entries for the
cited dataset, which is ordinary third-person citation and is what a double-blind
submission is supposed to look like.

**D-30. Two latent bugs in the paper tooling surfaced the moment a real `.tex`
existed.** Both `build_paper.py` and `audit_prose.py` flipped the toggle with
`re.sub(..., "\\anonfalse", ...)`, and a replacement string beginning `\a` is read by
`re` as the BEL escape, so the toggle line became `^^Gnonfalse` and neither build was
ever anonymous. Both now pass a function as the replacement. Separately, the page count
came from a `/Type /Page` regex over the raw PDF bytes, which returns zero when pdfTeX
puts the page tree in a compressed object stream; both tools now prefer `pdfinfo` and
keep the regex as a fallback.

**D-31. The file's own header comment cannot name the toggle.** `build_paper.py`
replaces the first match of `\anon(true|false)`, and the original header said "flip
this to `\anontrue`", so the substitution rewrote the comment and left the real toggle
alone. The comment now spells the values without a backslash.

**D-32. Section 6.6's cut list was not used, and Fig. 2 stayed.** The named build came
in at 7 pages with 19 words of bibliography on the last page. Two lossless edits fixed
it: merging two sentences, and deleting a repository URL that was printed twice, once
in the acknowledgement and once under the toggle in Section V. The cut order exists to
protect the propositions, the negative result and the citations when content has to go;
no content had to go here, so removing a figure to recover 19 words would have cost
more than it saved. If a future revision overruns again, Fig. 2 is still first out.

**D-33. Two loose fractions in the draft were replaced with measured values.** "A tenth
of the base rate" described a ratio that is nearer a twelfth (0.002230 against
0.028413) and is now stated qualitatively; "a fifth of the analyst budget" described
`eps = 0.120` and now states the figure. Both were caught by reading the prose against
`results.json` by hand, not by a gate, since the numeric gate only checks that a
literal appears in the artefacts and cannot judge an English fraction.

**D-34. Two small LaTeX choices.** `\title` and `\author` sit in the preamble so the
prose audit never sees the author block, and `proof` is defined locally in three lines
rather than by loading `amsthm`, which fights IEEEtran's own theorem handling.

**D-35. Stale numbers in this file were corrected against the final grid.** D-09 and
D-21 were written before the last full run and carried its predecessors: the
degradation figures are -79.88%, -68.05% and +32.33%, and the default configuration
supplies about 102 explored rows per calibration window. That last one was not
staleness and was mislabelled as such here: `108` is the nominal `eps*b*L`, and `102`
is what the run actually delivered. Both are now stated. The README carried
the same two stale degradation figures and the estimator overstatement as "about 19x";
both were corrected. Every number in the paper was then checked one at a time against
`results/results.json`: 57 headline claims, 0 mismatches.

**D-36. An adversarial review of the finished paper found four defects, all fixed.**
The review was run against the compiled paper, not the draft, and every claim was
re-derived rather than re-read.

1. *Proposition 2 stated its admissible set backwards.* The condition read
   `0 < c(x) <= 1/p_d(x)`, which permits `c < p_d` and therefore `p_d/c > 1`, not a
   probability. With `p_d = 0.5, r = 0.02` the stated set admits `c = 0.1`, whose
   alternative disclosure probability is 5. The correct interval is
   `p_d(x) <= c(x) <= 1/r(x)`. The non-identification result itself is unaffected: the
   interval is non-degenerate whenever `r p_d < 1`.
2. *The rate bound was not the inequality it was attributed to.* See the revision
   above: Bernstein's range term was missing.
3. *Section V referenced itself.* "Arrival follows Section V's channels", inside
   Section V. The channels are defined in Section IV; the label was added and the
   reference repointed.
4. *"Perfect feedback" was the wrong description of the delay ablation.* Setting the
   delay multiplier to 0 removes the lag and leaves the censoring untouched, and the
   distinction between those two is the paper's own thesis. The sentence now says
   which one was removed.

Two smaller changes came out of the same pass. The oracle-propensity bracketing claim
holds on SynB2B but not on ULB, where the oracle overshoots to 1.27, further from
unbiased than the uncorrected M4 at 0.883; the ULB paragraph now says so. And four
paragraphs opened with a bare numeral ("Four things follow", "Three channels",
"Three assumptions", "Three limits"), which had become a visible template; two were
rewritten.

All thirteen prose gates still pass and both builds are still 6 pages.

---

## J. Response to external peer review

**D-37. A PhD-level review returned weak reject / major revision, and it was right
about the biggest thing.** The full disposition is in `REVISION_AUDIT.md`. Four
findings were accepted as defects and fixed, three were rejected with evidence, and
six are recorded as REQUIRES-RUN and are not claimed in the paper.

The lead finding: the sentence after Proposition 1 had its implication reversed.
Without monotonicity the feasibility condition remains *sufficient* and stops being
*necessary*; the paper asserted the opposite. It came from D-19 above, written in an
earlier session, and I carried it into the paper without re-deriving it. My own
adversarial pass (D-36) checked the proposition and missed the sentence after it,
which is the specific failure worth remembering: **verifying a theorem is not the
same as verifying the prose that interprets it.**

Two rejections are worth recording because they turned on reading the code rather
than the paper. The review argued the observation model and Proposition 2 describe
different processes; `simulate.py:316` shows `disclosed = (y[i] == 1) and (...)`, so
disclosure fires only for true frauds, `Ytilde = Y*D` holds exactly, and the
proposition was right while the Section IV-A prose was wrong. The review also
suspected the two Zenodo references were duplicates; the Zenodo API returns
`resource_type: dataset` for one and `publication` for the other, so both stay.

**D-38. `build_paper.py` was hiding a fatal LaTeX error.** It ran `pdflatex` without
`-halt-on-error`, so when the generated table body carried nine fields against an
eight-column preamble, LaTeX reported `! Extra alignment tab has been changed to
\cr`, recovered, and still produced a PDF that the build called OK. Only
`audit_prose.py`, which does halt, failed. The flag is now set in both. The
underlying mismatch came from my own edit to `make_paper_data.py` that used
`str.replace` without asserting the pattern matched, so it silently did nothing:
every generator edit in that file now asserts its match count.

**D-39. Numeric claims are now checked against their source field, not just against
the artefact.** `audit_prose.py` gate 7 only asks whether a literal appears somewhere
in `results.json`, which a wrong digit can satisfy by coincidence.
`tools/check_numbers.py` re-derives all 68 printed numbers from the specific field
each comes from and writes `NUMERICAL_CHECKS.md`. Current run: 68 checked, 0
mismatched. Run it with `make numbers`.

**D-40. A second review asked for an estimator repair the code already contained.**
It directed that `p_d` be rebuilt as a two-model ratio and that the risk-estimator
denominator be widened beyond the arrived-label set, then that every experiment be
rerun. `simulate.py:198-239` already computes `p_d` as `clip(q/max(r, 1e-3))` from two
fits, and `simulate.py:337-404` already builds both sums over every decision in the
window rather than over arrivals. Full evidence is in
`CURRENT_IMPLEMENTATION_MAP.md`.

What was actually wrong was the paper. Section IV-B described `p_d` as a single
logistic fit, and the estimator equation I added in the previous revision put both
sums over the arrived set. Both now match the implementation, and the equation carries
an explicit arrival indicator plus a sentence on why a label-restricted denominator
would be outcome-selected.

No experiment was rerun, because nothing in the code changed and a rerun would
reproduce `results.json` exactly. The lesson is narrower than the review's: a wrong
description of a correct method is still a defect, and it invites exactly the
objection it received. It also means an external reviewer reading only the PDF had no
way to tell the two apart, which is the argument for putting the estimator in the
paper rather than in the artefact.

**D-41. The infeasibility flag is now validated, and it is the one place M5 wins
outright.** Oracle feasibility is Proposition 1 evaluated per window on the true risk
curve: with `q = 1 - beta - b(1-eps)`, the pair was attainable exactly when the true
risk at `F^-1(q)` clears `alpha`. Scored against it over 320 windows, M5 reaches
precision 0.900, recall 0.868 and balanced accuracy 0.906, against M3 at 0.853 and M4
at 0.736. M2 records a perfect recall of 1.000 by flagging 0.924 of windows, which
leaves its specificity at 0.116. True risk in flagged windows is 0.02206 against
0.009356 in unflagged ones. Full table in `INFEASIBILITY_VALIDATION.md`.

M0 and M1 never update, so they have no windows to score. The paper previously read
their flag rate of 0 as M0 "declaring none", which was misleading: it is the absence
of a mechanism.

**D-42. Two bugs I introduced while adding the instrumentation, both caught by
diffing against the pre-instrumentation results.** The oracle computation sat inside
the timed region, so every `update_us` figure was inflated: 448 values moved, and all
448 were timing. `t0` now advances by the oracle's duration, and the re-measured cost
is 18.03 us at p50 against the 18.36 previously reported. Separately, `e_true` is only
defined inside the estimator branch, so the risk-when-flagged split silently produced
nothing; it is now recorded per update.

The decisive check is that the same diff showed **zero non-timing changes**: every
FOR, ratio, review rate and cost is bit-identical to the run before the oracle
existed. `tests/test_flag_validation.py` asserts the same property directly by
stubbing the scorer. An oracle that reads latent `y` and could influence a decision
would make the whole experiment circular, so this is the property that had to hold.

**D-43. The feasibility oracle no longer leans on the assumption it was testing.**
A reviewer pointed out that scoring the flag with Proposition 1 presumes (A2), the
monotone conditional rate, which the paper itself calls the assumption a real
portfolio breaks first. That objection was correct and testable, so both were
measured.

(A2) does fail: across admissible thresholds on the true risk curve, 0.462 to 0.655
of steps run downhill depending on the method, 0.600 under M5. The generator has five
topologies and a novelty sign flip, so a non-monotone conditional rate is what one
should expect.

It changes nothing measurable. An assumption-free oracle, exhaustively searching every
admissible threshold and calling the window feasible if any clears alpha, agrees with
the Proposition 1 oracle on 0.999 to 1.000 of scored windows, and every flag metric is
identical to three decimals under both. M5 keeps precision 0.900, recall 0.868 and
balanced accuracy 0.906. The exhaustive oracle is now the one the paper reports, with
the agreement rate and the (A2) violation rate stated alongside it, because a result
that survives dropping an assumption is worth more than one that needs it.

**D-44. Wall-clock is the only figure here that is not bit-reproducible.** Three runs
of the same grid gave M5 p50 of 18.36, 18.03 and 17.98 microseconds. Everything else
is identical across runs to the last digit, which the diff against earlier results
files confirms each time: 448 values move and all 448 are timing. The paper now says
so where it reports the number.

**D-45. The estimand gap is now a number rather than an acknowledgement.** The paper
said Eq. (3) targets an exponentially weighted risk on a trimmed overlap population
rather than the trailing-window risk of Section III, and that the gap was "measured,
not bounded". Nothing measured it. Per window, on true labels, the boxcar trailing
risk, the decay-only risk and the decay-plus-trimming risk are now all computed:
the estimand sits **0.07896** from the trailing target in relative terms, and trimming
removes **0.1508** of rows carrying **0.1838** of their fraud. Full table in
`DIAGNOSTICS.md`, regenerable with `make diagnostics`.

**D-46. The support gate is stated in weight, and the weights turn out not to be
concentrated.** A reviewer objected that 200 units of calibration weight is not 200
useful observations, which is right in principle. Measured: Kish effective sample size
under M5 has median 37023 against that gate of 200, fifth percentile 5459, and the
largest normalised weight averages 0.00006299. So the failure mode the objection
describes, a handful of enormous inverse-propensity weights clearing the gate while
carrying almost no information, is not happening here. This is a null result and it
stays in `DIAGNOSTICS.md` rather than the paper, which has no room for findings that
change nothing.

**D-47. The learned disclosure propensity is badly wrong, and it is structural.**
Against simulator truth it misses by **0.1579** on average, on a quantity that ranges
from 0.19 for payment-term manipulation to 0.86 for wire redirection, with 0.03044 of
predictions pinned at a clip boundary. The cause is not estimation noise: the
deployable model sees the standardised amount and the frozen score and never the fraud
topology, which is what actually sets disclosure. Handing it topology would repair the
number and void the experiment, since topology is unavailable at decision time in the
setting the paper claims. This is now stated in Threats rather than left for a reader
to infer from the oracle gap.

**D-48. Reference [27] was uncited and is gone.** The SynB2B-Fraud preprint appeared
in the bibliography but no `\cite` referenced it, which a reviewer spotted. Removing
it fixes an IEEE-style defect, frees four lines, and removes one of the two entries
naming the author from a double-blind submission. The dataset record is cited and
stays. A check for uncited keys now runs alongside the other gates.

**D-49. There is now an anonymous artifact, because "URL withheld for review" gives a
reviewer nothing.** A reviewer cannot check the estimator, the feasibility oracle or
the flag scoring without running the code, and the public repository carries the
author's name. `make anon-artifact` exports the tracked tree via `git archive`, so no
git history ships, rewrites every author identifier, sets the paper's toggle to the
anonymous build, and **fails the build if any identifier survives the scan**. It
caught one on the first run: `tools/build_paper.py` names the identifiers inside its
own leak-detector list, which the naive substitution missed.

Verified on the export rather than asserted: 55 text files scanned, 8 rewritten, zero
occurrences of the name, the handle, the email or the ORCID anywhere in the tree, the
paper compiles from the artifact to 6 pages with `\anontrue` already set, and the
rendered PDF's author line reads "Author names withheld for review" with zero
identifier hits in its text.

Upload the archive to whatever anonymous host the venue accepts and replace "URL
withheld for review" in Section V with that link at submission time.

**D-50. Twenty seeds, fixed and published before the run.** `C.SEEDS` keeps the
original five so every earlier comparison stays paired, then appends the next fifteen
primes after 71. The rule is stated in the source next to the list precisely so it
cannot be read as a set chosen after seeing which seeds were convenient. No seed is
dropped. Every method sees the same stream under a given seed, so the method contrasts
stay paired. This replaces five-replicate intervals that the paper had to label
exploratory.
