# REVISION_AUDIT.md

Response to the PhD-level review of *Budget-Constrained Conformal Risk Control for
Fraud Decisioning under Drift and Endogenous Label Arrival* (weak reject / major
revision). One row per raised issue, with what was checked, what changed, and what
still needs a run.

Verification standard used here: a claim is only marked fixed if it was re-derived
from `results/results.json` or from the code, not re-read. `tools/check_numbers.py`
re-derives all 68 printed numbers from their specific source fields and writes
`NUMERICAL_CHECKS.md`; the current run reports **68 checked, 0 mismatched**.

---

## P0 — accepted and fixed

| # | Issue | Verified how | Disposition |
|---|---|---|---|
| 1 | The statement after Prop. 1 is reversed | Counterexample: scores uniform, `r = 0.9` on `[0,0.5]` and `0` above, `q = alpha = 0.5`. Then `R(F^-1(q)) = 0.90 > alpha` so the condition fails, yet `tau_lo = 1.0` is admissible and gives `R = 0.45 <= alpha`. Attainable while the condition reads false. | **Reviewer is right.** Without (A2) the condition is *sufficient, not necessary*. The paper said the opposite. Corrected, and the error is named in the text so a reader of the earlier version is warned. Same error corrected in `DECISIONS.md` D-19. |
| 2 | Prop. 1 lacks domain assumptions | Inspection | Added `0 <= eps <= 1` and `0 < b(1-eps) < 1-beta`, so `q in (0,1)` and the release region is non-empty. |
| 3 | Prop. 1 is static, the experiment is not | Inspection | Stated: the result is pointwise in the score law, `alpha_min` is a property of the current regime, and the implementation evaluates it per window on the *estimated* curve. |
| 4 | Observation model and Prop. 2 describe different processes | Read `simulate.py:316`: `disclosed = (y[i] == 1) and (u_disclose[i] < p_disc_eff[i])` | **Partly right, and the diagnosis matters.** Disclosure fires *only for true frauds*, so `Ytilde = Y*D` is exactly what the code does and Prop. 2 was correct. The Sec. IV-A *prose* wrongly implied any released item could disclose. Sec. III now defines `Y`, `D`, `A`, `Ytilde` and states `Ytilde = Y*D` on the release path as an equation; Sec. IV-A says only a fraud can surface. Interpretation A in the review does not apply to this simulator. |
| 5 | The M5 estimator is never displayed | Read `simulate.py:400-404` | Added as a numbered equation: `Rhat(tau) = sum w_i Ytilde_i / pi_i 1{s_i<=tau} / sum w_i 1{s_i<=tau}`, with `w_i = rho^(t-t_i) 1{pi_i >= pi_0}`. The propensity enters the numerator alone; that is what the code does and the paper now says so. |
| 6 | Trimming and decay change the estimand | Inspection: `keep = pi >= pi_floor` is applied to numerator *and* denominator | **Right.** The paper now states that the estimator targets risk on the overlap population `{pi_i >= pi_0}` and is exponentially weighted rather than the trailing-window mean of Sec. III, and repeats it in Threats. Both gaps are described as measured, not bounded. |
| 7 | Guarantee language exceeds what is proved | Audit of every use of *control*, *valid*, *guarantee* | **Right.** Title changed to *Budget-Constrained Risk **Calibration***. "controls the false-omission rate" became "targets". The paper no longer claims a conformal guarantee for M5. |
| 8 | The hard budget is not auditable in the table | `review_demand_rate` 0.022695 vs `review_served_rate` 0.01806 | Tables I and III now carry **demand and served as separate columns**, generated from `results.json`, so a reader can see the cap binding. Text states demand exceeds the cap in 0.44 of windows and served never does. |
| 9 | Overshoot arithmetic: 0.02269 - 0.0200 = 0.00269, not 0.002787 | `simulate.py:659`: `max(0, demand[sl].mean() - budget)` **per seed**, then averaged | **The number is right, the paper was unclear.** Per-seed clipping means `E[max(0,X-b)] != max(0,E[X]-b)`; the gap is 0.000093. Rather than explain a clipped statistic in six pages, the sentence now reports demand, the share of windows over cap, and served volume directly, and drops the derived overshoot. |
| 10 | Bootstrap unit unstated | Read `calib.py:80` and `experiment.py:79` | **Unit is the seed**, `n = 5`, percentile bootstrap over seed-level replicates. The review's worry about resampling correlated windows does not apply. But five replicates is coarse, and 10,000 resamples add nothing past that, so the paper now names the unit and labels the intervals **exploratory**. |
| 11 | M5 does not improve realised risk | `M4.for_overall` 0.020155 vs `M5` 0.020637; quiet 0.012344 vs 0.012511 | **Right, and the paper led with the wrong claim.** Abstract, Results and Conclusion now state plainly that realised risk does not improve and that the contribution is a believable estimate plus a usable infeasibility signal. |
| 12 | "near target", not "controlled at target" | 0.012511 > 0.0124 | Fixed; the abstract says both sit near the target rather than under it. |
| 13 | Exploration narrative overstates the ablation | 0.871 [0.823, 0.917] vs 0.881 [0.834, 0.929] | **Right.** The paper now says the step is 0.010 with intervals that overlap almost entirely, that most of the distance from M4 comes from the propensity correction, and that `eps = 0.120` was fixed as an operating assumption before the ablation ran. |
| 14 | "Bias" is a calibration ratio | `simulate.py:602`: `tru[warm].mean() / est[warm].mean()` | Renamed to *ratio* and defined in the caption as a quotient of two means over matured windows. |
| 15 | Active RCPS missing | arXiv API: 2406.10490, Xu, Karampatziakis, Mineiro, 2024-06-15 | **Right, and it weakens the capacity claim.** Cited in the Introduction and Related Work; the text now distinguishes budgeting *queries to an oracle* from a queue that also executes the decision. |
| 16 | Gamma parameterisation, cost units, relative AUC-PR | `config.py` | All stated: shape-scale, USD per thousand, "falls by a relative 0.7988". |
| 17 | "External validity" overstated | Inspection | Table III retitled **Cross-dataset transfer**, with a caption saying the disclosure and delay layers are imposed by us there too. ULB calendar mapping stated in Sec. V. |
| 18 | Hyperparameters only in the repo | — | `alpha, b, beta, eps, rho, gamma, pi_0`, both window caps, update cadence and support gate are now in Sec. V. |
| 19 | Infeasibility flag asserted, not validated | — | Not validated. The paper now says so and calls it a **diagnostic, not a verified detector**. See REQUIRES-RUN. |

## Rejected, with evidence

| Issue | Finding |
|---|---|
| "[25] and [26] are near-duplicate records for the same resource" | **They are distinct.** `10.5281/zenodo.21668281` is `resource_type: dataset`; `10.5281/zenodo.21670659` is `resource_type: publication`, different title. Both verified live against the Zenodo API by `tools/refcheck_arxiv.py`. Both retained. |
| "A naive transaction- or window-level bootstrap would be too narrow" | Does not apply. The unit is the independent seed (`calib.py:80`). The real weakness is `n = 5`, now labelled. |
| "Prop. 2 may not describe the observed process (Interpretation A)" | Interpretation A is not this simulator. Only frauds disclose, so `Ytilde = Y*D` holds exactly. The prose was wrong, not the proposition. |

## REQUIRES-RUN — not fixed, and the paper does not claim them

These need new simulation or a larger grid. None is asserted in the current text.

1. **Infeasibility-flag validation** against oracle feasibility per window: precision, recall, detection delay, risk when flagged vs not. The simulator has ground truth, so this is computable, but it is a new metric path.
2. **ESS-based support gate.** The gate is 200 units of weight, which is not 200 effective observations. Replacing it with `ESS = (sum w)^2 / sum w^2` changes policy behaviour and needs a re-run of the whole grid.
3. **Trimming diagnostics**: share of items and of fraud value dropped at `pi_0`, weight distribution, max weight, ESS. Not currently logged.
4. **>= 20 seeds** with paired per-method differences, to replace the exploratory five-seed intervals.
5. **Active RCPS as an implemented baseline.** It is cited and positioned, not run.
6. **Cross-fitting for the disclosure model.** It is fitted on closed-window allow rows and used to reweight overlapping data; the sample-splitting question is real and untouched.

## Deliverables

Produced: revised `paper/main.tex` (6 pages both toggle settings, 0 overfull boxes),
this file, `NUMERICAL_CHECKS.md` (generated), `WRITING_AUDIT.md` (generated, 13/13),
`refcheck/arxiv_verified.json` (27 entries, all live-verified), and the updated
artefact. `REPRODUCTION.md` is not a separate file: `README.md` plus the `Makefile`
targets already carry the one-command path, and duplicating it would create a second
thing to keep true. `CLAIM_TO_EVIDENCE.csv` is folded into `NUMERICAL_CHECKS.md`,
which is generated rather than typed and so cannot drift from the results file.

## Self-assessment after revision

| dimension | before | after | note |
|---|---|---|---|
| theoretical correctness | 4.5 | 7.0 | Prop. 1 direction fixed, domains stated, scope stated. Still no guarantee for M5, now not claimed. |
| estimator validity | 5.0 | 7.0 | Estimator displayed, estimand named. Corruption correction still rests on `pi` being right. |
| statistical validity | 4.5 | 5.5 | Unit named and labelled exploratory. Still five seeds. |
| reproducibility from paper | 5.0 | 7.5 | Hyperparameters, estimator, arrival model and ULB mapping in-paper. |
| experimental design | 6.0 | 6.0 | Unchanged; the flag validation is the missing piece. |
| novelty framing | 6.5 | 7.0 | Narrowed, and Active RCPS acknowledged. |
| natural-language quality | 8.5 | 8.5 | Aphorism density reduced; several signature lines cut. |

**Remaining reviewer risk.** The largest is that M5's headline is now a calibration
result rather than a risk result, which a reviewer may judge insufficient for a full
track. The second is the unvalidated flag. Both are stated in the paper rather than
left for a reader to discover.

---

# Round 2 — the estimator-repair prompt

A second review directed that the M5 estimator be repaired and every affected
experiment rerun, on three grounds. The implementation was audited against each
before any code was touched. Evidence with file and line references is in
`CURRENT_IMPLEMENTATION_MAP.md`.

| # | Directed change | Finding | Action |
|---|---|---|---|
| P0-2 | "`p_d` is fitted by logistic regression on closed release rows, so it estimates `q = r p_d`, not `p_d`. Implement two-model factorization (Option B1)." | **Already implemented.** `simulate.py:198-203` fits `m_obs` on the ledger label (target `q`) and `m_r` on analyst-adjudicated rows (target `r`); `simulate.py:234-239` returns `clip(q/max(r, 1e-3), 0.05, 0.97)`. That is Option B1, with the prescribed floors. | No code change. **The paper** described a single fit and so read exactly like the flawed design alleged. Sec. IV-B rewritten to state the ratio, its two training sets, and that neither sees topology. |
| P0-3 | "The denominator uses only records admitted to the label-arrival set, so it is outcome-selected. Rebuild it over the full risk set." | **Already correct in code.** `simulate.py:337-339` separates `idx_all` (every decision in the window) from `idx` (the arrived subset); `simulate.py:400-404` builds both sums over `idx_all`, with `ob_a` zeroed where no label arrived. `idx` is used only for the `min_calib_items` gate. | No code change. **The paper's displayed equation was wrong** — I wrote both sums over `C_t`, the arrived set, in the previous revision. Corrected to `D_t` with an explicit `O_i` arrival indicator, plus a sentence saying why a label-restricted denominator would be outcome-selected. |
| — | "Do not silently use simulator-only latent variables in the deployable M5." | **No leak.** `m_r` trains on `y` only for review-band and exploration rows (`simulate.py:370`), where an analyst adjudication reveals `y` in deployment. `oracle_pd` is reachable only via the `M5_oracle` condition. Topology is never a feature (`arrival.py:86`). | None needed; now stated in the paper. |
| P0-4 | "Trimming and decay change the estimand." | **Correct**, and unchanged from round 1: `keep = pi >= pi_floor` multiplies `w_all`, so low-propensity rows leave both sums. | Already stated in Sec. IV-B and Threats. Quantifying the gap stays REQUIRES-RUN. |

**Consequence.** Two of the three P0 items asked for a repair that the code already
contains. Since the implementation does not change, no experimental number changes,
and rerunning would reproduce the same `results.json` bit for bit. The defect was in
the paper's description of its own method, which is what was fixed. `make numbers`
still reports 68 claims checked, 0 mismatched, and both builds are 6 pages with 13/13
prose gates.

**Not done, and not claimed.** The round-2 prompt also asks for 20-30 seeds, an
exploration-only Horvitz-Thompson estimator, a doubly or triply robust estimator,
oracle-feasibility validation of the infeasibility flag, an ESS-based support gate,
propensity calibration diagnostics, a capacity-projection rewrite with per-window
quota tests, and roughly twenty deliverable files. None of that was done in this
round. Each needs new simulation and would rewrite every number in the paper; the
existing REQUIRES-RUN list in this file is the accurate statement of what is
outstanding. Nothing in the current manuscript depends on those results.
