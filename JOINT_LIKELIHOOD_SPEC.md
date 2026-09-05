# `t13.joint.JointTimeModel` — estimator specification

A reimplementation-grade description of `src/t13/joint.py`. Everything below was read
off the source; the empirical numbers are from `tests/test_joint.py`, run for this
document.

| item | value |
| --- | --- |
| file | `src/t13/joint.py`, 491 lines, md5 `8efdac59aae8396550579d7024fabe39` |
| git status | **untracked** — the module is not yet in git history |
| dependencies | `numpy`, `scipy.optimize.minimize`, `scipy.special.{expit, gammainc, log_expit}` |
| tests | `tests/test_joint.py`, 11 tests, all pass (0.58 s) |
| audited (UTC) | 2026-09-05T16:41:48Z; spec reconciled against the 16:52Z state |

> **The module is being actively rewritten.** It changed twice while this document was
> being written (md5 `d172e70c…` → `1bb9a1f7…` → `8efdac59…`, 396 → 491 lines). Everything
> below describes the pinned md5 above. Two of those edits are substantive and are marked
> **[new]** in §3, §5, §6 and §7. Re-check the hash before relying on this.

---

## 1. What it estimates and why the naive route fails

The release channel never reports a label. It reports an *event*: by the time we look, a
dispute / chargeback / reconciliation exception either has or has not surfaced. Two
functions are wanted:

- `r(x, u)` — the conditional fraud rate;
- `p(x, u)` — the probability that a fraud **leaves a trace at all**, i.e.
  `P(signal will ever surface | Y = 1, x, u)`.

On the release channel only the product `r(x,u) · p(x,u) · F(a)` is observable. The
deployed pipeline (`simulate.py::_predict_pd`) recovers `p` as `q-hat / r-hat` from two
separately-fitted logistics; that is unstable wherever `r` is small, and the two fits are
drawn from different transaction-time cohorts, so the quotient moves for reasons
unrelated to the estimand. This module fits both surfaces **in one maximum-likelihood
problem**, on every row that carries information.

## 2. Likelihood contributions, exactly

Rows are typed by `kind`: `0` = adjudicated, `1` = released, `2` = held. Only
`used = (kind == 0) | (kind == 1)` participates; any other code is excluded with the
held rows.

Write `eta_r = X_r · theta`, `eta_p = X_p · phi`, `r = sigma(eta_r)`, `p = sigma(eta_p)`,
`F_i = F(a_i)` the known delay CDF at row age `a_i`.

### 2a. Adjudicated row (analyst review or exploration draw), truthful label `y`

    y · log sigma(eta_r) + (1 - y) · log sigma(-eta_r)

```python
eta_a = np.clip(Xa @ th, -_ETA_CLIP, _ETA_CLIP)
ll += float(np.sum(ya * log_expit(eta_a) + (1.0 - ya) * log_expit(-eta_a)))
g_th += Xa.T @ (expit(eta_a) - ya)
```

`ya = np.clip(y_obs[adj], 0.0, 1.0)`. These rows are the **only** place `r` is directly
identified.

### 2b. Released row, signal has surfaced by now

    log( r · p · F(a) )  =  log sigma(eta_r) + log sigma(eta_p) + log F(a)

```python
log_Fs = np.log(np.clip(Fs, _EPS, 1.0))
const_ll = float(np.sum(log_Fs))          # parameter-free, added once
...
eta_rs = np.clip(Xs_r @ th, -_ETA_CLIP, _ETA_CLIP)
eta_ps = np.clip(Xs_p @ ph, -_ETA_CLIP, _ETA_CLIP)
ll += float(np.sum(log_expit(eta_rs) + log_expit(eta_ps)))
g_th += Xs_r.T @ (expit(eta_rs) - 1.0)
g_ph += Xs_p.T @ (expit(eta_ps) - 1.0)
```

`log F(a)` is hoisted into `const_ll` because `F` carries no parameters. It shifts the
objective but not the argmin.

### 2c. Released row, **no signal yet** — the censored term

    log( 1 - r · p · F(a) )

**This row is not treated as a truthful negative.** It is a row for which the joint event
`{it was fraud} AND {the fraud left a trace} AND {that trace arrived by now}` failed.
The code that implements exactly that:

```python
if len(Fn):
    rn = _sigmoid(Xn_r @ th)
    pn = _sigmoid(Xn_p @ ph)
    s = np.clip(Fn * rn * pn, _EPS, 1.0 - _EPS)
    ll += float(np.sum(np.log1p(-s)))
    # d log(1 - s)/d eta_r = -s (1 - r) / (1 - s), and likewise for p
    ratio = s / np.maximum(1.0 - s, _EPS)
    g_th += Xn_r.T @ (ratio * (1.0 - rn))
    g_ph += Xn_p.T @ (ratio * (1.0 - pn))
```

The contrast with the naive reading is worth stating numerically. `test_a_clean_release_
is_not_a_truthful_negative` generates 30 000 rows in which *no* release has surfaced and
fits twice — once with these rows typed as released (censored), once with them relabelled
as adjudicated negatives:

    mean r:  joint = 0.1343   naive = 0.0430   truth = 0.1401

Reading the clean releases as `y = 0` drags `r` down by roughly the whole clean mass —
a factor of 3.1 here. That bias is the one the censored term removes.

### 2d. Held row

    (nothing)

A payment that never went out is never adjudicated. Held rows are dropped by
`used = adj | rel` before the time frame is even set.
`test_held_rows_contribute_nothing` asserts bit-equality of `coef_r` and `coef_p` between
a fit on all rows and a fit on the held rows physically deleted.

### 2e. Assembled objective

```python
dph = ph - ph0                                    # [new] p shrinks toward ph0, not 0
pen = 0.5 * float(np.sum(lam_r * th * th) + np.sum(lam_p * dph * dph))
f = (-ll + pen) / n_scale
g = np.concatenate([g_th + lam_r * th, g_ph + lam_p * dph]) / n_scale
```

with `n_scale = float(max(self.n_fit, 1))` and `n_fit = n_adjudicated + n_released`; see
§5 for why `p` is penalised toward the moment-matched start rather than toward zero.
The gradient is supplied analytically (`jac=True`); it is the gradient of the *scaled,
penalised negative* log-likelihood, which is why the per-term `g_th` accumulations above
carry the sign they do.

## 3. Parameterisation of `r` and `p`

    r_theta(x, u) = sigma( [1, zeta, psi, psi^2, b_1(u), ..., b_K(u)] · theta )
    p_phi(x, u)   = sigma( [1, zeta, psi]                           · phi )      # "static"
    p_phi(x, u)   = sigma( [1, zeta, psi, tau(u)]                   · phi )      # "slow"

- `dim_r = 4 + K`, `K = len(self._knots)` (0 when `n_knots <= 0`).
- `dim_p = 3` (`static`) or `4` (`slow`).
- Columns: `zeta` is the standardised log invoice amount; **`psi` [new] is the frozen
  scorer's output on the *logit* scale, centred and scaled over the rows this fit saw**;
  `u` is transaction time in days. Nothing else.

**[new] The score enters on the link scale, and this is load-bearing.** The release
channel is the auto-allow region, so its scores sit in the extreme left tail — the
module docstring reports a median released score around `3e-4` — while the adjudicated
rows come from the review band an order of magnitude higher. On the raw probability
scale both are numerically zero, the linear and quadratic score columns carry no
contrast at all, and `r` collapses to an intercept fitted on the adjudicated pool and
then applied wholesale to a release channel with a several-fold lower fraud rate. The
likelihood faithfully matches the observed `r·p·F`, so all of that error in `r` is
pushed into `p`. The transform:

```python
_SCORE_EPS = 1e-6       # the scorer can return a hard 0 or 1; keep the logit finite
_PSI_CLIP  = 8.0        # standardised score, in fitted-window sigmas

def score_logit(score):
    s = np.clip(np.asarray(score, dtype=float), _SCORE_EPS, 1.0 - _SCORE_EPS)
    return np.log(s / (1.0 - s))

def _set_score_frame(self, score):          # called on u[used] rows at fit time
    psi = score_logit(score)
    mu, sd = float(np.mean(psi)), float(np.std(psi))
    if not np.isfinite(mu):                    mu = 0.0
    if not np.isfinite(sd) or sd < 1e-6:       sd = 1.0   # no score information
    self._s_mu, self._s_sd = mu, sd

def _psi(self, score):
    z = (score_logit(score) - self._s_mu) / max(self._s_sd, 1e-9)
    return np.clip(z, -_PSI_CLIP, _PSI_CLIP)
```

`(_s_mu, _s_sd)` are fixed at `fit` and reused unchanged at `predict`, exactly like the
knots. Standardisation also keeps `psi^2` on the same scale as `psi`, so the optimiser
is not handed a badly conditioned problem. `test_module_never_reaches_for_
  the_unobservable` greps the module source and fails on `"topology"`, `"y_true"` or
  `"p_disclose"`.

**The asymmetry is prespecified, and it is what makes the two surfaces separable.** Both
read the same product on the release channel, so an unconstrained time term in each would
trade off freely against the other. So: `r` carries the full RBF time basis; `p` carries
either no time term (`static`) or a single normalised-time slope under a 50× ridge
(`slow`). Both configurations are fixed in advance, reported side by side, and neither is
selected on the outcome. `disclosure` outside `{"static", "slow"}` raises `ValueError`.

## 4. Time basis and knot placement

Fixed at `fit` from `u[used]`, reused unchanged at `predict` (`_set_time_frame`).

```
lo, hi = min(u), max(u)
if not finite(lo) or not finite(hi) or hi <= lo:  hi = lo + 1.0
self._u_lo, self._u_hi = lo, hi
qs    = np.linspace(0.0, 1.0, n_knots)          # n_knots = 4 by default
knots = np.quantile(u, qs)                      # includes the 0th and 100th pctile
if n_knots > 1 and min(diff(knots)) <= 0:       # degenerate u
    knots = np.linspace(lo, hi, n_knots)        # spread over the range instead
span   = max(hi - lo, 1e-9)
width  = span / max(n_knots - 1, 1)
```

Basis (Gaussian RBF, unnormalised):

```
z = (u[:, None] - knots[None, :]) / max(width, 1e-9)
b = exp(-0.5 * clip(z * z, 0.0, 200.0))
```

Normalised time, used only by the `slow` variant of `p`:

```
tau(u) = clip(2 * (u - u_lo) / span - 1.0, -2.0, 2.0)     # ~[-1, 1] in-window, capped
```

Empty `u` at fit time yields `knots = zeros(0)`, `width = 1.0`, `u_lo = 0.0`,
`u_hi = 1.0`; `_time_basis` then returns an `(n, 0)` array and `dim_r = 4`.

## 5. Regularisation

| coefficient block | penalty vector entry | default |
| --- | --- | --- |
| `r` intercept, `lam_r[0]` | `0.0` — intercepts are never shrunk | 0.0 |
| `r` in `zeta, score, score^2`, `lam_r[1:4]` | `ridge_r` | **1.0** |
| `r` RBF time columns, `lam_r[4:]` | `ridge_r_time` | **`4.0 * ridge_r` = 4.0** |
| `p` intercept, `lam_p[0]` **[new]** | `ridge_p` — **no longer zero** | **1.0** |
| `p` in `zeta, score`, `lam_p[1:3]` | `ridge_p` | **1.0** |
| `p` time slope, `lam_p[3]` (`slow` only) | `ridge_p_time` | **`50.0 * ridge_p` = 50.0** |

`ridge_r_time` and `ridge_p_time` default to `None` and are then derived from
`ridge_r` / `ridge_p` by the `4×` and `50×` multipliers in `__init__`; passing a float
overrides the derivation.

**[new] The two blocks are penalised toward different points.** `r` is shrunk toward
zero in the usual way; `p` is shrunk toward the moment-matched start `ph0`:

```python
dph = ph - ph0
pen = 0.5 * float(np.sum(lam_r * th * th) + np.sum(lam_p * dph * dph))
g   = np.concatenate([g_th + lam_r * th, g_ph + lam_p * dph]) / n_scale
```

The reason, from the source comment, is the flat direction of the joint likelihood: the
release channel only ever sees the product `r·p·F`, so raising the `p` intercept and
lowering the `r` intercept moves along a ridge the data barely curves. Left unpenalised
the optimiser walks that ridge and lands wherever a handful of surfaced rows push it —
in practice at the `0.97` ceiling. So the surfacing intercept is shrunk, at the same
strength as every other `p` coefficient, toward the level that reproduces the observed
aggregate surfacing rate. `r` keeps a free intercept (`lam_r[0] = 0.0`) because the
truthfully adjudicated rows do pin it. Shrinking toward `ph0` rather than toward zero
matters: `ph0` asserts only an aggregate level, nothing about how surfacing varies with
the covariates.

`_irls` uses the plain `0.5 * sum(ridge * w^2)` form (shrinkage toward zero); the
toward-`ph0` form appears only in `objective`.

## 6. Optimiser and tolerances

**Start point** (two stages, both deterministic):

1. `th0 = _irls(Xa, ya, lam_r)` — ridge-penalised Newton **with backtracking [new]** on
   the adjudicated rows alone. `w[0] = _logit(mean(y))`, then up to `max_iter = 40`
   iterations of `H = X' diag(s) X + diag(ridge) + 1e-9 I` with
   `s = clip(mu(1-mu), 1e-6, None)` and `step = solve(H, X'(mu - y) + ridge*w)`. Each
   step is **accepted only if it lowers the penalised objective**
   `_pen_nll(X, y, ridge, w) = -sum(y log sigma(eta) + (1-y) log sigma(-eta)) + 0.5 sum(ridge w^2)`:

   ```python
   t, cand, fc = 1.0, None, np.inf
   for _ in range(12):                                  # up to 12 halvings
       trial = np.clip(w - t * step, -_COEF_BOUND, _COEF_BOUND)
       ft = _pen_nll(X, y, ridge, trial)
       if np.isfinite(ft) and ft < f:
           cand, fc = trial, ft
           break
       t *= 0.5
   if cand is None:                                     # no downhill step exists
       break
   moved, gained = float(np.max(np.abs(cand - w))), f - fc
   w, f = cand, fc
   if moved < 1e-9 or gained < 1e-12:
       break
   ```

   The damping is not cosmetic. A window of adjudicated rows is a few dozen fraud events
   over eight columns and is routinely quasi-separated; undamped Newton overshoots on the
   first step, lands where the working weights `mu(1-mu)` have already underflowed to the
   `1e-6` clip, and returns coefficients pinned at `±_COEF_BOUND`. That is a wrong start,
   not a slow one, and it propagates — this fit is also what the surfacing intercept is
   moment-matched against and now shrunk toward. Termination: `LinAlgError`, no downhill
   step, `max|Δw| < 1e-9`, or objective gain `< 1e-12`. The final `np.clip` to
   `±_COEF_BOUND` is now applied per trial step rather than once at the end.
2. Moment-match the surfacing intercept so the first step already reproduces the
   observed surfaced rate:

   ```python
   r0_rel = _sigmoid(concat([Xs_r @ th0, Xn_r @ th0]))   if n_released else zeros(0)
   f_rel  = concat([Fs, Fn])
   denom  = sum(r0_rel * f_rel)
   p0     = clip(s_rel.sum() / denom, PD_FLOOR, 0.95) if denom > 1e-9 else 0.45
   ph0    = zeros(dim_p);  ph0[0] = _logit(p0)
   self._p_fallback = p0
   ```

   Note the upper bound here is **0.95**, not `PD_CEIL = 0.97`. **[new]** `ph0` is no
   longer only a start point — it is also the shrinkage target for the entire `p` block
   (§5).

**Solver:**

```python
minimize(objective, w0, jac=True, method="L-BFGS-B",
         bounds=[(-25.0, 25.0)] * (dim_r + dim_p),
         options={"maxiter": 300,          # self.max_iter
                  "maxfun": 6000,          # 20 * self.max_iter
                  "ftol": 1e-12,
                  "gtol": 1e-8})
```

`converged = bool(res.success)`; `n_iter = res.nit`; `nll = res.fun * n_scale`;
`status = "ok"` on success, else `str(res.message)`.

Caveat on `nll`: it is the **scaled objective un-scaled**, so it contains the ridge
penalty *and* `const_ll = sum log F(a)` over surfaced rows. It is a fit diagnostic, not a
comparable log-likelihood across different `n` or different ridge settings.

The fit is deterministic — no RNG anywhere in the module.
`test_fit_is_deterministic` asserts `np.array_equal` on both coefficient vectors,
`n_iter`, `converged`, and both prediction vectors across repeated fits.

## 7. Numerical guards, in full

| guard | value | where |
| --- | --- | --- |
| linear-predictor clip `_ETA_CLIP` | `±30.0` | `_sigmoid`, and explicit `np.clip` on `eta_a`, `eta_rs`, `eta_ps` |
| coefficient bound `_COEF_BOUND` | `±25.0` | L-BFGS-B `bounds`, and the final clip in `_irls` |
| log-argument floor `_EPS` | `1e-12` | `log(clip(Fs, _EPS, 1.0))`; `s = clip(Fn*rn*pn, _EPS, 1 - _EPS)`; `max(1 - s, _EPS)` in `ratio` |
| IRLS working weight floor | `1e-6` | `s = np.clip(mu * (1 - mu), 1e-6, None)` |
| IRLS Hessian jitter | `1e-9 * I` | `H = X'SX + diag(ridge) + 1e-9 I` |
| `_logit` argument clip | `[1e-6, 1 - 1e-6]` | `_logit` |
| **[new]** score clip before the logit `_SCORE_EPS` | `[1e-6, 1 - 1e-6]` | `score_logit` |
| **[new]** standardised-score clip `_PSI_CLIP` | `±8.0` sigmas | `_psi` |
| **[new]** degenerate score frame | non-finite `mu` → `0.0`; non-finite or `sd < 1e-6` → `sd = 1.0` | `_set_score_frame` |
| **[new]** score-scale divisor floor | `max(self._s_sd, 1e-9)` | `_psi` |
| **[new]** IRLS backtracking | up to 12 halvings; step taken only if `_pen_nll` strictly decreases | `_irls` |
| RBF exponent clip | `clip(z*z, 0.0, 200.0)` | `_time_basis` |
| `tau` clip | `[-2.0, 2.0]` | `_tau` |
| span floors | `max(hi - lo, 1e-9)`, `max(width, 1e-9)` | `_set_time_frame`, `_time_basis`, `_tau` |
| degenerate-`u` fallback | `hi = lo + 1.0`; collided knots → `linspace(lo, hi, K)` | `_set_time_frame` |
| non-finite objective | return `(1e12, zeros_like(w))` | `objective` |
| non-finite gradient | `np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)` | `objective` |
| `minimize` raises | catch `Exception` → `coef = (th0, ph0)`, `converged = False`, `status = "optimiser-error: <TypeName>"` | `fit` |
| non-finite solution vector | `coef = (th0, ph0)`, `converged = False`, `status = "non-finite-solution"` | `fit` |
| `delay_cdf` age | `clip(age, 0.0, None)`; result `clip(f, 0.0, 1.0)` | `delay_cdf` |
| `LinAlgError` in IRLS | `break`, keep current `w` | `_irls` |

A surfaced row at age 0 has probability zero under the delay law; the `_EPS` floor on
`log_Fs` is what stops one malformed row poisoning the whole objective. It only shifts
`const_ll`.

## 8. Clipping bounds on the outputs

| output | bound |
| --- | --- |
| `predict_pd` (fitted) | `np.clip(sigma(eta), PD_FLOOR, PD_CEIL)` = **`[0.05, 0.97]`** |
| `predict_pd` (unfitted, `coef_p is None`) | `clip(self._p_fallback, 0.05, 0.97)`, `_p_fallback` default `0.45` |
| `predict_r` (fitted) | **not clipped** beyond `sigma(clip(eta, ±30))`, i.e. `≈[9.36e-14, 1 - 9.36e-14]` |
| `predict_r` (unfitted, `coef_r is None`) | `self._r_fallback`, default `0.05`, overwritten with `clip(mean(y_obs[adj]), 1e-4, 1 - 1e-4)` whenever `n_adjudicated > 0` |

`test_predicted_surfacing_probability_stays_in_range` checks the `p_d` bounds hold on
adversarial inputs (`zeta = ±50`, `score = ±10`, `u = ±1e4`) and on a never-fitted model.
That `r` is unbounded from below is a deliberate asymmetry: `r` is the quantity being
estimated, `p_d` is a divisor downstream.

## 9. Insufficient-data behaviour

```python
if self.n_fit < self.min_fit or self.n_adjudicated < 8:
    self.coef_r = None
    self.coef_p = None
    self.converged = False
    self.status = "insufficient-data"
    return self
```

`min_fit` defaults to **80**; the adjudicated floor is **8**, hard-coded. The rationale
is identification, not stability: `r` is identified only through the truthfully
adjudicated rows, and without them the release channel sees a product and nothing else,
so the fit would be shrinkage noise.

Two things still happen *before* that return, and callers depend on them:
`_set_time_frame(u[used])` runs, and `_r_fallback` is set from `mean(y_obs[adj])` when
any adjudicated rows exist. So the declined model still returns bounded, data-informed
fallbacks from `predict_r` / `predict_pd` rather than raising.
`test_thin_data_declines_rather_than_guessing` (n = 40) asserts
`not converged and status == "insufficient-data"` with predictions still inside
`[0.05, 0.97]`.

The caller in `simulate.py` (lines 468–479) treats non-convergence as abstention:
`m_joint = _jm if _jm.converged else None`, falling back to the scalar `pd_fallback`.

## 10. API

```python
score_logit(score) -> np.ndarray            # [new] module-level, public
```
`log(s / (1-s))` with `s` clipped to `[1e-6, 1 - 1e-6]`. The link-scale score, before
centring and scaling.

```python
delay_cdf(age, shape=2.0, scale=30.0, cap=180.0) -> np.ndarray
```
`P(delay <= age)` for a `Gamma(shape, scale)` truncated at `cap` days. Uses
`scipy.special.gammainc`, the regularised lower incomplete gamma, so it is exact rather
than tabulated; truncation is a point mass at the cap, hence `f = 1` for `age >= cap`.
Module constants `DELAY_SHAPE = 2.0`, `DELAY_SCALE = 30.0`, `DELAY_CAP = 180.0`
(days) are restated locally so the module carries **no import edge into the simulator**.
`test_delay_cdf_matches_the_stated_arrival_law` checks monotonicity, the truncation, and
the shape-2 closed form `1 - e^{-x}(1 + x)` to `atol=1e-12`.

```python
JointTimeModel(disclosure="static", ridge_r=1.0, ridge_p=1.0, n_knots=4,
               max_iter=300, ridge_r_time=None, ridge_p_time=None, min_fit=80,
               delay_shape=DELAY_SHAPE, delay_scale=DELAY_SCALE, delay_cap=DELAY_CAP)
```
Raises `ValueError` for `disclosure` outside `{"static", "slow"}`. `n_knots` is coerced
by `max(n_knots, 0)`.

```python
.fit(zeta, score, u, kind, y_obs, age, surfaced) -> self
```
All seven arguments are 1-D and must share length `n = len(zeta)`; a mismatch raises
`ValueError(f"{name} has length {len(arr)}, expected {n}")`.

| arg | meaning |
| --- | --- |
| `zeta` | standardised log invoice amount |
| `score` | frozen scorer output |
| `u` | transaction time (days), the time basis is built from this |
| `kind` | `0` adjudicated, `1` released, `2` held (anything else excluded) |
| `y_obs` | truthful label; **only `y_obs[kind == 0]` is read** |
| `age` | days elapsed since the transaction, i.e. `t_now - u` |
| `surfaced` | bool; **only `surfaced & (kind == 1)` is read** |

```python
.predict_r(zeta, score, u)  -> np.ndarray
.predict_pd(zeta, score, u) -> np.ndarray
.coefficients()             -> dict          # arrays are copies; callers cannot mutate
```

`coefficients()` returns `disclosure`, `coef_r`, `coef_p`, `knots`, **`score_centre`**
and **`score_scale`** **[new]**, `n_fit`, `n_adjudicated`, `n_released`, `converged`,
`n_iter`, `status`, `nll`.

Public attributes after `fit`: `coef_r`, `coef_p`, `n_fit`, `n_adjudicated`,
`n_released`, `converged`, `n_iter`, `status`, `nll`.

## 11. Identification assumptions the likelihood needs

Ordered by how much work each is doing.

1. **Truthful adjudication.** Rows with `kind == 0` carry `y = Y` exactly. `r` is
   identified *only* through these rows; the release channel supplies a product and
   cannot separate the factors on its own.
2. **Overlap of the adjudicated support with the region of interest.** Where no
   adjudicated row lies, `r` is extrapolated by the logistic form and nothing in the data
   can test it. The test fixture deliberately makes holding score-driven so the release
   channel must extrapolate above the adjudicated range — that is the honest version of
   the problem, not a convenience.
3. **Correct functional form of both surfaces.** Given (1) and (2), separation of `r`
   from `p` on the release channel rests on the parametric forms in §3 *plus* the
   prespecified time asymmetry *plus*, since the latest revision, the shrinkage of the
   surfacing intercept toward the moment-matched aggregate rate (§5). That last one is a
   prior, not an identifying restriction: it picks a point on a direction the data barely
   curves rather than making that direction identified. Non-parametrically the release channel alone identifies
   only `r · p · F` (this is Proposition 2 in `paper/main.tex`).
4. **The delay law is known and parameter-free.** `F(a)` enters as a per-row constant. If
   `F` is misspecified, `p` absorbs the error one-for-one on both release terms.
5. **The delay is independent of `(Y, D)` and of `(zeta, score, u)`.** `delay_cdf` takes
   no covariates, so a covariate-dependent surfacing delay is outside the model.
6. **`age` is measured, and `surfaced` is the true indicator at the evaluation instant.**
   No residual censoring beyond what `F(a)` describes.
7. **Held rows are ignorable given the design columns.** Their exclusion is innocuous
   only if the hold decision is a function of quantities already conditioned on (here,
   the score). The likelihood is conditional on the decision taken.
8. **`p` is fully described by the design columns.** In the simulator the true
   `p_disclose` is driven by fraud topology, which the policy cannot observe — if it
   could it would already know the label. So `p-hat` is a marginal approximation over
   topology. This is a stated misspecification, not something the likelihood can enforce.

**What is *not* assumed, and this is the point of the module.** No independence between
the disclosure draw `D` and the fraud state `Y` is required or used anywhere. `p` is
defined *conditionally on fraud* — `P(surfaces | Y = 1, x, u)` — so the release-row
probability `r · p · F` is a plain factorisation of a joint event, not a product of
independent events. The ratio pipeline it replaces needs
`D ⟂ Y | x` to write `P(obs = 1 | x) = r(x) p_d(x)` and then divide; this likelihood does
not.

## 12. Empirical recovery

`test_recovers_the_generating_surfaces`, n = 30 000, data generated from known logistic
surfaces under exactly the observation process the likelihood assumes (`r` carrying a
seasonal term, `p_d` static and amount-driven, holding score-driven):

| configuration | MAE `p_d` | MAE `r` | L-BFGS-B iterations |
| --- | --- | --- | --- |
| `disclosure="static"` | 0.0219 | 0.0175 | 59 |
| `disclosure="slow"` | 0.0232 | 0.0174 | 68 |

Asserted thresholds are `MAE p_d < 0.10` and `MAE r < 0.05`; both configurations clear
them with roughly 4× and 3× margin.

## 13. Wiring status (as of this audit)

`JointTimeModel` is reachable from `simulate.py::run_policy` via the `joint_mode: str =
""` keyword (lines 248, 454, 469–474, 492–496). **It is off by default and nothing in the
experiment grid turns it on**: `src/t13/experiment.py::method_comparison` builds its
variants from `(method, oracle_pd, cohort_aligned, cohort_fallback)` tuples only, and
`joint_mode` never appears there. Consequently no published result in
`results/results.json` uses this estimator, and `paper/main.tex` does not mention a joint
likelihood or a censored contribution anywhere. The module is validated in isolation and
not yet exercised end to end.
