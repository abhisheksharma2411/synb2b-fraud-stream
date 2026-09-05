"""Joint likelihood for the two nuisance functions behind the release channel.

The release channel never reports a label. It reports an *event*: by the time we
look, a dispute, chargeback or reconciliation exception either has or has not
surfaced. Writing q(x) = P(surfaced | x) and then recovering the surfacing
probability p(x) as q(x) / r(x) - two separate logistic fits, one divided by the
other - is unstable wherever r is small, and the two fits are not even drawn from
the same transaction-time cohort, so the ratio moves for reasons that have nothing
to do with the quantity being estimated.

This module fits both functions at once, by maximum likelihood, on every row that
carries information:

  adjudicated row (analyst review or exploration draw), truthful label y:
      y log r(x,u) + (1 - y) log(1 - r(x,u))
  released row of age a, signal has surfaced:
      log( r(x,u) p(x,u) F(a) )
  released row of age a, no signal yet:
      log( 1 - r(x,u) p(x,u) F(a) )
  held row:
      nothing - a payment that never went out is never adjudicated.

The third line is the point of the exercise. A release that closed clean is not a
truthful negative: it is a row for which the joint event {it was fraud, AND the
fraud left a trace, AND that trace arrived by now} failed, and the likelihood says
exactly that. A naive fit that reads such a row as y = 0 pushes r down by the whole
clean mass, which is precisely the bias the release channel injects.

F(a) is the delay CDF: Gamma(shape 2, scale 30 days) truncated at the 180-day
ledger close. It is an operational fact - the ledger closes on a stated day - not a
fitted quantity, so it enters as a known per-row constant. It is re-derived here in
closed form rather than imported, so this module stands alone.

Only columns observable at decision time are used: the standardised log invoice
amount, the frozen scorer's own output, and transaction time. Nothing that would
already reveal the label is available to either model.

The scorer's output enters on the *link* scale, standardised over the fitted rows,
not as a raw probability. That is not cosmetic. The release channel is the auto-allow
region, so its scores sit in the extreme left tail - on this generator the median
released score is around 3e-4 - while the adjudicated rows come from the review band
an order of magnitude higher up. On the raw probability scale both are numerically
zero, the linear and quadratic score columns carry no contrast at all, and r collapses
to an intercept fitted on the adjudicated pool and then applied wholesale to the
release channel, which has a several-fold lower fraud rate. The joint likelihood
faithfully matches the observed surfacing rate r p F, so every bit of that error in r
is pushed into p. Logit(score) is the natural linear predictor for a logistic model of
the same label the scorer was trained on, and it restores the contrast between the two
channels; the standardisation keeps the quadratic term on the same scale as the linear
one so the optimiser is not solving a badly conditioned problem.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammainc, log_expit

# The two arrival laws are stated operational parameters, restated here so that the
# module carries no import edge into the simulator.
DELAY_SHAPE = 2.0
DELAY_SCALE = 30.0      # days
DELAY_CAP = 180.0       # days; the ledger closes

PD_FLOOR = 0.05
PD_CEIL = 0.97

_EPS = 1e-12
_ETA_CLIP = 30.0        # |eta| beyond this is already saturated to float precision
_COEF_BOUND = 25.0
_SCORE_EPS = 1e-6       # the scorer can return a hard 0 or 1; keep the logit finite
_PSI_CLIP = 8.0         # standardised score, in fitted-window sigmas


def score_logit(score) -> np.ndarray:
    """The scorer's output on the link scale, which is where it is linear in the label."""
    s = np.clip(np.asarray(score, dtype=float), _SCORE_EPS, 1.0 - _SCORE_EPS)
    return np.log(s / (1.0 - s))


def delay_cdf(age, shape: float = DELAY_SHAPE, scale: float = DELAY_SCALE,
              cap: Optional[float] = DELAY_CAP) -> np.ndarray:
    """P(delay <= age) for a Gamma(shape, scale) delay truncated at `cap` days.

    `gammainc` is the regularised lower incomplete gamma, which *is* the Gamma CDF,
    so this is exact rather than tabulated. Truncation at the cap is a point mass at
    the cap, so the CDF is 1 from there on.
    """
    a = np.clip(np.asarray(age, dtype=float), 0.0, None)
    f = gammainc(shape, a / scale)
    if cap is not None:
        f = np.where(a >= cap, 1.0, f)
    return np.clip(f, 0.0, 1.0)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return expit(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _pen_nll(X: np.ndarray, y: np.ndarray, ridge: np.ndarray,
             w: np.ndarray) -> float:
    eta = np.clip(X @ w, -_ETA_CLIP, _ETA_CLIP)
    return float(-np.sum(y * log_expit(eta) + (1.0 - y) * log_expit(-eta))
                 + 0.5 * np.sum(ridge * w * w))


def _irls(X: np.ndarray, y: np.ndarray, ridge: np.ndarray,
          max_iter: int = 40) -> np.ndarray:
    """Ridge-penalised Newton with backtracking. Starts the joint optimiser.

    The damping is not a nicety. A window of adjudicated rows is a few dozen fraud
    events spread over eight columns, which is routinely quasi-separated, and undamped
    Newton on a quasi-separated design overshoots on the first step, lands where the
    working weights mu(1-mu) have already underflowed to the clip, and never comes
    back - it returns coefficients pinned at the coefficient bound. That is not a
    slow start, it is a wrong one, and it does not stay contained: this fit is also
    what the surfacing intercept is moment-matched against and shrunk toward. So a
    step is taken only when it actually lowers the penalised objective.
    """
    w = np.zeros(X.shape[1])
    if len(y) == 0:
        return w
    w[0] = _logit(float(np.mean(y)))
    f = _pen_nll(X, y, ridge, w)
    for _ in range(max_iter):
        mu = _sigmoid(X @ w)
        g = X.T @ (mu - y) + ridge * w
        s = np.clip(mu * (1.0 - mu), 1e-6, None)
        H = X.T @ (X * s[:, None]) + np.diag(ridge) + 1e-9 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t, cand, fc = 1.0, None, np.inf
        for _ in range(12):
            trial = np.clip(w - t * step, -_COEF_BOUND, _COEF_BOUND)
            ft = _pen_nll(X, y, ridge, trial)
            if np.isfinite(ft) and ft < f:
                cand, fc = trial, ft
                break
            t *= 0.5
        if cand is None:                     # no downhill step exists: already there
            break
        moved, gained = float(np.max(np.abs(cand - w))), f - fc
        w, f = cand, fc
        if moved < 1e-9 or gained < 1e-12:
            break
    return w


class JointTimeModel:
    """Time-aware joint ML fit of r(x,u) and p(x,u).

    Two logistic surfaces, one optimisation:

        r_theta(x,u) = sigma([1, zeta, psi, psi^2, b(u)] . theta)
        p_phi(x,u)   = sigma([1, zeta, psi, (tau(u))] . phi)

    psi is the frozen scorer's output on the logit scale, centred and scaled over the
    rows this fit actually saw. See the module docstring: on the raw probability scale
    the release channel and the review band are indistinguishable to float64, and r
    degenerates to the adjudicated pool's intercept.

    b(u) is a handful of RBF bumps on transaction time, knots at quantiles of the
    fitted u, and it sits only in r. The surfacing surface is deliberately poorer:
    under `disclosure="static"` it has no time term at all, and under
    `disclosure="slow"` it gets a single normalised-time slope carrying a much
    heavier penalty. That asymmetry is prespecified, not tuned. Both surfaces read
    the same product on the release channel, so an unconstrained time term in each
    would trade off against the other with nothing in the data to separate them; the
    truthfully adjudicated rows are what pin r down, and p is then whatever the
    release channel needs on top of it.

    Both configurations are fixed in advance and reported side by side. Neither is
    selected on the outcome.
    """

    def __init__(self, disclosure: str = "static", ridge_r: float = 1.0,
                 ridge_p: float = 1.0, n_knots: int = 4, max_iter: int = 300,
                 ridge_r_time: Optional[float] = None,
                 ridge_p_time: Optional[float] = None, min_fit: int = 80,
                 delay_shape: float = DELAY_SHAPE, delay_scale: float = DELAY_SCALE,
                 delay_cap: Optional[float] = DELAY_CAP):
        if disclosure not in ("static", "slow"):
            raise ValueError("disclosure must be 'static' or 'slow', got "
                             f"{disclosure!r}")
        self.disclosure = disclosure
        self.ridge_r = float(ridge_r)
        self.ridge_p = float(ridge_p)
        self.ridge_r_time = float(4.0 * ridge_r) if ridge_r_time is None \
            else float(ridge_r_time)
        self.ridge_p_time = float(50.0 * ridge_p) if ridge_p_time is None \
            else float(ridge_p_time)
        self.n_knots = int(max(n_knots, 0))
        self.max_iter = int(max_iter)
        self.min_fit = int(min_fit)
        self.delay_shape = float(delay_shape)
        self.delay_scale = float(delay_scale)
        self.delay_cap = None if delay_cap is None else float(delay_cap)

        self.coef_r: Optional[np.ndarray] = None
        self.coef_p: Optional[np.ndarray] = None
        self.n_fit = 0
        self.n_adjudicated = 0
        self.n_released = 0
        self.converged = False
        self.n_iter = 0
        self.status = "unfitted"
        self.nll = float("nan")
        self._knots: Optional[np.ndarray] = None
        self._width = 1.0
        self._u_lo = 0.0
        self._u_hi = 1.0
        self._s_mu = 0.0
        self._s_sd = 1.0
        self._r_fallback = 0.05
        self._p_fallback = 0.45

    # -- design ------------------------------------------------------------
    def _set_score_frame(self, score: np.ndarray) -> None:
        """Centre and scale the link-scale score. Fixed at fit, reused at predict."""
        psi = score_logit(score)
        if len(psi) == 0:
            self._s_mu, self._s_sd = 0.0, 1.0
            return
        mu = float(np.mean(psi))
        sd = float(np.std(psi))
        if not np.isfinite(mu):
            mu = 0.0
        # a window in which every row carries the same score has no score information;
        # fall back to a unit scale rather than dividing by ~0 and inventing some
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        self._s_mu, self._s_sd = mu, sd

    def _psi(self, score: np.ndarray) -> np.ndarray:
        z = (score_logit(score) - self._s_mu) / max(self._s_sd, 1e-9)
        return np.clip(z, -_PSI_CLIP, _PSI_CLIP)

    def _set_time_frame(self, u: np.ndarray) -> None:
        """Knots and the normalised-time frame. Fixed at fit, reused at predict."""
        if len(u) == 0:
            self._knots = np.zeros(0)
            self._width, self._u_lo, self._u_hi = 1.0, 0.0, 1.0
            return
        lo, hi = float(np.min(u)), float(np.max(u))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        self._u_lo, self._u_hi = lo, hi
        if self.n_knots <= 0:
            self._knots = np.zeros(0)
            self._width = 1.0
            return
        qs = np.linspace(0.0, 1.0, self.n_knots)
        knots = np.quantile(u, qs)
        # quantiles can collide on a degenerate u; spread them over the range instead
        if self.n_knots > 1 and np.min(np.diff(knots)) <= 0:
            knots = np.linspace(lo, hi, self.n_knots)
        self._knots = np.ascontiguousarray(knots, dtype=float)
        span = max(hi - lo, 1e-9)
        self._width = span / max(self.n_knots - 1, 1)

    def _time_basis(self, u: np.ndarray) -> np.ndarray:
        if self._knots is None or len(self._knots) == 0:
            return np.zeros((len(u), 0))
        z = (u[:, None] - self._knots[None, :]) / max(self._width, 1e-9)
        return np.exp(-0.5 * np.clip(z * z, 0.0, 200.0))

    def _tau(self, u: np.ndarray) -> np.ndarray:
        """Transaction time mapped to about [-1, 1] over the fitted window."""
        span = max(self._u_hi - self._u_lo, 1e-9)
        return np.clip(2.0 * (u - self._u_lo) / span - 1.0, -2.0, 2.0)

    def _design_r(self, zeta: np.ndarray, score: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
        psi = self._psi(score)
        return np.column_stack([np.ones(len(zeta)), zeta, psi, psi * psi,
                                self._time_basis(u)])

    def _design_p(self, zeta: np.ndarray, score: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(zeta)), zeta, self._psi(score)]
        if self.disclosure == "slow":
            cols.append(self._tau(u))
        return np.column_stack(cols)

    def _dim_r(self) -> int:
        return 4 + (0 if self._knots is None else len(self._knots))

    def _dim_p(self) -> int:
        return 4 if self.disclosure == "slow" else 3

    def _penalties(self) -> tuple:
        lam_r = np.full(self._dim_r(), self.ridge_r)
        lam_r[0] = 0.0                       # intercepts are never shrunk
        if self._dim_r() > 4:
            lam_r[4:] = self.ridge_r_time
        lam_p = np.full(self._dim_p(), self.ridge_p)
        # The surfacing intercept is the one coefficient that is *not* left free. It
        # is the flat direction of the joint likelihood: the release channel only ever
        # sees the product r p F, so raising the p intercept and lowering the r
        # intercept moves along a ridge the data barely curves. Left unpenalised the
        # optimiser walks that ridge and lands wherever a handful of surfaced rows
        # happen to push it - in practice at the 0.97 ceiling. It is therefore shrunk,
        # at the same strength as every other p coefficient, toward the level that
        # reproduces the observed aggregate surfacing rate (see `fit`). r keeps a free
        # intercept because the truthfully adjudicated rows do pin it.
        if self.disclosure == "slow":
            lam_p[3] = self.ridge_p_time
        return lam_r, lam_p

    # -- fit ---------------------------------------------------------------
    def fit(self, zeta, score, u, kind, y_obs, age, surfaced) -> "JointTimeModel":
        zeta = np.asarray(zeta, dtype=float)
        score = np.asarray(score, dtype=float)
        u = np.asarray(u, dtype=float)
        kind = np.asarray(kind).astype(int)
        y_obs = np.asarray(y_obs, dtype=float)
        age = np.asarray(age, dtype=float)
        surfaced = np.asarray(surfaced).astype(bool)

        n = len(zeta)
        for name, arr in (("score", score), ("u", u), ("kind", kind),
                          ("y_obs", y_obs), ("age", age), ("surfaced", surfaced)):
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")

        adj = kind == 0
        rel = kind == 1                      # kind == 2 is held: no contribution
        self.n_adjudicated = int(adj.sum())
        self.n_released = int(rel.sum())
        self.n_fit = self.n_adjudicated + self.n_released

        used = adj | rel
        self._set_time_frame(u[used])
        self._set_score_frame(score[used])
        if self.n_adjudicated:
            self._r_fallback = float(np.clip(np.mean(y_obs[adj]), 1e-4, 1.0 - 1e-4))

        if self.n_fit < self.min_fit or self.n_adjudicated < 8:
            # r is identified only through the truthfully adjudicated rows. Without
            # them the release channel sees a product and nothing else, and the fit
            # would be shrinkage noise. Say so and let the caller fall back.
            self.coef_r = None
            self.coef_p = None
            self.converged = False
            self.status = "insufficient-data"
            return self

        Xa = self._design_r(zeta[adj], score[adj], u[adj])
        ya = np.clip(y_obs[adj], 0.0, 1.0)

        s_rel = surfaced & rel
        n_rel = rel & ~surfaced
        Xs_r = self._design_r(zeta[s_rel], score[s_rel], u[s_rel])
        Xs_p = self._design_p(zeta[s_rel], score[s_rel], u[s_rel])
        Xn_r = self._design_r(zeta[n_rel], score[n_rel], u[n_rel])
        Xn_p = self._design_p(zeta[n_rel], score[n_rel], u[n_rel])
        Fs = delay_cdf(age[s_rel], self.delay_shape, self.delay_scale, self.delay_cap)
        Fn = delay_cdf(age[n_rel], self.delay_shape, self.delay_scale, self.delay_cap)
        # a surfaced row at age 0 has probability zero under the delay law; guard the
        # log rather than let a malformed row poison the whole objective
        log_Fs = np.log(np.clip(Fs, _EPS, 1.0))
        const_ll = float(np.sum(log_Fs))

        dr, dp = self._dim_r(), self._dim_p()
        lam_r, lam_p = self._penalties()
        n_scale = float(max(self.n_fit, 1))

        # start r from the truthful rows alone, then moment-match the surfacing
        # intercept so the first step already reproduces the observed surfaced rate
        th0 = _irls(Xa, ya, lam_r)
        r0_rel = _sigmoid(np.concatenate([Xs_r @ th0, Xn_r @ th0])) if self.n_released \
            else np.zeros(0)
        f_rel = np.concatenate([Fs, Fn])
        denom = float(np.sum(r0_rel * f_rel))
        p0 = float(np.clip(s_rel.sum() / denom, PD_FLOOR, 0.95)) if denom > 1e-9 else 0.45
        ph0 = np.zeros(dp)
        ph0[0] = _logit(p0)
        self._p_fallback = p0
        w0 = np.concatenate([th0, ph0])

        def objective(w):
            th, ph = w[:dr], w[dr:]
            ll = const_ll
            g_th = np.zeros(dr)
            g_ph = np.zeros(dp)

            eta_a = np.clip(Xa @ th, -_ETA_CLIP, _ETA_CLIP)
            ll += float(np.sum(ya * log_expit(eta_a) + (1.0 - ya) * log_expit(-eta_a)))
            g_th += Xa.T @ (expit(eta_a) - ya)

            if len(Fs):
                eta_rs = np.clip(Xs_r @ th, -_ETA_CLIP, _ETA_CLIP)
                eta_ps = np.clip(Xs_p @ ph, -_ETA_CLIP, _ETA_CLIP)
                ll += float(np.sum(log_expit(eta_rs) + log_expit(eta_ps)))
                g_th += Xs_r.T @ (expit(eta_rs) - 1.0)
                g_ph += Xs_p.T @ (expit(eta_ps) - 1.0)

            if len(Fn):
                rn = _sigmoid(Xn_r @ th)
                pn = _sigmoid(Xn_p @ ph)
                s = np.clip(Fn * rn * pn, _EPS, 1.0 - _EPS)
                ll += float(np.sum(np.log1p(-s)))
                # d log(1 - s)/d eta_r = -s (1 - r) / (1 - s), and likewise for p
                ratio = s / np.maximum(1.0 - s, _EPS)
                g_th += Xn_r.T @ (ratio * (1.0 - rn))
                g_ph += Xn_p.T @ (ratio * (1.0 - pn))

            # p is penalised toward the moment-matched start ph0, not toward zero:
            # ph0 is only the aggregate surfacing level implied by the truthful rows,
            # so this shrinks the flat direction without asserting anything about how
            # the surfacing probability varies with the covariates.
            dph = ph - ph0
            pen = 0.5 * float(np.sum(lam_r * th * th) + np.sum(lam_p * dph * dph))
            f = (-ll + pen) / n_scale
            g = np.concatenate([g_th + lam_r * th, g_ph + lam_p * dph]) / n_scale
            if not np.isfinite(f):
                return 1e12, np.zeros_like(w)
            return f, np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        bounds = [(-_COEF_BOUND, _COEF_BOUND)] * (dr + dp)
        try:
            res = minimize(objective, w0, jac=True, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": self.max_iter,
                                    "maxfun": 20 * self.max_iter,
                                    "ftol": 1e-12, "gtol": 1e-8})
        except Exception as exc:                      # pragma: no cover - defensive
            self.coef_r, self.coef_p = th0, ph0
            self.converged = False
            self.status = f"optimiser-error: {type(exc).__name__}"
            return self

        w = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(w)):
            self.coef_r, self.coef_p = th0, ph0
            self.converged = False
            self.status = "non-finite-solution"
            return self

        self.coef_r = w[:dr]
        self.coef_p = w[dr:]
        self.n_iter = int(getattr(res, "nit", 0))
        self.converged = bool(res.success)
        self.nll = float(res.fun) * n_scale
        self.status = str(getattr(res, "message", "")) if not res.success else "ok"
        return self

    # -- predict -----------------------------------------------------------
    def predict_r(self, zeta, score, u) -> np.ndarray:
        zeta = np.asarray(zeta, dtype=float)
        score = np.asarray(score, dtype=float)
        u = np.asarray(u, dtype=float)
        if self.coef_r is None:
            return np.full(len(zeta), self._r_fallback)
        return _sigmoid(self._design_r(zeta, score, u) @ self.coef_r)

    def predict_pd(self, zeta, score, u) -> np.ndarray:
        zeta = np.asarray(zeta, dtype=float)
        score = np.asarray(score, dtype=float)
        u = np.asarray(u, dtype=float)
        if self.coef_p is None:
            return np.full(len(zeta), float(np.clip(self._p_fallback,
                                                    PD_FLOOR, PD_CEIL)))
        eta = self._design_p(zeta, score, u) @ self.coef_p
        return np.clip(_sigmoid(eta), PD_FLOOR, PD_CEIL)

    def coefficients(self) -> dict:
        """Flat diagnostic view; the arrays are copies, so callers cannot mutate."""
        return {
            "disclosure": self.disclosure,
            "coef_r": None if self.coef_r is None else self.coef_r.copy(),
            "coef_p": None if self.coef_p is None else self.coef_p.copy(),
            "knots": None if self._knots is None else self._knots.copy(),
            "score_centre": self._s_mu,
            "score_scale": self._s_sd,
            "n_fit": self.n_fit,
            "n_adjudicated": self.n_adjudicated,
            "n_released": self.n_released,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "status": self.status,
            "nll": self.nll,
        }
