"""Layer B: endogenous label arrival.

The policy chooses which labels it will ever see. Three channels, and they are not
symmetric:

  review  - an analyst adjudicates within hours. The label is correct and it is fast.
  allow   - the ledger closes at 180 days. If a dispute, chargeback or reconciliation
            exception surfaced first, the item is booked as fraud at the moment it
            surfaced. If nothing surfaced, the item is booked *clean* at day 180 -
            whether or not it actually was. That second case is the censoring.
  block   - no label. Ever. Nobody adjudicates a payment that never went out.

All randomness is drawn once per row, before any policy runs, so every method faces
the same potential outcomes and differences between methods are not draw noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


def disclosure_probability(stream: str, topo: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    n = len(zeta)
    p = np.zeros(n, dtype=float)
    if stream == "synb2b":
        base = np.full(n, 0.42, dtype=float)
        for k, v in C.P_DISCLOSE_BASE.items():
            base[topo == k] = v
    else:
        base = np.full(n, C.P_DISCLOSE_ULB_INTERCEPT, dtype=float)
    logit = np.log(base / (1.0 - base))
    coef = C.P_DISCLOSE_AMOUNT_COEF if stream == "synb2b" else C.P_DISCLOSE_ULB_COEF
    logit = logit + coef * zeta
    p = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(p, C.P_DISCLOSE_FLOOR, C.P_DISCLOSE_CEIL)


@dataclass
class ArrivalDraws:
    p_disclose: np.ndarray
    u_disclose: np.ndarray
    delay_review: np.ndarray
    delay_disclose: np.ndarray
    u_explore: np.ndarray


def draw_arrivals(stream: str, topo: np.ndarray, zeta: np.ndarray, seed: int) -> ArrivalDraws:
    rng = np.random.default_rng(500_000_017 + 104_729 * seed)
    n = len(zeta)
    p = disclosure_probability(stream, topo, zeta)
    return ArrivalDraws(
        p_disclose=p,
        u_disclose=rng.random(n),
        delay_review=rng.gamma(C.REVIEW_DELAY_SHAPE, C.REVIEW_DELAY_SCALE, size=n),
        delay_disclose=np.minimum(
            rng.gamma(C.DISCLOSE_DELAY_SHAPE, C.DISCLOSE_DELAY_SCALE, size=n),
            C.DISCLOSE_DELAY_CAP,
        ),
        u_explore=rng.random(n),
    )


class DisclosureModel:
    """Estimator for p_disclose on the auto-allow path.

    The true disclosure probability is a function of the fraud topology, which the
    policy cannot observe - if it could observe the topology it would already know the
    label. So the model is fitted on what is actually visible at decision time: the
    standardised invoice amount and the frozen scorer's own output. It can only be
    fitted on allow-path rows whose 180-day ledger window has fully closed, which
    makes the correction itself lagged. That is a real cost and it is reported.

    Plain Newton-step logistic regression on three columns; no external dependency,
    and deterministic given its inputs.
    """

    def __init__(self, ridge: float = 1.0, max_iter: int = 25):
        self.ridge = ridge
        self.max_iter = max_iter
        self.coef = None
        self.n_fit = 0

    @staticmethod
    def _design(zeta: np.ndarray, score: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(zeta)), zeta, score, score * score])

    def fit(self, zeta: np.ndarray, score: np.ndarray, disclosed: np.ndarray) -> "DisclosureModel":
        X = self._design(zeta, score)
        y = disclosed.astype(float)
        w = np.zeros(X.shape[1])
        w[0] = np.log(max(y.mean(), 1e-3) / max(1.0 - y.mean(), 1e-3))
        for _ in range(self.max_iter):
            eta = X @ w
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            g = X.T @ (mu - y) + self.ridge * w
            s = np.clip(mu * (1.0 - mu), 1e-6, None)
            H = X.T @ (X * s[:, None]) + self.ridge * np.eye(X.shape[1])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            w = w - step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.coef = w
        self.n_fit = len(y)
        return self

    def predict(self, zeta: np.ndarray, score: np.ndarray) -> np.ndarray:
        if self.coef is None:
            return np.full(len(zeta), 0.45)
        eta = self._design(zeta, score) @ self.coef
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), 0.05, 0.97)
