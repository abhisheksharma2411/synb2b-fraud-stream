"""Layer A of SynB2B-Fraud-Stream: three drift events with different mechanisms.

The published SynB2B-Fraud generator places fraud as explicit topology firings, so
there is no conditional law to perturb out of the box. This module supplies one, as a
second fraud channel layered on top of the benchmark rather than in place of it:

    y_i = b_i  OR  Bernoulli(p_i),
    logit p_i = a0 + gamma * zeta_i + beta(t) * nu_i + lambda(t)

`b_i` is the generator's own topology label, `nu_i` a standardised supplier-novelty
score, `zeta_i` a standardised log invoice amount. Nothing the generator marked as
fraud is ever un-marked, so SynB2B-Fraud's published ground truth survives intact and
the extension is additive. Outside the declared windows a0 is solved so the second
channel adds at most 5% on top of the local base rate - real label noise, not a
rewrite. a0, beta and lambda are solved per stream by tools/calibrate_drift.py.

Event 1 is covariate: the invoice amount law moves, P(y | x) does not.
Event 2 is concept: beta flips sign, so newly generated fraud migrates off young
suppliers and onto established relationships - which is where a scorer that learned
"unfamiliar counterparty means risk" stops being able to see it.
Event 3 is prior: lambda rises and decays, lifting the base rate at fixed shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class DriftEvent:
    kind: str
    index: int
    day: float
    magnitude: float
    duration_days: float
    description: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "index": int(self.index),
            "day": round(float(self.day), 4),
            "magnitude": round(float(self.magnitude), 6),
            "duration_days": round(float(self.duration_days), 4),
            "description": self.description,
        }


def plan_events(days: np.ndarray, n_train: int, severity: float = 1.0,
                stream: str = "synb2b") -> List[DriftEvent]:
    n = len(days)
    n_eval = n - n_train
    idx = [n_train + int(round(f * n_eval)) for f in C.DRIFT_FRACTIONS]
    return [
        DriftEvent(
            "covariate", idx[0], float(days[idx[0]]),
            C.COVARIATE_LOGAMT_SHIFT * severity, float("inf"),
            "persistent upward shift of the log invoice amount law; P(y|x) unchanged",
        ),
        DriftEvent(
            "concept", idx[1], float(days[idx[1]]),
            C.lgp_params(stream)[1] * severity, C.CONCEPT_DURATION_DAYS,
            "sign flip of the supplier-novelty coefficient in the conditional law",
        ),
        DriftEvent(
            "prior", idx[2], float(days[idx[2]]),
            C.lgp_params(stream)[2] * severity,
            C.PRIOR_RAMP_DAYS + C.PRIOR_TAIL_DAYS,
            "base-rate surge over a four-day ramp with a nine-day half-life decay",
        ),
    ]


def apply_covariate_shift(raw: pd.DataFrame, ev: DriftEvent) -> pd.DataFrame:
    """Move the amount law upward in log space, ramped in over three days."""
    out = raw.copy()
    day = out["day"].to_numpy(float)
    ramp = np.clip((day - ev.day) / C.COVARIATE_RAMP_DAYS, 0.0, 1.0)
    ramp[day < ev.day] = 0.0
    out["amount"] = out["amount"].to_numpy(float) * np.exp(ev.magnitude * ramp)
    return out


def beta_path(day: np.ndarray, ev: DriftEvent) -> np.ndarray:
    """ev.magnitude already carries the severity multiplier applied in plan_events."""
    b = np.full(len(day), C.LGP_BETA_BASE, dtype=float)
    inside = (day >= ev.day) & (day < ev.day + ev.duration_days)
    b[inside] = ev.magnitude
    return b


def lambda_path(day: np.ndarray, ev: DriftEvent) -> np.ndarray:
    lam = np.zeros(len(day), dtype=float)
    d = day - ev.day
    up = (d >= 0) & (d < C.PRIOR_RAMP_DAYS)
    lam[up] = ev.magnitude * (d[up] / C.PRIOR_RAMP_DAYS)
    down = (d >= C.PRIOR_RAMP_DAYS) & (d < C.PRIOR_RAMP_DAYS + C.PRIOR_TAIL_DAYS)
    lam[down] = ev.magnitude * np.exp(
        -np.log(2.0) * (d[down] - C.PRIOR_RAMP_DAYS) / C.PRIOR_HALFLIFE_DAYS
    )
    return lam


def _zscore(v: np.ndarray, n_ref: int, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    m = float(np.mean(v[:n_ref]))
    s = float(np.std(v[:n_ref])) or 1.0
    return np.clip((v - m) / s, lo, hi)


class _Fenwick:
    __slots__ = ("n", "t")

    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Count of entries in bins [0, i]."""
        i += 1
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s


def rolling_rank(v: np.ndarray, window: int = 5000, n_bins: int = 256) -> np.ndarray:
    """Causal rank of v[i] among the previous `window` values, mapped to [-1, 1].

    Absolute supplier age is not stationary in SynB2B-Fraud: buyers hold fixed
    supplier rosters, so almost every relationship is established by the second half
    of the stream and any fixed threshold on age degenerates. Rank against what the
    book is actually transacting with right now is stationary by construction, and it
    is also the quantity an analyst reasons about - "new *for us, this quarter*".
    """
    n = len(v)
    lo, hi = float(np.min(v)), float(np.max(v))
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    b = np.searchsorted(edges, v, side="right").astype(int)
    out = np.zeros(n, dtype=float)
    ft = _Fenwick(n_bins)
    count = 0
    for i in range(n):
        if count > 0:
            below = ft.prefix(b[i] - 1) if b[i] > 0 else 0
            out[i] = 2.0 * (below / count) - 1.0
        ft.add(b[i], 1)
        count += 1
        if i >= window:
            ft.add(b[i - window], -1)
            count -= 1
    return out


def novelty_score(stream: str, feats: pd.DataFrame, n_ref: int) -> np.ndarray:
    """Relative novelty in [-1, 1]. Higher means "newer than what we usually see"."""
    if stream == "synb2b":
        base = -np.log1p(feats["supplier_age_days"].to_numpy(float))
    else:
        # ULB is anonymised; V14 is the component most strongly associated with the
        # positive class in this portfolio, so -V14 stands in for the "unfamiliar"
        # direction. See DECISIONS.md D-04.
        base = -feats["V14"].to_numpy(float)
    return rolling_rank(base)


def apply_drift(
    stream: str,
    raw: pd.DataFrame,
    build_features,
    n_train: int,
    seed: int,
    severity: float = 1.0,
):
    """Returns (raw_shifted, features, labels, events, diagnostics)."""
    days = raw["day"].to_numpy(float)
    a0 = C.lgp_params(stream)[0]
    events = plan_events(days, n_train, severity=severity, stream=stream)
    ev_cov, ev_con, ev_pri = events

    shifted = apply_covariate_shift(raw, ev_cov) if stream == "synb2b" else raw.copy()
    if stream == "ulb":
        # the same multiplicative move, applied to the only interpretable ULB column
        day = shifted["day"].to_numpy(float)
        ramp = np.clip((day - ev_cov.day) / C.COVARIATE_RAMP_DAYS, 0.0, 1.0)
        ramp[day < ev_cov.day] = 0.0
        shifted["amount"] = shifted["amount"].to_numpy(float) * np.exp(ev_cov.magnitude * ramp)

    feats = build_features(shifted)
    nu = novelty_score(stream, feats, n_train)
    zeta = _zscore(np.log1p(shifted["amount"].to_numpy(float)), n_train)

    base = shifted["base_label"].to_numpy(float)
    beta = beta_path(days, ev_con)
    lam = lambda_path(days, ev_pri)

    logit = a0 + C.LGP_GAMMA_AMT * zeta + beta * nu + lam
    p = 1.0 / (1.0 + np.exp(-logit))
    rng = np.random.default_rng(1_000_003 + 7919 * seed)
    extra = (rng.random(len(p)) < p) & (base == 0)
    y = np.maximum(base, extra.astype(float)).astype(np.int8)

    topo = shifted["topology"].to_numpy(object).copy()
    topo[extra] = "D_drift_induced"

    ev_slice = slice(n_train, len(y))
    d_ev = days[n_train:]
    win_con = (d_ev >= ev_con.day) & (d_ev < ev_con.day + ev_con.duration_days)
    win_pri = (d_ev >= ev_pri.day) & (d_ev < ev_pri.day + ev_pri.duration_days)
    quiet = ~(win_con | win_pri)
    ex_ev, base_ev = extra[ev_slice], base[ev_slice]
    diag = {
        "n_base_fraud": int(base.sum()),
        "n_realised_fraud": int(y.sum()),
        "n_drift_induced": int(extra.sum()),
        "agreement": float(np.mean(y == base)),
        "quiet_rate": float(np.mean(np.maximum(base_ev, ex_ev)[quiet])),
        "concept_rate": float(np.mean(np.maximum(base_ev, ex_ev)[win_con])),
        "prior_rate": float(np.mean(np.maximum(base_ev, ex_ev)[win_pri])),
        "n_quiet_induced": int(ex_ev[quiet].sum()),
        "n_concept_induced": int(ex_ev[win_con].sum()),
        "n_prior_induced": int(ex_ev[win_pri].sum()),
        "concept_established_share": float(
            np.mean(nu[ev_slice][win_con & ex_ev] < 0.0)
        ) if int(ex_ev[win_con].sum()) else None,
    }
    return shifted, feats, y, topo, events, diag, {"nu": nu, "zeta": zeta}
