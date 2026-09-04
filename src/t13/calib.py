"""Calibration primitives.

Everything the decision layer needs is a monotone curve over a fixed score grid:
the trailing score CDF (for the budget projection) and the risk curve
R(tau) = E[y | s <= tau] (for the threshold search). Both are accumulated with
`np.bincount` over grid bins rather than by sorting, which keeps the per-update cost
linear in the window length. That matters: the update runs 320 times per pass and the
window holds tens of thousands of items.
"""
from __future__ import annotations

import numpy as np


def build_grid(train_scores: np.ndarray, n_coarse: int = 257, n_fine: int = 513,
               n_tail: int = 129) -> np.ndarray:
    """Quantile grid, refined where the decision thresholds actually live."""
    levels = np.concatenate([
        np.linspace(0.0, 0.90, n_coarse),
        np.linspace(0.90, 0.999, n_fine),
        np.linspace(0.999, 1.0, n_tail),
    ])
    g = np.quantile(train_scores, np.unique(levels))
    g = np.unique(np.concatenate([[0.0], g, [1.0]]))
    return np.ascontiguousarray(g, dtype=float)


def bin_of(grid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Smallest grid index g with scores <= grid[g]."""
    return np.clip(np.searchsorted(grid, scores, side="left"), 0, len(grid) - 1)


def accumulate(bins: np.ndarray, weights: np.ndarray, n_grid: int) -> np.ndarray:
    return np.cumsum(np.bincount(bins, weights=weights, minlength=n_grid))


def risk_curve(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """R(tau) = weighted mean of the response over every item scoring at or below tau.

    Both inputs are already cumulative over the grid, so this is a ratio and nothing
    more. No monotone projection is applied. An earlier version forced the curve
    non-decreasing with a running maximum, which turned out to be actively harmful:
    the low-score head of the grid holds a handful of items, one positive among ten of
    them reads as a risk of 0.1, and the running maximum then propagates that single
    accident across the whole curve. Support is checked instead, at the point of use.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def invert_cdf(cdf: np.ndarray, level: float) -> int:
    """Grid index whose trailing CDF first reaches `level`."""
    level = float(np.clip(level, 0.0, 1.0))
    return int(np.clip(np.searchsorted(cdf, level, side="left"), 0, len(cdf) - 1))


def crc_threshold(risk: np.ndarray, support: np.ndarray, target: float,
                  min_support: float = 200.0) -> int:
    """Largest grid index whose estimated risk sits at or below `target`.

    A grid point is only a candidate if the calibration window puts at least
    `min_support` units of weight at or below it. Certifying a risk of order 1% from a
    dozen observations is not conservative, it is arithmetic that happens to come out
    small. When nothing clears the target the tightest *supported* threshold comes
    back, and the caller's feasibility floor turns that into a raised flag rather than
    a silently violated constraint.
    """
    supported = support >= min_support
    ok = np.nonzero(supported & (risk <= target))[0]
    if len(ok):
        return int(ok[-1])
    sup = np.nonzero(supported)[0]
    return int(sup[0]) if len(sup) else 0


def decay_weights(age_days: np.ndarray, rho: float) -> np.ndarray:
    return np.power(rho, np.maximum(age_days, 0.0))


def bootstrap_ci(values, n_resamples: int = 10000, seed: int = 12345, alpha: float = 0.05):
    """Percentile bootstrap over the seed-level replicates."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (None, None, None)
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(v.mean()), float(lo), float(hi))
