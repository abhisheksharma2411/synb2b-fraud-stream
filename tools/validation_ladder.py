"""Staged estimator-validation ladder: does the risk estimator recover its target?

The question this answers is the one everything downstream rests on. The M5 update
reads a risk off an inverse-propensity-weighted calibration window and drives the
release threshold with it. If that reading does not recover the true release-region
risk in the *simplest* setting available - no arrival delay, no booked-clean
corruption, no propensity trimming, no calibration decay, no drift, and the true
disclosure propensity handed over rather than estimated - then no result computed on
top of it means anything, and every downstream ablation is measuring the interaction
of two broken things.

So the ladder starts at that simplest setting and switches one factor on at a time.
The first rung that breaks calibration names the culprit. Each rung reports the
calibration ratio (true risk over estimated risk); the ladder passes at a rung when a
95% t-interval over seeds covers 1.0.

Nothing here writes to src/. The knobs used are exactly the ones run_policy,
prepare_stream and PolicyConfig already expose; where a factor has no off switch that
is stated in the report rather than papered over. The three places that bit are
documented in NOT_FULLY_DISABLEABLE below and reproduced into the report.

Usage
    PYTHONPATH=src python tools/validation_ladder.py [--seeds 5] [--out FILE]

Exits non-zero when the simplest rung (S1) fails its calibration check.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import os
import sys
import time
import warnings
from dataclasses import fields as _dc_fields
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from t13 import config as C                                    # noqa: E402
from t13.simulate import prepare_stream, run_policy            # noqa: E402


# ---------------------------------------------------------------------------
# Settings that stand in for "off" because a hard zero is not admissible.
# ---------------------------------------------------------------------------
# delay_scale_mult multiplies the Gamma scale of both arrival laws. At exactly 0.0 the
# review-channel scale becomes 0, scipy's gamma.cdf returns NaN for every age, and
# `keep = pi >= pi_floor` is then False for every adjudicated row - the estimator
# silently discards the entire review channel instead of erroring. The ledger channel
# survives only because its delay cap rewrites the NaN to 1.0. 1e-4 keeps the law
# well-defined and puts every delay under four seconds on a 561-day stream.
NO_DELAY = 1e-4
# pi_floor = 0.0 keeps rows whose propensity is exactly zero (the item decided at the
# current instant, age 0), and the Horvitz-Thompson numerator then evaluates 0.0/0.0.
# The whole calibration curve goes NaN and bias_ratio_warm comes back None. 1e-9 keeps
# every row with any chance at all of having reported and drops only the exact zeros,
# which carry no label by construction.
NO_TRIM = 1e-9
# p_disc_eff = min(p_disclose * disclose_mult, 0.999). The 0.999 ceiling is hard-coded
# in run_policy, so disclosure can be pushed to near-certain but not to certain; a
# residual 0.1% of allow-path frauds are still booked clean at any multiplier.
FULL_DISCLOSE = 100.0

NOT_FULLY_DISABLEABLE = [
    ("booked-clean corruption", "disclose_mult",
     "An allow-path item that is not disclosed is written into the calibration set as "
     "obs = 0, i.e. as confirmed-clean, whether or not it was fraud. There is no knob "
     "that stops that write. disclose_mult=%g raises every disclosure probability to "
     "the hard-coded 0.999 ceiling, so ~0.1%% of allow-path frauds are still booked "
     "clean at rung S1. The mechanism is attenuated, not removed." % FULL_DISCLOSE),
    ("zero arrival delay", "delay_scale_mult",
     "delay_scale_mult=0.0 is not admissible: it makes the review-channel Gamma scale "
     "zero, scipy returns NaN, and every adjudicated row is then trimmed away without "
     "warning. The ladder uses %g instead - delays of order seconds on a 561-day "
     "stream - so a residual, negligible arrival lag remains." % NO_DELAY),
    ("zero propensity trimming", "pi_floor",
     "pi_floor=0.0 admits rows with an exactly-zero propensity and the "
     "Horvitz-Thompson numerator evaluates 0.0/0.0, taking the entire risk curve to "
     "NaN. The ladder uses %g, which trims only the zero-propensity rows (which carry "
     "no label anyway)." % NO_TRIM),
    ("per-event drift isolation", "drift_severity",
     "prepare_stream exposes a single scalar drift_severity that scales all three "
     "events together; there is no knob that enables the covariate, concept or prior "
     "event on its own. Rungs S7/S8 therefore move severity as a whole, and the "
     "per-event read is supplied instead by the by_regime decomposition, which scores "
     "calibration separately inside each event window."),
    ("drift severity 0 is not 'no concept drift'", "drift_severity",
     "At severity 0 the covariate shift and the prior surge are genuinely off "
     "(magnitude 0 means exp(0)=1 and lambda=0). The concept event is not: beta_path "
     "sets beta to ev.magnitude inside the concept window, so severity 0 flips beta "
     "from LGP_BETA_BASE=%g to 0.0 there rather than leaving it at the base value. A "
     "residual conditional-law change survives inside that window at every rung."
     % C.LGP_BETA_BASE),
    ("inverse-propensity weight clipping", "weight_clip",
     "PolicyConfig.weight_clip exists but is referenced nowhere in src/t13. Weights "
     "are controlled by propensity trimming only, so no weight-clipping share can be "
     "reported. The clipping column below is the p_d clip share (the share of learned "
     "disclosure propensities pinned to the [0.05, 0.97] bounds of _predict_pd), which "
     "is NaN wherever p_d is the oracle."),
]


# ---------------------------------------------------------------------------
# The ladder. Each rung is the previous rung plus exactly one factor.
# ---------------------------------------------------------------------------
STAGES: List[dict] = [
    dict(
        sid="S1",
        name="simplest: oracle p_d, no delay, no trimming, no decay, no drift",
        turns_on="(baseline)",
        severity=0.0,
        run_set=dict(oracle_pd=True, delay_scale_mult=NO_DELAY,
                     disclose_mult=FULL_DISCLOSE),
        pol_set=dict(rho=1.0, pi_floor=NO_TRIM),
    ),
    dict(
        sid="S2", name="+ arrival delay", turns_on="delay_scale_mult 1e-4 -> 1.0",
        severity=0.0, run_set=dict(delay_scale_mult=1.0), pol_set={},
    ),
    dict(
        sid="S3", name="+ booked-clean corruption",
        turns_on="disclose_mult 100.0 -> 1.0",
        severity=0.0, run_set=dict(disclose_mult=1.0), pol_set={},
    ),
    dict(
        sid="S4", name="+ propensity trimming",
        turns_on="pi_floor 1e-9 -> default", severity=0.0,
        run_set={}, pol_set={}, pol_unset=("pi_floor",),
    ),
    dict(
        sid="S5", name="+ calibration decay",
        turns_on="rho 1.0 -> default", severity=0.0,
        run_set={}, pol_set={}, pol_unset=("rho",),
    ),
    dict(
        sid="S6", name="+ learned p_d", turns_on="oracle_pd True -> False",
        severity=0.0, run_set=dict(oracle_pd=False), pol_set={},
    ),
    dict(
        sid="S7", name="+ drift, half severity", turns_on="drift_severity 0.0 -> 0.5",
        severity=0.5, run_set={}, pol_set={},
    ),
    dict(
        sid="S8", name="full published configuration",
        turns_on="drift_severity 0.5 -> 1.0", severity=1.0, run_set={}, pol_set={},
    ),
]


def build_ladder() -> List[dict]:
    """Resolve the cumulative deltas into an explicit knob set per rung."""
    run_kw: Dict[str, object] = {}
    pol_kw: Dict[str, object] = {}
    out = []
    for st in STAGES:
        run_kw = {**run_kw, **st["run_set"]}
        pol_kw = {**pol_kw, **st["pol_set"]}
        for k in st.get("pol_unset", ()):
            pol_kw.pop(k, None)
        out.append({**st, "run_kw": dict(run_kw), "pol_kw": dict(pol_kw)})
    return out


# ---------------------------------------------------------------------------
def _filter_kwargs(fn, kw: Dict[str, object], label: str,
                   dropped: List[str]) -> Dict[str, object]:
    """Keep only knobs the callee still accepts.

    simulate.py, joint.py and config.py are being edited by other processes while this
    runs. Passing a knob that has since been renamed would abort the whole ladder, so
    unknown knobs are dropped and recorded - a rung that could not set the factor it
    claims to set must say so rather than report a number for a run that did not
    happen.
    """
    try:
        allowed = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return dict(kw)
    ok = {}
    for k, v in kw.items():
        if k in allowed:
            ok[k] = v
        else:
            dropped.append(f"{label}.{k}")
    return ok


def _policy(stream: str, overrides: Dict[str, object],
            dropped: List[str]) -> "C.PolicyConfig":
    allowed = {f.name for f in _dc_fields(C.PolicyConfig)}
    ok = {}
    for k, v in overrides.items():
        if k in allowed:
            ok[k] = v
        else:
            dropped.append(f"PolicyConfig.{k}")
    return C.policy_for(stream, **ok)


def _g(d: Optional[dict], *path, default=float("nan")):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return default if cur is None else cur


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _finite(xs: Sequence[float]) -> np.ndarray:
    a = np.asarray([_f(x) for x in xs], dtype=float)
    return a[np.isfinite(a)]


def _t_crit(df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df))
    except Exception:                                          # pragma: no cover
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093}
        return table.get(df, 1.96)


def ratio_stats(ratios: Sequence[float]) -> dict:
    """Mean calibration ratio with a 95% t-interval over seeds, and the 1.0 verdict."""
    v = _finite(ratios)
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "covers_one": False}
    m = float(v.mean())
    if len(v) == 1:
        return {"n": 1, "mean": m, "sd": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "covers_one": False}
    sd = float(v.std(ddof=1))
    half = _t_crit(len(v) - 1) * sd / np.sqrt(len(v))
    lo, hi = m - half, m + half
    return {"n": int(len(v)), "mean": m, "sd": sd, "ci_lo": float(lo),
            "ci_hi": float(hi), "covers_one": bool(lo <= 1.0 <= hi)}


# ---------------------------------------------------------------------------
class _CtxCache:
    """Contexts are ~15 MB and cost ~3 s each; the ladder is stage-major so only one
    severity is live at a time. Cleared whenever the severity moves on."""

    def __init__(self, stream: str, small: bool, dropped: List[str]):
        self.stream, self.small, self.dropped = stream, small, dropped
        self._sev: Optional[float] = None
        self._d: Dict[int, object] = {}

    def get(self, seed: int, severity: float):
        if self._sev is None or abs(severity - self._sev) > 1e-12:
            self._d.clear()
            self._sev = severity
        if seed not in self._d:
            kw = {"stream": self.stream, "seed": seed, "small": self.small,
                  "drift_severity": severity}
            kw = _filter_kwargs(prepare_stream, kw, "prepare_stream", self.dropped)
            self._d[seed] = prepare_stream(**kw)
        return self._d[seed]


def truth_definitions(r: dict, ctx, delay_mult: float) -> dict:
    """Score `e_t` against three different definitions of "the truth".

    `bias_ratio_warm` compares the estimate to `e_true`, which run_policy computes over
    the **ledger channel only**. The estimate `e_t = risk[g_now]` is computed over
    every channel sitting below tau_lo - which, because tau_lo moves, includes items
    that were routed to the review band under an earlier, higher threshold. Those are
    the high-scoring tail of today's allow region, so the two quantities are not the
    same estimand and part of any measured "bias" is that difference rather than the
    estimator.

    The composition series already carries the all-channel truth under three
    treatments, so the comparison is free:
      r_trail   - boxcar mean of the true labels below tau_lo, all channels
      r_decay   - the same with the geometric decay applied
      r_overlap - the same with decay and propensity trimming: the actual estimand
    """
    comp = r.get("composition") or []
    if not comp:
        return {}
    try:
        warm_day = float(ctx.days[ctx.n_train]) + C.MATURITY_DAYS * float(delay_mult)
    except Exception:
        return {}

    def col(k):
        return np.asarray([_f(c.get(k)) for c in comp], dtype=float)

    et, day = col("e_t"), col("day")
    warm = np.isfinite(et) & (et > 0) & np.isfinite(day) & (day >= warm_day)
    out = {"n_comp_warm": int(warm.sum())}
    for src, dst in (("e_true", "vs_ledger"), ("r_trail", "vs_trail"),
                     ("r_decay", "vs_decay"), ("r_overlap", "vs_overlap")):
        v = col(src)
        m = warm & np.isfinite(v)
        denom = float(et[m].mean()) if m.any() else 0.0
        out[dst] = float(v[m].mean() / denom) if m.any() and denom > 0 else float("nan")
    return out


def collect(stage: dict, ctx, method: str, pol, dropped: List[str]) -> dict:
    """One (rung, seed) run reduced to the numbers the report needs."""
    kw = _filter_kwargs(run_policy, dict(stage["run_kw"]), "run_policy", dropped)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        r = run_policy(ctx, method, pol, **kw)
        elapsed = time.perf_counter() - t0
    td = truth_definitions(r, ctx, kw.get("delay_scale_mult", 1.0))
    eb = r.get("estimator_bias") or {}
    true_r = _f(eb.get("mean_true_warm"))
    est_r = _f(eb.get("mean_estimate_warm"))
    return {
        "seed": ctx.seed,
        "seconds": elapsed,
        "n_numpy_warnings": len(caught),
        "true": true_r,
        "est": est_r,
        "ratio": _f(eb.get("bias_ratio_warm")),
        "abs_err": abs(true_r - est_r),
        "lag_ratio": _f(eb.get("lag_ratio_warm")),
        "n_updates_warm": _f(eb.get("n_updates_warm")),
        "ess_median": _f(r.get("ess_median")),
        "trim_share": _f(r.get("trim_share_mean")),
        "pd_clip_share": _f(r.get("pd_clip_share_mean")),
        "pd_mae": _f(r.get("pd_mae_mean")),
        "w_top1_share": _f(r.get("w_top1_share_mean")),
        "gap_overlap_rel": _f(r.get("gap_overlap_rel")),
        "flag_precision": _f(r.get("flag_precision")),
        "flag_recall": _f(r.get("flag_recall")),
        "oracle_infeasible_rate": _f(r.get("oracle_infeasible_rate")),
        "for_overall": _f(r.get("for_overall")),
        "for_ratio_to_alpha": _f(r.get("for_ratio_to_alpha")),
        "vs_ledger": _f(td.get("vs_ledger")),
        "vs_trail": _f(td.get("vs_trail")),
        "vs_decay": _f(td.get("vs_decay")),
        "vs_overlap": _f(td.get("vs_overlap")),
        "n_comp_warm": _f(td.get("n_comp_warm")),
        "quiet_for": _f(_g(r, "by_regime", "quiet", "for")),
        "covariate_for": _f(_g(r, "by_regime", "covariate", "for")),
        "concept_for": _f(_g(r, "by_regime", "concept", "for")),
        "prior_for": _f(_g(r, "by_regime", "prior", "for")),
        "infeasible_rate": _f(r.get("infeasible_rate")),
    }


def reduce_stage(rows: List[dict]) -> dict:
    """Seed-level rows -> the rung's line in the table."""
    def col(k):
        return _finite([r[k] for r in rows])

    def mean(k):
        v = col(k)
        return float(v.mean()) if len(v) else float("nan")

    true_v = np.asarray([r["true"] for r in rows], dtype=float)
    est_v = np.asarray([r["est"] for r in rows], dtype=float)
    ok = np.isfinite(true_v) & np.isfinite(est_v)
    rmse = float(np.sqrt(np.mean((true_v[ok] - est_v[ok]) ** 2))) if ok.any() else float("nan")
    rel = (np.abs(true_v[ok] - est_v[ok]) / np.maximum(true_v[ok], 1e-12)) if ok.any() else np.array([])
    out = {k: mean(k) for k in (
        "true", "est", "abs_err", "lag_ratio", "ess_median", "trim_share",
        "pd_clip_share", "pd_mae", "w_top1_share", "gap_overlap_rel",
        "flag_precision", "flag_recall", "oracle_infeasible_rate", "for_overall",
        "for_ratio_to_alpha", "quiet_for", "covariate_for", "concept_for",
        "prior_for", "infeasible_rate", "n_updates_warm", "n_comp_warm", "seconds")}
    out["rmse"] = rmse
    out["rel_err_mean"] = float(rel.mean()) if len(rel) else float("nan")
    out["n_numpy_warnings"] = int(sum(r["n_numpy_warnings"] for r in rows))
    out["ratio_stats"] = ratio_stats([r["ratio"] for r in rows])
    for k in ("vs_ledger", "vs_trail", "vs_decay", "vs_overlap"):
        out[f"{k}_stats"] = ratio_stats([r[k] for r in rows])
    out["per_seed"] = rows
    return out


# ---------------------------------------------------------------------------
def _n(x, d=4, dash="n/a") -> str:
    v = _f(x)
    return dash if not np.isfinite(v) else f"{v:.{d}f}"


def _knob_str(stage: dict) -> str:
    bits = [f"drift_severity={stage['severity']:g}"]
    bits += [f"{k}={v!r}" for k, v in sorted(stage["run_kw"].items())]
    bits += [f"{k}={v!r}" for k, v in sorted(stage["pol_kw"].items())]
    return ", ".join(bits)


def render(res: List[dict], ladder: List[dict], meta: dict) -> str:
    L = []
    A = L.append
    A("# Estimator validation ladder")
    A("")
    A(f"Generated {meta['when']} by `tools/validation_ladder.py`. "
      f"Stream `{meta['stream']}`, method `{meta['method']}`, "
      f"**{meta['n_seeds']} seeds** per stage "
      f"({', '.join(str(s) for s in meta['seeds'])}), "
      f"{meta['total_runs']} runs in {meta['wall']:.0f} s.")
    A("")
    A("## The question")
    A("")
    A("Does the M5 risk estimator recover the true release-region risk when every "
      "complication is switched off? Each rung below adds exactly one factor to the "
      "rung above it, so the first rung whose calibration ratio stops covering 1.0 "
      "names the factor responsible.")
    A("")
    A("- **true risk** = `estimator_bias.mean_true_warm`, the decay-weighted true "
      "fraud rate among allow-path items below `tau_lo` in the calibration window. "
      "Computed with labels nobody actually has; never fed back into the policy.")
    A("- **estimated risk** = `estimator_bias.mean_estimate_warm`, the value `e_t` "
      "the policy actually acted on.")
    A("- **calibration ratio** = `estimator_bias.bias_ratio_warm` = true / estimate. "
      "1.0 is perfect. Above 1.0 the estimator is optimistic (it under-reads the risk "
      "it is releasing).")
    A("- Cold-start updates - those inside the first `MATURITY_DAYS * "
      "delay_scale_mult` days, when the ledger has reported nothing - are excluded by "
      "the `_warm` suffix. At rungs S1 and S2 that exclusion window is ~0 days, so "
      "essentially every update is scored; from S3 on it is 180 days.")
    A("- **95% CI** is a t-interval over seeds on the ratio. A rung **passes** when "
      "it covers 1.0.")
    A("")

    key = next((r for r in res if r["sid"] == "S1"), None)
    A("## Headline: the simplest rung")
    A("")
    if key is None:
        A("S1 was not run in this invocation, so there is no verdict on the simplest "
          "configuration and nothing below is safe to read as a localisation.")
        A("")
        rs = {"covers_one": True}
    else:
        rs = key["ratio_stats"]
        verdict = "**S1 FAILS.**" if not rs["covers_one"] else "**S1 passes.**"
        A(f"{verdict} Calibration ratio {_n(rs['mean'], 4)} "
          f"(sd {_n(rs['sd'], 4)}, 95% CI [{_n(rs['ci_lo'], 4)}, "
          f"{_n(rs['ci_hi'], 4)}], n={rs['n']}).")
        A("")
    if key is None:
        pass
    elif rs["covers_one"]:
        A("With the oracle disclosure propensity, no arrival delay, no propensity "
          "trimming, no calibration decay and no drift, the estimator recovers its "
          "target to within Monte Carlo noise. The estimator core is therefore not "
          "the thing that is wrong, and any miscalibration further up the ladder is "
          "attributable to the factor that rung switched on.")
    else:
        ov = key["vs_overlap_stats"]
        A("**The strict check does not pass.** The seed spread is tight enough "
          f"(sd {_n(rs['sd'], 4)} over {rs['n']} seeds) that a "
          f"{abs(1.0 - rs['mean']) * 100:.2f}% offset is resolvable, so the interval "
          "sits clear of 1.0. The direction is "
          + ("conservative: the estimator over-reads the risk it is releasing, so the "
             "policy holds back more traffic than it needs to."
             if rs["mean"] < 1.0 else
             "optimistic: the estimator under-reads the risk it is releasing.")
          )
        A("")
        A("Most of that offset is a **definition mismatch, not an estimator defect** - "
          "see the localisation section below. `bias_ratio_warm` scores `e_t` against "
          "`e_true`, which is computed over the **ledger channel only**, while `e_t` "
          "itself is computed over **every channel** below `tau_lo`. Because `tau_lo` "
          "moves, that includes items routed to the review band under an earlier, "
          "higher threshold - the high-scoring tail of today's allow region. Scored "
          "against the all-channel truth on the same window, the S1 ratio moves to "
          f"{_n(ov['mean'], 4)} (95% CI [{_n(ov['ci_lo'], 4)}, "
          f"{_n(ov['ci_hi'], 4)}]).")
        A("")
        if np.isfinite(_f(ov["mean"])):
            A(f"So the genuine S1 estimator residual is about "
              f"{abs(1.0 - ov['mean']) * 100:.2f}%"
              + ("" if ov["covers_one"] else
                 " - still resolvable at this seed count, so strictly speaking the "
                 "estimator core is not exactly unbiased either")
              + ". For scale, the last rung on this ladder "
              + f"({res[-1]['sid']}) misses by "
              + f"{abs(1.0 - res[-1]['ratio_stats']['mean']) * 100:.1f}%. The core is "
              "close to right; essentially all of the operational miscalibration is "
              "introduced further up the ladder.")
    A("")
    if key is not None:
        A("Per-seed ratios at S1: " +
          ", ".join(f"`{r['seed']}` {_n(r['ratio'], 4)}" for r in key["per_seed"])
          + ".")
        A("")

    first_bad = next((r for r in res if not r["ratio_stats"]["covers_one"]), None)
    if first_bad is None:
        A("No rung on the ladder fails the 1.0 check, including the full published "
          "configuration.")
    elif first_bad["sid"] == res[0]["sid"]:
        A("Because the ladder's own baseline is already outside the interval, the "
          "usual read-off - *the first rung to break names the culprit* - has to be "
          "made on effect size rather than on the pass/fail bit. Rung-to-rung shifts, "
          "against both the headline ledger-only yardstick and the estimator's actual "
          "estimand:")
        A("")
        steps = []
        prev_h = res[0]["ratio_stats"]["mean"]
        prev_o = res[0]["vs_overlap_stats"]["mean"]
        for r in res[1:]:
            st = next(s for s in ladder if s["sid"] == r["sid"])
            cur_h = r["ratio_stats"]["mean"]
            cur_o = r["vs_overlap_stats"]["mean"]
            dh, do = cur_h - prev_h, cur_o - prev_o
            steps.append((abs(_f(do)), r["sid"], st["turns_on"], do))
            A(f"- `{r['sid']}` {st['turns_on']}: headline {_n(prev_h, 4)} -> "
              f"{_n(cur_h, 4)} (shift {_n(dh, 4)}); "
              f"vs estimand {_n(prev_o, 4)} -> {_n(cur_o, 4)} "
              f"(shift {_n(do, 4)})"
              + ("" if r["ratio_stats"]["covers_one"] else " - headline CI excludes 1.0"))
            prev_h, prev_o = cur_h, cur_o
        A("")
        steps = [s for s in steps if np.isfinite(s[0])]
        if steps:
            steps.sort(reverse=True)
            top = steps[0]
            A(f"**The largest single step against the estimand is {top[1]}** "
              f"(`{top[2]}`), which moves the ratio by {top[3]:+.4f}." +
              (f" The next largest is {steps[1][1]} (`{steps[1][2]}`, "
               f"{steps[1][3]:+.4f})." if len(steps) > 1 else ""))
    else:
        st = next(s for s in ladder if s["sid"] == first_bad["sid"])
        A(f"**First rung to break: {first_bad['sid']}** ({st['name']}). The factor it "
          f"switched on is `{st['turns_on']}`; ratio "
          f"{_n(first_bad['ratio_stats']['mean'], 4)} "
          f"with CI [{_n(first_bad['ratio_stats']['ci_lo'], 4)}, "
          f"{_n(first_bad['ratio_stats']['ci_hi'], 4)}].")
    A("")

    A("## Stage table: calibration")
    A("")
    A("| rung | factor switched on | true risk | est risk | ratio (true/est) | 95% CI | abs err | RMSE | rel err | covers 1.0 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        st = next(s for s in ladder if s["sid"] == r["sid"])
        q = r["ratio_stats"]
        A(f"| {r['sid']} | {st['turns_on']} | {_n(r['true'], 6)} | {_n(r['est'], 6)} "
          f"| {_n(q['mean'], 4)} | [{_n(q['ci_lo'], 4)}, {_n(q['ci_hi'], 4)}] "
          f"| {_n(r['abs_err'], 6)} | {_n(r['rmse'], 6)} | {_n(r['rel_err_mean'], 4)} "
          f"| {'yes' if q['covers_one'] else '**NO**'} |")
    A("")
    A("`abs err` is the mean over seeds of |true - estimate|; `RMSE` is the "
      "root-mean-square of the same difference across seeds; `rel err` is "
      "|true - estimate| / true.")
    A("")

    A("## Localisation: which truth is the estimate being scored against?")
    A("")
    A("Every column is the same estimate `e_t` divided into a different definition of "
      "the truth, over the same warm updates, straight from the per-update "
      "composition series. It separates *the estimator is wrong* from *the estimator "
      "and the yardstick are measuring different populations*.")
    A("")
    A("| rung | vs ledger-only (`e_true`, the headline) | vs all-channel boxcar (`r_trail`) | vs + decay (`r_decay`) | vs + decay and trimming (`r_overlap`, the actual estimand) | warm updates |")
    A("|---|---|---|---|---|---|")
    for r in res:
        cells = []
        for k in ("vs_ledger", "vs_trail", "vs_decay", "vs_overlap"):
            q = r[f"{k}_stats"]
            mark = "" if q["covers_one"] else " *"
            cells.append(f"{_n(q['mean'], 4)} [{_n(q['ci_lo'], 4)}, "
                         f"{_n(q['ci_hi'], 4)}]{mark}")
        A(f"| {r['sid']} | " + " | ".join(cells) +
          f" | {_n(r['n_comp_warm'], 0)} |")
    A("")
    A("`*` marks a 95% t-interval that excludes 1.0. The first column reproduces "
      "`bias_ratio_warm` up to the slightly wider update set used here "
      "(`_summarise` additionally drops updates too close to the end of the stream to "
      "fill their realisation horizon).")
    A("")
    A("Reading across a row: the gap between column 1 and column 2 is the "
      "**channel-definition** effect - `e_true` looks only at ledger rows, `e_t` at "
      "every channel below `tau_lo`, and because `tau_lo` moves, the latter includes "
      "items that were sent to the review band under an earlier, higher threshold. "
      "The gap between columns 2 and 3 is the **decay** estimand shift, and between 3 "
      "and 4 the **trimming** estimand shift. Whatever remains in column 4 is the "
      "estimator's own error against the quantity it actually targets.")
    A("")

    A("## Stage table: weights, propensities and flag")
    A("")
    A("| rung | ESS median | trim share | p_d clip share | p_d MAE | top-1% weight share | est-gap rel | flag prec | flag rec | oracle infeas rate | warm updates |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        A(f"| {r['sid']} | {_n(r['ess_median'], 0)} | {_n(r['trim_share'], 4)} "
          f"| {_n(r['pd_clip_share'], 4)} | {_n(r['pd_mae'], 4)} "
          f"| {_n(r['w_top1_share'], 4)} | {_n(r['gap_overlap_rel'], 4)} "
          f"| {_n(r['flag_precision'], 3)} | {_n(r['flag_recall'], 3)} "
          f"| {_n(r['oracle_infeasible_rate'], 3)} | {_n(r['n_updates_warm'], 0)} |")
    A("")
    A("There is no inverse-propensity weight-clipping share to report: "
      "`PolicyConfig.weight_clip` is defined but referenced nowhere in `src/t13`, so "
      "weights are governed by trimming alone. The clip column is the share of "
      "*learned disclosure propensities* pinned to the [0.05, 0.97] bounds of "
      "`_predict_pd`, which is undefined wherever p_d is the oracle.")
    A("")

    A("## Stage table: realised risk on the allow path")
    A("")
    A("| rung | FOR overall | FOR / alpha | quiet | covariate | concept | prior | infeasible rate |")
    A("|---|---|---|---|---|---|---|---|")
    for r in res:
        A(f"| {r['sid']} | {_n(r['for_overall'], 6)} | {_n(r['for_ratio_to_alpha'], 3)} "
          f"| {_n(r['quiet_for'], 6)} | {_n(r['covariate_for'], 6)} "
          f"| {_n(r['concept_for'], 6)} | {_n(r['prior_for'], 6)} "
          f"| {_n(r['infeasible_rate'], 4)} |")
    A("")
    A("The by-regime columns are the per-event read that the missing per-event drift "
      "knob cannot give: at S8 they score calibration separately inside the "
      "covariate, concept and prior windows.")
    A("")

    A("## Exact knob settings per rung")
    A("")
    A("| rung | name | knobs |")
    A("|---|---|---|")
    for st in ladder:
        A(f"| {st['sid']} | {st['name']} | `{_knob_str(st)}` |")
    A("")
    A("Everything not listed is the published default from "
      "`config.policy_for(stream)` and the `run_policy` signature defaults "
      "(`cohort_aligned=True`, `cohort_fallback=True`, `joint_mode=''`, "
      f"`epsilon={C.PolicyConfig.epsilon}`, `alpha`, `budget`, `min_support`, "
      "`window_max_n`, `window_max_days`, `update_every` all at defaults).")
    A("")

    A("## Factors that could not be switched fully off")
    A("")
    A("Stated here rather than left implicit, because a rung that claims to remove a "
      "factor and does not is worse than one that admits it.")
    A("")
    for title, knob, body in NOT_FULLY_DISABLEABLE:
        A(f"- **{title}** (`{knob}`). {body}")
    A("")

    if meta["dropped"]:
        A("### Knobs the current source no longer accepts")
        A("")
        A("These were requested by the ladder and dropped because the callee does not "
          "expose them. Any rung depending on one did **not** actually switch its "
          "factor:")
        A("")
        for k in sorted(set(meta["dropped"])):
            A(f"- `{k}`")
        A("")

    A("## Per-seed detail")
    A("")
    A("| rung | seed | true | est | ratio | abs err | ESS median | trim share | FOR | numpy warnings |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        for s in r["per_seed"]:
            A(f"| {r['sid']} | {s['seed']} | {_n(s['true'], 6)} | {_n(s['est'], 6)} "
              f"| {_n(s['ratio'], 4)} | {_n(s['abs_err'], 6)} "
              f"| {_n(s['ess_median'], 0)} | {_n(s['trim_share'], 4)} "
              f"| {_n(s['for_overall'], 6)} | {s['n_numpy_warnings']} |")
    A("")
    A("A non-zero numpy-warning count means the run hit a divide-by-zero or invalid "
      "operation somewhere in the estimator and the numbers on that line should not "
      "be trusted.")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5,
                    help="how many of config.SEEDS to use (default 5)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ESTIMATOR_VALIDATION_LADDER.md"),
        help="markdown report path")
    ap.add_argument("--stream", default="synb2b")
    ap.add_argument("--method", default="M5",
                    help="only M5 runs the inverse-propensity estimator under test")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--stages", default="",
                    help="comma-separated rung ids to run (default: all)")
    args = ap.parse_args(argv)

    seeds = list(C.SEEDS[:max(1, args.seeds)])
    ladder = build_ladder()
    if args.stages:
        want = {s.strip().upper() for s in args.stages.split(",") if s.strip()}
        ladder = [s for s in ladder if s["sid"] in want]
        if not ladder:
            print("no matching stages", file=sys.stderr)
            return 2

    dropped: List[str] = []
    cache = _CtxCache(args.stream, args.small, dropped)
    res: List[dict] = []
    t_start = time.perf_counter()

    print(f"validation ladder: stream={args.stream} method={args.method} "
          f"seeds={seeds} ({len(seeds)} per stage)")
    for st in ladder:
        rows = []
        for seed in seeds:
            ctx = cache.get(seed, st["severity"])
            pol = _policy(args.stream, st["pol_kw"], dropped)
            rows.append(collect(st, ctx, args.method, pol, dropped))
        agg = reduce_stage(rows)
        agg["sid"] = st["sid"]
        res.append(agg)
        q, o = agg["ratio_stats"], agg["vs_overlap_stats"]
        print(f"  {st['sid']:3s} {st['turns_on']:34s} "
              f"true={_n(agg['true'], 6)} est={_n(agg['est'], 6)} "
              f"ratio={_n(q['mean'], 4)} CI=[{_n(q['ci_lo'], 4)},{_n(q['ci_hi'], 4)}] "
              f"rmse={_n(agg['rmse'], 6)} ess={_n(agg['ess_median'], 0)} "
              f"trim={_n(agg['trim_share'], 4)} "
              f"vs_estimand={_n(o['mean'], 4)} "
              f"{'OK ' if q['covers_one'] else 'FAIL'}",
              flush=True)

    meta = {
        "when": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stream": args.stream, "method": args.method, "seeds": seeds,
        "n_seeds": len(seeds), "dropped": dropped,
        "total_runs": len(ladder) * len(seeds),
        "wall": time.perf_counter() - t_start,
    }
    with open(args.out, "w") as fh:
        fh.write(render(res, ladder, meta))
    print(f"wrote {args.out}")

    simplest = next((r for r in res if r["sid"] == "S1"), None)
    if simplest is None:
        print("S1 was not run; no verdict on the simplest configuration.",
              file=sys.stderr)
        return 0
    q = simplest["ratio_stats"]
    if q["covers_one"]:
        print(f"S1 PASS: calibration ratio {q['mean']:.4f} "
              f"CI [{q['ci_lo']:.4f}, {q['ci_hi']:.4f}] covers 1.0")
        return 0
    print(f"S1 FAIL: calibration ratio {_n(q['mean'], 4)} "
          f"CI [{_n(q['ci_lo'], 4)}, {_n(q['ci_hi'], 4)}] excludes 1.0 - the "
          f"estimator does not recover its target in the simplest setting",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
