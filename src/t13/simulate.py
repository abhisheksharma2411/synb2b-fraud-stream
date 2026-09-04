"""The stream harness: one pass, one policy, hard budget.

Action codes
    0 allow            1 review (band)      2 block
    3 review (explore) 4 overflow-allow     5 overflow-block

The manual queue holds a fixed number of slots per reporting block. When the policy
asks for more reviews than the queue can take, the surplus is auto-decided against the
midpoint of the band. Every method faces that same ceiling, so "review rate" is
reported twice: what the thresholds *demanded*, and what the queue actually *served*.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import calib, config as C
from .arrival import DisclosureModel, draw_arrivals
from .data import load_stream, build_synb2b_features, build_ulb_features
from .drift import apply_drift
from .scorer import post_event_degradation, scorer_report, train_frozen_scorer

ACT_ALLOW, ACT_REVIEW, ACT_BLOCK, ACT_EXPLORE, ACT_OVF_ALLOW, ACT_OVF_BLOCK = range(6)


class FutureLabelError(AssertionError):
    """A label that has not arrived yet was offered to the calibration set."""


def assert_calibration_causal(arrival: np.ndarray, idx: np.ndarray, now: float) -> None:
    """The one invariant the whole design rests on.

    At decision time `now`, every item the estimator is allowed to look at must
    already carry an arrived label. Blocked items never arrive at all, so an infinite
    arrival time trips this too.
    """
    if len(idx) == 0:
        return
    a = arrival[idx]
    bad = ~np.isfinite(a) | (a > now + 1e-9)
    if np.any(bad):
        j = int(idx[np.nonzero(bad)[0][0]])
        raise FutureLabelError(
            f"row {j} has arrival {arrival[j]!r} but the clock reads {now!r}"
        )


ALLOW_ACTS = (ACT_ALLOW, ACT_OVF_ALLOW)
REVIEW_ACTS = (ACT_REVIEW, ACT_EXPLORE)
CH_REVIEW, CH_LEDGER, CH_EXPLORE = 0, 1, 2


@dataclass
class StreamContext:
    stream: str
    seed: int
    n: int
    n_train: int
    days: np.ndarray
    amount: np.ndarray
    y: np.ndarray
    topo: np.ndarray
    zeta: np.ndarray
    scores: np.ndarray
    grid: np.ndarray
    events: list
    draws: object
    scorer: dict
    degradation: dict
    drift_diag: dict


_CTX_CACHE: Dict[tuple, StreamContext] = {}


def prepare_stream(stream: str, seed: int, small: bool = False,
                   drift_severity: float = 1.0) -> StreamContext:
    key = (stream, seed, small, round(drift_severity, 6))
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]

    raw, _, feat_names = load_stream(stream, small=small)
    builder = build_synb2b_features if stream == "synb2b" else build_ulb_features
    n = len(raw)
    n_train = int(round(C.TRAIN_FRACTION * n))

    shifted, feats, y, topo, events, diag, aux = apply_drift(
        stream, raw, builder, n_train, seed, severity=drift_severity
    )
    X = feats[feat_names].to_numpy(dtype=np.float64)
    _, scores = train_frozen_scorer(X, y, n_train, seed)
    days = shifted["day"].to_numpy(float)
    amount = shifted["amount"].to_numpy(float)

    ctx = StreamContext(
        stream=stream, seed=seed, n=n, n_train=n_train, days=days, amount=amount,
        y=np.asarray(y, dtype=np.int8), topo=np.asarray(topo, dtype=object),
        zeta=aux["zeta"], scores=scores,
        grid=calib.build_grid(scores[:n_train]),
        events=events, draws=draw_arrivals(stream, topo, aux["zeta"], seed),
        scorer=scorer_report(scores, y, n_train),
        degradation=post_event_degradation(scores, y, days, events),
        drift_diag=diag,
    )
    # Contexts are large (several float64 arrays over the whole stream) and building
    # one costs seconds, so keep a couple around but do not hoard them.
    while len(_CTX_CACHE) >= 2:
        _CTX_CACHE.pop(next(iter(_CTX_CACHE)))
    _CTX_CACHE[key] = ctx
    return ctx


# ---------------------------------------------------------------------------
def _place_thresholds(cdf, g_star, pol, eps, respects_budget):
    """Turn a risk-driven candidate into a legal (tau_lo, tau_hi) pair.

    Analyst capacity is standing capacity. The queue is staffed and it is worked, so
    the band is held at b(1 - eps) of traffic and the question is which items land in
    it, not whether anyone is reading. tau_lo is therefore the single free threshold,
    and it lives in

        1 - block_cap - b(1-eps)  <=  F(tau_lo)  <=  1 - b(1-eps).

    The upper end is the queue staying full; the lower end is the hold allowance
    running out. When risk control asks for a tau_lo below that floor, the pair
    (alpha, b) is not jointly attainable and the flag goes up rather than the
    constraint quietly going away. That is Proposition 1 in one line of code.
    """
    band = pol.budget * (1.0 - eps)
    g_cap = calib.invert_cdf(cdf, 1.0 - band)
    g_floor = calib.invert_cdf(cdf, 1.0 - pol.block_cap - band)
    infeasible = 0
    g_lo = min(int(g_star), g_cap)
    if respects_budget and g_lo < g_floor:
        g_lo, infeasible = g_floor, 1
    g_hi = calib.invert_cdf(cdf, min(float(cdf[g_lo]) + band, 1.0))
    return g_lo, max(g_hi, g_lo), infeasible


def _initial_thresholds(ctx: StreamContext, pol: C.PolicyConfig, method: str):
    """Thresholds fixed on the initial split, used as-is by M0/M1 and as the warm
    start for everything else."""
    grid = ctx.grid
    s_tr = ctx.scores[:ctx.n_train]
    y_tr = ctx.y[:ctx.n_train].astype(float)
    bins = calib.bin_of(grid, s_tr)
    den = calib.accumulate(bins, np.ones(len(s_tr)), len(grid))
    num = calib.accumulate(bins, y_tr, len(grid))
    cdf = den / max(den[-1], 1.0)

    if method == "M1":
        # split conformal: the calibration slice is the last quarter of the training
        # split, and the level carries the finite-sample (n+1) correction.
        k0 = int(round(0.75 * ctx.n_train))
        sc, yc = ctx.scores[k0:ctx.n_train], ctx.y[k0:ctx.n_train].astype(float)
        b2 = calib.bin_of(grid, sc)
        d2 = calib.accumulate(b2, np.ones(len(sc)), len(grid))
        n2 = calib.accumulate(b2, yc, len(grid))
        m = len(sc)
        risk = calib.risk_curve(n2, d2)
        g_star = calib.crc_threshold(risk, d2, pol.alpha * (m + 1.0) / m,
                                     pol.min_support)
    else:
        risk = calib.risk_curve(num, den)
        g_star = calib.crc_threshold(risk, den, pol.alpha, pol.min_support)

    eps = pol.epsilon if method == "M5" else 0.0
    g_lo, g_hi, _ = _place_thresholds(cdf, g_star, pol, eps,
                                      respects_budget=method not in ("M0", "M1"))
    return g_lo, g_hi, float(cdf[g_lo])


def _fit_disclosure(ctx, pol, idx_ledger, idx_truth, n_explore, obs, floor=80):
    """p_disclose on the allow path is a ratio of two fitted probabilities.

    On the ledger channel only obs = y * D is ever seen, so P(obs=1|x) = r(x) p_d(x)
    and the two factors cannot be pulled apart from that channel alone. The second fit
    supplies r(x) from the rows whose labels are adjudicated rather than inferred: the
    review band, plus whatever exploration bought.

    Pooling the band in is a real modelling decision and it is not free. The band is
    mid-score by construction, so using it for r means extrapolating a three-term
    logistic down into the allow region, and nothing in the band can test that
    extrapolation. The exploration sample is what anchors it - the only labels drawn
    from the allow region itself. This is Proposition 2(a): at eps = 0 the risk on the
    auto-allow path is identified only through an untestable functional form.
    """
    if len(idx_truth) < floor or len(idx_ledger) < floor:
        return None, None, n_explore, len(idx_ledger)
    rng = np.random.default_rng(9176 + ctx.seed)
    if len(idx_ledger) > 8000:
        idx_ledger = rng.choice(idx_ledger, size=8000, replace=False)
    if len(idx_truth) > 8000:
        idx_truth = rng.choice(idx_truth, size=8000, replace=False)
    m_obs = DisclosureModel().fit(
        ctx.zeta[idx_ledger], ctx.scores[idx_ledger], obs[idx_ledger]
    )
    m_r = DisclosureModel().fit(
        ctx.zeta[idx_truth], ctx.scores[idx_truth], ctx.y[idx_truth].astype(float)
    )
    return m_obs, m_r, n_explore, len(idx_ledger)


_GCDF_CACHE: Dict[tuple, tuple] = {}


def _gamma_cdf(age: np.ndarray, shape: float, scale: float, cap) -> np.ndarray:
    """P(delay <= age) for a Gamma(shape, scale), optionally capped at `cap` days.

    The two arrival laws are operational facts, not fitted quantities: the queue works
    to a stated SLA and the ledger closes on a stated day. Only the disclosure
    probability has to be estimated. Tabulated once per (shape, scale, cap) and read
    back by interpolation, because this runs 320 times per pass over tens of thousands
    of rows and scipy's own cdf is far too slow for that.
    """
    key = (round(shape, 6), round(scale, 6), None if cap is None else round(cap, 6))
    tab = _GCDF_CACHE.get(key)
    if tab is None:
        from scipy.stats import gamma as _g

        xs = np.concatenate([np.linspace(0.0, 20.0, 401), np.linspace(20.5, 400.0, 761)])
        ys = _g.cdf(xs, shape, scale=scale)
        if cap is not None:
            ys = np.where(xs >= cap, 1.0, ys)
        tab = (xs, ys)
        _GCDF_CACHE[key] = tab
    xs, ys = tab
    return np.interp(np.maximum(age, 0.0), xs, ys)


def _predict_pd(m_obs, m_r, zeta, score, fallback: float) -> np.ndarray:
    if m_obs is None or m_r is None:
        return np.full(len(zeta), fallback)
    num = m_obs.predict(zeta, score)
    den = np.maximum(m_r.predict(zeta, score), 1e-3)
    return np.clip(num / den, 0.05, 0.97)


# ---------------------------------------------------------------------------
def run_policy(ctx: StreamContext, method: str, pol: C.PolicyConfig,
               delay_scale_mult: float = 1.0, disclose_mult: float = 1.0,
               check_invariants: bool = True, oracle_pd: bool = False,
               keep_arrays: bool = False) -> dict:
    n, n0 = ctx.n, ctx.n_train
    grid, G = ctx.grid, len(ctx.grid)
    scores, days, y, amount = ctx.scores, ctx.days, ctx.y, ctx.amount
    dr = ctx.draws
    eps = pol.epsilon if method == "M5" else 0.0
    cst = C.costs_for(ctx.stream)
    uses_iap = method == "M5"
    adaptive = method in ("M2", "M3", "M4", "M5")

    action = np.full(n, -1, dtype=np.int8)
    channel = np.full(n, -1, dtype=np.int8)
    obs = np.full(n, -1.0, dtype=float)
    arrival = np.full(n, np.inf, dtype=float)

    g_lo, g_hi, allow_mass0 = _initial_thresholds(ctx, pol, method)
    tau_lo, tau_hi = grid[g_lo], grid[g_hi]
    alpha_t = pol.alpha
    u_t = allow_mass0  # M3 carries an allow-mass level rather than a risk target
    # Exploration is deliberately not gated on having calibration data. Waiting for
    # labels before sampling is a deadlock: with no reviews nothing is adjudicated,
    # and the ledger channel does not report for six months.
    eps_rate = float(np.clip(eps * pol.budget / max(allow_mass0, 1e-6), 0.0, 0.25))
    m_obs = m_r = None
    pd_fallback = 0.45

    block_size = pol.metric_block
    slots_per_block = int(np.floor(pol.budget * block_size))
    slots_left = slots_per_block
    block_start = n0

    infeasible_flags: List[int] = []
    oracle_flags: List[int] = []
    flag_days: List[float] = []
    flag_etrue: List[float] = []
    update_times: List[float] = []
    comp_series: List[dict] = []
    demand_review = np.zeros(n, dtype=np.int8)
    n_updates = 0
    since_update = 0
    disc_refit_countdown = 0

    p_disc_eff = np.minimum(dr.p_disclose * disclose_mult, 0.999)

    for i in range(n0, n):
        if i >= block_start + block_size:
            block_start = i
            slots_left = slots_per_block
        s = scores[i]
        d = days[i]

        if s <= tau_lo:
            want_review = eps_rate > 0.0 and dr.u_explore[i] < eps_rate
            a = ACT_EXPLORE if want_review else ACT_ALLOW
        elif s > tau_hi:
            a = ACT_BLOCK
        else:
            a = ACT_REVIEW

        if a in REVIEW_ACTS:
            demand_review[i] = 1
            if slots_left > 0:
                slots_left -= 1
            else:  # queue full: fall back to an automatic decision at the band midpoint
                a = ACT_OVF_ALLOW if s <= 0.5 * (tau_lo + tau_hi) else ACT_OVF_BLOCK

        action[i] = a
        if a in REVIEW_ACTS:
            channel[i] = CH_EXPLORE if a == ACT_EXPLORE else CH_REVIEW
            obs[i] = float(y[i])
            arrival[i] = d + dr.delay_review[i] * delay_scale_mult
        elif a in ALLOW_ACTS:
            channel[i] = CH_LEDGER
            disclosed = (y[i] == 1) and (dr.u_disclose[i] < p_disc_eff[i])
            if disclosed:
                obs[i] = 1.0
                arrival[i] = d + dr.delay_disclose[i] * delay_scale_mult
            else:
                obs[i] = 0.0
                arrival[i] = d + C.MATURITY_DAYS * delay_scale_mult
        else:
            channel[i] = -1
            arrival[i] = np.inf

        since_update += 1
        if not adaptive or since_update < pol.update_every:
            continue
        since_update = 0
        n_updates += 1
        t0 = time.perf_counter()

        lo = max(n0, i - pol.window_max_n)
        sl = slice(lo, i + 1)
        in_win = (days[sl] >= d - pol.window_max_days) & (channel[sl] >= 0)
        idx_all = np.nonzero(in_win)[0] + lo          # decided, not blocked
        has_label = arrival[idx_all] <= d
        idx = idx_all[has_label]
        if check_invariants:
            assert_calibration_causal(arrival, idx, d)

        # trailing score law needs no labels at all
        sc_recent = scores[lo:i + 1]
        cdf = calib.accumulate(calib.bin_of(grid, sc_recent), np.ones(len(sc_recent)), G)
        cdf = cdf / max(cdf[-1], 1.0)

        # Oracle feasibility, for validating the flag rather than for driving it.
        # Timed out of the update: t0 is advanced by however long this takes, so the
        # reported per-update cost stays the cost of the calibration update itself and
        # not of the validation instrumentation wrapped around it.
        _t_oracle = time.perf_counter()
        # Proposition 1 evaluated on the TRUE risk curve of this window: the tightest
        # admissible release threshold is the hold-cap floor, so (alpha, b) was
        # attainable here exactly when the true risk at that floor clears alpha.
        # Nothing below is fed back into any decision; it exists to be scored.
        _band_o = pol.budget * (1.0 - eps)
        _g_floor_o = calib.invert_cdf(cdf, 1.0 - pol.block_cap - _band_o)
        _bins_o = calib.bin_of(grid, scores[idx_all])
        _w_o = calib.decay_weights(d - days[idx_all], pol.rho)
        _den_o = calib.accumulate(_bins_o, _w_o, G)
        _num_o = calib.accumulate(_bins_o, _w_o * y[idx_all].astype(float), G)
        if _den_o[_g_floor_o] >= pol.min_support:
            _r_o = float(_num_o[_g_floor_o] / max(_den_o[_g_floor_o], 1e-12))
            oracle_infeasible = int(_r_o > pol.alpha)
        else:
            _r_o = float("nan")
            oracle_infeasible = -1          # undetermined: too little support to judge
        t0 += time.perf_counter() - _t_oracle

        infeasible = 0
        e_true = float("nan")
        if len(idx) >= pol.min_calib_items:
            if uses_iap:
                # Horvitz-Thompson. The denominator counts every decision taken in the
                # window, labelled or not, because the policy knows perfectly well what
                # it decided. The numerator counts the labels that did arrive, each
                # divided by its probability of having arrived at all. Clean items sit
                # in the ledger for 180 days; frauds surface in about sixty. Taking the
                # arrived pool at face value therefore reads the recent past as far
                # more fraudulent than it was, and dividing by that probability is what
                # puts it back.
                ch_a = channel[idx_all]
                age_a = d - days[idx_all]
                w_all = calib.decay_weights(age_a, pol.rho)
                bins_all = calib.bin_of(grid, scores[idx_all])
                ob_a = np.where(has_label, obs[idx_all], 0.0)

                if disc_refit_countdown <= 0:
                    mature = idx_all[(ch_a == CH_LEDGER) & (age_a >= C.MATURITY_DAYS
                                                            * delay_scale_mult)]
                    ch_l = channel[idx]
                    expl = idx[ch_l == CH_EXPLORE]
                    truth = idx[(ch_l == CH_EXPLORE) | (ch_l == CH_REVIEW)]
                    m_obs, m_r, _, _ = _fit_disclosure(ctx, pol, mature, truth,
                                                       len(expl), obs)
                    if m_obs is None and len(truth) > 0 and len(mature) > 0:
                        r_ex = float(np.mean(y[truth]))
                        r_ob = float(np.mean(obs[mature]))
                        pd_fallback = float(np.clip(r_ob / max(r_ex, 1e-6), 0.05, 0.97)) \
                            if r_ex > 0 else 0.45
                    disc_refit_countdown = 20
                disc_refit_countdown -= 1

                if oracle_pd:
                    pd_hat = p_disc_eff[idx_all]
                else:
                    pd_hat = _predict_pd(m_obs, m_r, ctx.zeta[idx_all],
                                         scores[idx_all], pd_fallback)
                pi = np.where(
                    ch_a == CH_LEDGER,
                    pd_hat * _gamma_cdf(age_a, C.DISCLOSE_DELAY_SHAPE,
                                        C.DISCLOSE_DELAY_SCALE * delay_scale_mult,
                                        C.DISCLOSE_DELAY_CAP * delay_scale_mult),
                    _gamma_cdf(age_a, C.REVIEW_DELAY_SHAPE,
                               C.REVIEW_DELAY_SCALE * delay_scale_mult, None),
                )
                # Propensity trimming rather than weight clipping. A decision taken
                # nine days ago has had almost no chance to report, so its inverse
                # propensity is enormous and its contribution is nearly all variance.
                # Dropping it is cleaner than capping it, and it says something
                # honest: the estimator only looks at decisions old enough to have had
                # a fair chance of coming back. That is also where the lag comes from.
                keep = pi >= pol.pi_floor
                pi = np.clip(pi, pol.pi_floor, 1.0)
                wk = w_all * keep
                den = calib.accumulate(bins_all, wk, G)
                num = calib.accumulate(bins_all, wk * ob_a / pi, G)
                w_report, ch_report = wk, ch_a
            else:
                # the arrived pool, taken at face value
                ch_report = channel[idx]
                w_report = calib.decay_weights(d - days[idx], pol.rho)
                bins = calib.bin_of(grid, scores[idx])
                ob = np.clip(obs[idx], 0.0, 1.0)
                den = calib.accumulate(bins, w_report, G)
                num = calib.accumulate(bins, w_report * ob, G)

            risk = calib.risk_curve(num, den)
            ch, w = ch_report, w_report

            # the estimator's own reading of the current allow region, for ACI
            g_now = calib.bin_of(grid, np.asarray([tau_lo]))[0]
            e_t = float(risk[g_now]) if den[g_now] >= pol.min_support else float(alpha_t)

            # the same window scored with labels nobody actually has. Reported only,
            # never fed back into the policy: it is what separates "the estimator is
            # biased" from "the estimator is looking at last month".
            _m = (channel[idx_all] == CH_LEDGER) & (scores[idx_all] <= tau_lo)
            if _m.sum() >= 50:
                _w = calib.decay_weights(d - days[idx_all][_m], pol.rho)
                e_true = float(np.sum(_w * y[idx_all][_m]) / max(np.sum(_w), 1e-9))
            else:
                e_true = float("nan")

            if method == "M2":
                # cost-optimal tau_lo, band pinned to the remaining review budget
                fn_cost = cst.fn_rate * amount[idx] + cst.fn_fixed
                fp_cost = cst.fp_rate * amount[idx] + cst.fp_fixed
                A = calib.accumulate(bins, w * ob * fn_cost, G)
                B = calib.accumulate(bins, w * (1.0 - ob) * fp_cost, G)
                W = calib.accumulate(bins, w, G)
                band = pol.budget * (1.0 - eps)
                lvl = np.clip(cdf + band, 0.0, 1.0)
                g_hi_all = np.clip(np.searchsorted(cdf, lvl, side="left"), 0, G - 1)
                total = A + (B[-1] - B[g_hi_all]) + cst.review * (W[g_hi_all] - W)
                g_star = int(np.argmin(total))
            elif method == "M3":
                rel = float(np.clip((pol.alpha - e_t) / max(pol.alpha, 1e-9), -5.0, 1.0))
                u_t = float(np.clip(u_t + 0.35 * pol.gamma_aci * rel, 0.50, 1.0))
                g_star = calib.invert_cdf(cdf, u_t)
            else:  # M4, M5
                alpha_t = float(np.clip(alpha_t + pol.gamma_aci * (pol.alpha - e_t),
                                        0.05 * pol.alpha, 3.0 * pol.alpha))
                g_star = calib.crc_threshold(risk, den, alpha_t, pol.min_support)

            g_lo, g_hi, infeasible = _place_thresholds(cdf, g_star, pol, eps,
                                                       respects_budget=True)
            if method == "M3":
                u_t = float(cdf[g_lo])
            tau_lo, tau_hi = grid[g_lo], grid[g_hi]
            if eps > 0:
                eps_rate = float(np.clip(
                    eps * pol.budget / max(float(cdf[g_lo]), 1e-6), 0.0, 0.25))

            tw = float(np.sum(w))
            comp_series.append({
                "i": int(i), "day": float(d),
                "w_review": float(np.sum(w[ch == CH_REVIEW]) / max(tw, 1e-9)),
                "w_ledger": float(np.sum(w[ch == CH_LEDGER]) / max(tw, 1e-9)),
                "w_explore": float(np.sum(w[ch == CH_EXPLORE]) / max(tw, 1e-9)),
                "n_review": int(np.sum(ch == CH_REVIEW)),
                "n_ledger": int(np.sum(ch == CH_LEDGER)),
                "n_explore": int(np.sum(ch == CH_EXPLORE)),
                "tau_lo": float(tau_lo), "tau_hi": float(tau_hi),
                "alpha_t": float(alpha_t), "infeasible": int(infeasible),
                "oracle_infeasible": int(oracle_infeasible),
                "oracle_floor_risk": float(_r_o),
                "e_t": float(e_t), "e_true": float(e_true),
            })
        infeasible_flags.append(infeasible)
        oracle_flags.append(oracle_infeasible)
        flag_days.append(float(d))
        flag_etrue.append(float(e_true))
        update_times.append(time.perf_counter() - t0)

    summary = _summarise(ctx, method, pol, action, obs, arrival, channel, demand_review,
                         infeasible_flags, update_times, comp_series, eps,
                         delay_scale_mult, disclose_mult, oracle_pd,
                         oracle_flags, flag_days, flag_etrue)
    if keep_arrays:
        # evaluation-stream slices, for the invariant tests. Never serialised.
        sl2 = slice(n0, n)
        summary["_action"] = action[sl2]
        summary["_channel"] = channel[sl2]
        summary["_arrival"] = arrival[sl2]
        summary["_obs"] = obs[sl2]
        summary["_dec_day"] = days[sl2]
        summary["_score"] = scores[sl2]
        summary["_y"] = y[sl2]
    return summary


# ---------------------------------------------------------------------------

def _score_flag(flags, oracle, days, etrue, ctx) -> dict:
    """Score the budget-infeasibility flag against oracle feasibility.

    The oracle is Proposition 1 evaluated on the window's true risk curve, so this
    asks a narrow question: when the target genuinely was not attainable at this
    capacity, did the policy say so? Windows where the true curve has too little
    support at the hold-cap floor are marked undetermined and excluded rather than
    guessed at.
    """
    out = {"flag_n_scored": 0, "flag_undetermined": 0}
    if not flags or not oracle:
        return out
    f = np.asarray(flags, dtype=int)
    o = np.asarray(oracle, dtype=int)
    d = np.asarray(days, dtype=float)
    ok = o >= 0
    out["flag_undetermined"] = int((~ok).sum())
    if not ok.any():
        return out
    f, o, d_ok = f[ok], o[ok], d[ok]
    tp = int(((f == 1) & (o == 1)).sum())
    fp = int(((f == 1) & (o == 0)).sum())
    fn = int(((f == 0) & (o == 1)).sum())
    tn = int(((f == 0) & (o == 0)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    out.update({
        "flag_n_scored": int(ok.sum()),
        "oracle_infeasible_rate": float(o.mean()),
        "flag_tp": tp, "flag_fp": fp, "flag_fn": fn, "flag_tn": tn,
        "flag_precision": float(tp / max(tp + fp, 1)) if (tp + fp) else None,
        "flag_recall": float(rec) if (tp + fn) else None,
        "flag_specificity": float(spec) if (tn + fp) else None,
        "flag_balanced_acc": float(0.5 * (rec + spec)) if (tp + fn) and (tn + fp) else None,
        "flag_false_infeasible_rate": float(fp / max(fp + tn, 1)) if (fp + tn) else None,
        "flag_missed_infeasible_rate": float(fn / max(tp + fn, 1)) if (tp + fn) else None,
    })
    # true release-region risk in flagged vs unflagged windows
    et = np.asarray(etrue if etrue is not None else [], dtype=float)
    if len(et) == len(ok):
        et = et[ok]
        m1, m0 = (f == 1) & np.isfinite(et), (f == 0) & np.isfinite(et)
        out["risk_when_flagged"] = float(et[m1].mean()) if m1.any() else None
        out["risk_when_not_flagged"] = float(et[m0].mean()) if m0.any() else None
    # detection delay: first flag after the oracle turns infeasible, per drift event
    for ev in getattr(ctx, "events", []) or []:
        ev_day = float(ev["day"] if isinstance(ev, dict) else ev.day)
        kind = ev["kind"] if isinstance(ev, dict) else ev.kind
        after = d_ok >= ev_day
        if not after.any():
            continue
        o_idx = np.nonzero(after & (o == 1))[0]
        f_idx = np.nonzero(after & (f == 1))[0]
        if len(o_idx) == 0:
            out[f"flag_delay_{kind}"] = None          # oracle never infeasible after it
            continue
        first_o = d_ok[o_idx[0]]
        later = f_idx[d_ok[f_idx] >= first_o] if len(f_idx) else np.array([], dtype=int)
        out[f"flag_delay_{kind}"] = float(d_ok[later[0]] - first_o) if len(later) else None
    return out


def _summarise(ctx, method, pol, action, obs, arrival, channel, demand_review,
               infeasible_flags, update_times, comp_series, eps,
               delay_scale_mult, disclose_mult, oracle_pd=False,
               oracle_flags=None, flag_days=None, flag_etrue=None) -> dict:
    n0, n = ctx.n_train, ctx.n
    sl = slice(n0, n)
    a = action[sl]
    y = ctx.y[sl].astype(int)
    amt = ctx.amount[sl]
    days = ctx.days[sl]
    topo = ctx.topo[sl]

    allowed = np.isin(a, ALLOW_ACTS)
    reviewed = np.isin(a, REVIEW_ACTS)
    blocked = np.isin(a, (ACT_BLOCK, ACT_OVF_BLOCK))
    explored = a == ACT_EXPLORE
    n_eval = len(a)

    for_overall = float(y[allowed].sum() / max(allowed.sum(), 1))
    fraud_dollars_allowed = float(amt[allowed & (y == 1)].sum())

    cst = C.costs_for(ctx.stream)
    cost = (
        float(np.sum(cst.fn_rate * amt[allowed & (y == 1)] + cst.fn_fixed))
        + float(np.sum(cst.fp_rate * amt[blocked & (y == 0)] + cst.fp_fixed))
        + cst.review * float(reviewed.sum())
    )
    cost_per_1k = 1000.0 * cost / n_eval

    # ---- per reporting block ----
    B = pol.metric_block
    nb = n_eval // B
    blk = {k: [] for k in ("for", "served", "demand", "overflow", "day", "allowed")}
    for k in range(nb):
        s2 = slice(k * B, (k + 1) * B)
        al = allowed[s2]
        blk["for"].append(float(y[s2][al].sum() / max(al.sum(), 1)))
        blk["served"].append(float(reviewed[s2].mean()))
        blk["demand"].append(float(demand_review[sl][s2].mean()))
        blk["overflow"].append(float(np.isin(a[s2], (ACT_OVF_ALLOW, ACT_OVF_BLOCK)).mean()))
        blk["day"].append(float(days[s2].mean()))
        blk["allowed"].append(int(al.sum()))
    for_series = np.asarray(blk["for"], dtype=float)
    day_series = np.asarray(blk["day"], dtype=float)
    dev = for_series - pol.alpha

    post = {}
    for ev in ctx.events:
        m = (day_series >= ev.day) & (day_series < ev.day + 10.0)
        post[ev.kind] = {
            "n_blocks": int(m.sum()),
            "mean_dev": float(dev[m].mean()) if m.any() else None,
            "max_dev": float(dev[m].max()) if m.any() else None,
            "mean_for": float(for_series[m].mean()) if m.any() else None,
            "recovery_days": _recovery_days(day_series, for_series, ev.day, pol.alpha),
        }

    # ---- by drift regime ----
    ev_cov, ev_con, ev_pri = ctx.events
    regimes = {
        "covariate": (days >= ev_cov.day) & (days < ev_cov.day + 30.0),
        "concept": (days >= ev_con.day) & (days < ev_con.day + ev_con.duration_days),
        "prior": (days >= ev_pri.day) & (days < ev_pri.day + ev_pri.duration_days),
    }
    regimes["quiet"] = ~(regimes["covariate"] | regimes["concept"] | regimes["prior"])
    by_regime = {}
    for name, m in regimes.items():
        al = allowed & m
        by_regime[name] = {
            "n": int(m.sum()),
            "for": float(y[al].sum() / max(al.sum(), 1)),
            "n_allowed": int(al.sum()),
            "n_fraud_allowed": int(y[al].sum()),
            "dollars_allowed": float(amt[al & (y == 1)].sum()),
            "review_demand": float(demand_review[sl][m].mean()) if m.any() else None,
        }

    # ---- how far off is the policy's own risk reading? ----
    # For each update, compare the estimate the policy acted on against the risk it
    # actually went on to realise over the next stretch of traffic.
    bias_pairs = []
    horizon = 4 * pol.update_every
    for c in comp_series:
        j = c["i"] - n0
        k = min(j + horizon, n_eval)
        if k - j < pol.update_every:
            continue
        al = allowed[j:k]
        if al.sum() < 200:
            continue
        realised = float(y[j:k][al].sum() / al.sum())
        if realised > 0:
            bias_pairs.append((c["e_t"], realised, c["day"],
                               c.get("e_true", float("nan"))))
    est = np.asarray([p[0] for p in bias_pairs], dtype=float)
    rea = np.asarray([p[1] for p in bias_pairs], dtype=float)
    tru = np.asarray([p[3] for p in bias_pairs], dtype=float)
    dayv = np.asarray([p[2] for p in bias_pairs], dtype=float)
    ok = np.isfinite(tru) & (tru > 0)
    # The ledger reports nothing for its first 180 days, so early updates see an
    # allow-region pool made up entirely of disclosed frauds. That cold start is a real
    # finding, but averaging it into a steady-state bias number would not be a fair
    # comparison, so the warm figure excludes it.
    warm = ok & (dayv >= ctx.days[n0] + C.MATURITY_DAYS * delay_scale_mult)
    cold = ok & ~warm
    est_bias = {
        "n_updates_scored": len(bias_pairs),
        "n_updates_warm": int(warm.sum()),
        "bias_ratio_warm": float(tru[warm].mean() / est[warm].mean())
        if warm.any() and est[warm].mean() > 0 else None,
        "lag_ratio_warm": float(rea[warm].mean() / tru[warm].mean())
        if warm.any() and tru[warm].mean() > 0 else None,
        "mean_estimate_warm": float(est[warm].mean()) if warm.any() else None,
        "mean_true_warm": float(tru[warm].mean()) if warm.any() else None,
        "mean_realised_warm": float(rea[warm].mean()) if warm.any() else None,
        "bias_ratio_coldstart": float(tru[cold].mean() / est[cold].mean())
        if cold.any() and est[cold].mean() > 0 else None,
        "mean_estimate": float(est.mean()) if len(est) else None,
        "mean_realised": float(rea.mean()) if len(rea) else None,
        "mean_true_window": float(tru[ok].mean()) if ok.any() else None,
        "bias_ratio_true_over_estimate": float(tru[ok].mean() / est[ok].mean())
        if ok.any() and est[ok].mean() > 0 else None,
        "lag_ratio_realised_over_true": float(rea[ok].mean() / tru[ok].mean())
        if ok.any() and tru[ok].mean() > 0 else None,
        "ratio_realised_over_estimate": float(rea.mean() / est.mean())
        if len(est) and est.mean() > 0 else None,
        "median_ratio": float(np.median(rea / np.maximum(est, 1e-12)))
        if len(est) else None,
    }

    topo_for = {}
    for t in C.TOPOLOGIES + ["D_drift_induced"]:
        m = (topo == t) & (y == 1)
        if m.sum() > 0:
            topo_for[t] = {
                "n": int(m.sum()),
                "miss_rate": float(allowed[m].mean()),
                "dollars_allowed": float(amt[m & allowed].sum()),
            }

    ut = np.asarray(update_times, dtype=float) * 1e6 / max(pol.update_every, 1)
    comp_last = comp_series[-1] if comp_series else {}

    return {
        "method": method,
        "oracle_pd": bool(oracle_pd),
        "stream": ctx.stream,
        "seed": ctx.seed,
        "n_eval": int(n_eval),
        "n_pos_eval": int(y.sum()),
        "alpha": pol.alpha,
        "budget": pol.budget,
        "epsilon": eps,
        "rho": pol.rho,
        "gamma_aci": pol.gamma_aci,
        "window_max_n": pol.window_max_n,
        "delay_scale_mult": delay_scale_mult,
        "disclose_mult": disclose_mult,
        "for_overall": for_overall,
        "for_ratio_to_alpha": for_overall / pol.alpha,
        "fraud_dollars_allowed": fraud_dollars_allowed,
        "fraud_dollars_total": float(amt[y == 1].sum()),
        "cost_per_1k": cost_per_1k,
        "review_served_rate": float(reviewed.mean()),
        "review_demand_rate": float(demand_review[sl].mean()),
        "review_demand_overshoot": float(max(0.0, demand_review[sl].mean() - pol.budget)),
        "review_demand_max_block": float(np.max(blk["demand"])) if nb else None,
        "frac_blocks_demand_over_budget": float(
            np.mean(np.asarray(blk["demand"]) > pol.budget + 1e-12)) if nb else None,
        "overflow_rate": float(np.isin(a, (ACT_OVF_ALLOW, ACT_OVF_BLOCK)).mean()),
        "block_rate": float(blocked.mean()),
        "explore_rate": float(explored.mean()),
        "n_explore": int(explored.sum()),
        "budget_rmse": float(np.sqrt(np.mean((np.asarray(blk["served"]) - pol.budget) ** 2)))
        if nb else None,
        "coverage_mean_dev": float(dev.mean()) if nb else None,
        "coverage_max_dev": float(dev.max()) if nb else None,
        "coverage_mean_abs_dev": float(np.abs(dev).mean()) if nb else None,
        "frac_blocks_over_alpha": float(np.mean(for_series > pol.alpha)) if nb else None,
        "infeasible_rate": float(np.mean(infeasible_flags)) if infeasible_flags else 0.0,
        "n_updates": len(infeasible_flags),
        **_score_flag(infeasible_flags, oracle_flags, flag_days, flag_etrue, ctx),
        "post_event": post,
        "by_regime": by_regime,
        "estimator_bias": est_bias,
        "topology_miss": topo_for,
        "update_us_p50": float(np.percentile(ut, 50)) if len(ut) else None,
        "update_us_p99": float(np.percentile(ut, 99)) if len(ut) else None,
        "calib_w_review_final": comp_last.get("w_review"),
        "calib_w_ledger_final": comp_last.get("w_ledger"),
        "calib_w_explore_final": comp_last.get("w_explore"),
        "tau_lo_final": comp_last.get("tau_lo"),
        "tau_hi_final": comp_last.get("tau_hi"),
        "series": {
            "day": [round(x, 4) for x in day_series.tolist()],
            "for": [round(x, 8) for x in for_series.tolist()],
            "served": [round(x, 8) for x in blk["served"]],
            "demand": [round(x, 8) for x in blk["demand"]],
        },
        "composition": comp_series,
    }


def _recovery_days(day_series, for_series, ev_day, alpha, streak: int = 3):
    idx = np.nonzero(day_series >= ev_day)[0]
    if len(idx) == 0:
        return None
    run = 0
    for j in idx:
        if for_series[j] <= alpha:
            run += 1
            if run >= streak:
                return float(day_series[j] - ev_day)
        else:
            run = 0
    return None
