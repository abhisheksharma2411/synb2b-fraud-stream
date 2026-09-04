"""Solve the drift-channel constants against stated targets, per stream.

Targets, expressed relative to each stream's own quiet-regime fraud rate so that the
same three events mean the same thing on a 2.1% B2B book and a 0.23% card portfolio:

  quiet regime   the drift channel adds at most 5% on top of the local base rate
  concept window local fraud rate reaches 2.2x the quiet rate, and the new fraud sits
                 on established counterparties rather than novel ones
  prior window   local fraud rate reaches 3.4x the quiet rate

Writes src/t13/drift_params.json. Re-run with `make drift-params`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from scipy.optimize import brentq

from t13 import config as C
from t13.data import load_stream, build_synb2b_features, build_ulb_features
from t13.drift import plan_events, apply_covariate_shift, _zscore, novelty_score

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "t13", "drift_params.json")
QUIET_ADD = 0.05
CONCEPT_MULT = 2.2
PRIOR_MULT = 3.4

params = {}
for stream in ("synb2b", "ulb"):
    raw, _, _ = load_stream(stream)
    builder = build_synb2b_features if stream == "synb2b" else build_ulb_features
    n = len(raw)
    n_train = int(round(C.TRAIN_FRACTION * n))
    days = raw["day"].to_numpy(float)
    ev_cov, ev_con, ev_pri = plan_events(days, n_train, stream=stream)

    shifted = apply_covariate_shift(raw, ev_cov)
    feats = builder(shifted)
    nu = novelty_score(stream, feats, n_train)
    zeta = _zscore(np.log1p(shifted["amount"].to_numpy(float)), n_train)
    base = shifted["base_label"].to_numpy(float)

    ev = slice(n_train, n)
    d = days[n_train:]
    win_con = (d >= ev_con.day) & (d < ev_con.day + ev_con.duration_days)
    win_pri = (d >= ev_pri.day) & (d < ev_pri.day + ev_pri.duration_days)
    quiet = ~(win_con | win_pri)
    free = base[ev] == 0
    q_rate = float(base[ev][quiet].mean())

    shape = np.zeros(len(d))
    dd = d - ev_pri.day
    up = (dd >= 0) & (dd < C.PRIOR_RAMP_DAYS)
    shape[up] = dd[up] / C.PRIOR_RAMP_DAYS
    dn = (dd >= C.PRIOR_RAMP_DAYS) & (dd < C.PRIOR_RAMP_DAYS + C.PRIOR_TAIL_DAYS)
    shape[dn] = np.exp(-np.log(2.0) * (dd[dn] - C.PRIOR_RAMP_DAYS) / C.PRIOR_HALFLIFE_DAYS)

    def expect(mask, a0, beta, lam_peak=0.0, use_shape=False):
        m = mask & free
        lam = lam_peak * shape[m] if use_shape else lam_peak
        logit = a0 + C.LGP_GAMMA_AMT * zeta[ev][m] + beta * nu[ev][m] + lam
        return float(np.sum(1.0 / (1.0 + np.exp(-logit))))

    tgt_quiet = QUIET_ADD * q_rate * quiet.sum()
    a0 = brentq(lambda a: expect(quiet, a, C.LGP_BETA_BASE) - tgt_quiet, -22.0, -1.0)

    tgt_con = (CONCEPT_MULT * q_rate - float(base[ev][win_con].mean())) * win_con.sum()
    beta_f = brentq(lambda b: expect(win_con, a0, b) - tgt_con, -30.0, 0.0)

    tgt_pri = (PRIOR_MULT * q_rate - float(base[ev][win_pri].mean())) * win_pri.sum()
    lam_p = brentq(lambda L: expect(win_pri, a0, C.LGP_BETA_BASE, L, True) - tgt_pri, 0.0, 30.0)

    m = win_con & free
    lg = a0 + C.LGP_GAMMA_AMT * zeta[ev][m] + beta_f * nu[ev][m]
    pp = 1.0 / (1.0 + np.exp(-lg))
    est_share = float(np.sum(pp[nu[ev][m] < 0]) / max(np.sum(pp), 1e-9))

    params[stream] = {
        "a0": round(float(a0), 4),
        "beta_flipped": round(float(beta_f), 4),
        "lambda_peak": round(float(lam_p), 4),
        "_quiet_base_rate": round(q_rate, 6),
        "_expected_quiet_induced": round(expect(quiet, a0, C.LGP_BETA_BASE), 1),
        "_expected_concept_induced": round(expect(win_con, a0, beta_f), 1),
        "_expected_prior_induced": round(expect(win_pri, a0, C.LGP_BETA_BASE, lam_p, True), 1),
        "_concept_established_share": round(est_share, 4),
        "_concept_target_rate": round(CONCEPT_MULT * q_rate, 6),
        "_prior_target_rate": round(PRIOR_MULT * q_rate, 6),
    }
    print(stream, json.dumps(params[stream], indent=2))

with open(OUT, "w") as f:
    json.dump(params, f, indent=2)
    f.write("\n")
print("wrote", OUT)
