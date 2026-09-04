"""Where can (alpha, b) actually live? Proposition 1 evaluated empirically, per regime."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from t13 import config as C
from t13.simulate import prepare_stream

for stream in ("synb2b", "ulb"):
    ctx = prepare_stream(stream, seed=11)
    n0 = ctx.n_train
    s, y, d = ctx.scores[n0:], ctx.y[n0:].astype(int), ctx.days[n0:]
    ev_con, ev_pri = ctx.events[1], ctx.events[2]
    win_con = (d >= ev_con.day) & (d < ev_con.day + ev_con.duration_days)
    win_pri = (d >= ev_pri.day) & (d < ev_pri.day + ev_pri.duration_days)
    ev_cov = ctx.events[0]
    win_cov = (d >= ev_cov.day) & (d < ev_cov.day + 30.0)
    quiet = ~(win_con | win_pri | win_cov)

    print("=" * 78)
    print(stream, "eval n", len(y), "pos", y.sum(), "rate", round(float(y.mean()), 6))
    print("  drift diag", ctx.drift_diag)
    print("  scorer", {k: round(v, 5) if isinstance(v, float) else v for k, v in ctx.scorer.items()})

    def floor(mask, allow_mass):
        ss, yy = s[mask], y[mask]
        o = np.argsort(ss, kind="mergesort")
        ys = yy[o].astype(float)
        k = max(int(allow_mass * len(ys)) - 1, 0)
        return float(np.cumsum(ys)[k] / (k + 1))

    for name, m in (("quiet", quiet), ("covariate+30d", win_cov),
                    ("concept", win_con), ("prior", win_pri), ("all", np.ones_like(quiet))):
        if m.sum() < 200:
            continue
        rate = float(y[m].mean())
        ap = average_precision_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else None
        line = f"  {name:14s} n={int(m.sum()):6d} rate={rate:.6f} apr={ap:.4f} | floors:"
        for am in (0.9675, 0.9800, 0.9900):
            line += f"  q{am:.4f}={floor(m, am):.6f}"
        print(line)

    # capture at the block cap
    o = np.argsort(s, kind="mergesort")[::-1]
    for topk in (0.0125, 0.0325, 0.0500):
        k = int(topk * len(s))
        print(f"  top {topk:.4f} of traffic holds {y[o][:k].sum()}/{y.sum()} frauds "
              f"({y[o][:k].sum() / max(y.sum(), 1):.4f})")
