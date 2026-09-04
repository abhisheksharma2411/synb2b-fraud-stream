"""How steep is R(tau) = E[y | s <= tau]? If it is flat, threshold choice cannot matter."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from t13 import config as C
from t13.simulate import prepare_stream

for stream in ("synb2b", "ulb"):
    ctx = prepare_stream(stream, seed=11)
    n0 = ctx.n_train
    s, y, d = ctx.scores[n0:], ctx.y[n0:].astype(int), ctx.days[n0:]
    ev_con, ev_pri, ev_cov = ctx.events[1], ctx.events[2], ctx.events[0]
    win = ((d >= ev_con.day) & (d < ev_con.day + ev_con.duration_days)) | \
          ((d >= ev_pri.day) & (d < ev_pri.day + ev_pri.duration_days)) | \
          ((d >= ev_cov.day) & (d < ev_cov.day + 30.0))
    print("=" * 76)
    print(stream, "quiet ROC", round(roc_auc_score(y[~win], s[~win]), 4),
          "APR", round(average_precision_score(y[~win], s[~win]), 4),
          "rate", round(float(y[~win].mean()), 6))
    for label, m in (("quiet", ~win), ("drifted", win), ("all", np.ones_like(win))):
        ss, yy = s[m], y[m]
        o = np.argsort(ss, kind="mergesort")
        cum = np.cumsum(yy[o].astype(float))
        row = f"  {label:8s} R(tau) by allow mass: "
        for q in (0.85, 0.90, 0.9224, 0.95, 0.9674, 0.98, 0.99):
            k = max(int(q * len(yy)) - 1, 0)
            row += f"{q:.4f}={cum[k] / (k + 1):.5f}  "
        print(row)
