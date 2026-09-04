import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import config as C
from t13.simulate import prepare_stream, run_policy

t0 = time.time()
ctx = prepare_stream("synb2b", seed=11, small=False)
print("prepare", round(time.time() - t0, 2), "s")
print("drift diag", ctx.drift_diag)
print("scorer", json.dumps(ctx.scorer, indent=None))
print("events", [e.to_dict() for e in ctx.events])
print("degradation", json.dumps(ctx.degradation, indent=1, default=str))

n0 = ctx.n_train
print("n_train", n0, "train end day", round(ctx.days[n0], 2), "total days", round(ctx.days[-1], 2))
ev_s, ev_y = ctx.scores[n0:], ctx.y[n0:]
print("eval pos rate", ev_y.mean())

# what is the achievable risk floor?
order = np.argsort(ev_s)
ys = ev_y[order].astype(float)
cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)
for q in (0.9500, 0.9600, 0.9675, 0.9700, 0.9800, 0.9875, 0.9900):
    k = int(q * len(ys)) - 1
    print(f"  allow mass {q:.4f} -> FOR floor {cum[k]:.6f}  (n_fraud allowed {ys[:k+1].sum():.0f})")

print("topology fraud counts (eval):")
import collections
print(collections.Counter(ctx.topo[n0:][ev_y == 1]))
