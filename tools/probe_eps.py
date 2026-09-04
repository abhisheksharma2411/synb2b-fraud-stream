import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import config as C
from t13.simulate import prepare_stream, run_policy

stream = sys.argv[1] if len(sys.argv) > 1 else "synb2b"
oracle = len(sys.argv) > 2
ctx = prepare_stream(stream, seed=11)


def f(v, d=3):
    return ("%.*f" % (d, v)) if v is not None and np.isfinite(v) else "  -  "


print(f"{'eps':>6s} {'FOR':>9s} {'quietFOR':>9s} {'est':>9s} {'true':>9s} {'real':>9s} "
      f"{'BIAS':>6s} {'LAG':>6s} {'nexp':>6s} {'dem':>7s} {'blk':>7s} {'infeas':>7s} "
      f"{'cost/1k':>9s}")
for eps in (0.0, 0.05, 0.12, 0.25, 0.40, 0.60, 0.80):
    pol = C.policy_for(stream, epsilon=eps)
    r = run_policy(ctx, "M5", pol, oracle_pd=oracle)
    eb, br = r["estimator_bias"], r["by_regime"]
    print(f"{eps:6.2f} {r['for_overall']:9.6f} {br['quiet']['for']:9.6f} "
          f"{f(eb['mean_estimate'], 6):>9s} {f(eb['mean_true_window'], 6):>9s} "
          f"{f(eb['mean_realised'], 6):>9s} "
          f"{f(eb['bias_ratio_true_over_estimate']):>6s} "
          f"{f(eb['lag_ratio_realised_over_true']):>6s} "
          f"{r['n_explore']:6d} {r['review_demand_rate']:7.4f} {r['block_rate']:7.4f} "
          f"{r['infeasible_rate']:7.3f} {r['cost_per_1k']:9.1f}")
