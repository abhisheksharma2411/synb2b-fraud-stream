import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import config as C
from t13.simulate import prepare_stream, run_policy

stream = sys.argv[1] if len(sys.argv) > 1 else "synb2b"
ctx = prepare_stream(stream, seed=11)
pol = C.policy_for(stream)
print(f"stream={stream} alpha={pol.alpha} b={pol.budget} eps={pol.epsilon}")
hdr = (f"{'M':4s} {'FOR':>9s} {'/a':>6s} {'dem':>7s} {'srv':>7s} {'ovf':>7s} {'blk':>7s} "
       f"{'infeas':>7s} {'cost/1k':>9s} {'$allow':>10s} {'p50us':>7s} {'p99us':>7s} {'sec':>5s}")
print(hdr)
for m in C.METHODS:
    t0 = time.time()
    r = run_policy(ctx, m, pol)
    el = time.time() - t0
    print(f"{m:4s} {r['for_overall']:9.6f} {r['for_ratio_to_alpha']:6.2f} "
          f"{r['review_demand_rate']:7.4f} {r['review_served_rate']:7.4f} "
          f"{r['overflow_rate']:7.4f} {r['block_rate']:7.4f} {r['infeasible_rate']:7.3f} "
          f"{r['cost_per_1k']:9.2f} {r['fraud_dollars_allowed']:10.0f} "
          f"{(r['update_us_p50'] or 0):7.2f} {(r['update_us_p99'] or 0):7.2f} {el:5.1f}")
    br = r["by_regime"]; eb = r["estimator_bias"]
    g = lambda v, d=4: ("%.*f" % (d, v)) if v is not None else "-"
    print(f"      quiet={g(br['quiet']['for'],5)} dev={g(br['quiet']['for']-pol.alpha,5)} "
          f"cov={g(br['covariate']['for'],5)} con={g(br['concept']['for'],5)} "
          f"pri={g(br['prior']['for'],5)} | est={g(eb['mean_estimate'],5)} "
          f"true={g(eb['mean_true_window'],5)} BIAS={g(eb['bias_ratio_true_over_estimate'],3)} "
          f"LAG={g(eb['lag_ratio_realised_over_true'],3)} "
          f"| WARM bias={g(eb['bias_ratio_warm'],3)} lag={g(eb['lag_ratio_warm'],3)} "
          f"cold={g(eb['bias_ratio_coldstart'],3)}")
    if m == "M5":
        print("   calib weight shares  review=%.4f ledger=%.4f explore=%.4f  n_explore=%d"
              % (r['calib_w_review_final'] or 0, r['calib_w_ledger_final'] or 0,
                 r['calib_w_explore_final'] or 0, r['n_explore']))
        print("   post-event:", {k: (round(v['mean_dev'], 5) if v['mean_dev'] is not None else None,
                                     v['recovery_days']) for k, v in r['post_event'].items()})
        print("   topology miss:", {k: round(v['miss_rate'], 4) for k, v in r['topology_miss'].items()})
