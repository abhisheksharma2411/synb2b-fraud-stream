import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import config as C
from t13.simulate import prepare_stream, run_policy

ctx = prepare_stream("synb2b", seed=11)
for mci in (int(sys.argv[1]) if len(sys.argv) > 1 else 400,):
    pol = C.policy_for("synb2b", min_calib_items=mci)
    for m in ("M4", "M5"):
        r = run_policy(ctx, m, pol)
        dem = np.asarray(r["series"]["demand"])
        day = np.asarray(r["series"]["day"])
        print(f"mci={mci} {m}: updates={r['n_updates']} demand mean={dem.mean():.4f} "
              f"max={dem.max():.4f} overflow={r['overflow_rate']:.4f} FOR={r['for_overall']:.5f}")
        print("   demand by decile of stream:",
              np.round([dem[i::10].mean() for i in range(10)], 4).tolist())
        comp = r["composition"]
        if comp:
            print(f"   first update at day {comp[0]['day']:.1f} (idx {comp[0]['i']}), "
                  f"n_review={comp[0]['n_review']} n_ledger={comp[0]['n_ledger']} "
                  f"n_explore={comp[0]['n_explore']}")
            k = len(comp) // 2
            print(f"   mid  update at day {comp[k]['day']:.1f} tau_lo={comp[k]['tau_lo']:.5f} "
                  f"tau_hi={comp[k]['tau_hi']:.5f} alpha_t={comp[k]['alpha_t']:.6f} "
                  f"w_rev={comp[k]['w_review']:.3f} w_led={comp[k]['w_ledger']:.3f}")
