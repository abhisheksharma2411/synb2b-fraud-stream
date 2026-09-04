"""Dump the internals of one M4/M5 update so the threshold search can be inspected."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import calib, config as C
from t13.simulate import (prepare_stream, run_policy, _place_thresholds,
                          _initial_thresholds, CH_REVIEW, CH_LEDGER, CH_EXPLORE)
import t13.simulate as S

ctx = prepare_stream("synb2b", seed=11)
pol = C.policy_for("synb2b")
grid, G = ctx.grid, len(ctx.grid)

orig = S.calib.crc_threshold
DUMP = []


def traced(risk, target):
    g = orig(risk, target)
    DUMP.append((float(target), int(g), risk.copy()))
    return g


S.calib.crc_threshold = traced
r = run_policy(ctx, "M4", pol)
S.calib.crc_threshold = orig

print("M4 FOR", r["for_overall"], "infeasible", r["infeasible_rate"])
comp = r["composition"]
for k in (0, len(DUMP) // 4, len(DUMP) // 2, 3 * len(DUMP) // 4, len(DUMP) - 1):
    tgt, g, risk = DUMP[k]
    c = comp[min(k, len(comp) - 1)]
    print(f"\nupdate {k}: day={c['day']:.1f} target(alpha_t)={tgt:.6f} g_star={g} "
          f"tau_lo={c['tau_lo']:.6f} infeas={c['infeasible']}")
    qs = [0, G // 8, G // 4, G // 2, int(0.75 * G), int(0.90 * G), G - 1]
    print("   risk curve at grid idx", qs)
    print("   ->", np.round(risk[qs], 6).tolist())
    print("   grid values ->", np.round(grid[qs], 6).tolist())
