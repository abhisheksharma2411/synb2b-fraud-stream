"""Run the grid, aggregate over seeds with bootstrap intervals, emit results.json.

Two blocks of work. The method comparison runs every method on both streams. The
ablation moves one factor at a time around the stated default - not a factorial, which
would be a hundred times the compute for a result nobody reads.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

from . import calib, config as C
from .simulate import prepare_stream, run_policy

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"
)

# Headline metrics that get a bootstrap interval. Anything reported in the paper is
# in this list; anything in this list gets an interval.
CI_METRICS = [
    "for_overall", "for_ratio_to_alpha", "quiet_for", "quiet_dev",
    "review_demand_rate", "review_served_rate", "overflow_rate", "block_rate",
    "budget_rmse", "infeasible_rate", "cost_per_1k", "fraud_dollars_allowed",
    "coverage_mean_abs_dev", "coverage_max_dev", "frac_blocks_over_alpha",
    "bias_ratio_warm", "lag_ratio_warm", "update_us_p50", "update_us_p99",
    "explore_rate", "n_explore", "concept_for", "prior_for", "covariate_for",
    # flag validation against oracle feasibility
    "oracle_infeasible_rate", "flag_precision", "flag_recall", "flag_specificity",
    "flag_balanced_acc", "risk_when_flagged", "risk_when_not_flagged",
    "exh_flag_precision", "exh_flag_recall", "exh_flag_specificity",
    "exh_flag_balanced_acc", "exh_oracle_infeasible_rate", "oracle_agreement",
    "gap_overlap_mean", "gap_overlap_rel", "ess_median", "pd_mae_mean",
    "cohort_gap_days_mean",
]


def _flatten(r: dict) -> dict:
    out = {k: r.get(k) for k in (
        "for_overall", "for_ratio_to_alpha", "review_demand_rate", "review_served_rate",
        "overflow_rate", "block_rate", "budget_rmse", "infeasible_rate", "cost_per_1k",
        "fraud_dollars_allowed", "fraud_dollars_total", "coverage_mean_dev",
        "coverage_max_dev", "coverage_mean_abs_dev", "frac_blocks_over_alpha",
        "update_us_p50", "update_us_p99", "explore_rate", "n_explore",
        "review_demand_overshoot", "review_demand_max_block",
        "frac_blocks_demand_over_budget", "n_eval", "n_pos_eval",
        "calib_w_review_final", "calib_w_ledger_final", "calib_w_explore_final",
        # flag validation against oracle feasibility
        "flag_n_scored", "flag_undetermined", "oracle_infeasible_rate",
        "flag_tp", "flag_fp", "flag_fn", "flag_tn",
        "flag_precision", "flag_recall", "flag_specificity", "flag_balanced_acc",
        "flag_false_infeasible_rate", "flag_missed_infeasible_rate",
        "risk_when_flagged", "risk_when_not_flagged",
        "flag_delay_covariate", "flag_delay_concept", "flag_delay_prior",
        "cohort_gap_days_mean",
        # assumption-free (exhaustive) oracle + how much A2 matters
        "exh_flag_precision", "exh_flag_recall", "exh_flag_specificity", "exh_flag_balanced_acc", "exh_oracle_infeasible_rate", "exh_flag_n_scored", "exh_flag_tp", "exh_flag_fp", "exh_flag_fn", "exh_flag_tn", "mono_violation_mean", "oracle_agreement",
        # estimand gap, weight information, propensity error
        "gap_decay_mean", "gap_decay_p90", "gap_decay_rel", "gap_overlap_mean", "gap_overlap_p90", "gap_overlap_rel", "r_trail_mean", "r_overlap_mean", "ess_mean", "ess_median", "ess_p05", "w_max_norm_mean", "w_top1_share_mean", "trim_share_mean", "trim_fraud_share_mean", "pd_mae_mean", "pd_brier_mean", "pd_clip_share_mean",
    )}
    br = r.get("by_regime", {})
    for k in ("quiet", "covariate", "concept", "prior"):
        if k in br:
            out[f"{k}_for"] = br[k]["for"]
            out[f"{k}_dollars_allowed"] = br[k]["dollars_allowed"]
    out["quiet_dev"] = (br["quiet"]["for"] - r["alpha"]) if "quiet" in br else None
    eb = r.get("estimator_bias", {})
    for k in ("bias_ratio_warm", "lag_ratio_warm", "mean_estimate_warm",
              "mean_true_warm", "mean_realised_warm", "bias_ratio_coldstart",
              "ratio_realised_over_estimate", "n_updates_warm"):
        out[k] = eb.get(k)
    for ev, d in r.get("post_event", {}).items():
        out[f"post_{ev}_mean_dev"] = d.get("mean_dev")
        out[f"post_{ev}_max_dev"] = d.get("max_dev")
        out[f"post_{ev}_recovery_days"] = d.get("recovery_days")
    return out


def aggregate(runs: List[dict], label: str, extra: dict,
              with_series: bool = False) -> dict:
    flat = [_flatten(r) for r in runs]
    keys = sorted({k for f in flat for k in f})
    agg = {"label": label, "n_seeds": len(runs), **extra}
    for k in keys:
        vals = [f.get(k) for f in flat]
        num = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if not num:
            agg[k] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
            continue
        if k in CI_METRICS:
            m, lo, hi = calib.bootstrap_ci(num, C.BOOTSTRAP_RESAMPLES, seed=90210 + len(k))
        else:
            m, lo, hi = float(np.mean(num)), None, None
        agg[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(num),
                  "sd": float(np.std(num, ddof=1)) if len(num) > 1 else 0.0,
                  # every seed, so paired differences can be computed rather than
                  # inferred from whether two marginal intervals happen to overlap
                  "per_seed": [None if (v is None or not np.isfinite(v)) else float(v)
                               for v in vals]}
    # per-topology miss rates, averaged over seeds
    topo = {}
    for t in C.TOPOLOGIES + ["D_drift_induced"]:
        vs = [r["topology_miss"][t]["miss_rate"] for r in runs if t in r.get("topology_miss", {})]
        ns = [r["topology_miss"][t]["n"] for r in runs if t in r.get("topology_miss", {})]
        if vs:
            m, lo, hi = calib.bootstrap_ci(vs, C.BOOTSTRAP_RESAMPLES, seed=4242)
            topo[t] = {"miss_rate": m, "ci_lo": lo, "ci_hi": hi, "n_mean": float(np.mean(ns))}
    agg["topology_miss"] = topo
    if with_series:
        agg["series"] = _mean_series(runs)
        agg["composition"] = _mean_composition(runs)
    return agg


def _mean_series(runs: List[dict]) -> dict:
    """Per-block series averaged across seeds, truncated to the shortest run."""
    k = min(len(r["series"]["day"]) for r in runs)
    out = {}
    for field in ("day", "for", "served", "demand"):
        m = np.mean([r["series"][field][:k] for r in runs], axis=0)
        out[field] = [round(float(x), 8) for x in m]
    return out


def _mean_composition(runs: List[dict]) -> dict:
    """Calibration-set composition over time, averaged across seeds.

    Update cadence is identical across seeds but the first update waits on labels, so
    the series can differ in length by an update or two; align on the tail.
    """
    comps = [r["composition"] for r in runs if r["composition"]]
    if not comps:
        return {}
    k = min(len(c) for c in comps)
    if k == 0:
        return {}
    out = {}
    for field in ("day", "w_review", "w_ledger", "w_explore", "tau_lo", "tau_hi",
                  "alpha_t", "e_t", "e_true", "infeasible"):
        m = np.nanmean([[c[i].get(field, np.nan) for i in range(-k, 0)] for c in comps],
                       axis=0)
        out[field] = [None if not np.isfinite(x) else round(float(x), 8) for x in m]
    return out


def method_comparison(streams, seeds, small=False, log=print) -> dict:
    """Seed-outer, method-inner: building a stream context costs seconds (ULB more
    than that), so each seed's context is built once and every method runs against it.
    That also means all methods see identical potential outcomes, seed by seed."""
    out = {}
    # M5_oracle is the proposed method with the disclosure propensity handed to it:
    # an upper bound on what the correction can buy if the model were error-free.
    # M5_oracle hands the true disclosure propensity to the proposed method, bounding
    # what the correction can buy. M5_misaligned is the estimator as it stood before
    # cohort alignment: q fitted on releases at least 180 days old, r on analyst rows
    # from hours ago. It is kept as a measured ablation rather than deleted, because
    # the size of that error is the argument for aligning them.
    # (method, oracle_pd, cohort_aligned, cohort_fallback)
    # (method, oracle_pd, cohort_aligned, cohort_fallback, joint_mode)
    variants = ([(m, False, True, True, "") for m in C.METHODS]
                + [("M5", True, True, True, ""),      # oracle p_d, diagnostic only
                   ("M5", False, False, True, ""),    # legacy: mismatched cohorts
                   ("M5", False, True, False, ""),    # strict match, abstains when thin
                   ("M5", False, True, True, "static"),   # joint likelihood
                   ("M5", False, True, True, "slow")])
    for stream in streams:
        pol = C.policy_for(stream)
        collected: Dict[str, list] = {}
        for sd in seeds:
            t0 = time.time()
            ctx = prepare_stream(stream, seed=sd, small=small)
            log(f"  {stream}/seed{sd} prepared in {time.time() - t0:.1f}s "
                f"(n={ctx.n}, pos={int(ctx.y.sum())})")
            for method, oracle, aligned, fb, jm in variants:
                key = ("M5_oracle" if oracle
                       else f"M5_joint_{jm}" if jm
                       else "M5_legacy" if not aligned
                       else method if fb else "M5_strictmatch")
                t1 = time.time()
                r = run_policy(ctx, method, pol, oracle_pd=oracle,
                               cohort_aligned=aligned, cohort_fallback=fb,
                               joint_mode=jm)
                collected.setdefault(key, []).append(r)
                log(f"    {key:10s} {time.time() - t1:5.1f}s FOR={r['for_overall']:.6f}")
        out[stream] = {
            key: aggregate(runs, f"{stream}/{key}",
                           {"stream": stream, "method": key,
                            "method_label": C.METHOD_LABELS.get(
                                key, {"M5_oracle": "M5 with oracle propensity",
                                      "M5_misaligned": "M5, mismatched cohorts"}
                                .get(key, key)),
                            **pol.as_dict()},
                           with_series=True)
            for key, runs in collected.items()
        }
    return out


ABLATIONS = {
    "epsilon": [0.0, 0.05, 0.12, 0.25, 0.40, 0.60, 0.80],
    "rho": [0.9800, 0.9900, 0.9950, 0.9975, 0.9990, 1.0000],
    "budget": [0.0075, 0.0125, 0.0200, 0.0300, 0.0450],
    "gamma_aci": [0.0045, 0.0090, 0.0180, 0.0360, 0.0720],
    "window_max_n": [12000, 24000, 45000, 70000],
    "alpha": None,          # filled per stream, relative to the stream default
    "delay_scale_mult": [0.0, 0.25, 0.50, 1.0, 2.0],
    "disclose_mult": [1.0, 1.4, 1.8, 2.4],   # 2.4 saturates p_disclose near 1
    "drift_severity": [0.0, 0.5, 1.0, 1.5, 2.0],
}


def ablation(stream, seeds, small=False, log=print) -> dict:
    base_alpha = C.STREAM_ALPHA[stream]
    grid = dict(ABLATIONS)
    grid["alpha"] = [round(base_alpha * m, 8) for m in (0.60, 0.80, 1.0, 1.35, 1.80)]
    out = {}
    for factor, levels in grid.items():
        rows = []
        for lv in levels:
            runs = []
            for sd in seeds:
                sev = lv if factor == "drift_severity" else 1.0
                ctx = prepare_stream(stream, seed=sd, small=small, drift_severity=sev)
                kw = {}
                if factor in ("epsilon", "rho", "budget", "gamma_aci",
                              "window_max_n", "alpha"):
                    kw[factor] = lv
                pol = C.policy_for(stream, **kw)
                runs.append(run_policy(
                    ctx, "M5", pol,
                    delay_scale_mult=lv if factor == "delay_scale_mult" else 1.0,
                    disclose_mult=lv if factor == "disclose_mult" else 1.0))
            rows.append(aggregate(runs, f"{stream}/M5/{factor}={lv}",
                                  {"stream": stream, "method": "M5",
                                   "factor": factor, "level": lv}))
            log(f"  ablation {factor}={lv}: FOR={rows[-1]['for_overall']['mean']:.6f} "
                f"bias={rows[-1]['bias_ratio_warm']['mean']}")
        out[factor] = rows
    return out


def provenance() -> dict:
    import importlib.metadata as md

    def v(p):
        try:
            return md.version(p)
        except Exception:
            return None

    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sh("sysctl -n machdep.cpu.brand_string") or platform.processor(),
        "n_cpu": os.cpu_count(),
        "packages": {p: v(p) for p in
                     ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow")},
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(small: bool = False, streams=("synb2b", "ulb"), with_ablation: bool = True):
    t_start = time.time()
    seeds = C.SEEDS[:2] if small else C.SEEDS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    log(f"== method comparison ==  streams={streams} seeds={seeds} small={small}")
    methods = method_comparison(streams, seeds, small=small, log=log)

    stream_facts = {}
    for stream in streams:
        c = prepare_stream(stream, seed=seeds[0], small=small)
        stream_facts[stream] = {
            "n": c.n, "n_train": c.n_train, "n_eval": c.n - c.n_train,
            "span_days": float(c.days[-1] - c.days[0]),
            "train_end_day": float(c.days[c.n_train]),
            "scorer": c.scorer,
            "degradation": c.degradation,
            "drift_diag": c.drift_diag,
            "alpha": C.STREAM_ALPHA[stream],
            "costs": C.costs_for(stream).__dict__,
            "events": [e.to_dict() for e in c.events],
        }

    abl = {}
    if with_ablation:
        log("== ablation (synb2b) ==")
        abl["synb2b"] = ablation("synb2b", seeds, small=small, log=log)

    payload = {
        "schema": "t13-results/1",
        "small": small,
        "seeds": seeds,
        "policy_default": C.PolicyConfig().as_dict(),
        "stream_alpha": C.STREAM_ALPHA,
        "drift_params": C.DRIFT_PARAMS,
        "stream_facts": stream_facts,
        "methods": methods,
        "ablation": abl,
        "wall_clock_s": round(time.time() - t_start, 2),
        "provenance": provenance(),
    }
    path = os.path.join(RESULTS_DIR, "results_small.json" if small else "results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    log(f"wrote {path}  ({time.time() - t_start:.1f}s)")

    ev_path = os.path.join(RESULTS_DIR, "drift_events.json")
    with open(ev_path, "w") as f:
        json.dump({s: stream_facts[s]["events"] for s in streams}, f, indent=1)
        f.write("\n")
    log(f"wrote {ev_path}")
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--streams", default="synb2b,ulb")
    a = ap.parse_args()
    main(small=a.small, streams=tuple(a.streams.split(",")),
         with_ablation=not a.no_ablation)
