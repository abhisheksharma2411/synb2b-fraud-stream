"""results/SUMMARY.md - the headline numbers in plain text, with provenance."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from t13 import config as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def cell(d, key, digits=6, scale=1.0, ci=True):
    c = d.get(key)
    if not c or c.get("mean") is None:
        return "n/a"
    m = c["mean"] * scale
    if ci and c.get("ci_lo") is not None:
        return f"{m:.{digits}f} [{c['ci_lo'] * scale:.{digits}f}, {c['ci_hi'] * scale:.{digits}f}]"
    return f"{m:.{digits}f}"


def main(small=False):
    path = os.path.join(RES, "results_small.json" if small else "results.json")
    with open(path) as f:
        R = json.load(f)
    L = []
    A = L.append
    A("# SynB2B-Fraud-Stream / T13 - headline results")
    A("")
    A("Budget-constrained conformal risk control for fraud decisioning under drift and")
    A("endogenous label arrival. Every interval is a percentile bootstrap over the seed")
    A(f"replicates, {C.BOOTSTRAP_RESAMPLES:,} resamples, 95%.")
    A("")
    A(f"Seeds: {R['seeds']}.  Reduced-size pass: {R['small']}.")
    A("")

    for stream, facts in R["stream_facts"].items():
        alpha = facts["alpha"]
        A(f"## Stream: {stream}")
        A("")
        A(f"- rows {facts['n']:,}, evaluation stream {facts['n_eval']:,}, "
          f"span {facts['span_days']:.2f} days, frozen-scorer split ends day "
          f"{facts['train_end_day']:.2f}")
        A(f"- scorer AUC-PR {facts['scorer']['auc_pr_eval']:.4f}, "
          f"AUC-ROC {facts['scorer']['auc_roc_eval']:.4f}, "
          f"positives {facts['scorer']['n_pos_eval']:,} "
          f"({facts['scorer']['pos_rate_eval']:.6f} of the evaluation stream)")
        A(f"- target alpha = {alpha}, review budget b = {R['policy_default']['budget']}, "
          f"hold allowance = {R['policy_default']['block_cap']}, "
          f"exploration share eps = {R['policy_default']['epsilon']}")
        d = facts["drift_diag"]
        A(f"- drift channel: quiet rate {d['quiet_rate']:.6f}, "
          f"concept-window rate {d['concept_rate']:.6f}, "
          f"prior-window rate {d['prior_rate']:.6f}; "
          f"{d['n_drift_induced']:,} rows added, benchmark labels untouched")
        for ev in facts["events"]:
            deg = facts["degradation"].get(ev["kind"], {})
            rd = deg.get("rel_drop")
            A(f"  - {ev['kind']:9s} at index {ev['index']:,} (day {ev['day']:.2f}): "
              f"AUC-PR {deg.get('auc_pr_pre')} -> {deg.get('auc_pr_post')}"
              + (f"  ({rd:+.4f} relative)" if rd is not None else ""))
        A("")
        A("| method | realised FOR | quiet-regime FOR | review demand | hold rate | "
          "infeasible | cost/1k | bias ratio |")
        A("|---|---|---|---|---|---|---|---|")
        for key, m in R["methods"][stream].items():
            A(f"| {key} ({m['method_label']}) | {cell(m, 'for_overall')} | "
              f"{cell(m, 'quiet_for')} | {cell(m, 'review_demand_rate', 5)} | "
              f"{cell(m, 'block_rate', 5)} | {cell(m, 'infeasible_rate', 4)} | "
              f"{cell(m, 'cost_per_1k', 2)} | {cell(m, 'bias_ratio_warm', 4)} |")
        A("")
        m5 = R["methods"][stream]["M5"]
        A(f"- M5 calibration update: p50 {cell(m5, 'update_us_p50', 2)} us, "
          f"p99 {cell(m5, 'update_us_p99', 2)} us per transaction "
          f"(amortised over {R['policy_default']['update_every']} transactions)")
        A(f"- M5 fraud value allowed through: {cell(m5, 'fraud_dollars_allowed', 0)} "
          f"of {m5['fraud_dollars_total']['mean']:,.0f} total")
        if m5.get("topology_miss"):
            A("- per-topology miss rate under M5 (share of that topology's fraud auto-allowed):")
            for t, v in sorted(m5["topology_miss"].items()):
                A(f"  - {t}: {v['miss_rate']:.4f} "
                  f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] over {v['n_mean']:.1f} items")
        A("")

    if R.get("ablation"):
        A("## Ablation (SynB2B-Fraud, M5, one factor at a time)")
        A("")
        for factor, rows in R["ablation"]["synb2b"].items():
            A(f"### {factor}")
            A("")
            A("| level | realised FOR | quiet FOR | bias ratio | hold rate | infeasible | cost/1k |")
            A("|---|---|---|---|---|---|---|")
            for r in rows:
                A(f"| {r['level']} | {cell(r, 'for_overall')} | {cell(r, 'quiet_for')} | "
                  f"{cell(r, 'bias_ratio_warm', 4)} | {cell(r, 'block_rate', 5)} | "
                  f"{cell(r, 'infeasible_rate', 4)} | {cell(r, 'cost_per_1k', 2)} |")
            A("")

    p = R["provenance"]
    A("## Provenance")
    A("")
    A(f"- Python {p['python']} on {p['platform']} ({p['machine']})")
    A(f"- CPU: {p['cpu']}, {p['n_cpu']} logical cores")
    A("- packages: " + ", ".join(f"{k} {v}" for k, v in sorted(p["packages"].items()) if v))
    A(f"- git commit at the time of the run: {p['git_commit']}"
      + ("  (working tree dirty)" if p["git_dirty"] else ""))
    A("  This is the commit the grid executed at, not the commit that carries this file;"
      " later commits added the paper and did not recompute any number here.")
    A(f"- total wall clock: {R['wall_clock_s']:.1f} s")
    A(f"- generated: {p['utc']}")
    A(f"- bootstrap: {C.BOOTSTRAP_RESAMPLES:,} resamples, percentile method")
    A("")

    out = os.path.join(RES, "SUMMARY_small.md" if small else "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    main(**vars(ap.parse_args()))
