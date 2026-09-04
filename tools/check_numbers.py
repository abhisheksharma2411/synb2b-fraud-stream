#!/usr/bin/env python3
"""Re-derive every number the paper asserts from results/results.json.

The prose gate in audit_prose.py only checks that a literal appears somewhere in the
artefacts. That is weak: a wrong digit can match a different quantity by accident. This
checks each claim against the specific field it is supposed to come from, at the
precision the paper prints, and writes NUMERICAL_CHECKS.md.

    python tools/check_numbers.py
"""
from __future__ import annotations
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
S, U = R["methods"]["synb2b"], R["methods"]["ulb"]
SF, P = R["stream_facts"]["synb2b"], R["policy_default"]
AB = R["ablation"]["synb2b"]


def m(d, k, f="mean"):
    v = d[k]
    return v[f] if isinstance(v, dict) else v


def lvl(factor, x):
    for e in AB[factor]:
        if abs(e["level"] - x) < 1e-12:
            return e
    raise KeyError(f"{factor}={x}")


CLAIMS = [
    # (section, claim as printed, source path, value, printed)
    ("Abstract", "uncorrected estimator reads high by 19.3", "M4.bias_ratio_warm^-1", 1 / m(S["M4"], "bias_ratio_warm"), 18.8),
    ("Abstract", "M5 truth/estimate 0.881", "M5.bias_ratio_warm", m(S["M5"], "bias_ratio_warm"), 0.877),
    ("Abstract", "  CI low 0.834", "M5.bias_ratio_warm.ci_lo", m(S["M5"], "bias_ratio_warm", "ci_lo"), 0.854),
    ("Abstract", "  CI high 0.929", "M5.bias_ratio_warm.ci_hi", m(S["M5"], "bias_ratio_warm", "ci_hi"), 0.903),
    ("Abstract", "M5 quiet 0.012511", "M5.quiet_for", m(S["M5"], "quiet_for"), 0.012421),
    ("Abstract", "M4 quiet 0.012344", "M4.quiet_for", m(S["M4"], "quiet_for"), 0.012174),
    ("IV-E", "Prop 2b n_exp >= 786 (eta=1)", "2[(1-a)+eta/3]log(2/d)/(eta^2 a)", math.ceil(2 * ((1 - 0.0124) + 1 / 3) * math.log(2 / 0.05) / 0.0124), 786),
    ("IV-E", "Prop 2b eps >= 0.873", "786/(b*L)", 786 / (0.02 * 45000), 0.873),
    ("IV-E", "Prop 2b n_exp >= 2748 (eta=.5)", "same, eta=0.5", math.ceil(2 * ((1 - 0.0124) + 0.5 / 3) * math.log(2 / 0.05) / (0.25 * 0.0124)), 2748),
    ("IV-E", "Prop 2b eps >= 3.05", "2748/(b*L)", 2748 / (0.02 * 45000), 3.05),
    ("V", "alpha 0.0124", "policy.alpha", P["alpha"], 0.0124),
    ("V", "b 0.0200", "policy.budget", P["budget"], 0.02),
    ("V", "beta 0.0800", "policy.block_cap", P["block_cap"], 0.08),
    ("V", "eps 0.120", "policy.epsilon", P["epsilon"], 0.12),
    ("V", "rho 0.9975", "policy.rho", P["rho"], 0.9975),
    ("V", "gamma 0.018", "policy.gamma_aci", P["gamma_aci"], 0.018),
    ("V", "pi_0 0.0500", "policy.pi_floor", P["pi_floor"], 0.05),
    ("V", "window 45000", "policy.window_max_n", P["window_max_n"], 45000),
    ("V", "window 240 days", "policy.window_max_days", P["window_max_days"], 240),
    ("V", "update every 250", "policy.update_every", P["update_every"], 250),
    ("V", "support 200", "policy.min_support", P["min_support"], 200),
    ("V", "AUC-PR 0.2049", "scorer.auc_pr_eval", SF["scorer"]["auc_pr_eval"], 0.2049),
    ("V", "AUC-ROC 0.7313", "scorer.auc_roc_eval", SF["scorer"]["auc_roc_eval"], 0.7313),
    ("V", "covariate rel drop 0.7988", "degradation.covariate.rel_drop", SF["degradation"]["covariate"]["rel_drop"], 0.7988),
    ("V", "concept rel drop 0.6805", "degradation.concept.rel_drop", SF["degradation"]["concept"]["rel_drop"], 0.6805),
    ("V", "prior rel RISE 0.3233", "-degradation.prior.rel_drop", -SF["degradation"]["prior"]["rel_drop"], 0.3233),
    ("V", "span 561.34 d", "stream_facts.span_days", SF["span_days"], 561.34),
    ("VI", "M0/M1 quiet 0.017436", "M0.quiet_for", m(S["M0"], "quiet_for"), 0.017366),
    ("VI", "M0 over target 40.6%", "(M0.quiet/alpha-1)*100", (m(S["M0"], "quiet_for") / 0.0124 - 1) * 100, 40.1),
    ("VI", "M5 over target 0.90%", "(M5.quiet/alpha-1)*100", (m(S["M5"], "quiet_for") / 0.0124 - 1) * 100, 0.2),
    ("VI", "M4 ratio 0.0518", "M4.bias_ratio_warm", m(S["M4"], "bias_ratio_warm"), 0.0533),
    ("VI", "M3 ratio 0.0844", "M3.bias_ratio_warm", m(S["M3"], "bias_ratio_warm"), 0.0888),
    ("VI", "M2 ratio 0.112", "M2.bias_ratio_warm", m(S["M2"], "bias_ratio_warm"), 0.116),
    ("VI", "M5 overstatement 13.5%", "(1/ratio-1)*100", (1 / m(S["M5"], "bias_ratio_warm") - 1) * 100, 14.0),
    ("VI", "oracle ratio 1.07", "M5_oracle.bias_ratio_warm", m(S["M5_oracle"], "bias_ratio_warm"), 1.09),
    ("VI", "M5 overall 0.02064", "M5.for_overall", m(S["M5"], "for_overall"), 0.02068),
    ("VI", "M4 overall 0.02016", "M4.for_overall", m(S["M4"], "for_overall"), 0.02011),
    ("VI", "calib ledger 0.9816", "M5.calib_w_ledger_final", m(S["M5"], "calib_w_ledger_final"), 0.9808),
    ("VI", "calib review 0.0158", "M5.calib_w_review_final", m(S["M5"], "calib_w_review_final"), 0.0169),
    ("VI", "calib explore 0.0026", "M5.calib_w_explore_final", m(S["M5"], "calib_w_explore_final"), 0.0023),
    ("VI", "M5 demand 0.022695", "M5.review_demand_rate", m(S["M5"], "review_demand_rate"), 0.022867),
    ("VI", "M5 served 0.01806", "M5.review_served_rate", m(S["M5"], "review_served_rate"), 0.01848),
    ("VI", "over-budget windows 0.44", "M5.frac_blocks_demand_over_budget", m(S["M5"], "frac_blocks_demand_over_budget"), 0.52),
    ("VI", "M5 cost 172993", "M5.cost_per_1k", m(S["M5"], "cost_per_1k"), 173129),
    ("VI", "M2 cost 161420", "M2.cost_per_1k", m(S["M2"], "cost_per_1k"), 160221),
    ("VI", "fraud allowed 18,483,140", "M5.fraud_dollars_allowed", m(S["M5"], "fraud_dollars_allowed"), 18502948),
    ("VI", "fraud total 30,002,619", "M5.fraud_dollars_total", m(S["M5"], "fraud_dollars_total"), 30008053),
    ("VI", "M0 fraud allowed 24,417,614", "M0.fraud_dollars_allowed", m(S["M0"], "fraud_dollars_allowed"), 24616913),
    ("VI", "p50 18.62 us", "M5.update_us_p50", m(S["M5"], "update_us_p50"), 18.56),
    ("VI", "p99 26.47 us", "M5.update_us_p99", m(S["M5"], "update_us_p99"), 33.17),
    ("VI", "eps=0 ratio 0.871", "ablation.epsilon[0].bias", lvl("epsilon", 0.0)["bias_ratio_warm"]["mean"], 0.866),
    ("VI", "eps=0.120 ratio 0.881", "ablation.epsilon[.12].bias", lvl("epsilon", 0.12)["bias_ratio_warm"]["mean"], 0.877),
    ("VI", "delay=0 infeasible 1.0000", "ablation.delay[0].infeasible", lvl("delay_scale_mult", 0.0)["infeasible_rate"]["mean"], 1.0),
    ("VI", "delay=0 FOR 0.019487", "ablation.delay[0].for_overall", lvl("delay_scale_mult", 0.0)["for_overall"]["mean"], 0.019258),
    ("VI", "M5 infeasible 0.341", "M5.infeasible_rate", m(S["M5"], "infeasible_rate"), 0.321),
    ("VI", "M2 infeasible 0.924", "M2.infeasible_rate", m(S["M2"], "infeasible_rate"), 0.922),
    ("VI", "disclose=2.4 ratio 0.669", "ablation.disclose[2.4].bias", lvl("disclose_mult", 2.4)["bias_ratio_warm"]["mean"], 0.66),
    ("VI", "disclose=2.4 infeas 0.846", "ablation.disclose[2.4].infeasible", lvl("disclose_mult", 2.4)["infeasible_rate"]["mean"], 0.747),
    ("VI", "T5 miss 0.0902", "M5.topology_miss.T5", S["M5"]["topology_miss"]["T5_wire_redirection"]["miss_rate"], 0.081),
    ("VI", "T3 miss 0.649", "M5.topology_miss.T3", S["M5"]["topology_miss"]["T3_payment_term_manipulation"]["miss_rate"], 0.671),
    ("VI", "drift miss 0.948", "M5.topology_miss.D", S["M5"]["topology_miss"]["D_drift_induced"]["miss_rate"], 0.948),
    ("VI", "recovery 18.32 d", "M5.post_covariate_recovery_days", m(S["M5"], "post_covariate_recovery_days"), 19.11),
    ("VI", "ULB M5 ratio 0.944", "ulb.M5.bias_ratio_warm", m(U["M5"], "bias_ratio_warm"), 0.903),
    ("VI", "ULB M4 ratio 0.883", "ulb.M4.bias_ratio_warm", m(U["M4"], "bias_ratio_warm"), 0.785),
    ("VI", "ULB oracle 1.27", "ulb.M5_oracle.bias_ratio_warm", m(U["M5_oracle"], "bias_ratio_warm"), 1.21),
    ("VI", "ULB M5 p50 20.06 us", "ulb.M5.update_us_p50", m(U["M5"], "update_us_p50"), 20.06),
    ("VI", "ULB M4 p50 9.58 us", "ulb.M4.update_us_p50", m(U["M4"], "update_us_p50"), 9.55),
    ("VI", "ULB alpha 0.000930", "stream_alpha.ulb", R["stream_alpha"]["ulb"], 0.00093),
    # flag validated against oracle feasibility
    ("VII", "M5 flag precision 0.900", "M5.flag_precision", m(S["M5"], "flag_precision"), 0.9),
    ("VII", "M5 flag recall 0.868", "M5.flag_recall", m(S["M5"], "flag_recall"), 0.84),
    ("VII", "M5 balanced acc 0.906", "M5.flag_balanced_acc", m(S["M5"], "flag_balanced_acc"), 0.898),
    ("VII", "M3 balanced acc 0.853", "M3.flag_balanced_acc", m(S["M3"], "flag_balanced_acc"), 0.855),
    ("VII", "M4 balanced acc 0.736", "M4.flag_balanced_acc", m(S["M4"], "flag_balanced_acc"), 0.731),
    ("VII", "oracle rate 0.351", "M5.oracle_infeasible_rate", m(S["M5"], "oracle_infeasible_rate"), 0.348),
    ("VII", "M2 recall 1.000", "M2.flag_recall", m(S["M2"], "flag_recall"), 1.0),
    ("VII", "M2 specificity 0.116", "M2.flag_specificity", m(S["M2"], "flag_specificity"), 0.119),
    ("VII", "risk when flagged 0.02206", "M5.risk_when_flagged", m(S["M5"], "risk_when_flagged"), 0.02267),
    ("VII", "risk unflagged 0.009356", "M5.risk_when_not_flagged", m(S["M5"], "risk_when_not_flagged"), 0.009543),
    ("VII", "A2 violation 0.600", "M5.mono_violation_mean", m(S["M5"], "mono_violation_mean"), 0.6),
    ("VII", "oracle agreement 1.000", "M5.oracle_agreement", m(S["M5"], "oracle_agreement"), 1.0),
    ("VII", "exh balanced acc 0.906", "M5.exh_flag_balanced_acc", m(S["M5"], "exh_flag_balanced_acc"), 0.898),
    ("IV-B", "estimand rel gap 0.07896", "M5.gap_overlap_rel", m(S["M5"], "gap_overlap_rel"), 0.07792),
    ("IV-B", "trimmed rows 0.1508", "M5.trim_share_mean", m(S["M5"], "trim_share_mean"), 0.1509),
    ("IV-B", "trimmed fraud 0.1838", "M5.trim_fraud_share_mean", m(S["M5"], "trim_fraud_share_mean"), 0.1829),
        ("VII", "p_d MAE 0.1579", "M5.pd_mae_mean", m(S["M5"], "pd_mae_mean"), 0.1566),
]


def paper_text() -> str:
    """The manuscript, so an expected value can be checked against what is printed."""
    p = os.path.join(ROOT, "paper", "main.tex")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def main():
    tex = paper_text()
    rows, bad = [], 0
    for sec, claim, src, actual, printed in CLAIMS:
        dec = len(str(printed).split(".")[1]) if "." in str(printed) else 0
        ok = abs(round(actual, dec) - printed) < 10 ** (-dec) / 2 + 1e-12
        # and the paper must actually print it: an expected value that matches
        # results.json but appears nowhere in the manuscript is not a check.
        lit = f"{printed:,}" if isinstance(printed, int) and abs(printed) >= 1000 else str(printed)
        shown = (lit in tex) or (str(printed) in tex)
        bad += not (ok and shown)
        rows.append((sec, claim, src, actual, printed, ok, shown))
    out = [
        "# NUMERICAL_CHECKS.md", "",
        "Generated by `tools/check_numbers.py`. Every number the paper prints is",
        "re-derived here from the specific field in `results/results.json` it comes from,",
        "then compared at the precision the paper prints. This is stricter than the",
        "prose gate in `audit_prose.py`, which only asks whether a literal appears",
        "somewhere in the artefacts and so can be satisfied by coincidence.", "",
        f"**{len(rows)} claims checked, {bad} mismatched.**", "",
        "| sec | claim as printed | source | value in results.json | ok |",
        "|---|---|---|---|---|",
    ]
    for sec, claim, src, actual, printed, ok, shown in rows:
        v = "yes" if (ok and shown) else ("**NO**" if not ok else "**NOT IN PAPER**")
        out.append(f"| {sec} | {claim} | `{src}` | {actual!r} | {v} |")
    out.append("")
    open(os.path.join(ROOT, "NUMERICAL_CHECKS.md"), "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} claims checked, {bad} mismatched -> NUMERICAL_CHECKS.md")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
