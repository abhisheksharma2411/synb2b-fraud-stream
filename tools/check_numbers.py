#!/usr/bin/env python3
"""Re-derive every number the paper asserts from results/results.json.

The prose gate in audit_prose.py only checks that a literal appears somewhere in the
artefacts. That is weak: a wrong digit can match a different quantity by accident. This
checks each claim against the specific field it is supposed to come from, at the
precision the paper prints, and writes NUMERICAL_CHECKS.md.

    python tools/check_numbers.py

Which variant a claim reads from matters more than it used to. `methods[stream]` now
carries M0..M5 plus M5_oracle, M5_legacy, M5_strictmatch, M5_joint_static and
M5_joint_slow. The paper reports **M5_legacy** -- the temporally mismatched ratio
estimator -- as "the reported estimator", because it is the best-calibrated deployable
variant, which is the paper's uncomfortable finding. M5 is the cohort-aligned variant
and is what Table I's "M5 -iap" row and the ablation grid are run under. Prose numbers
therefore mostly come from M5_legacy and table numbers from M5; each claim below names
its variant explicitly.

Some paper numbers are paired seed-level differences against M5_legacy (the same
quantity tools/make_paired_diffs.py reports) and exist nowhere in results.json as a
scalar. Those are computed here from the `per_seed` arrays rather than hard-coded, so
the check stays live if the run changes.
"""
from __future__ import annotations
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
S, U = R["methods"]["synb2b"], R["methods"]["ulb"]
SF, P = R["stream_facts"]["synb2b"], R["policy_default"]
AB = R["ablation"]["synb2b"]

# The variant the paper calls "the reported estimator", and the paired baseline used by
# tools/make_paired_diffs.py.
BASELINE = "M5_legacy"


def m(d, k, f="mean"):
    v = d[k]
    return v[f] if isinstance(v, dict) else v


def lvl(factor, x):
    for e in AB[factor]:
        if abs(e["level"] - x) < 1e-12:
            return e
    raise KeyError(f"{factor}={x}")


def miss(d, topology):
    return d["topology_miss"][topology]["miss_rate"]


# --- paired seed-level differences -------------------------------------------------
# method_comparison is seed-outer / variant-inner, so seed s fixes arrivals, labels and
# potential outcomes for every variant at once and the seed-level values of two variants
# are matched pairs. aggregate() persists that as `per_seed`, aligned to results["seeds"].


def per_seed(stream, variant, metric):
    node = R["methods"][stream][variant].get(metric)
    if not isinstance(node, dict) or "per_seed" not in node:
        raise KeyError(f"{stream}.{variant}.{metric}: no per_seed array to pair on")
    return node["per_seed"]


def _pairs(stream, cand, metric, target):
    base = per_seed(stream, BASELINE, metric)
    cnd = per_seed(stream, cand, metric)
    if len(base) != len(cnd):
        raise ValueError(f"{stream}.{cand}.{metric}: per_seed lengths differ")
    out = []
    for b, c in zip(base, cnd):
        if b is None or c is None:
            continue
        if not (isinstance(b, (int, float)) and isinstance(c, (int, float))):
            continue
        if not (math.isfinite(b) and math.isfinite(c)):
            continue
        if target is None:
            out.append(c - b)
        else:
            out.append(abs(c - target) - abs(b - target))
    if not out:
        raise ValueError(f"{stream}.{cand}.{metric}: no usable pairs")
    return out


def pdiff(stream, cand, metric, target=None):
    """Mean over seeds of (candidate - M5_legacy).

    With `target` set the difference is taken in distance from that target,
    d_s = |cand_s - target| - |base_s - target|, which is what the paper means by
    "costs X in distance from calibration": positive is worse calibrated.
    """
    d = _pairs(stream, cand, metric, target)
    return sum(d) / len(d)


def pcount(stream, cand, metric, target=None, sign=1):
    """How many seeds moved in the stated direction: the paper's "on k of n seeds"."""
    d = _pairs(stream, cand, metric, target)
    return sum(1 for x in d if (x > 0 if sign > 0 else x < 0))


ALPHA = P["alpha"]

CLAIMS = [
    # (section, claim as printed, source path, live value, printed literal)

    # ---- Abstract ----------------------------------------------------------------
    ("Abstract", "best deployable estimator misses the ratio by 0.123",
     "1 - M5_legacy.bias_ratio_warm", 1 - m(S[BASELINE], "bias_ratio_warm"), "0.123"),
    ("Abstract", "cohort matching costs 0.047",
     "paired abs(ratio-1): M5 - M5_legacy", pdiff("synb2b", "M5", "bias_ratio_warm", 1.0),
     "0.047"),
    ("Abstract", "cohort matching is worse on 20 of 20 seeds",
     "paired abs(ratio-1) sign count: M5 - M5_legacy",
     pcount("synb2b", "M5", "bias_ratio_warm", 1.0), 20),
    ("Abstract", "joint censored likelihood costs 0.147",
     "paired abs(ratio-1): M5_joint_static - M5_legacy",
     pdiff("synb2b", "M5_joint_static", "bias_ratio_warm", 1.0), "0.147"),
    ("Abstract", "an oracle propensity still leaves 0.086",
     "abs(M5_oracle.bias_ratio_warm - 1)", abs(m(S["M5_oracle"], "bias_ratio_warm") - 1),
     "0.086"),
    ("Abstract", "0.0200 review budget", "policy.budget", P["budget"], "0.0200"),
    ("Abstract", "feasibility flag reaches 0.898 balanced accuracy",
     "M5_legacy.flag_balanced_acc", m(S[BASELINE], "flag_balanced_acc"), "0.898"),

    # ---- IV-B, the estimand the estimator actually targets -------------------------
    ("IV-B", "estimand sits 0.07792 from the trailing-window risk",
     "M5_legacy.gap_overlap_rel", m(S[BASELINE], "gap_overlap_rel"), "0.07792"),
    ("IV-B", "trimming removes 0.1509 of rows",
     "M5_legacy.trim_share_mean", m(S[BASELINE], "trim_share_mean"), "0.1509"),
    ("IV-B", "those rows carry 0.1829 of the fraud",
     "M5_legacy.trim_fraud_share_mean", m(S[BASELINE], "trim_fraud_share_mean"),
     "0.1829"),
    ("IV-C", "support gate of 200 weight units", "policy.min_support",
     P["min_support"], 200),

    # ---- IV-E, the minimum-exploration bound ---------------------------------------
    ("IV-E", "Prop 2b n_exp >= 786 (eta=1)", "2[(1-a)+eta/3]log(2/d)/(eta^2 a)",
     math.ceil(2 * ((1 - ALPHA) + 1 / 3) * math.log(2 / 0.05) / ALPHA), 786),
    ("IV-E", "Prop 2b eps >= 0.873", "786/(b*window_max_n)",
     786 / (P["budget"] * P["window_max_n"]), "0.873"),
    ("IV-E", "Prop 2b n_exp >= 2748 (eta=0.5)", "same bound at eta=0.5",
     math.ceil(2 * ((1 - ALPHA) + 0.5 / 3) * math.log(2 / 0.05) / (0.25 * ALPHA)), 2748),
    ("IV-E", "Prop 2b eps of 3.05", "2748/(b*window_max_n)",
     2748 / (P["budget"] * P["window_max_n"]), "3.05"),

    # ---- V, experimental setup -----------------------------------------------------
    ("V", "100,000 invoices", "stream_facts.n", SF["n"], 100000),
    ("V", "span 561.34 d", "stream_facts.span_days", SF["span_days"], "561.34"),
    ("V", "1,500 published fraud labels", "stream_facts.drift_diag.n_base_fraud",
     SF["drift_diag"]["n_base_fraud"], 1500),
    ("V", "60-day concept sign flip", "stream_facts.events[concept].duration_days",
     next(e["duration_days"] for e in SF["events"] if e["kind"] == "concept"), 60),
    ("V", "AUC-PR 0.2049", "scorer.auc_pr_eval", SF["scorer"]["auc_pr_eval"], "0.2049"),
    ("V", "AUC-ROC 0.7313", "scorer.auc_roc_eval", SF["scorer"]["auc_roc_eval"],
     "0.7313"),
    ("V", "covariate rel drop 0.7988", "degradation.covariate.rel_drop",
     SF["degradation"]["covariate"]["rel_drop"], "0.7988"),
    ("V", "concept rel drop 0.6805", "degradation.concept.rel_drop",
     SF["degradation"]["concept"]["rel_drop"], "0.6805"),
    ("V", "prior rel RISE 0.3233", "-degradation.prior.rel_drop",
     -SF["degradation"]["prior"]["rel_drop"], "0.3233"),
    ("V", "alpha 0.0124", "policy.alpha", ALPHA, "0.0124"),
    ("V", "b 0.0200", "policy.budget", P["budget"], "0.0200"),
    ("V", "beta 0.0800", "policy.block_cap", P["block_cap"], "0.0800"),
    ("V", "eps 0.120", "policy.epsilon", P["epsilon"], "0.120"),
    ("V", "rho 0.9975", "policy.rho", P["rho"], "0.9975"),
    ("V", "gamma 0.018", "policy.gamma_aci", P["gamma_aci"], "0.018"),
    ("V", "pi_0 0.0500", "policy.pi_floor", P["pi_floor"], "0.0500"),
    ("V", "window 45000 items", "policy.window_max_n", P["window_max_n"], 45000),
    ("V", "window 240 days", "policy.window_max_days", P["window_max_days"], 240),
    ("V", "update every 250", "policy.update_every", P["update_every"], 250),
    ("V", "blocks of 2000 transactions (Fig. 3 caption)", "policy.metric_block",
     P["metric_block"], 2000),
    ("V", "ULB alpha 0.000930 (Table III caption)", "stream_alpha.ulb",
     R["stream_alpha"]["ulb"], "0.000930"),

    # ---- VI, results ---------------------------------------------------------------
    ("VI", "M0/M1 quiet 0.017366", "M0.quiet_for", m(S["M0"], "quiet_for"), "0.017366"),
    ("VI", "M0 overshoots alpha by 40.1%", "(M0.quiet_for/alpha - 1)*100",
     (m(S["M0"], "quiet_for") / ALPHA - 1) * 100, "40.1"),
    ("VI", "reported estimator quiet 0.012421", "M5_legacy.quiet_for",
     m(S[BASELINE], "quiet_for"), "0.012421"),
    ("VI", "0.17% above target", "(M5_legacy.quiet_for/alpha - 1)*100",
     (m(S[BASELINE], "quiet_for") / ALPHA - 1) * 100, "0.17"),
    ("VI", "M4 ratio 0.053", "M4.bias_ratio_warm", m(S["M4"], "bias_ratio_warm"),
     "0.053"),
    ("VI", "M4 reads high by a factor of 18.8", "1/M4.bias_ratio_warm",
     1 / m(S["M4"], "bias_ratio_warm"), "18.8"),
    ("VI", "M3 ratio 0.089", "M3.bias_ratio_warm", m(S["M3"], "bias_ratio_warm"),
     "0.089"),
    ("VI", "M2 ratio 0.116", "M2.bias_ratio_warm", m(S["M2"], "bias_ratio_warm"),
     "0.116"),
    ("VI", "reported estimator ratio 0.877", "M5_legacy.bias_ratio_warm",
     m(S[BASELINE], "bias_ratio_warm"), "0.877"),
    ("VI", "its two nuisance factors are fitted 87.0 days apart",
     "M5_legacy.cohort_gap_days_mean", m(S[BASELINE], "cohort_gap_days_mean"), "87.0"),
    ("VI", "strict cohort matching costs 0.383",
     "paired abs(ratio-1): M5_strictmatch - M5_legacy",
     pdiff("synb2b", "M5_strictmatch", "bias_ratio_warm", 1.0), "0.383"),
    ("VI", "joint likelihood overshoots to 1.168", "M5_joint_static.bias_ratio_warm",
     m(S["M5_joint_static"], "bias_ratio_warm"), "1.168"),
    ("VI", "calib weight, ledger 0.9808", "M5_legacy.calib_w_ledger_final",
     m(S[BASELINE], "calib_w_ledger_final"), "0.9808"),
    ("VI", "calib weight, review band 0.0169", "M5_legacy.calib_w_review_final",
     m(S[BASELINE], "calib_w_review_final"), "0.0169"),
    ("VI", "calib weight, exploration 0.0023", "M5_legacy.calib_w_explore_final",
     m(S[BASELINE], "calib_w_explore_final"), "0.0023"),
    ("VI", "demand averages 0.022867", "M5_legacy.review_demand_rate",
     m(S[BASELINE], "review_demand_rate"), "0.022867"),
    ("VI", "over budget in 0.52 of windows",
     "M5_legacy.frac_blocks_demand_over_budget",
     m(S[BASELINE], "frac_blocks_demand_over_budget"), "0.52"),
    ("VI", "served load 0.018476", "M5_legacy.review_served_rate",
     m(S[BASELINE], "review_served_rate"), "0.018476"),
    ("VI", "cost 173129", "M5_legacy.cost_per_1k", m(S[BASELINE], "cost_per_1k"),
     173129),
    ("VI", "M2 cost 160221", "M2.cost_per_1k", m(S["M2"], "cost_per_1k"), 160221),
    ("VI", "fraud allowed 18,502,948", "M5_legacy.fraud_dollars_allowed",
     m(S[BASELINE], "fraud_dollars_allowed"), 18502948),
    ("VI", "fraud total 30,008,053", "M5_legacy.fraud_dollars_total",
     m(S[BASELINE], "fraud_dollars_total"), 30008053),
    ("VI", "M0 fraud allowed 24,616,913", "M0.fraud_dollars_allowed",
     m(S["M0"], "fraud_dollars_allowed"), 24616913),
    ("VI", "update p50 18.09 us", "M5_legacy.update_us_p50",
     m(S[BASELINE], "update_us_p50"), "18.09"),
    ("VI", "update p99 28.16 us", "M5_legacy.update_us_p99",
     m(S[BASELINE], "update_us_p99"), "28.16"),
    ("VI", "Table I M5 -iap ratio 0.829", "M5.bias_ratio_warm",
     m(S["M5"], "bias_ratio_warm"), "0.829"),
    ("VI", "eps=0 ratio 0.8288", "ablation.epsilon[0].bias_ratio_warm",
     lvl("epsilon", 0.0)["bias_ratio_warm"]["mean"], "0.8288"),
    ("VI", "eps=0.120 ratio 0.8286", "ablation.epsilon[0.12].bias_ratio_warm",
     lvl("epsilon", 0.12)["bias_ratio_warm"]["mean"], "0.8286"),
    ("VI", "disclose x2.4 ratio 0.6788", "ablation.disclose_mult[2.4].bias_ratio_warm",
     lvl("disclose_mult", 2.4)["bias_ratio_warm"]["mean"], "0.6788"),
    ("VI", "disclose x2.4 infeasibility 0.7794",
     "ablation.disclose_mult[2.4].infeasible_rate",
     lvl("disclose_mult", 2.4)["infeasible_rate"]["mean"], "0.7794"),
    ("VI", "T5 wire redirection missed at 0.0810",
     "M5_legacy.topology_miss.T5_wire_redirection",
     miss(S[BASELINE], "T5_wire_redirection"), "0.0810"),
    ("VI", "T3 payment-term manipulation missed at 0.671",
     "M5_legacy.topology_miss.T3_payment_term_manipulation",
     miss(S[BASELINE], "T3_payment_term_manipulation"), "0.671"),
    ("VI", "recovery 19.11 d after the covariate shift",
     "M5_legacy.post_covariate_recovery_days",
     m(S[BASELINE], "post_covariate_recovery_days"), "19.11"),
    ("VI", "ULB reported estimator overshoots to 1.385",
     "ulb.M5_legacy.bias_ratio_warm", m(U[BASELINE], "bias_ratio_warm"), "1.385"),
    ("VI", "ULB uncorrected M4 sits at 0.432", "ulb.M4.bias_ratio_warm",
     m(U["M4"], "bias_ratio_warm"), "0.432"),
    ("VI", "ULB joint likelihood best at 0.594",
     "ulb.M5_joint_static.bias_ratio_warm",
     m(U["M5_joint_static"], "bias_ratio_warm"), "0.594"),

    # ---- VII, discussion: the flag, which is the part that survives -----------------
    ("VII", "(A2) fails on 0.598 of admissible steps", "M5_legacy.mono_violation_mean",
     m(S[BASELINE], "mono_violation_mean"), "0.598"),
    ("VII", "oracle and Prop. 1 agree on 1.000 of scored windows",
     "M5_legacy.oracle_agreement", m(S[BASELINE], "oracle_agreement"), "1.000"),
    ("VII", "flag precision 0.924", "M5_legacy.flag_precision",
     m(S[BASELINE], "flag_precision"), "0.924"),
    ("VII", "flag recall 0.840", "M5_legacy.flag_recall",
     m(S[BASELINE], "flag_recall"), "0.840"),
    ("VII", "flags 0.321 of windows", "M5_legacy.infeasible_rate",
     m(S[BASELINE], "infeasible_rate"), "0.321"),
    ("VII", "against an oracle rate of 0.348", "M5_legacy.oracle_infeasible_rate",
     m(S[BASELINE], "oracle_infeasible_rate"), "0.348"),
    ("VII", "true risk in flagged windows 0.02267", "M5_legacy.risk_when_flagged",
     m(S[BASELINE], "risk_when_flagged"), "0.02267"),
    ("VII", "true risk in unflagged windows 0.00954",
     "M5_legacy.risk_when_not_flagged", m(S[BASELINE], "risk_when_not_flagged"),
     "0.00954"),
    ("VII", "cohort matching helps the flag by 0.013",
     "paired flag_balanced_acc: M5 - M5_legacy",
     pdiff("synb2b", "M5", "flag_balanced_acc"), "0.013"),
    ("VII", "it helps on 16 of 20 seeds",
     "paired flag_balanced_acc sign count: M5 - M5_legacy",
     pcount("synb2b", "M5", "flag_balanced_acc"), 16),
    ("VII", "learned p_d misses the truth by 0.1566", "M5_legacy.pd_mae_mean",
     m(S[BASELINE], "pd_mae_mean"), "0.1566"),
]


def spec(printed):
    """(value, decimals, literals) for an expected number as the paper prints it.

    A string keeps the paper's trailing zeros, so "0.840" is checked to three decimals
    rather than the two a float would leave. Integers are also tried thousands-grouped,
    which is how the paper writes the dollar figures.
    """
    if isinstance(printed, str):
        val = float(printed.replace(",", ""))
        dec = len(printed.split(".")[1]) if "." in printed else 0
        return val, dec, (printed,)
    if isinstance(printed, int):
        return float(printed), 0, (str(printed), f"{printed:,}")
    txt = repr(printed)
    dec = len(txt.split(".")[1]) if "." in txt else 0
    return float(printed), dec, (txt,)


def paper_text() -> str:
    """The manuscript, so an expected value can be checked against what is printed."""
    p = os.path.join(ROOT, "paper", "main.tex")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def main():
    tex = paper_text()
    rows, bad = [], 0
    for sec, claim, src, actual, printed in CLAIMS:
        val, dec, lits = spec(printed)
        ok = abs(round(actual, dec) - val) < 10 ** (-dec) / 2 + 1e-12
        # and the paper must actually print it: an expected value that matches
        # results.json but appears nowhere in the manuscript is not a check.
        shown = any(l in tex for l in lits)
        bad += not (ok and shown)
        rows.append((sec, claim, src, actual, printed, ok, shown))
    out = [
        "# NUMERICAL_CHECKS.md", "",
        "Generated by `tools/check_numbers.py`. Every number the paper prints is",
        "re-derived here from the specific field in `results/results.json` it comes from,",
        "then compared at the precision the paper prints. This is stricter than the",
        "prose gate in `audit_prose.py`, which only asks whether a literal appears",
        "somewhere in the artefacts and so can be satisfied by coincidence.", "",
        "Prose numbers read from `M5_legacy`, the variant the paper reports; Table I and",
        "the ablation grid read from `M5`. Paired rows are seed-level differences against",
        "`M5_legacy`, computed from `per_seed` rather than hard-coded.", "",
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
