#!/usr/bin/env python3
"""Cohort-alignment / propensity-calibration audit, read from results/results.json.

Writes COHORT_ALIGNMENT_AUDIT.md: one row per (stream, variant) for the cohort
gap, pd calibration, weight-concentration and outcome diagnostics that the
results file actually carries, plus an explicit inventory of the diagnostics it
does NOT carry. Every field that is absent is printed as "not recorded" -- it is
never filled in, defaulted, or inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MISSING = "not recorded"
SIGFIGS = 6

# (results key, column header) -- split into two tables so each stays readable.
COHORT_FIELDS = [
    ("cohort_gap_days_mean", "cohort gap (days)"),
    ("pd_mae_mean", "pd MAE"),
    ("pd_brier_mean", "pd Brier"),
    ("pd_clip_share_mean", "pd clip share"),
]
WEIGHT_FIELDS = [
    ("ess_median", "ESS median"),
    ("ess_p05", "ESS p05"),
    ("trim_share_mean", "trim share"),
]
OUTCOME_INTERVAL_FIELDS = [("bias_ratio_warm", "bias ratio (warm)")]
OUTCOME_FIELDS = [("flag_balanced_acc", "flag balanced acc")]

# Diagnostics we want for a cohort audit but may not be recorded. Each entry is
# (human name, [candidate keys searched anywhere in a variant record]).
WANTED_DIAGNOSTICS = [
    ("per-update cohort fallback counts",
     ["n_fallback", "n_cohort_fallback", "cohort_fallback", "cohort_fallback_count",
      "fallback_share", "fallback_rate"]),
    ("per-update refit counts",
     ["n_refit", "n_refits", "pd_refits", "refit_count"]),
    ("thin / zero-weight / abstain update counts",
     ["n_thin", "n_zero", "n_abstain", "thin_share", "abstain_share"]),
    ("q / r cohort row counts per update",
     ["n_q_rows", "n_r_rows", "q_rows", "r_rows", "n_q", "n_r",
      "q_row_count", "r_row_count"]),
    ("cohort gap distribution beyond the mean",
     ["cohort_gap_days_median", "cohort_gap_days_p90", "cohort_gap_days_max",
      "cohort_gap_days_sd", "cohort_gap_days_p05"]),
    ("interval construction method / bootstrap replicate count",
     ["ci_method", "bootstrap", "n_boot", "n_bootstrap"]),
]


def fmt(value) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, (int, float)):
        return f"{value:.{SIGFIGS}g}"
    return str(value)


def mean_of(variant: dict, key: str):
    node = variant.get(key)
    if isinstance(node, dict):
        v = node.get("mean")
        return v if isinstance(v, (int, float)) else None
    if isinstance(node, (int, float)):
        return node
    return None


def mean_with_interval(variant: dict, key: str) -> str:
    node = variant.get(key)
    if not isinstance(node, dict):
        return MISSING
    mean = node.get("mean")
    if not isinstance(mean, (int, float)):
        return MISSING
    lo, hi = node.get("ci_lo"), node.get("ci_hi")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]"
    return f"{fmt(mean)} [interval {MISSING}]"


def n_of(variant: dict, key: str):
    node = variant.get(key)
    if isinstance(node, dict):
        n = node.get("n")
        return n if isinstance(n, (int, float)) else None
    return None


def find_seedlike_arrays(obj, n_seeds, path="", hits=None):
    """Locate numeric arrays whose length equals the seed count."""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_seedlike_arrays(v, n_seeds, f"{path}/{k}", hits)
    elif isinstance(obj, list):
        if (
            n_seeds
            and len(obj) == n_seeds
            and obj
            and all(isinstance(x, (int, float)) for x in obj)
        ):
            hits.append(path)
    return hits


def table(streams_variants, fields, key_getter, header_label="variant"):
    lines = []
    headers = [header_label] + [label for _, label in fields]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for label, variant in streams_variants:
        cells = [label] + [key_getter(variant, key) for key, _ in fields]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(REPO / "results" / "results.json"))
    ap.add_argument("--out-md", default=str(REPO / "COHORT_ALIGNMENT_AUDIT.md"))
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        return 2
    with results_path.open() as fh:
        results = json.load(fh)

    methods = results.get("methods")
    if not isinstance(methods, dict) or not methods:
        print(f"ERROR: no 'methods' block in {results_path}", file=sys.stderr)
        return 2

    prov = results.get("provenance", {}) if isinstance(results.get("provenance"), dict) else {}
    seeds = results.get("seeds")
    n_seeds = len(seeds) if isinstance(seeds, list) else None

    all_keys = set()
    for stream in methods:
        for variant in methods[stream].values():
            if isinstance(variant, dict):
                all_keys |= set(variant.keys())

    seedlike = find_seedlike_arrays(methods, n_seeds)

    md = []
    md.append("# Cohort alignment audit")
    md.append("")
    md.append(f"Source: `{results_path}`")
    if prov.get("git_commit"):
        md.append(f"Run commit: `{prov['git_commit']}` (dirty={prov.get('git_dirty', MISSING)})")
    if prov.get("utc"):
        md.append(f"Run UTC: {prov['utc']}")
    md.append(f"Seeds declared: {n_seeds if n_seeds is not None else MISSING}")
    md.append("")
    md.append(
        "Every cell is the aggregate `mean` recorded in the results file, over the "
        "seed replicates, unless stated otherwise. A field the file does not carry "
        f"is printed as `{MISSING}`; nothing is imputed."
    )
    md.append("")

    n_cells = 0
    n_missing = 0

    for stream in sorted(methods):
        variants = methods[stream]
        ordered = sorted(variants)
        pairs = []
        for key in ordered:
            v = variants[key]
            if not isinstance(v, dict):
                continue
            label_extra = v.get("method_label")
            label = f"`{key}`" + (f" ({label_extra})" if label_extra else "")
            pairs.append((label, v))

        md.append(f"## {stream}")
        md.append("")

        def cell(variant, key):
            nonlocal n_cells, n_missing
            n_cells += 1
            value = mean_of(variant, key)
            if value is None:
                n_missing += 1
                return MISSING
            return fmt(value)

        md.append("### Cohort matching and propensity calibration")
        md.append("")
        md += table(pairs, COHORT_FIELDS, cell)
        md.append("")

        md.append("### Weight concentration")
        md.append("")
        md += table(pairs, WEIGHT_FIELDS, cell)
        md.append("")

        md.append("### Estimator bias and flag quality")
        md.append("")

        def interval_cell(variant, key):
            nonlocal n_cells, n_missing
            n_cells += 1
            out = mean_with_interval(variant, key)
            if out == MISSING:
                n_missing += 1
            return out

        combined = OUTCOME_INTERVAL_FIELDS + OUTCOME_FIELDS
        headers = ["variant"] + [label for _, label in combined] + ["n (replicates)"]
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join("---" for _ in headers) + " |")
        for label, variant in pairs:
            cells = [label]
            for key, _ in OUTCOME_INTERVAL_FIELDS:
                cells.append(interval_cell(variant, key))
            for key, _ in OUTCOME_FIELDS:
                cells.append(cell(variant, key))
            n_rep = n_of(variant, OUTCOME_INTERVAL_FIELDS[0][0])
            if n_rep is None:
                n_rep = variant.get("n_seeds")
            cells.append(fmt(n_rep) if isinstance(n_rep, (int, float)) else MISSING)
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
        md.append(
            "The interval on the bias ratio is the per-variant interval stored in "
            "the results file. It describes that variant's own mean; it is not an "
            "interval on any between-variant difference."
        )
        md.append("")

    # ---- what is NOT in the file ------------------------------------------
    md.append("## Diagnostics this results file does not contain")
    md.append("")
    md.append(
        "Each row was probed against the union of keys across all "
        f"{sum(len(v) for v in methods.values())} (stream, variant) records "
        f"({len(all_keys)} distinct keys). Anything marked absent needs new "
        "instrumentation upstream before it can be audited."
    )
    md.append("")
    md.append("| diagnostic | status | keys searched |")
    md.append("| --- | --- | --- |")

    if seedlike:
        seed_status = "present at " + ", ".join(f"`{p}`" for p in seedlike[:5])
    else:
        seed_status = "**absent**"
    md.append(
        f"| per-seed (replicate-level) metric values | {seed_status} | "
        f"any numeric array of length {n_seeds if n_seeds is not None else '(seed count)'} "
        "under a variant record |"
    )

    absent_names = [] if seedlike else ["per-seed (replicate-level) metric values"]
    for name, keys in WANTED_DIAGNOSTICS:
        found = [k for k in keys if k in all_keys]
        if found:
            status = "present as " + ", ".join(f"`{k}`" for k in found)
        else:
            status = "**absent**"
            absent_names.append(name)
        md.append(f"| {name} | {status} | " + ", ".join(f"`{k}`" for k in keys) + " |")

    # seed attribution of the stored traces
    trace_notes = []
    for stream in sorted(methods):
        for vkey, variant in sorted(methods[stream].items()):
            if not isinstance(variant, dict):
                continue
            for trace in ("composition", "series"):
                node = variant.get(trace)
                if isinstance(node, dict) and not any(
                    "seed" in k.lower() for k in node
                ):
                    trace_notes.append(trace)
    if trace_notes:
        md.append(
            "| seed attribution for the stored `composition` / `series` traces | "
            "**absent** | any key containing `seed` inside those trace objects |"
        )
        absent_names.append("seed attribution for the composition / series traces")
    md.append("")

    if absent_names:
        md.append("In short, the following still need instrumentation:")
        md.append("")
        for name in absent_names:
            md.append(f"- {name}")
        md.append("")
        if not seedlike:
            md.append(
                "The per-seed gap is the consequential one. Without replicate-level "
                "values, no paired comparison between variants can be computed from "
                "this file, and no interval on a between-variant difference is "
                "recoverable from the marginal sds alone."
            )
            md.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))

    n_variants = sum(len(v) for v in methods.values())
    print(
        f"make_cohort_audit: {n_variants} (stream, variant) records across "
        f"{len(methods)} streams, {n_cells} audited cells ({n_missing} {MISSING}), "
        f"{len(absent_names)} missing diagnostic families -> {out_md.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
