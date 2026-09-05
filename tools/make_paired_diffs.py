#!/usr/bin/env python3
"""Paired variant-vs-baseline differences for the method grid, from results.json.

Writes PAIRED_VARIANT_DIFFERENCES.csv and PAIRED_VARIANT_DIFFERENCES.md.

Why this is a paired analysis and not a difference of two averages
-----------------------------------------------------------------
`method_comparison` in src/t13/experiment.py is seed-outer, variant-inner: one
stream context is prepared per seed and *every* variant is run against that same
context.  Seed s therefore fixes the arrivals, the labels and the potential
outcomes for all variants at once, so the seed-level values of any two variants
are matched pairs, not independent samples.

`aggregate()` persists that structure: each metric node carries

    {"mean", "ci_lo", "ci_hi", "n", "sd", "per_seed"}

where `per_seed[i]` is the value under `results["seeds"][i]` (or None when the
run did not produce a finite value).  This script reads `per_seed` and forms the
seed-level difference

    d_s = candidate_s - baseline_s

then reports its mean, median, sd, range, sign counts and a percentile bootstrap
interval taken over the *differences*.  Pairing removes the between-seed
variance that dominates the marginal intervals, which is the whole point: two
marginal intervals can overlap heavily while every single seed moves the same
way.

When a metric node has no `per_seed` array -- `aggregate()` omits it when no
seed produced a finite value -- the row is emitted with a "per_seed unavailable"
note and no statistics.  It is deliberately *not* backfilled with a difference
of means: that quantity has no computable interval here and reporting it beside
genuine paired rows would invite it to be read as one.

    python tools/make_paired_diffs.py [--results results/results.json] [--out .]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CSV_NAME = "PAIRED_VARIANT_DIFFERENCES.csv"
MD_NAME = "PAIRED_VARIANT_DIFFERENCES.md"

# The variant whose per-seed values are subtracted from every candidate's.
BASELINE = "M5_legacy"

# Preferred display order.  Any variant present in the results file but missing
# from this list is still analysed, appended in sorted order, so a new arm added
# to the grid does not silently drop out of the report.
VARIANT_ORDER = [
    "M0", "M1", "M2", "M3", "M4", "M5",
    "M5_oracle", "M5_strictmatch", "M5_joint_static", "M5_joint_slow",
]

# Metrics carried through the paired analysis.  Names are exactly the keys
# written by aggregate(); nothing here is derived from a hard-coded value.
METRICS = [
    "bias_ratio_warm",
    "for_overall",
    "quiet_for",
    "flag_precision",
    "flag_recall",
    "flag_balanced_acc",
    "review_demand_rate",
    "review_served_rate",
    "block_rate",
    "cost_per_1k",
    "pd_mae_mean",
    "ess_median",
    "cohort_gap_days_mean",
    "infeasible_rate",
]

# A calibration ratio is best at 1 and can miss in either direction, so the raw
# signed difference does not say whether a variant is better calibrated.  For
# these metrics an extra derived row reports the paired difference in distance
# from the target: d_s = |cand_s - target| - |base_s - target|, negative meaning
# the candidate sits closer to the target on that seed.
RATIO_TARGET = {"bias_ratio_warm": 1.0}
ABS_DEV_SUFFIX = "_abs_dev_from_target"

# The two comparisons the report leads with.
HEADLINE_METRICS = ["bias_ratio_warm", "flag_balanced_acc"]

DEFAULT_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 90210
DEFAULT_ALPHA = 0.05

NA = ""
UNAVAILABLE_NOTE = "per_seed unavailable"

COLUMNS = [
    "stream",
    "baseline",
    "candidate",
    "candidate_label",
    "metric",
    "source_metric",
    "transform",
    "n_seeds_declared",
    "n_pairs",
    "n_skipped",
    "baseline_mean_paired",
    "candidate_mean_paired",
    "mean_diff",
    "median_diff",
    "sd_diff",
    "min_diff",
    "max_diff",
    "n_positive",
    "n_negative",
    "n_zero",
    "ci_lo",
    "ci_hi",
    "ci_method",
    "bootstrap_resamples",
    "bootstrap_seed",
    "note",
]

SIGFIGS = 6


def die(msg: str, code: int = 2) -> "int":
    print(f"make_paired_diffs: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def fmt(value) -> str:
    """Round a number for the Markdown tables; blank when there is nothing to say."""
    if value is None:
        return NA
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{SIGFIGS}g}"
    return str(value)


def csv_cell(value) -> str:
    """Full-precision CSV cell.

    The CSV is the machine-readable artifact, so floats go in at shortest
    round-tripping precision rather than the six significant figures used for
    the human-readable tables; a reader recomputing from
    SEED_LEVEL_RESULTS.csv should get these numbers back exactly.
    """
    if value is None:
        return NA
    if isinstance(value, float):
        return "nan" if math.isnan(value) else repr(value)
    return str(value)


def load_results(path: Path) -> dict:
    if not path.is_file():
        die(f"results file not found: {path}")
    try:
        with path.open() as fh:
            results = json.load(fh)
    except json.JSONDecodeError as exc:
        die(f"{path} is not valid JSON: {exc}")
    except OSError as exc:
        die(f"cannot read {path}: {exc}")
    if not isinstance(results, dict):
        die(f"{path}: top level is {type(results).__name__}, expected an object")
    methods = results.get("methods")
    if not isinstance(methods, dict) or not methods:
        die(f"{path}: missing or empty 'methods' object -- "
            "this file was not written by t13.experiment.main()")
    for stream, variants in methods.items():
        if not isinstance(variants, dict) or not variants:
            die(f"{path}: methods['{stream}'] is not a non-empty object")
    return results


def per_seed_values(variant: dict, metric: str):
    """Return (values, reason).  values is None when no per-seed array exists."""
    node = variant.get(metric)
    if node is None:
        return None, f"metric '{metric}' absent"
    if not isinstance(node, dict):
        return None, f"metric '{metric}' is not an aggregate object"
    if "per_seed" not in node:
        # aggregate() drops per_seed only when no seed produced a finite value.
        return None, f"{UNAVAILABLE_NOTE} for '{metric}' (n={node.get('n')})"
    raw = node["per_seed"]
    if not isinstance(raw, list):
        return None, f"{UNAVAILABLE_NOTE} for '{metric}' (per_seed is not a list)"
    out = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out.append(None)
        elif not math.isfinite(v):
            out.append(None)
        else:
            out.append(float(v))
    return out, ""


def quantile(sorted_vals, q: float) -> float:
    """Linear-interpolated quantile, matching numpy's default `np.quantile`."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def bootstrap_ci(diffs, resamples: int, seed: int, alpha: float):
    """Percentile bootstrap of the mean of the seed-level differences.

    Same construction as src/t13/calib.py:bootstrap_ci -- resample the units with
    replacement, take the mean of each resample, read off the two percentiles --
    but applied to the *differences* rather than to either arm's levels, and
    written against the standard library so the report can be regenerated from a
    results.json alone, without the simulation environment.
    """
    if not diffs:
        return None, None
    if len(diffs) == 1:
        return float(diffs[0]), float(diffs[0])
    rng = random.Random(seed)
    pick = rng.choices
    k = len(diffs)
    means = sorted(sum(pick(diffs, k=k)) / k for _ in range(resamples))
    return quantile(means, alpha / 2.0), quantile(means, 1.0 - alpha / 2.0)


def row_seed(base_seed: int, stream: str, candidate: str, metric: str) -> int:
    """Deterministic, row-specific bootstrap seed.

    crc32 rather than hash() so the value does not move with PYTHONHASHSEED and
    the report is byte-reproducible across runs and machines.
    """
    tag = f"{stream}|{candidate}|{metric}".encode()
    return (base_seed + zlib.crc32(tag)) % (2 ** 32)


def metric_specs():
    """(name, source_metric, transform_label, transform_fn) for every emitted row."""
    specs = []
    for m in METRICS:
        specs.append((m, m, "identity", None))
        if m in RATIO_TARGET:
            target = RATIO_TARGET[m]
            specs.append((m + ABS_DEV_SUFFIX, m, f"abs(x - {target:g})",
                          lambda x, _t=target: abs(x - _t)))
    return specs


def analyse(stream, variants, candidate, label, spec, n_seeds_declared, args):
    """One CSV row: the paired comparison of `candidate` against BASELINE."""
    name, source, transform_label, transform = spec
    baseline = args.baseline
    base_vals, base_why = per_seed_values(variants[baseline], source)
    cand_vals, cand_why = per_seed_values(variants[candidate], source)

    row = {
        "stream": stream,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_label": label,
        "metric": name,
        "source_metric": source,
        "transform": transform_label,
        "n_seeds_declared": n_seeds_declared,
        "n_pairs": 0,
        "n_skipped": None,
        "baseline_mean_paired": None,
        "candidate_mean_paired": None,
        "mean_diff": None,
        "median_diff": None,
        "sd_diff": None,
        "min_diff": None,
        "max_diff": None,
        "n_positive": None,
        "n_negative": None,
        "n_zero": None,
        "ci_lo": None,
        "ci_hi": None,
        "ci_method": None,
        "bootstrap_resamples": None,
        "bootstrap_seed": None,
        "note": "",
    }

    if base_vals is None or cand_vals is None:
        why = "; ".join(w for w in (
            f"baseline {baseline}: {base_why}" if base_vals is None else "",
            f"candidate {candidate}: {cand_why}" if cand_vals is None else "",
        ) if w)
        row["note"] = (f"{UNAVAILABLE_NOTE} -- no paired difference computed "
                       f"({why}). Not backfilled with a difference of means.")
        return row, "unavailable"

    if len(base_vals) != len(cand_vals):
        row["note"] = (f"{UNAVAILABLE_NOTE} -- per_seed lengths differ "
                       f"(baseline {len(base_vals)}, candidate {len(cand_vals)}); "
                       "the two arms cannot be aligned seed by seed.")
        return row, "unavailable"

    kept_base, kept_cand, skipped_idx = [], [], []
    for i, (b, c) in enumerate(zip(base_vals, cand_vals)):
        if b is None or c is None:
            skipped_idx.append(i)
        else:
            kept_base.append(b)
            kept_cand.append(c)

    if transform is not None:
        kept_base = [transform(x) for x in kept_base]
        kept_cand = [transform(x) for x in kept_cand]

    diffs = [c - b for b, c in zip(kept_base, kept_cand)]
    n = len(diffs)
    row["n_pairs"] = n
    row["n_skipped"] = len(skipped_idx)

    notes = []
    if skipped_idx:
        notes.append(f"{len(skipped_idx)} pair(s) skipped at seed index "
                     f"{','.join(str(i) for i in skipped_idx)} "
                     "(None/NaN on one or both arms)")
    if n == 0:
        row["note"] = "; ".join(notes + ["no usable pairs"])
        return row, "empty"

    row["baseline_mean_paired"] = statistics.fmean(kept_base)
    row["candidate_mean_paired"] = statistics.fmean(kept_cand)
    row["mean_diff"] = statistics.fmean(diffs)
    row["median_diff"] = float(statistics.median(diffs))
    row["sd_diff"] = statistics.stdev(diffs) if n > 1 else None
    row["min_diff"] = min(diffs)
    row["max_diff"] = max(diffs)
    row["n_positive"] = sum(1 for d in diffs if d > 0)
    row["n_negative"] = sum(1 for d in diffs if d < 0)
    row["n_zero"] = sum(1 for d in diffs if d == 0)

    seed = row_seed(args.bootstrap_seed, stream, candidate, name)
    lo, hi = bootstrap_ci(diffs, args.resamples, seed, args.alpha)
    row["ci_lo"] = lo
    row["ci_hi"] = hi
    conf = int(round((1.0 - args.alpha) * 100))
    row["ci_method"] = f"{conf}% percentile bootstrap over seed-level differences"
    row["bootstrap_resamples"] = args.resamples
    row["bootstrap_seed"] = seed
    if n == 1:
        notes.append("one pair only: sd undefined and the interval is degenerate")
    elif n < 3:
        notes.append(f"only {n} pairs: the interval is coarse")
    row["note"] = "; ".join(notes)
    return row, "paired"


def md_table(rows, conf: int):
    out = [f"| variant | n | mean d | median d | sd d | {conf}% CI | min d | max d "
           "| +/-/0 | note |",
           "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | --- |"]
    for r in rows:
        ci = (f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]"
              if r["ci_lo"] is not None else "--")
        signs = (f"{r['n_positive']}/{r['n_negative']}/{r['n_zero']}"
                 if r["n_positive"] is not None else "--")
        cells = [fmt(r[k]) or "--" for k in
                 ("mean_diff", "median_diff", "sd_diff")]
        rng = [fmt(r[k]) or "--" for k in ("min_diff", "max_diff")]
        out.append(
            f"| `{r['candidate']}` | {r['n_pairs']} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {ci} | {rng[0]} | {rng[1]} | {signs} | "
            f"{r['note'].replace('|', '/')} |"
        )
    return out


def build_md(rows, results, results_path, args, streams, candidates, labels):
    prov = results.get("provenance") if isinstance(results.get("provenance"), dict) else {}
    prov = prov or {}
    seeds = results.get("seeds")
    conf = int(round((1.0 - args.alpha) * 100))

    md = ["# Paired variant differences", ""]
    md.append(f"Source: `{results_path}`")
    if results.get("schema"):
        md.append(f"Schema: `{results['schema']}`  (small={results.get('small')})")
    if prov.get("git_commit"):
        md.append(f"Run commit: `{prov['git_commit']}` (dirty={prov.get('git_dirty')})")
    if prov.get("utc"):
        md.append(f"Run UTC: {prov['utc']}")
    if isinstance(seeds, list):
        md.append(f"Seeds ({len(seeds)}): `{seeds}`")
    md += [
        "",
        f"Baseline: `{args.baseline}`. Every difference is "
        "**candidate minus baseline**, formed seed by seed.",
        "",
        "## How the numbers were made",
        "",
        "`method_comparison` prepares one stream context per seed and runs every "
        "variant against that same context, so under seed *s* all variants see "
        "identical arrivals, labels and potential outcomes. `aggregate()` stores the "
        "seed-level value of each metric in `per_seed`, aligned to `results[\"seeds\"]`. "
        "This report forms `d_s = candidate_s - baseline_s` and summarises the "
        "differences directly.",
        "",
        f"- Interval: {conf}% percentile bootstrap of the mean of `d_s`, "
        f"{args.resamples} resamples, resampling **seed-level differences** "
        "(not levels). The seed is derived per row from the stream, candidate and "
        "metric name, so the report is reproducible.",
        "- A pair is dropped when either arm is `None`/NaN on that seed; the count of "
        "dropped pairs is in `n_skipped` and repeated in the row note.",
        "- Marginal `ci_lo`/`ci_hi` from `results.json` describe each variant's own "
        "mean and are **not** used here. Overlap between two marginal intervals is "
        "not a test of the difference; the interval below is.",
        f"- `{'`, `'.join(m + ABS_DEV_SUFFIX for m in RATIO_TARGET)}` is a derived "
        "row: a calibration ratio is best at its target and can miss either way, so "
        "the signed difference alone cannot say which arm is better calibrated. "
        "The derived row differences `|x - target|`, so **negative means the "
        "candidate is closer to the target**.",
        f"- Rows with no `per_seed` array are marked `{UNAVAILABLE_NOTE}` and left "
        "blank rather than being filled with a difference of means.",
        "",
    ]

    md.append("## Headline comparisons")
    md.append("")
    for metric in HEADLINE_METRICS:
        names = [metric] + ([metric + ABS_DEV_SUFFIX] if metric in RATIO_TARGET else [])
        for name in names:
            md.append(f"### `{name}`")
            if name.endswith(ABS_DEV_SUFFIX):
                md.append("")
                md.append("Distance from the calibrated target; negative = candidate "
                          "closer to target.")
            md.append("")
            any_rows = False
            for stream in streams:
                sub = [r for r in rows if r["stream"] == stream and r["metric"] == name]
                if not sub:
                    continue
                any_rows = True
                md.append(f"**{stream}**")
                md.append("")
                md += md_table(sub, conf)
                md.append("")
            if not any_rows:
                md.append("_No rows: metric absent from every variant._")
                md.append("")

    md.append("## All metrics")
    md.append("")
    for stream in streams:
        md.append(f"### {stream}")
        md.append("")
        for name, _src, _tl, _fn in metric_specs():
            sub = [r for r in rows if r["stream"] == stream and r["metric"] == name]
            if not sub:
                continue
            md.append(f"#### `{name}`")
            md.append("")
            md += md_table(sub, conf)
            md.append("")

    gaps = [r for r in rows
            if UNAVAILABLE_NOTE in r["note"] or bool(r["n_skipped"])]
    md.append("## Coverage gaps")
    md.append("")
    if gaps:
        md.append("| stream | candidate | metric | n_pairs | n_skipped | note |")
        md.append("| --- | --- | --- | ---: | ---: | --- |")
        for r in gaps:
            md.append(f"| {r['stream']} | `{r['candidate']}` | `{r['metric']}` | "
                      f"{r['n_pairs']} | {fmt(r['n_skipped']) or '--'} | "
                      f"{r['note'].replace('|', '/')} |")
    else:
        md.append("None: every candidate/metric pair had a complete `per_seed` array "
                  "on both arms.")
    md += [
        "",
        f"Variants compared: {', '.join('`' + c + '`' for c in candidates)}.",
        "",
        "Full machine-readable output, one row per stream x candidate x metric, "
        f"is in `{CSV_NAME}`. Seed-level inputs are in `SEED_LEVEL_RESULTS.csv` "
        "(`tools/export_seed_results.py`).",
        "",
    ]
    # labels are informative only; keep them out of the tables but record them once
    if any(labels.values()):
        md.append("Variant labels as recorded in the results file:")
        md.append("")
        for k in candidates:
            if labels.get(k):
                md.append(f"- `{k}`: {labels[k]}")
        md.append("")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "results" / "results.json"),
                    help="path to results.json written by t13.experiment")
    ap.add_argument("--out", default=str(REPO),
                    help=f"output directory for {CSV_NAME} and {MD_NAME}")
    ap.add_argument("--baseline", default=BASELINE,
                    help="variant key used as the paired baseline")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                    help="bootstrap resamples for the interval on the difference")
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
                    help="base seed; each row derives its own seed from this")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="two-sided miss rate for the interval (0.05 -> 95%%)")
    args = ap.parse_args()

    if args.resamples < 1:
        die("--resamples must be >= 1")
    if not 0.0 < args.alpha < 1.0:
        die("--alpha must lie strictly between 0 and 1")

    results_path = Path(args.results)
    results = load_results(results_path)
    methods = results["methods"]

    seeds = results.get("seeds")
    n_seeds_declared = len(seeds) if isinstance(seeds, list) else None
    if n_seeds_declared is None:
        print(f"make_paired_diffs: WARNING: {results_path} has no 'seeds' list; "
              "pairs are still formed by position in per_seed, but the report "
              "cannot name the seed behind each index", file=sys.stderr)

    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        die(f"--out must be a directory, got a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = sorted(methods)
    missing_baseline = [s for s in streams if args.baseline not in methods[s]]
    if len(missing_baseline) == len(streams):
        die(f"{results_path}: baseline '{args.baseline}' is absent from every stream "
            f"(streams have: {sorted({k for s in streams for k in methods[s]})})")

    all_variants = sorted({k for s in streams for k in methods[s]})
    ordered = [v for v in VARIANT_ORDER if v in all_variants and v != args.baseline]
    ordered += [v for v in all_variants
                if v not in VARIANT_ORDER and v != args.baseline]

    labels = {}
    for v in ordered:
        for s in streams:
            node = methods[s].get(v)
            if isinstance(node, dict) and node.get("method_label"):
                labels[v] = node["method_label"]
                break
        labels.setdefault(v, "")

    rows, kinds = [], {"paired": 0, "unavailable": 0, "empty": 0}
    skipped_streams = []
    for stream in streams:
        variants = methods[stream]
        if args.baseline not in variants:
            skipped_streams.append(stream)
            continue
        for cand in ordered:
            if cand not in variants:
                continue
            for spec in metric_specs():
                row, kind = analyse(stream, variants, cand, labels.get(cand, ""),
                                    spec, n_seeds_declared, args)
                rows.append(row)
                kinds[kind] += 1

    if not rows:
        die(f"{results_path}: no candidate variants to compare against "
            f"'{args.baseline}'")

    csv_path = out_dir / CSV_NAME
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, restval=NA)
        w.writeheader()
        w.writerows({k: csv_cell(v) for k, v in r.items()} for r in rows)

    present_streams = [s for s in streams if s not in skipped_streams]
    md = build_md(rows, results, results_path, args, present_streams, ordered, labels)
    if skipped_streams:
        md += ("\n## Streams skipped\n\n"
               + "".join(f"- {s}: baseline `{args.baseline}` absent\n"
                         for s in skipped_streams))
    md_path = out_dir / MD_NAME
    md_path.write_text(md)

    n_skipped_pairs = sum(r["n_skipped"] or 0 for r in rows)
    print(f"make_paired_diffs: {len(rows)} rows "
          f"({kinds['paired']} paired, {kinds['unavailable']} per_seed-unavailable, "
          f"{kinds['empty']} no-usable-pairs) over {len(present_streams)} stream(s) x "
          f"{len(ordered)} candidate(s) vs {args.baseline}; "
          f"{n_skipped_pairs} seed pair(s) "
          f"dropped; wrote {csv_path} and {md_path}")
    if kinds["paired"] == 0:
        # Not one comparison could be paired: the file predates the per_seed
        # field, or the grid wrote it from a build that did not persist seeds.
        # The report is still written so the notes are readable, but this is a
        # failure -- a caller must not treat an all-blank report as a result.
        print(f"make_paired_diffs: ERROR: no metric on any candidate could be "
              f"paired against '{args.baseline}' -- every metric node in "
              f"{results_path} is missing its 'per_seed' array. The report was "
              "written with per-row notes, but it contains no paired statistics.",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
