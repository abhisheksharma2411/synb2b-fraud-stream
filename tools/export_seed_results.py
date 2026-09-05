#!/usr/bin/env python3
"""Export every seed-level metric value from results.json to SEED_LEVEL_RESULTS.csv.

One row per (stream, variant, seed).  Columns: the stream, the variant key and
its label, the seed's index in `results["seeds"]`, the seed value itself, and one
column for every metric that carries a `per_seed` array.

`aggregate()` in src/t13/experiment.py writes each metric as

    {"mean", "ci_lo", "ci_hi", "n", "sd", "per_seed"}

with `per_seed[i]` holding the value obtained under `results["seeds"][i]`, or
None when that seed produced nothing finite.  The aggregates in results.json are
a summary of exactly these numbers; publishing them makes every reported mean and
interval recomputable by a reader, and makes the seed-level pairing between
variants -- every variant sees the same prepared stream under a given seed --
checkable rather than asserted.

Values are written at full float precision (shortest round-tripping repr), not
rounded for display.  A missing seed value is written as an empty cell.

    python tools/export_seed_results.py [--results results/results.json] [--out .]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CSV_NAME = "SEED_LEVEL_RESULTS.csv"

# Identity columns that precede the metric columns.
ID_COLUMNS = ["stream", "variant", "method_label", "block", "factor", "level",
              "seed_index", "seed"]

# Preferred variant order; anything else found in the file is appended sorted, so
# a new arm in the grid is exported rather than dropped.
VARIANT_ORDER = [
    "M0", "M1", "M2", "M3", "M4", "M5",
    "M5_oracle", "M5_legacy", "M5_strictmatch", "M5_joint_static", "M5_joint_slow",
]

NA = ""


def die(msg: str, code: int = 2) -> None:
    print(f"export_seed_results: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


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
    seeds = results.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        die(f"{path}: missing or empty 'seeds' list -- seed-level rows cannot be "
            "labelled with the seed they came from")
    return results


def per_seed_metrics(node: dict) -> dict:
    """Metric name -> per_seed list, for every metric in one aggregate node."""
    out = {}
    for key, val in node.items():
        if isinstance(val, dict) and isinstance(val.get("per_seed"), list):
            out[key] = val["per_seed"]
    return out


def cell(v) -> str:
    """Full-precision cell; empty when the seed produced no finite value."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return NA
    if isinstance(v, float) and not math.isfinite(v):
        return NA
    return repr(float(v))


def emit(rows, warnings, stream, variant, node, seeds, block, factor, level):
    """Append one row per seed for a single aggregate node."""
    metrics = per_seed_metrics(node)
    if not metrics:
        warnings.append(f"{stream}/{variant}"
                        + (f" [{factor}={level}]" if factor != NA else "")
                        + ": no metric carries a per_seed array; no rows emitted")
        return set()
    lengths = {len(v) for v in metrics.values()}
    if len(lengths) > 1:
        warnings.append(f"{stream}/{variant}: per_seed arrays have inconsistent "
                        f"lengths {sorted(lengths)}; rows are indexed by position "
                        "and short arrays leave empty cells")
    n_rows = max(lengths)
    if n_rows != len(seeds):
        warnings.append(f"{stream}/{variant}: {n_rows} per-seed value(s) but "
                        f"{len(seeds)} seed(s) declared; seed labels beyond the "
                        "declared list are left empty")
    label = node.get("method_label") or node.get("label") or NA
    for i in range(n_rows):
        row = {
            "stream": stream,
            "variant": variant,
            "method_label": label,
            "block": block,
            "factor": factor,
            "level": level,
            "seed_index": i,
            "seed": seeds[i] if i < len(seeds) else NA,
        }
        for name, vals in metrics.items():
            row[name] = cell(vals[i]) if i < len(vals) else NA
        rows.append(row)
    return set(metrics)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "results" / "results.json"),
                    help="path to results.json written by t13.experiment")
    ap.add_argument("--out", default=str(REPO),
                    help=f"output directory for {CSV_NAME}")
    ap.add_argument("--include-ablation", action="store_true",
                    help="also export the seed-level values of the ablation sweeps "
                         "(extra rows, block=ablation, with factor/level filled in)")
    args = ap.parse_args()

    results_path = Path(args.results)
    results = load_results(results_path)
    methods = results["methods"]
    seeds = results["seeds"]

    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        die(f"--out must be a directory, got a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings, metric_names = [], [], set()

    for stream in sorted(methods):
        variants = methods[stream]
        ordered = [v for v in VARIANT_ORDER if v in variants]
        ordered += [v for v in sorted(variants) if v not in VARIANT_ORDER]
        for variant in ordered:
            node = variants[variant]
            if not isinstance(node, dict):
                warnings.append(f"{stream}/{variant}: not an object; skipped")
                continue
            metric_names |= emit(rows, warnings, stream, variant, node, seeds,
                                 "methods", NA, NA)

    if args.include_ablation:
        abl = results.get("ablation")
        if not isinstance(abl, dict):
            warnings.append("--include-ablation given but there is no 'ablation' "
                            "object in the results file")
        else:
            for stream in sorted(abl):
                factors = abl[stream]
                if not isinstance(factors, dict):
                    continue
                for factor in sorted(factors):
                    entries = factors[factor]
                    if not isinstance(entries, list):
                        continue
                    for node in entries:
                        if not isinstance(node, dict):
                            continue
                        metric_names |= emit(
                            rows, warnings, stream,
                            node.get("method", "?"), node, seeds,
                            "ablation", factor, node.get("level", NA))

    if not rows:
        die(f"{results_path}: no seed-level values found -- every metric node is "
            "missing its 'per_seed' array (was this file written by an older "
            "version of aggregate()?)")

    columns = ID_COLUMNS + sorted(metric_names)
    csv_path = out_dir / CSV_NAME
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, restval=NA)
        w.writeheader()
        w.writerows(rows)

    for msg in warnings:
        print(f"export_seed_results: WARNING: {msg}", file=sys.stderr)

    n_streams = len({r["stream"] for r in rows})
    n_combos = len({(r["stream"], r["variant"], r["factor"], r["level"])
                    for r in rows})
    print(f"export_seed_results: {len(rows)} rows over {n_combos} "
          f"(stream, variant) combination(s) in {n_streams} stream(s) and "
          f"{len(seeds)} declared seed(s), {len(metric_names)} per-seed metric "
          f"column(s)"
          + (f", {len(warnings)} warning(s) on stderr" if warnings else "")
          + f"; wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
