#!/usr/bin/env python3
"""Emit FINAL_CLAIM_EVIDENCE.csv: one row per checked claim, with its provenance.

`tools/check_numbers.py` already re-derives every number the paper prints from the
specific field of `results/results.json` it comes from. What it does not say is which
script *produced* that field, and whether re-running that script would land on the
same digits. A reviewer asking "where does this number come from, and would I get it
again?" needs both, in one machine-readable table.

The claim list is not restated here. It is imported from `check_numbers`, so the two
files cannot drift: adding a claim there adds a row here on the next run.

This script reads `results/results.json` and `paper/main.tex`. It never runs the
experiment.

    python tools/make_claim_evidence.py [--out FINAL_CLAIM_EVIDENCE.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import check_numbers as CN  # noqa: E402  - imports results/results.json at load
except FileNotFoundError as exc:
    sys.exit(
        f"cannot import tools/check_numbers.py: {exc}\n"
        "It reads results/results.json at import time. Run `make reproduce-full` first."
    )

COLUMNS = [
    "section",
    "claim",
    "source_field",
    "generating_script",
    "value",
    "appears_in_paper",
    "deterministic",
    "notes",
]

# The only claims not read out of results/results.json are the Proposition 2b sample
# sizes, which check_numbers.py evaluates in closed form from stipulated constants.
# They are recognisable by their source label carrying algebra rather than a field path.
CLOSED_FORM = re.compile(r"log\(|\(b\*L\)|^same,")

# A source label that is not a bare dotted path is an arithmetic transform of one or
# more results.json fields (an inverse, a sign flip, a percentage) done inside
# check_numbers.py rather than a value read straight out of the file.
PLAIN_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\[\]]*$")

# Wall-clock measurements are the only quantities that move between runs of identical
# work. Everything else in the grid is seeded and reproduces to the last digit.
NON_DETERMINISTIC_FIELDS = ("update_us_p50", "update_us_p99")

GRID_SCRIPT = "src/t13/experiment.py (make reproduce-full) -> results/results.json"
CLOSED_FORM_SCRIPT = "tools/check_numbers.py (closed form over stipulated constants)"


def appears_in_paper(printed, tex: str) -> bool:
    """Whether the manuscript actually prints this literal.

    Same rule as the `shown` test in check_numbers.main(): an expected value that
    agrees with results.json but appears nowhere in the paper is not a check.
    """
    lit = f"{printed:,}" if isinstance(printed, int) and abs(printed) >= 1000 else str(printed)
    return (lit in tex) or (str(printed) in tex)


def matches_at_printed_precision(actual, printed) -> bool:
    """Same comparison check_numbers.main() makes, at the precision the paper prints."""
    dec = len(str(printed).split(".")[1]) if "." in str(printed) else 0
    return abs(round(actual, dec) - printed) < 10 ** (-dec) / 2 + 1e-12


def classify(source: str):
    """(generating_script, deterministic, provenance_note) for one claim source."""
    if CLOSED_FORM.search(source):
        return (
            CLOSED_FORM_SCRIPT,
            True,
            "closed form; depends on stipulated constants only, no simulation input",
        )
    deterministic = not any(f in source for f in NON_DETERMINISTIC_FIELDS)
    if deterministic:
        note = (
            "read from results.json"
            if PLAIN_PATH.match(source)
            else "arithmetic transform of results.json fields, evaluated in check_numbers.py"
        )
    else:
        note = (
            "wall-clock measurement; re-runs of identical work land on a different "
            "value, so this row is reproducible in method but not in digits"
        )
    return GRID_SCRIPT, deterministic, note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "FINAL_CLAIM_EVIDENCE.csv"))
    args = ap.parse_args(argv)

    tex = CN.paper_text()
    if not tex:
        print("warning: paper/main.tex not found; appears_in_paper will be false for "
              "every row", file=sys.stderr)

    rows = []
    for section, claim, source, actual, printed in CN.CLAIMS:
        script, deterministic, prov_note = classify(source)
        shown = appears_in_paper(printed, tex)
        ok = matches_at_printed_precision(actual, printed)

        notes = [f"paper prints {printed}", prov_note]
        if not ok:
            notes.insert(1, "MISMATCH: re-derived value does not round to the printed one")
        if not shown:
            notes.insert(1, "NOT FOUND in paper/main.tex")
        rows.append(
            {
                "section": section,
                "claim": claim.strip(),
                "source_field": source,
                "generating_script": script,
                "value": repr(actual),
                "appears_in_paper": str(bool(shown)).lower(),
                "deterministic": str(bool(deterministic)).lower(),
                "notes": "; ".join(notes),
            }
        )

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_shown = sum(r["appears_in_paper"] == "true" for r in rows)
    n_nondet = sum(r["deterministic"] == "false" for r in rows)
    n_closed = sum(r["generating_script"] == CLOSED_FORM_SCRIPT for r in rows)
    print(
        f"{len(rows)} claims -> {os.path.relpath(args.out, ROOT)}  "
        f"({n_shown} printed in the paper, {len(rows) - n_closed} from the grid, "
        f"{n_closed} closed form, {n_nondet} not bit-reproducible)"
    )
    for r in rows:
        if r["appears_in_paper"] != "true":
            print(f"  note: not printed in paper -> {r['section']} {r['claim']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
