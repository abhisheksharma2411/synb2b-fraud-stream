#!/bin/sh
# One-command reproduction of every artifact in this repository.
#
#   ./reproduce.sh --quick   reduced-size pass over 2 seeds; target under 5 minutes
#   ./reproduce.sh --full    the complete 20-seed grid; hours, not minutes
#
# Every stage is a `make` target or a tool under tools/. Nothing here computes a
# number of its own; the script only orders the existing targets and checks that
# the environment can run them. FINAL_REPRODUCTION.md documents what each stage
# regenerates, the measured runtimes, and which outputs are bit-reproducible.

set -eu

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

MODE=""
SKIP_REFCHECK=0
SKIP_ANON=0

usage() {
    cat <<'USAGE'
reproduce.sh - end-to-end reproduction for the SynB2B fraud-stream artifact

Usage:
  ./reproduce.sh --quick [options]    reduced-size pass, target under 5 minutes
  ./reproduce.sh --full  [options]    the complete 20-seed grid, takes HOURS
  ./reproduce.sh --help

Options:
  --skip-refcheck   do not re-verify citations against arXiv/Crossref/Zenodo
                    (that stage is the only one that needs network access
                    once the raw data is cached)
  --skip-anon       do not build the anonymous review artifact (--full only)

What --quick does
  1. prerequisites          python3, pdflatex, make, the virtualenv
  2. unit tests             make test
  3. reduced grid           make reproduce-small   -> results/results_small.json
  4. paper gates            make audit && make paper
  The reduced pass runs 2 seeds and does NOT overwrite results/results.json, so
  the number-checking stages are skipped: they read the full grid by design.

What --full does
  1. prerequisites
  2. unit tests             make test
  3. FULL grid (HOURS)      make reproduce-full    -> results/results.json
  4. paper data             make paper-data
  5. numbers                make numbers           -> NUMERICAL_CHECKS.md
  6. diagnostics            make diagnostics       -> DIAGNOSTICS.md,
                                                     INFEASIBILITY_VALIDATION.md
  7. paired + cohort        tools/make_paired_diffs.py, tools/make_cohort_audit.py
  8. estimator ladder       tools/validation_ladder.py
  9. claim evidence         tools/make_claim_evidence.py
 10. citations             make refcheck          (network)
 11. prose audit           make audit             -> WRITING_AUDIT.md
 12. paper build           make paper
 13. anonymous artifact    make anon-artifact

Exit status is non-zero as soon as any stage fails. The stages are gates, not
reports: `make numbers` fails if a printed number no longer matches its source,
and `make audit` fails if a writing threshold is missed.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --quick|-q) MODE=quick ;;
        --full|-f)  MODE=full ;;
        --skip-refcheck) SKIP_REFCHECK=1 ;;
        --skip-anon)     SKIP_ANON=1 ;;
        --help|-h)  usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$MODE" ]; then
    echo "error: choose a mode -- --quick or --full" >&2
    echo >&2
    usage >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# staged output
# ---------------------------------------------------------------------------
STEP=0
TOTAL=0
RUN_START=$(date +%s)

rule() { echo "==================================================================="; }

stage() {
    STEP=$((STEP + 1))
    echo
    rule
    echo "[$STEP/$TOTAL] $1"
    rule
}

note() { echo "  - $1"; }

die() {
    echo >&2
    echo "FAILED: $1" >&2
    exit 1
}

elapsed() {
    _now=$(date +%s)
    echo "$((_now - RUN_START))"
}

if [ "$MODE" = quick ]; then
    TOTAL=4
else
    TOTAL=13
fi

# ---------------------------------------------------------------------------
# 1. prerequisites
# ---------------------------------------------------------------------------
stage "prerequisites"

command -v make >/dev/null 2>&1 || \
    die "make not found on PATH. Install the platform build tools (macOS: xcode-select --install)."

command -v python3 >/dev/null 2>&1 || \
    die "python3 not found on PATH. Install CPython 3.14 (the pinned build; see requirements.txt) and retry."
note "python3: $(python3 --version 2>&1)"

command -v pdflatex >/dev/null 2>&1 || \
    die "pdflatex not found on PATH. Install a TeX distribution (macOS: brew install --cask mactex-no-gui; Debian: apt install texlive-latex-recommended texlive-latex-extra). Re-run with the paper stage removed only if you do not need the PDF gate."
note "pdflatex: $(pdflatex --version 2>/dev/null | head -1)"

# The Makefile's `venv` target creates ./.venv. A worktree checkout may instead
# share one virtualenv with its siblings at ../.venv, so accept either.
VENV_PY=""
for _cand in "$ROOT/.venv/bin/python" "$ROOT/../.venv/bin/python"; do
    if [ -x "$_cand" ]; then
        VENV_PY=$(cd "$(dirname "$_cand")" && pwd)/$(basename "$_cand")
        break
    fi
done
[ -n "$VENV_PY" ] || \
    die "no virtualenv found at ./.venv or ../.venv. Create one with:  make venv   (that is: python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt)"
note "virtualenv: $VENV_PY ($("$VENV_PY" --version 2>&1))"

"$VENV_PY" - <<'PYCHECK' || die "the virtualenv is missing pinned dependencies. Reinstall them with:  ./.venv/bin/python -m pip install -r requirements.txt"
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "pyarrow", "pytest")
           if importlib.util.find_spec(m) is None]
if missing:
    print("missing modules: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PYCHECK
note "pinned dependencies import cleanly"

# Not fatal: the pipeline runs on other 3.x builds, but results/results.json was
# recorded under 3.14.7 and requirements.txt is pinned against it.
VENV_MINOR=$("$VENV_PY" -c 'import sys;print("%d.%d" % sys.version_info[:2])')
if [ "$VENV_MINOR" != "3.14" ]; then
    note "WARNING: virtualenv is Python $VENV_MINOR; the recorded run and the pins in requirements.txt are CPython 3.14.7"
fi

# The raw inputs are never committed. src/t13/data.py fetches SynB2B-Fraud from
# Zenodo and the ULB card portfolio from OpenML on first use, so a cold cache
# needs network for the grid stage as well.
if [ -f "$ROOT/data/raw/synb2b_fraud.csv" ] && [ -f "$ROOT/data/raw/ulb_creditcard.parquet" ]; then
    note "raw data cached under data/raw/"
else
    note "raw data NOT fully cached; the grid stage will download it (Zenodo record 21668281, OpenML data_id 1597). Network required."
fi

# A function, not a variable: the interpreter path may contain spaces, and an
# unquoted "$MAKE target" would word-split it into two arguments.
run_make() { make "PY=$VENV_PY" "$@"; }
note "stages will run as: make PY=$VENV_PY <target>"

if [ "$MODE" = full ]; then
    echo
    rule
    echo "  WARNING - --full runs the complete grid and takes HOURS, not minutes."
    echo
    echo "  Measured on an Apple M4 Pro (12 cores), 20 seeds, both streams,"
    echo "  6 methods + oracle, 9 ablation factors, 10000 bootstrap resamples:"
    echo "      5409 s and 6064 s  (1h30m to 1h41m) for the grid stage alone."
    echo "  The remaining stages add a few minutes on top."
    echo
    echo "  It rewrites results/results.json and every report derived from it."
    echo "  Use --quick for a reduced 2-seed pass that leaves results.json alone."
    rule
fi

# ---------------------------------------------------------------------------
# 2. unit tests
# ---------------------------------------------------------------------------
stage "unit tests  (make test)"
run_make test

# ---------------------------------------------------------------------------
# 3. the grid
# ---------------------------------------------------------------------------
if [ "$MODE" = quick ]; then
    stage "reduced-size grid  (make reproduce-small)"
    note "2 seeds, reduced streams; writes results/results_small.json and results/SUMMARY_small.md"
    note "results/results.json is NOT touched"
    run_make reproduce-small
else
    stage "full grid  (make reproduce-full)  -- HOURS"
    note "20 seeds, both streams; writes results/results.json, results/SUMMARY.md, results/drift_events.json"
    run_make reproduce-full
fi

# ---------------------------------------------------------------------------
# quick mode stops after the paper gates
# ---------------------------------------------------------------------------
if [ "$MODE" = quick ]; then
    stage "paper gates  (make audit && make paper)"
    note "these read paper/main.tex against the committed results/results.json,"
    note "not the reduced pass just produced"
    if [ -f "$ROOT/results/results.json" ]; then
        run_make audit
        run_make paper
    else
        note "SKIPPED: results/results.json is absent. Run ./reproduce.sh --full first."
    fi

    echo
    rule
    echo "QUICK PASS COMPLETE in $(elapsed) s"
    rule
    echo "Reduced-pass outputs:"
    echo "  results/results_small.json"
    echo "  results/SUMMARY_small.md"
    echo
    echo "The quick pass deliberately does not regenerate NUMERICAL_CHECKS.md,"
    echo "DIAGNOSTICS.md, INFEASIBILITY_VALIDATION.md, the paired/cohort reports"
    echo "or the estimator ladder: those are defined against the full grid."
    echo "Run ./reproduce.sh --full for those. See FINAL_REPRODUCTION.md."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. paper data
# ---------------------------------------------------------------------------
stage "paper data  (make paper-data)"
note "regenerates results/paper_data.tex and splices it into paper/main.tex"
note "between the INLINE-DATA markers"
run_make paper-data

# ---------------------------------------------------------------------------
# 5. numbers
# ---------------------------------------------------------------------------
stage "printed numbers  (make numbers)  -> NUMERICAL_CHECKS.md"
note "re-derives every number the paper prints from the field it comes from"
note "in results/results.json; fails if any claim no longer matches"
run_make numbers

# ---------------------------------------------------------------------------
# 6. diagnostics
# ---------------------------------------------------------------------------
stage "diagnostics  (make diagnostics)  -> DIAGNOSTICS.md, INFEASIBILITY_VALIDATION.md"
note "estimand gap, effective sample size, propensity error, flag scoring"
run_make diagnostics

# ---------------------------------------------------------------------------
# 7. paired differences and cohort alignment
# ---------------------------------------------------------------------------
stage "paired differences and cohort alignment"
note "-> PAIRED_VARIANT_DIFFERENCES.csv, PAIRED_VARIANT_DIFFERENCES.md"
PYTHONPATH=src "$VENV_PY" tools/make_paired_diffs.py
note "-> COHORT_ALIGNMENT_AUDIT.md"
PYTHONPATH=src "$VENV_PY" tools/make_cohort_audit.py

# ---------------------------------------------------------------------------
# 8. estimator validation ladder
# ---------------------------------------------------------------------------
stage "estimator validation ladder  -> ESTIMATOR_VALIDATION_LADDER.md"
note "8 rungs x 5 seeds; switches one confounder on at a time"
note "this tool exits non-zero when rung S1 fails its calibration check, which is"
note "the currently recorded state, so its status is reported rather than fatal"
LADDER_RC=0
PYTHONPATH=src "$VENV_PY" tools/validation_ladder.py --seeds 5 || LADDER_RC=$?
if [ "$LADDER_RC" -eq 0 ]; then
    note "ladder gate: PASS (S1 covers 1.0)"
else
    note "ladder gate: rung S1 does not cover 1.0 (exit $LADDER_RC). See ESTIMATOR_VALIDATION_LADDER.md."
fi

# ---------------------------------------------------------------------------
# 9. claim evidence
# ---------------------------------------------------------------------------
stage "claim evidence  -> FINAL_CLAIM_EVIDENCE.csv"
note "one row per checked claim: source field, generating script, determinism"
PYTHONPATH=src "$VENV_PY" tools/make_claim_evidence.py

# ---------------------------------------------------------------------------
# 10. citations
# ---------------------------------------------------------------------------
if [ "$SKIP_REFCHECK" -eq 1 ]; then
    stage "citations  (SKIPPED: --skip-refcheck)"
    note "run 'make refcheck' separately to re-verify refcheck/arxiv_verified.json"
else
    stage "citations  (make refcheck)  -> refcheck/arxiv_verified.json"
    note "re-verifies every citation against arXiv, Crossref and Zenodo; needs network"
    run_make refcheck
fi

# ---------------------------------------------------------------------------
# 11. prose audit
# ---------------------------------------------------------------------------
stage "prose audit  (make audit)  -> WRITING_AUDIT.md"
note "hard writing thresholds over paper/main.tex; fails if any gate is missed"
run_make audit

# ---------------------------------------------------------------------------
# 12. paper build
# ---------------------------------------------------------------------------
stage "paper build  (make paper)"
note "pdflatex under both anonymisation settings; checks page limit and leaks"
run_make paper

# ---------------------------------------------------------------------------
# 13. anonymous artifact
# ---------------------------------------------------------------------------
if [ "$SKIP_ANON" -eq 1 ]; then
    stage "anonymous artifact  (SKIPPED: --skip-anon)"
else
    stage "anonymous artifact  (make anon-artifact)"
    note "history-free export with every author identifier stripped;"
    note "writes ../anon-artifact/synb2b-fraud-stream-anonymous{,.tar.gz,.zip}"
    run_make anon-artifact
fi

# ---------------------------------------------------------------------------
echo
rule
echo "FULL REPRODUCTION COMPLETE in $(elapsed) s"
rule
cat <<'DONE'
Regenerated:
  results/results.json            the grid
  results/SUMMARY.md              headline numbers with provenance
  results/drift_events.json       the three drift events per stream
  results/paper_data.tex          spliced into paper/main.tex
  NUMERICAL_CHECKS.md             every printed number re-derived
  DIAGNOSTICS.md                  estimand gap, ESS, propensity error
  INFEASIBILITY_VALIDATION.md     flag scored against the feasibility oracle
  PAIRED_VARIANT_DIFFERENCES.csv  and .md
  COHORT_ALIGNMENT_AUDIT.md       cohort alignment per (stream, variant)
  ESTIMATOR_VALIDATION_LADDER.md  staged estimator validation
  FINAL_CLAIM_EVIDENCE.csv        claim -> source -> determinism
  WRITING_AUDIT.md                prose gates
  ../anon-artifact/               anonymous review export

Wall-clock timing figures (update_us_p50, update_us_p99) are the only numbers
that will not match the recorded run bit for bit. See FINAL_REPRODUCTION.md.
DONE
