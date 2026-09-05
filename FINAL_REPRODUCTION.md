# FINAL_REPRODUCTION.md

How to regenerate every artifact in this repository, what each command produces, how
long it takes, and which outputs are bit-reproducible.

One command does the whole thing:

    ./reproduce.sh --quick    # reduced 2-seed pass, target under 5 minutes
    ./reproduce.sh --full     # the complete 20-seed grid; HOURS, not minutes
    ./reproduce.sh --help

`reproduce.sh` only orders the `make` targets and the tools under `tools/`. Everything
below can also be run one target at a time.

---

## 1. Environment

### Interpreter

    $ ../.venv/bin/python --version
    Python 3.14.7

The recorded full-grid run in `results/results.json` carries its own provenance block:

| field | recorded value |
|---|---|
| `python` | `3.14.7` |
| `platform` | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| `machine` | `arm64` |
| `n_cpu` | `12` |
| `git_commit` | `50f7e8f73800ed85027164cefbc6d7f183d3a34d` |
| `utc` | `2026-09-05T09:08:53Z` |

Read it back at any time with:

    python3 -c "import json;print(json.load(open('results/results.json'))['provenance'])"

### Packages

`requirements.txt` is fully pinned, including transitives, so the environment resolves
identically. Its header records the build target: *CPython 3.14.7, macOS 15
(Darwin 25.5.0), Apple M4 Pro*. The direct pins are:

    lightgbm==4.7.0      numpy==2.5.2        pandas==3.0.5      pyarrow==25.0.1
    pytest==9.1.1        requests==2.34.2    scikit-learn==1.9.0  scipy==1.18.1

Create the environment:

    make venv       # python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt

`make venv` creates `./.venv`. A worktree checkout may instead share one virtualenv
with its siblings at `../.venv`; `reproduce.sh` accepts either and passes the one it
finds to `make` as `PY=`.

### External tools

- `pdflatex` — required by `make paper`. `pdfinfo` and `pdftotext` (poppler) are
  optional; `tools/build_paper.py` falls back to parsing the PDF directly without them.
- `make`.
- Network — needed twice: once to fetch the raw data on a cold cache, and by
  `make refcheck`, which re-verifies every citation against arXiv, Crossref and Zenodo.

### Raw data

Never committed (`.gitignore` excludes `data/raw/`). `src/t13/data.py` fetches it on
first use and caches it under `data/raw/`:

| stream | source | cached as |
|---|---|---|
| `synb2b` | Zenodo record 21668281, `synb2b_fraud.csv` | `data/raw/synb2b_fraud.csv` |
| `ulb` | OpenML `data_id=1597` via `sklearn.datasets.fetch_openml` | `data/raw/ulb_creditcard.parquet` |

---

## 2. Seeds

The seed list is not chosen after seeing results and is not passed on the command
line. It is `C.SEEDS` in `src/t13/config.py`, quoted here verbatim:

```python
# Twenty seeds, fixed before the run and published here rather than chosen after
# seeing results. The first five are the original set, kept in place so every earlier
# comparison stays paired; the remaining fifteen are simply the next fifteen primes
# after 71. No seed is dropped for being inconvenient, and every method sees the same
# stream under a given seed, so method contrasts are paired throughout.
SEEDS = [11, 23, 37, 53, 71,
         73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
```

`src/t13/experiment.py` uses all twenty for the full grid and `C.SEEDS[:2]` — that is,
`[11, 23]` — for `--small`. `tools/validation_ladder.py` defaults to the first five,
`[11, 23, 37, 53, 71]`, and prints them in its report header.

Confirm the list without editing anything:

    PYTHONPATH=src python3 -c "from t13 import config as C; print(C.SEEDS)"

Also fixed in `config.py`: `TRAIN_FRACTION = 0.20` and
`BOOTSTRAP_RESAMPLES = 10000`.

---

## 3. Make targets

`PYTHONPATH=src` is exported by the Makefile; `PY` defaults to `python3` and can be
pointed at the virtualenv (`make PY=./.venv/bin/python <target>`).

| target | runs | regenerates |
|---|---|---|
| `venv` | `python -m venv` + pip install | `.venv/` |
| `test` | `pytest tests -q` | nothing (gate) |
| `reproduce-small` | `t13.experiment --small`, `tools/make_summary.py --small` | `results/results_small.json`, `results/SUMMARY_small.md`, `results/drift_events.json` |
| `reproduce-full` | `t13.experiment`, `tools/make_summary.py` | `results/results.json`, `results/SUMMARY.md`, `results/drift_events.json` |
| `paper-data` | `tools/make_paper_data.py`, `tools/inline_paper_data.py` | `results/paper_data.tex`, spliced into `paper/main.tex` between the `INLINE-DATA` markers |
| `drift-params` | `tools/calibrate_drift.py` | `src/t13/drift_params.json` |
| `numbers` | `tools/check_numbers.py` | `NUMERICAL_CHECKS.md` |
| `diagnostics` | `tools/make_diagnostics_report.py`, `tools/make_flag_report.py` | `DIAGNOSTICS.md`, `INFEASIBILITY_VALIDATION.md` |
| `refcheck` | `tools/refcheck_arxiv.py` | `refcheck/arxiv_verified.json` (network) |
| `audit` | `tools/audit_prose.py paper/main.tex` | `WRITING_AUDIT.md` |
| `paper` | `tools/build_paper.py` | `paper/.build/` PDFs, page counts, leak scan (the PDF is a build check, not a deliverable; `.gitignore` excludes it) |
| `anon-artifact` | `tools/make_anon_artifact.py` | `../anon-artifact/synb2b-fraud-stream-anonymous/` plus `.tar.gz` and `.zip` |
| `clean` | — | removes LaTeX scratch and `__pycache__` |

`numbers`, `diagnostics`, `paper-data` and `audit` all read `results/results.json`
specifically — never `results_small.json`. That is why `./reproduce.sh --quick` does
not regenerate the numeric reports: a 2-seed pass is a smoke test of the pipeline, not
a source for the paper's numbers.

`make drift-params` is deliberately outside both `reproduce.sh` paths. It re-solves the
drift constants that `config.py` loads from `drift_params.json`; running it changes the
label-generating process and therefore every downstream number. Run it only when
re-deriving the calibration itself.

---

## 4. Generated reports and the command that produces each

| report | command | source |
|---|---|---|
| `results/SUMMARY.md` | `make reproduce-full` | `results/results.json` |
| `results/SUMMARY_small.md` | `make reproduce-small` | `results/results_small.json` |
| `NUMERICAL_CHECKS.md` | `make numbers` | `results/results.json` + `paper/main.tex` |
| `DIAGNOSTICS.md` | `make diagnostics` | `results/results.json` |
| `INFEASIBILITY_VALIDATION.md` | `make diagnostics` | `results/results.json` |
| `WRITING_AUDIT.md` | `make audit` | `paper/main.tex`, `results/results.json` |
| `ESTIMATOR_VALIDATION_LADDER.md` | `PYTHONPATH=src python3 tools/validation_ladder.py --seeds 5` | re-runs the policy; does not read `results.json` |
| `PAIRED_VARIANT_DIFFERENCES.csv` | `PYTHONPATH=src python3 tools/make_paired_diffs.py` | `results/results.json` |
| `PAIRED_VARIANT_DIFFERENCES.md` | same command | `results/results.json` |
| `COHORT_ALIGNMENT_AUDIT.md` | `PYTHONPATH=src python3 tools/make_cohort_audit.py` | `results/results.json` |
| `FINAL_CLAIM_EVIDENCE.csv` | `python3 tools/make_claim_evidence.py` | imports `tools/check_numbers.py`; reads `results/results.json` + `paper/main.tex` |
| anonymous artifact | `make anon-artifact` | `git archive HEAD`, then identifier substitution and a leak scan |

`tools/validation_ladder.py` accepts `--seeds`, `--stream`, `--method`, `--small`,
`--stages` and `--out`. `tools/make_paired_diffs.py` and `tools/make_cohort_audit.py`
accept `--results` and their `--out-csv` / `--out-md` paths, so both can be pointed at
`results/results_small.json` for a dry run.

### `SEED_LEVEL_RESULTS.csv` — not produced by anything in this tree

There is no script, make target or committed file named `SEED_LEVEL_RESULTS.csv`
anywhere in the repository or its history. What exists instead:

- `src/t13/experiment.py` now writes a `per_seed` array alongside `mean`/`sd`/`n`/
  `ci_lo`/`ci_hi` for every aggregated metric, so the replicate-level values are
  persisted inside `results/results.json` itself.
- `tools/make_paired_diffs.py` reads those arrays when present (it looks for
  `per_seed`, `seeds`, `values`, `raw`, or a sibling `<metric>_per_seed`) and switches
  from a difference-of-means report to a genuine paired one; its header line states
  which mode it ran in.

The `results/results.json` currently on disk predates that change and carries no
`per_seed` arrays, so `PAIRED_VARIANT_DIFFERENCES.md` still reports means-only. A
completed `make reproduce-full` on the current source fixes that. If a standalone
seed-level CSV is wanted, it would be a flattening of those arrays and does not exist
yet.

---

## 5. Runtimes

Measured figures come from the artifacts themselves and are cited with their source.
Estimates are marked as such and are not evidence.

| stage | runtime | basis |
|---|---|---|
| `make reproduce-full` (20 seeds, both streams) | **5409 s** (1 h 30 m) | measured — `wall_clock_s = 5408.99` in `results/results.json`, Apple M4 Pro, 12 cores |
| `make reproduce-full`, second recorded run | **6064 s** (1 h 41 m) | measured on the same machine; reported by the operator, not recoverable from the tree |
| `make reproduce-small` (2 seeds) | **41.7 s** | measured — `wall_clock_s = 41.69` in `results/results_small.json` |
| `tools/validation_ladder.py --seeds 5` | **121 s** for 40 runs | measured — stated in the header of `ESTIMATOR_VALIDATION_LADDER.md` |
| `make test` | ~1-2 min | estimate |
| `make numbers` / `make diagnostics` / `make paper-data` | seconds each | estimate; they only re-read a 2.4 MB JSON |
| `tools/make_paired_diffs.py`, `tools/make_cohort_audit.py`, `tools/make_claim_evidence.py` | seconds each | estimate; same |
| `make audit` | under a minute | estimate; it invokes `pdflatex` for the page-count gates |
| `make paper` | 1-2 min | estimate; two `pdflatex` passes per anonymisation setting |
| `make refcheck` | minutes, network-bound | estimate |
| `make anon-artifact` | under a minute | estimate |
| `./reproduce.sh --quick` end to end | under 5 min | estimate, from the measured 41.7 s grid plus the estimated stages above |
| `./reproduce.sh --full` end to end | 1.5-2 h | estimate, dominated by the measured grid |

The grid dominates everything. Each of the two full-grid timings covers 6 methods plus
the oracle, 2 streams, 20 seeds, 9 ablation factors and 10,000 bootstrap resamples.

---

## 6. What is bit-reproducible and what is not

**Bit-reproducible.** Every simulated and estimated quantity. The streams, the drift
events, the scorer, the policy trajectories, the ablation grid and the bootstrap
intervals are all seeded — `C.SEEDS` for the grid, fixed constants inside
`experiment.py` for the bootstrap draws. Re-running `make reproduce-full` on the same
source and pinned environment reproduces `results/results.json` to the last digit,
apart from the exceptions below. `DECISIONS.md` D-44 records the check that established
this: diffing a rerun against the previous results file moved 448 values, and all 448
were timing.

**Not reproducible: wall-clock timing.** `update_us_p50` and `update_us_p99` are the
only non-reproducible numbers in the artifact. They are host measurements of how long a
calibration update takes, so they move with the machine, its thermal state and whatever
else it is doing. Across five runs of identical work the M5 p50 has landed between
**17.98 and 18.62 microseconds** — the paper prints 18.56, and `DECISIONS.md` D-44
records 18.36, 18.03 and 17.98 from three further runs. The paper says so where it
reports the figure: *"the only figure here not bit-reproducible."*

That 17.98-18.62 band is the spread over runs of *identical* work. The
`results/results.json` currently on disk reads `M5.update_us_p50 = 17.393`, below the
band, and it was produced from a modified working tree — `git status` shows
`src/t13/simulate.py`, `src/t13/config.py` and `src/t13/experiment.py` as changed. A
timing figure from a changed workload is not evidence about run-to-run variance either
way; only reruns of one fixed source belong in the band.

Four claim rows are affected — `M5.update_us_p50`, `M5.update_us_p99`,
`ulb.M5.update_us_p50`, `ulb.M4.update_us_p50` — and they are the four rows with
`deterministic=false` in `FINAL_CLAIM_EVIDENCE.csv`. Also non-reproducible, and not a
result: `wall_clock_s` and the `utc` timestamp in the provenance block.

A rerun that shifts only those numbers has reproduced the artifact. A rerun that shifts
anything else has not, and `make numbers` will say so.

---

## 7. Verifying a reproduction

    make numbers      # fails if any printed number no longer matches its source field
    make audit        # fails if any writing gate is missed
    make paper        # fails on a LaTeX error, an over-length build or an anonymity leak

    python3 tools/make_claim_evidence.py     # FINAL_CLAIM_EVIDENCE.csv

`FINAL_CLAIM_EVIDENCE.csv` carries one row per checked claim with `section`, `claim`,
`source_field`, `generating_script`, `value`, `appears_in_paper`, `deterministic` and
`notes`. It imports the claim list from `tools/check_numbers.py` rather than restating
it, so the two cannot drift, and it never runs the experiment. Of the 85 claims, 81 are
read or derived from `results/results.json` and 4 — the Proposition 2b sample sizes in
Section IV-E — are closed forms over stipulated constants with no simulation input.

`make numbers` is the gate; the CSV is the evidence trail behind it. When the CSV shows
`MISMATCH` in `notes` for rows that are not the four timing rows, `results/results.json`
and `paper/main.tex` have fallen out of step and `make paper-data` has not been re-run
since the last grid.
