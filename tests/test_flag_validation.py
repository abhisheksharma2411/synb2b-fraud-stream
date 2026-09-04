"""The infeasibility flag is scored against oracle feasibility, and that scoring
must not leak back into the policy.

The oracle reads latent y. If it ever influenced a decision the whole experiment
would be circular, so these tests pin two things: the confusion matrix is
internally consistent, and running with the oracle computation present produces
exactly the actions a run without it would.
"""
import numpy as np
import pytest

from t13 import config as C
from t13 import simulate


def _run(ctx, method="M5"):
    return simulate.run_policy(ctx, method, C.PolicyConfig())


@pytest.fixture(scope="module")
def run(ctx_small):
    return _run(ctx_small)


def test_confusion_matrix_partitions_the_scored_windows(run):
    tp, fp, fn, tn = (run[k] for k in ("flag_tp", "flag_fp", "flag_fn", "flag_tn"))
    assert tp + fp + fn + tn == run["flag_n_scored"]


def test_undetermined_windows_are_excluded_not_guessed(run):
    # every update is either scored or explicitly undetermined
    assert run["flag_n_scored"] + run["flag_undetermined"] == run["n_updates"]


def test_rates_are_consistent_with_the_counts(run):
    tp, fp, fn, tn = (run[k] for k in ("flag_tp", "flag_fp", "flag_fn", "flag_tn"))
    if tp + fn:
        assert run["flag_recall"] == pytest.approx(tp / (tp + fn))
        assert run["flag_missed_infeasible_rate"] == pytest.approx(fn / (tp + fn))
    if tp + fp:
        assert run["flag_precision"] == pytest.approx(tp / (tp + fp))
    if tn + fp:
        assert run["flag_specificity"] == pytest.approx(tn / (tn + fp))
        assert run["flag_false_infeasible_rate"] == pytest.approx(fp / (fp + tn))


def test_oracle_rate_matches_the_scored_oracle_labels(run):
    tp, fp, fn, tn = (run[k] for k in ("flag_tp", "flag_fp", "flag_fn", "flag_tn"))
    assert run["oracle_infeasible_rate"] == pytest.approx((tp + fn) / (tp + fp + fn + tn))


def test_oracle_never_changes_the_policy(run, ctx_small):
    """Same seed, same actions, whatever the oracle computes.

    The oracle is a function of latent y. If the action stream depended on it, a
    second run whose oracle labels were forced to a constant would diverge. Patch
    the flag to a fixed value and confirm the decisions are untouched.
    """
    orig = simulate._score_flag
    try:
        simulate._score_flag = lambda *a, **k: {"flag_n_scored": 0, "flag_undetermined": 0}
        other = _run(ctx_small)
    finally:
        simulate._score_flag = orig
    for k in ("for_overall", "review_demand_rate", "review_served_rate",
              "block_rate", "infeasible_rate", "cost_per_1k"):
        assert other[k] == pytest.approx(run[k]), f"{k} moved when the oracle was stubbed"
