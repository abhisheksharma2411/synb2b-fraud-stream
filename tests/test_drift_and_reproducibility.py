"""Drift has to actually bite, and a rerun has to reproduce the run."""
import json
import os

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from t13 import config as C
from t13.simulate import prepare_stream, run_policy


def test_injected_drift_degrades_the_scorer_where_it_was_injected(ctx_full):
    """Covariate and concept events both move what the scorer needs to know.

    The prior event is deliberately not asserted here. Lifting the base rate raises
    AUC-PR rather than lowering it - a larger positive class makes average precision
    easier - which is exactly why the paper does not use AUC-PR as a drift alarm.
    """
    deg = ctx_full.degradation
    for kind in ("covariate", "concept"):
        d = deg[kind]
        assert d["auc_pr_pre"] is not None and d["auc_pr_post"] is not None
        assert d["rel_drop"] > 0.25, f"{kind} drift barely moved AUC-PR: {d}"
    assert deg["prior"]["rel_drop"] < 0.0, "prior shift is expected to inflate AUC-PR"


def test_drift_events_land_where_they_are_logged(ctx_full):
    days, ev = ctx_full.days, ctx_full.events
    for e in ev:
        assert abs(days[e.index] - e.day) < 1e-6
        assert e.index >= ctx_full.n_train, "an event landed inside the training split"
    assert ev[0].index < ev[1].index < ev[2].index


def test_the_drift_channel_does_not_overwrite_the_benchmark(ctx_full):
    """SynB2B-Fraud's own 1,500 topology labels must all survive."""
    d = ctx_full.drift_diag
    assert d["n_base_fraud"] == 1500
    assert d["n_realised_fraud"] >= d["n_base_fraud"]
    assert d["concept_rate"] > 1.7 * d["quiet_rate"]
    assert d["prior_rate"] > 2.5 * d["quiet_rate"]
    # concept drift is supposed to move fraud onto established counterparties
    assert d["concept_established_share"] > 0.75


@pytest.mark.parametrize("method", ["M0", "M2", "M3", "M4", "M5"])
def test_a_rerun_reproduces_the_run_exactly(method):
    ctx_a = prepare_stream("synb2b", seed=37, small=True)
    pol = C.policy_for("synb2b")
    a = run_policy(ctx_a, method, pol)
    ctx_b = prepare_stream("synb2b", seed=37, small=True)
    b = run_policy(ctx_b, method, pol)
    for k in ("for_overall", "cost_per_1k", "review_served_rate", "block_rate",
              "infeasible_rate", "fraud_dollars_allowed", "n_explore"):
        assert a[k] == b[k], f"{method}.{k}: {a[k]!r} != {b[k]!r}"
    assert a["series"]["for"] == b["series"]["for"]


def test_stream_context_is_seed_deterministic():
    a = prepare_stream("synb2b", seed=53, small=True)
    b = prepare_stream("synb2b", seed=53, small=True)
    assert np.array_equal(a.y, b.y)
    assert np.allclose(a.scores, b.scores, rtol=0, atol=0)
    assert a.scorer == b.scorer


def test_different_seeds_give_different_label_realisations():
    a = prepare_stream("synb2b", seed=11, small=True)
    b = prepare_stream("synb2b", seed=23, small=True)
    assert not np.array_equal(a.y, b.y)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "results.json")),
    reason="full results.json not built yet",
)
def test_results_json_is_wellformed():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "results.json")
    with open(path) as f:
        r = json.load(f)
    assert r["schema"] == "t13-results/1"
    assert set(r["methods"]["synb2b"]) >= set(C.METHODS)
    for m in C.METHODS:
        cell = r["methods"]["synb2b"][m]["for_overall"]
        assert cell["ci_lo"] is not None and cell["ci_lo"] <= cell["mean"] <= cell["ci_hi"]
