"""The invariants the whole design rests on.

Written before the policies were, and they caught two real bugs while it was being
built: a calibration window whose item cap silently excluded the entire 180-day ledger
channel, and a monotone projection that let one accidental positive in the low-score
head pin the risk curve at 1.0.
"""
import numpy as np
import pytest

from t13 import config as C
from t13.simulate import (ACT_BLOCK, ACT_OVF_BLOCK, CH_LEDGER, FutureLabelError,
                          assert_calibration_causal, prepare_stream, run_policy)


def test_guard_rejects_a_label_that_has_not_arrived():
    """Feed the guard a row whose label lands tomorrow and it must refuse it."""
    arrival = np.array([1.0, 2.0, 9.0, np.inf])
    assert_calibration_causal(arrival, np.array([0, 1]), now=5.0)  # fine
    with pytest.raises(FutureLabelError):
        assert_calibration_causal(arrival, np.array([0, 2]), now=5.0)


def test_guard_rejects_a_blocked_row():
    """A blocked payment is never adjudicated, so its arrival is infinite and it must
    never reach the estimator - not even as a zero."""
    arrival = np.array([1.0, np.inf])
    with pytest.raises(FutureLabelError):
        assert_calibration_causal(arrival, np.array([0, 1]), now=100.0)


@pytest.mark.parametrize("method", ["M2", "M3", "M4", "M5"])
def test_no_future_label_reaches_calibration(ctx_small, method):
    """The in-loop guard runs on every update of every adaptive method."""
    pol = C.policy_for("synb2b")
    run_policy(ctx_small, method, pol, check_invariants=True)


@pytest.mark.parametrize("method", ["M2", "M3", "M4", "M5"])
def test_blocked_transactions_are_never_labelled(ctx_small, method):
    pol = C.policy_for("synb2b")
    r = run_policy(ctx_small, method, pol, keep_arrays=True)
    a, ch = r["_action"], r["_channel"]
    blocked = np.isin(a, (ACT_BLOCK, ACT_OVF_BLOCK))
    # M0/M1 freeze a threshold from a training split that carries almost no fraud
    # and can end up blocking nothing at all, which would make this vacuous.
    assert blocked.sum() > 0, "the test is vacuous if nothing was ever blocked"
    assert np.all(ch[blocked] == -1)
    assert np.all(~np.isfinite(r["_arrival"][blocked]))


def test_allow_path_labels_land_no_earlier_than_the_disclosure_law_allows(ctx_small):
    pol = C.policy_for("synb2b")
    r = run_policy(ctx_small, "M5", pol, keep_arrays=True)
    ch, arr, dec = r["_channel"], r["_arrival"], r["_dec_day"]
    led = (ch == CH_LEDGER) & np.isfinite(arr)
    lag = arr[led] - dec[led]
    assert lag.min() >= 0.0
    # an undisclosed item is booked clean at exactly the reconciliation horizon
    assert lag.max() <= C.MATURITY_DAYS + 1e-6
    assert np.isclose(lag.max(), C.MATURITY_DAYS)


def test_budget_is_never_exceeded_within_a_reporting_block(ctx_small):
    """The queue holds floor(b * block) slots. Nothing may be served beyond that."""
    pol = C.policy_for("synb2b")
    cap = int(np.floor(pol.budget * pol.metric_block))
    for method in C.METHODS:
        r = run_policy(ctx_small, method, pol)
        served = np.asarray(r["series"]["served"]) * pol.metric_block
        assert served.max() <= cap + 1, (
            f"{method} served {served.max()} reviews in a block, cap is {cap}")
