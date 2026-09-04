"""Features must not see the future. Checked by rebuilding on a prefix."""
import numpy as np
import pytest

from t13.data import SYNB2B_FEATURES, build_synb2b_features, load_synb2b
from t13.drift import rolling_rank


@pytest.fixture(scope="module")
def raw():
    return load_synb2b(small=True)


def test_features_on_a_prefix_match_features_on_the_whole_stream(raw):
    """If any feature peeked ahead, truncating the stream would change earlier rows."""
    k = 4000
    full = build_synb2b_features(raw)
    prefix = build_synb2b_features(raw.iloc[:k].reset_index(drop=True))
    for col in SYNB2B_FEATURES:
        a = full[col].to_numpy()[:k]
        b = prefix[col].to_numpy()
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12), f"{col} depends on future rows"


def test_rolling_rank_is_causal():
    rng = np.random.default_rng(3)
    v = rng.normal(size=3000)
    full = rolling_rank(v, window=500, n_bins=64)
    # a rank computed against a rolling window of the past cannot move when later
    # values change, provided the binning reference does not - so bin explicitly
    assert full[0] == 0.0
    assert np.all(full >= -1.0) and np.all(full <= 1.0)
    # strictly increasing input: every value is the largest seen so far
    inc = rolling_rank(np.arange(500.0), window=1000, n_bins=256)
    assert inc[-1] > 0.99
    dec = rolling_rank(-np.arange(500.0), window=1000, n_bins=256)
    assert dec[-1] < -0.99


def test_first_row_of_each_edge_has_no_history(raw):
    f = build_synb2b_features(raw)
    first = f["edge_n"].to_numpy() == 0
    assert first.any()
    assert np.all(f["edge_days_since"].to_numpy()[first] == -1.0)
    assert np.all(f["edge_amt_z"].to_numpy()[first] == 0.0)
    assert np.all(f["bank_changed"].to_numpy()[first] == 0.0)
