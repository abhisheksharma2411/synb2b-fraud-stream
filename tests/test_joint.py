"""The joint likelihood has to be right, not merely stable.

The load-bearing test is the first one: data is generated from a known r and a known
surfacing probability under exactly the observation process the likelihood claims,
and the fit has to get both back. If the likelihood were mis-specified - if a
released row that closed clean were being read as a truthful negative, say - the
recovered r would be biased low and the recovered surfacing probability biased high,
and no amount of regularisation would hide it at n = 30000.
"""
import numpy as np
import pytest

from t13.joint import JointTimeModel, delay_cdf

N = 30000


def synth(n: int = N, seed: int = 7, t_now: float = 400.0, all_clean: bool = False):
    """Ground truth plus the exact observation process the likelihood assumes.

    r and the surfacing probability are both known logistic surfaces of observable
    columns only. r carries a slow seasonal term on transaction time; the surfacing
    probability is static, and depends on invoice size the way a real dispute
    process does - a large invoice is far more likely to be chased.

    Holding is score-driven, so the release channel covers only the lower part of the
    score range and the fitted surfaces have to extrapolate above it. That is the
    honest version of the problem, not a convenience.
    """
    rng = np.random.default_rng(seed)
    zeta = rng.normal(size=n)
    score = rng.beta(2.0, 5.0, size=n)
    u = rng.uniform(0.0, 365.0, size=n)

    tau = u / 365.0
    r = 1.0 / (1.0 + np.exp(-(-3.1 + 0.45 * zeta + 3.0 * score + 1.2 * score ** 2
                              + 0.8 * np.sin(2.0 * np.pi * tau))))
    pd = 1.0 / (1.0 + np.exp(-(0.10 + 0.62 * zeta - 0.9 * score)))

    ku = rng.random(n)
    kind = np.where(score > 0.55, 2, np.where(ku < 0.28, 0, 1))
    kind = np.where((kind == 2) & (ku < 0.10), 0, kind)   # exploration into the held band

    y = (rng.random(n) < r).astype(float)
    d = (rng.random(n) < pd).astype(float)
    delay = np.minimum(rng.gamma(2.0, 30.0, size=n), 180.0)
    age = np.clip(t_now - u, 0.0, None)                   # older rows have had longer
    surfaced = (kind == 1) & (y > 0) & (d > 0) & (delay <= age)
    if all_clean:
        surfaced = np.zeros(n, dtype=bool)
    return dict(zeta=zeta, score=score, u=u, kind=kind.astype(int), y=y, age=age,
                surfaced=surfaced, r=r, pd=pd)


def fit_on(d, **kw):
    m = JointTimeModel(**kw)
    return m.fit(d["zeta"], d["score"], d["u"], d["kind"], d["y"], d["age"],
                 d["surfaced"])


@pytest.fixture(scope="module")
def data():
    return synth()


# --------------------------------------------------------------------------
@pytest.mark.parametrize("disclosure", ["static", "slow"])
def test_recovers_the_generating_surfaces(data, disclosure):
    """Both prespecified configurations recover r and the surfacing probability."""
    m = fit_on(data, disclosure=disclosure, n_knots=4)
    assert m.converged, m.status
    assert m.n_fit == int((data["kind"] != 2).sum())
    assert m.n_iter > 0

    pd_hat = m.predict_pd(data["zeta"], data["score"], data["u"])
    r_hat = m.predict_r(data["zeta"], data["score"], data["u"])
    mae_pd = float(np.mean(np.abs(pd_hat - data["pd"])))
    mae_r = float(np.mean(np.abs(r_hat - data["r"])))
    print(f"\n[{disclosure}] MAE p_d = {mae_pd:.4f}   MAE r = {mae_r:.4f}   "
          f"iters = {m.n_iter}")
    assert mae_pd < 0.10, f"surfacing probability MAE {mae_pd:.4f}"
    assert mae_r < 0.05, f"r MAE {mae_r:.4f}"


def test_a_clean_release_is_not_a_truthful_negative():
    """The whole point of the censored term.

    Same rows, two readings. Reading A is the correct one: the row was released, its
    ledger window has not closed on a signal, so the joint event {fraud, it left a
    trace, the trace arrived} simply failed. Reading B is the naive one that a
    ratio-of-two-fits pipeline effectively takes: call it a clean label and be done.
    Reading B has to drag r toward zero; reading A must not.
    """
    d = synth(seed=11, all_clean=True)
    m_joint = fit_on(d, disclosure="static", n_knots=4)
    assert m_joint.converged, m_joint.status

    naive = dict(d)
    rel = d["kind"] == 1
    naive["kind"] = np.where(rel, 0, d["kind"])           # wrongly adjudicated
    naive["y"] = np.where(rel, 0.0, d["y"])               # wrongly clean
    m_naive = fit_on(naive, disclosure="static", n_knots=4)
    assert m_naive.converged, m_naive.status

    r_joint = m_joint.predict_r(d["zeta"], d["score"], d["u"])
    r_naive = m_naive.predict_r(d["zeta"], d["score"], d["u"])
    print(f"\nmean r: joint = {r_joint.mean():.4f}  naive = {r_naive.mean():.4f}  "
          f"truth = {d['r'].mean():.4f}")
    assert r_joint.mean() > r_naive.mean()
    # and the censored reading is the one that is actually closer to the truth
    assert (np.mean(np.abs(r_joint - d["r"]))
            < np.mean(np.abs(r_naive - d["r"])))


@pytest.mark.parametrize("disclosure", ["static", "slow"])
def test_predicted_surfacing_probability_stays_in_range(data, disclosure):
    m = fit_on(data, disclosure=disclosure, n_knots=4)
    for z, s, u in (
        (data["zeta"], data["score"], data["u"]),
        (np.array([-50.0, 0.0, 50.0]), np.array([-10.0, 0.5, 10.0]),
         np.array([-1e4, 180.0, 1e4])),
    ):
        p = m.predict_pd(z, s, u)
        assert np.all(np.isfinite(p))
        assert p.min() >= 0.05 - 1e-12 and p.max() <= 0.97 + 1e-12

    # the unfitted fallback path is bounded too
    blank = JointTimeModel(disclosure=disclosure)
    assert np.all((blank.predict_pd(np.zeros(3), np.zeros(3), np.zeros(3)) >= 0.05)
                  & (blank.predict_pd(np.zeros(3), np.zeros(3), np.zeros(3)) <= 0.97))


@pytest.mark.parametrize("disclosure", ["static", "slow"])
def test_fit_is_deterministic(data, disclosure):
    a = fit_on(data, disclosure=disclosure, n_knots=4)
    b = fit_on(data, disclosure=disclosure, n_knots=4)
    assert np.array_equal(a.coef_r, b.coef_r)
    assert np.array_equal(a.coef_p, b.coef_p)
    assert a.n_iter == b.n_iter and a.converged == b.converged
    q = (data["zeta"], data["score"], data["u"])
    assert np.array_equal(a.predict_pd(*q), b.predict_pd(*q))
    assert np.array_equal(a.predict_r(*q), b.predict_r(*q))


def test_held_rows_contribute_nothing(data):
    """A payment that never went out is never adjudicated, so it cannot move the fit."""
    keep = data["kind"] != 2
    trimmed = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v)
               for k, v in data.items()}
    a = fit_on(data, disclosure="slow", n_knots=4)
    b = fit_on(trimmed, disclosure="slow", n_knots=4)
    assert np.array_equal(a.coef_r, b.coef_r)
    assert np.array_equal(a.coef_p, b.coef_p)


def test_module_never_reaches_for_the_unobservable():
    """Neither surface may be built from anything the policy cannot see."""
    import inspect

    import t13.joint as J

    src = inspect.getsource(J)
    for forbidden in ("topology", "y_true", "p_disclose"):
        assert forbidden not in src, f"joint.py references {forbidden!r}"


def test_delay_cdf_matches_the_stated_arrival_law():
    a = np.array([-5.0, 0.0, 30.0, 60.0, 179.0, 180.0, 1e4])
    f = delay_cdf(a)
    assert f[0] == 0.0 and f[1] == 0.0
    assert np.all(np.diff(f) >= 0.0)
    assert f[-1] == 1.0 and f[-2] == 1.0          # truncated at the ledger close
    # closed form for shape 2: 1 - e^{-x}(1 + x), x = a / scale
    x = a[2:5] / 30.0
    assert np.allclose(f[2:5], 1.0 - np.exp(-x) * (1.0 + x), atol=1e-12)


def test_thin_data_declines_rather_than_guessing():
    d = synth(n=40, seed=3)
    m = fit_on(d, disclosure="static")
    assert not m.converged and m.status == "insufficient-data"
    p = m.predict_pd(d["zeta"], d["score"], d["u"])
    assert np.all((p >= 0.05) & (p <= 0.97))
