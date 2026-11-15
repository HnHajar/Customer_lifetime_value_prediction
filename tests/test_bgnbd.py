"""
Tests numeriques de src/clv_bgnbd.py   ->   pytest -q tests/
  1. recuperation de parametres connus (simulation)
  2. calibration des predictions contre une simulation Monte-Carlo
  3. concordance avec `lifetimes` (saute si non installe)
  4. robustesse (a < 1, grands x, donnees incoherentes)
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.clv_bgnbd import BGNBDModel, GammaGammaModel  # noqa: E402


def simulate_bgnbd(n, r, alpha, a, b, T, horizon, seed=0):
    """Simule calibration (0,T] et holdout (T, T+horizon]. Pas de mort a t=0."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1.0 / alpha, n)
    p = rng.beta(a, b, n)
    t = np.zeros(n); alive = np.ones(n, bool)
    x = np.zeros(n); tx = np.zeros(n); fut = np.zeros(n)
    while True:
        act = alive & (t < T + horizon)
        if not act.any():
            break
        idx = np.where(act)[0]
        t[idx] += rng.exponential(1.0 / lam[idx])
        ev = idx[t[idx] < T + horizon]
        cal = ev[t[ev] <= T]; fu = ev[t[ev] > T]
        x[cal] += 1; tx[cal] = t[cal]; fut[fu] += 1
        die = ev[rng.random(len(ev)) < p[ev]]
        alive[die] = False
        alive[idx[t[idx] >= T + horizon]] = False
    return x, tx, np.full(n, float(T)), fut


@pytest.fixture(scope="module")
def sim():
    true = dict(r=0.8, alpha=30.0, a=1.5, b=3.0)
    x, tx, T, fut = simulate_bgnbd(40000, true["r"], true["alpha"], true["a"], true["b"], 300, 120)
    return true, x, tx, T, fut


def test_parameter_recovery(sim):
    true, x, tx, T, _ = sim
    m = BGNBDModel().fit(x, tx, T)
    for k, v in true.items():
        assert abs(m.params_[k] - v) / v < 0.15, (k, m.params_[k], v)


def test_calibration_vs_montecarlo(sim):
    """Moyenne des predictions ~ moyenne reelle, globalement ET par tranche de x."""
    true, x, tx, T, fut = sim
    m = BGNBDModel(); m.params_ = true
    pred = m.expected_purchases(120, x, tx, T)
    assert abs(pred.mean() - fut.mean()) / fut.mean() < 0.03
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 8), (9, 999)]:
        s = (x >= lo) & (x <= hi)
        if s.sum() > 300:
            assert abs(pred[s].mean() - fut[s].mean()) / max(fut[s].mean(), 1e-3) < 0.10, (lo, hi)


def test_prob_alive_bounds_and_monotone(sim):
    true, x, tx, T, _ = sim
    m = BGNBDModel(); m.params_ = true
    p = m.prob_alive(x, tx, T)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert (p[x == 0] == 1).all()
    # plus recent => plus vivant (x fixe)
    pa = m.prob_alive(np.array([5.]), np.array([50.]), np.array([300.]))
    pb = m.prob_alive(np.array([5.]), np.array([290.]), np.array([300.]))
    assert pb > pa


def test_expected_purchases_monotone_in_t(sim):
    true, x, tx, T, _ = sim
    m = BGNBDModel(); m.params_ = true
    prev = np.zeros(len(x))
    for t in (30, 90, 180, 365, 1095):
        cur = m.expected_purchases(t, x, tx, T)
        assert (cur >= prev - 1e-9).all()
        prev = cur


@pytest.mark.parametrize("a", [0.6, 0.874, 1.0, 1.3, 2.5])
def test_calibration_montecarlo_all_regimes(a):
    """Regression : a < 1 donnait des predictions nulles. Verifie aussi a = 1 et a > 1."""
    true = dict(r=0.8, alpha=30.0, a=a, b=2.5)
    x, tx, T, fut = simulate_bgnbd(40000, true["r"], true["alpha"], a, true["b"], 300, 120, seed=11)
    m = BGNBDModel(); m.params_ = true
    pred = m.expected_purchases(120, x, tx, T)
    assert pred.mean() > 0
    assert abs(pred.mean() - fut.mean()) / fut.mean() < 0.04
    for lo, hi in [(0, 0), (1, 2), (3, 8), (9, 999)]:
        s = (x >= lo) & (x <= hi)
        if s.sum() > 300:
            assert abs(pred[s].mean() - fut[s].mean()) / max(fut[s].mean(), 1e-3) < 0.12, (a, lo, hi)


@pytest.mark.parametrize("a", [0.4, 0.999, 1.0, 1.0001, 0.7, 2.5])
def test_no_nan_when_a_near_or_below_one(a):
    """lifetimes renvoie des NaN pour a <= 1 : ici jamais."""
    m = BGNBDModel(); m.params_ = dict(r=0.7, alpha=20.0, a=a, b=2.0)
    x = np.array([0, 1, 5, 40, 120.]); tx = np.array([0, 10, 100, 280, 299.]); T = np.full(5, 300.)
    out = m.expected_purchases(365 * 3, x, tx, T)
    assert np.isfinite(out).all() and (out >= 0).all()
    assert out[2:].min() > 0          # un client actif doit avoir une esperance > 0


def test_gamma_gamma_recovery():
    rng = np.random.default_rng(1)
    p, q, v = 4.0, 3.0, 15.0
    n = 20000
    x = rng.integers(1, 15, n).astype(float)
    nu = rng.gamma(q, 1.0 / v, n)
    m = np.array([rng.gamma(p * xi, 1.0 / (nu_i * xi)) for xi, nu_i in zip(x, nu)])
    g = GammaGammaModel().fit(x, m, clip_quantile=None)
    pop_true = p * v / (q - 1)
    pop_est = g.params_["p"] * g.params_["v"] / (g.params_["q"] - 1)
    assert abs(pop_est - pop_true) / pop_true < 0.05
    assert g.params_["q"] > 1
    # x = 0 -> moyenne de population
    e0 = g.expected_average_value(np.array([0.]), np.array([0.]))[0]
    assert abs(e0 - pop_est) < 1e-9


def test_rejects_inconsistent_data():
    with pytest.raises(ValueError):
        BGNBDModel().fit([3, 1], [0, 5], [100, 100])       # frequency>0 & recency=0
    with pytest.raises(ValueError):
        BGNBDModel().fit([1, 1], [5, 500], [100, 100])     # recency > T
    with pytest.raises(ValueError):
        BGNBDModel().fit([1, np.nan], [5, 5], [100, 100])


def test_matches_lifetimes(sim):
    pytest.importorskip("lifetimes", reason="lifetimes non installe : comparaison sautee")
    from lifetimes import BetaGeoFitter
    true, x, tx, T, _ = sim
    mine = BGNBDModel().fit(x, tx, T)
    ref = BetaGeoFitter(penalizer_coef=0.0).fit(x, tx, T)
    # meme optimum (log-vraisemblance) a quelques unites pres
    ll_ref = float(np.sum(__import__("src.clv_bgnbd", fromlist=["x"]).bgnbd_log_likelihood(
        [ref.params_[k] for k in ("r", "alpha", "a", "b")], x, tx, T)))
    assert mine.loglik_ >= ll_ref - 1e-3 * abs(ll_ref)
    # memes predictions SOUS LES MEMES parametres
    m2 = BGNBDModel(); m2.params_ = {k: float(ref.params_[k]) for k in ("r", "alpha", "a", "b")}
    ours = m2.expected_purchases(90, x, tx, T)
    theirs = ref.conditional_expected_number_of_purchases_up_to_time(90, x, tx, T)
    ok = np.isfinite(theirs)
    assert np.allclose(ours[ok], theirs[ok], rtol=1e-6, atol=1e-9)
    assert np.allclose(m2.prob_alive(x, tx, T), ref.conditional_probability_alive(x, tx, T), atol=1e-8)
