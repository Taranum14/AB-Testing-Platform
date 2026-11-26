import numpy as np
from src.simulate import simulate_experiment
from src.cuped import cuped_adjust, variance_reduction, diff_in_means
from src.sequential import msprt_always_valid_p
from src.peeking_simulation import peeking_false_positive_rates


def test_variance_reduction_matches_theory():
    df = simulate_experiment(n=200000, rho=0.7, lift=0.0, seed=1)
    adj, _ = cuped_adjust(df.metric, df.pre_metric)
    assert abs(variance_reduction(df.metric, adj) - 0.49) < 0.02   # rho squared


def test_cuped_is_unbiased_and_tighter():
    raw, cuped = [], []
    for s in range(200):
        df = simulate_experiment(n=4000, rho=0.7, lift=0.03, seed=s)
        adj, _ = cuped_adjust(df.metric, df.pre_metric)
        raw.append(diff_in_means(df.metric, df.group)["diff"])
        cuped.append(diff_in_means(adj, df.group)["diff"])
    assert abs(np.mean(cuped) - 3.0) < 0.15      # true effect is +3 units
    assert np.std(cuped) < np.std(raw)


def test_sequential_p_value_never_increases():
    rng = np.random.default_rng(3)
    p = msprt_always_valid_p(rng.normal(100, 20, 1000), rng.normal(100, 20, 1000), 3.0)
    assert np.all(np.diff(p) <= 1e-12)


def test_sequential_detects_a_real_effect():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(50):
        t, c = rng.normal(103, 20, 5000), rng.normal(100, 20, 5000)
        hits += msprt_always_valid_p(t, c, 3.0)[-1] < 0.05
    assert hits / 50 > 0.9


def test_peeking_inflates_false_positives_but_msprt_does_not():
    r = peeking_false_positive_rates(n_sims=500, seed=7)
    assert r["naive_peeking"] > 0.10
    assert r["msprt_peeking"] < 0.07