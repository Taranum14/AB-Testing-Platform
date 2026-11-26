import numpy as np
from src.stat_tests import srm_check, two_proportion_test, bootstrap_diff
from src.design import sample_size_per_group


def test_srm_balanced_is_not_flagged():
    assert srm_check(5000, 5000)["p_value"] > 0.99


def test_srm_detects_imbalance():
    assert srm_check(6000, 4000)["p_value"] < 0.001


def test_two_proportion_known_values():
    r = two_proportion_test(1900, 10000, 1820, 10000)  # 19.0% vs 18.2%
    assert abs(r["diff"] - (-0.008)) < 1e-9
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_no_difference_is_not_significant():
    assert two_proportion_test(1000, 10000, 1000, 10000)["p_value"] > 0.9


def test_false_positive_rate_is_near_alpha():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(1000):
        xc, xt = rng.binomial(5000, 0.2), rng.binomial(5000, 0.2)
        hits += two_proportion_test(xc, 5000, xt, 5000)["p_value"] < 0.05
    assert 0.03 < hits / 1000 < 0.07


def test_bootstrap_interval_contains_observed_diff():
    r = bootstrap_diff(1900, 10000, 1820, 10000)
    assert r["ci_low"] < -0.008 < r["ci_high"]


def test_sample_size_matches_notebook():
    assert abs(sample_size_per_group(0.19, 0.01) - 23664) < 5