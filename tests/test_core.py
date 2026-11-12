import numpy as np
from src.simulate import simulate_experiment
from src.cuped import cuped_adjust, variance_reduction
from src.peeking_simulation import peeking_false_positive_rate


def test_cuped_variance_reduction_matches_theory():
    df = simulate_experiment(n=50_000, rho=0.7, seed=1)
    y_adj, _ = cuped_adjust(df.metric, df.pre_metric)
    assert abs(variance_reduction(df.metric.values, y_adj) - 0.49) < 0.03  # rho^2


def test_cuped_is_unbiased():
    diffs_raw, diffs_cuped = [], []
    for s in range(200):
        df = simulate_experiment(n=4000, rho=0.7, lift=0.02, seed=s)
        y_adj, _ = cuped_adjust(df.metric, df.pre_metric)
        g = df.group.values
        diffs_raw.append(df.metric.values[g == 1].mean() - df.metric.values[g == 0].mean())
        diffs_cuped.append(y_adj[g == 1].mean() - y_adj[g == 0].mean())
    assert abs(np.mean(diffs_cuped) - 1.0) < 0.1      # true effect = 0.02 * 50
    assert np.std(diffs_cuped) < np.std(diffs_raw)


def test_aa_false_positive_rate_controlled():
    res = peeking_false_positive_rate(n_sims=500, seed=3)
    naive = [v for k, v in res.items() if "naive" in k][0]
    msprt = [v for k, v in res.items() if "mSPRT" in k][0]
    assert naive > 0.10 and msprt < 0.07


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)