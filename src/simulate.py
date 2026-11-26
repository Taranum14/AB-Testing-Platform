"""Simulated A/B experiment with a pre-experiment metric."""
import numpy as np
import pandas as pd


def simulate_experiment(n=20000, rho=0.7, lift=0.03, seed=42):
    """Metric = baseline 100. `rho` is the correlation with the pre-experiment metric.
    `lift` is the true treatment effect as a share of the baseline (0.03 = +3 units)."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)
    pre = rng.normal(100, 20, n)
    noise = rng.normal(0, 20, n)
    metric = 100 + rho * (pre - 100) + np.sqrt(1 - rho**2) * noise + lift * 100 * group
    return pd.DataFrame({"group": group, "pre_metric": pre, "metric": metric})