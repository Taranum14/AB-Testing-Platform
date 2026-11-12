import numpy as np
import pandas as pd


def simulate_experiment(n=20_000, rho=0.7, mean=50.0, sd=20.0, lift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(mean, sd, n)
    noise = rng.normal(0, sd, n)
    base = mean + rho * (pre - mean) + np.sqrt(1 - rho**2) * noise
    group = rng.integers(0, 2, n)          # 0 = control, 1 = treatment
    metric = base + group * lift * mean
    return pd.DataFrame(
        {"user_id": np.arange(n), "group": group,
         "pre_metric": pre, "metric": metric}
    )