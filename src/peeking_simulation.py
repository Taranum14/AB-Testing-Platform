import numpy as np
from scipy import stats
from .sequential import log_lambda


def peeking_false_positive_rate(n_sims=1000, n_per_arm=5000, n_looks=20,
                                alpha=0.05, tau=0.1, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_sims, n_per_arm)).cumsum(axis=1)
    y = rng.standard_normal((n_sims, n_per_arm)).cumsum(axis=1)
    looks = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    naive_hit = np.zeros(n_sims, bool)
    p_always = np.ones(n_sims)
    final_only = None
    for n in looks:
        d = (y[:, n - 1] - x[:, n - 1]) / n
        v = 2.0 / n                                   # sigma = 1 known
        p_naive = 2 * stats.norm.sf(np.abs(d) / np.sqrt(v))
        naive_hit |= p_naive < alpha                  # any look significant
        p_always = np.minimum(p_always, np.exp(-log_lambda(d, v, tau)))
        final_only = p_naive < alpha
    return {
        "fixed_horizon (look once)": final_only.mean(),
        f"naive peeking ({n_looks} looks)": naive_hit.mean(),
        f"mSPRT ({n_looks} looks)": (p_always < alpha).mean(),
    }