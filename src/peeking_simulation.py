"""How often does 'stop at the first significant result' give a false winner?"""
import numpy as np
from scipy import stats
from src.sequential import cumulative_stats, msprt_always_valid_p


def peeking_false_positive_rates(n_sims=1000, n_per_arm=5000, n_looks=20,
                                 alpha=0.05, tau=3.0, seed=0):
    """A/A tests (no true effect). Returns the share of tests that declare a winner."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int) - 1
    fixed = naive = sequential = 0
    for _ in range(n_sims):
        t = rng.normal(100, 20, n_per_arm)
        c = rng.normal(100, 20, n_per_arm)
        k, d, v = cumulative_stats(t, c)
        p = 2 * stats.norm.sf(np.abs(d / np.sqrt(v / k)))
        fixed += p[-1] < alpha                                   # look once, at the end
        naive += (p[looks] < alpha).any()                        # peek 20 times, stop at first p < alpha
        sequential += msprt_always_valid_p(t, c, tau)[looks].min() < alpha
    return {"fixed_horizon": fixed / n_sims, "naive_peeking": naive / n_sims,
            "msprt_peeking": sequential / n_sims}