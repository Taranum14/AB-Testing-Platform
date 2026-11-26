"""Always-valid sequential testing with the mixture SPRT (Johari et al., 2015)."""
import numpy as np


def cumulative_stats(treatment, control):
    """Running sample size k, mean difference d and variance V = var_t + var_c.
    The i-th treatment and i-th control user are treated as arriving together."""
    t = np.asarray(treatment, float)
    c = np.asarray(control, float)
    n = min(len(t), len(c))
    t, c = t[:n], c[:n]
    k = np.arange(1, n + 1)
    mt, mc = np.cumsum(t) / k, np.cumsum(c) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = (np.cumsum(t**2) - k * mt**2) / (k - 1)
        vc = (np.cumsum(c**2) - k * mc**2) / (k - 1)
    return k, mt - mc, vt + vc


def msprt_always_valid_p(treatment, control, tau):
    """Always-valid p-value after each pair of users.
    `tau` is the prior standard deviation of the true effect (use the size of effect you care about)."""
    k, d, v = cumulative_stats(treatment, control)
    tau2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lr = 0.5 * np.log(v / (v + k * tau2)) + (k**2 * tau2 * d**2) / (2 * v * (v + k * tau2))
    log_lr = np.nan_to_num(log_lr, nan=0.0, posinf=700.0, neginf=0.0)
    log_lr[:2] = 0.0                       # too few points to estimate the variance
    running_max = np.maximum.accumulate(log_lr)
    return np.minimum(1.0, np.exp(-running_max))