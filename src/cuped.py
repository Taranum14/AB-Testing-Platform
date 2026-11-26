"""CUPED variance reduction (Deng et al., 2013)."""
import numpy as np


def cuped_adjust(metric, pre_metric):
    """Return (adjusted metric, theta). Adjusted = Y - theta * (X - mean(X))."""
    y = np.asarray(metric, float)
    x = np.asarray(pre_metric, float)
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def variance_reduction(metric, adjusted):
    """Share of the variance removed by the adjustment."""
    return 1 - np.var(adjusted, ddof=1) / np.var(metric, ddof=1)


def diff_in_means(y, group):
    """Treatment minus control, with a normal-approximation 95% interval."""
    y = np.asarray(y, float)
    g = np.asarray(group)
    t, c = y[g == 1], y[g == 0]
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"diff": diff, "se": se, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se}