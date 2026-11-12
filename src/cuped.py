import numpy as np


def cuped_adjust(y, x, theta=None):
    y, x = np.asarray(y, float), np.asarray(x, float)
    if theta is None:
        theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def variance_reduction(y, y_adj):
    return 1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1)