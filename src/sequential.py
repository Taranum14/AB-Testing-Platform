import numpy as np


def log_lambda(diff, var_diff, tau):
    return 0.5 * np.log(var_diff / (var_diff + tau**2)) + \
        tau**2 * diff**2 / (2 * var_diff * (var_diff + tau**2))


def msprt_path(control, treatment, tau, sigma=None, step=100):
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    m = min(len(control), len(treatment))
    ns = np.arange(step, m + 1, step)
    cc, ct = np.cumsum(control[:m]), np.cumsum(treatment[:m])
    cc2, ct2 = np.cumsum(control[:m] ** 2), np.cumsum(treatment[:m] ** 2)
    diffs, pvals, p_run = [], [], 1.0
    for n in ns:
        mc, mt = cc[n - 1] / n, ct[n - 1] / n
        if sigma is None:
            var_c = (cc2[n - 1] - n * mc**2) / (n - 1)
            var_t = (ct2[n - 1] - n * mt**2) / (n - 1)
            v = (var_c + var_t) / n
        else:
            v = 2 * sigma**2 / n
        d = mt - mc
        p_run = min(p_run, float(np.exp(-log_lambda(d, v, tau))))  # running minimum
        diffs.append(d)
        pvals.append(p_run)
    return ns, np.array(diffs), np.array(pvals)


def first_stop(ns, pvals, alpha=0.05):
    idx = np.where(pvals < alpha)[0]
    return int(ns[idx[0]]) if len(idx) else None