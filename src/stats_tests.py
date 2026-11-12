import numpy as np
from scipy import stats


def welch_ttest(control, treatment, alpha=0.05):
    control, treatment = np.asarray(control), np.asarray(treatment)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / len(control)
                 + treatment.var(ddof=1) / len(treatment))
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    crit = stats.norm.ppf(1 - alpha / 2)
    return {"diff": diff, "se": se, "ci_low": diff - crit * se,
            "ci_high": diff + crit * se, "p_value": p}


def sample_size_per_group(sd, mde_abs, alpha=0.05, power=0.8):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * (sd * (z_a + z_b) / mde_abs) ** 2))