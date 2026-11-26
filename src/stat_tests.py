"""Core statistical tests for A/B experiments."""
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


def srm_check(n_control, n_treatment, expected_ratio=0.5):
    """Chi-square test that the group sizes match the planned split."""
    total = n_control + n_treatment
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {"chi2": float(chi2), "p_value": float(p)}


def two_proportion_test(x_control, n_control, x_treatment, n_treatment, alpha=0.05):
    """Two-sided z-test for a difference in proportions (treatment minus control)."""
    p_c = x_control / n_control
    p_t = x_treatment / n_treatment
    _, p_value = proportions_ztest([x_treatment, x_control], [n_treatment, n_control])
    low, high = confint_proportions_2indep(
        x_treatment, n_treatment, x_control, n_control,
        compare="diff", method="newcomb", alpha=alpha)
    return {"control_rate": p_c, "treatment_rate": p_t, "diff": p_t - p_c,
            "ci_low": float(low), "ci_high": float(high), "p_value": float(p_value)}


def bootstrap_diff(x_control, n_control, x_treatment, n_treatment, n_boot=10000, seed=42):
    """Bootstrap interval for a difference in proportions."""
    rng = np.random.default_rng(seed)
    pc, pt = x_control / n_control, x_treatment / n_treatment
    diffs = (rng.binomial(n_treatment, pt, n_boot) / n_treatment
             - rng.binomial(n_control, pc, n_boot) / n_control)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), "share_negative": float((diffs < 0).mean())}