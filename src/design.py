"""Sample-size planning."""
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def sample_size_per_group(baseline, mde, alpha=0.05, power=0.8):
    """Players needed per group to detect an absolute drop of `mde` from `baseline`."""
    es = abs(proportion_effectsize(baseline, baseline - mde))
    return int(round(NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power, ratio=1)))