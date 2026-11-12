import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .simulate import simulate_experiment
from .stats_tests import welch_ttest, sample_size_per_group
from .cuped import cuped_adjust, variance_reduction
from .sequential import msprt_path, first_stop
from .peeking_simulation import peeking_false_positive_rate

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    df = simulate_experiment(n=20_000, rho=0.7, lift=0.03, seed=42)
    ctrl, trt = df[df.group == 0], df[df.group == 1]

    # 1) Raw vs CUPED (fixed horizon)
    raw = welch_ttest(ctrl.metric, trt.metric)
    y_adj, theta = cuped_adjust(df.metric, df.pre_metric)
    df["metric_cuped"] = y_adj
    cup = welch_ttest(df[df.group == 0].metric_cuped, df[df.group == 1].metric_cuped)
    vr = variance_reduction(df.metric.values, y_adj)
    print(f"theta = {theta:.3f} | variance reduction = {vr:.1%}")
    for name, r in [("Raw", raw), ("CUPED", cup)]:
        print(f"{name:6s} diff={r['diff']:.3f}  CI=({r['ci_low']:.3f}, {r['ci_high']:.3f})  "
              f"p={r['p_value']:.4f}  CI width={r['ci_high'] - r['ci_low']:.3f}")

    sd = df.metric.std()
    n_raw = sample_size_per_group(sd, mde_abs=1.0)
    n_cup = sample_size_per_group(y_adj.std(), mde_abs=1.0)
    print(f"Sample size/arm to detect +1.0: raw={n_raw:,}  CUPED={n_cup:,}  "
          f"(saves {1 - n_cup / n_raw:.0%})")

    # 2) Sequential monitoring
    c = df[df.group == 0].metric_cuped.values
    t = df[df.group == 1].metric_cuped.values
    ns, diffs, pv = msprt_path(c, t, tau=1.5, step=100)
    print(f"mSPRT first stop (CUPED): n/arm = {first_stop(ns, pv)}")
    ns_r, _, pv_r = msprt_path(ctrl.metric.values, trt.metric.values, tau=1.5, step=100)
    print(f"mSPRT first stop (raw):   n/arm = {first_stop(ns_r, pv_r)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, pv, label="CUPED + mSPRT")
    ax.plot(ns_r, pv_r, label="Raw + mSPRT")
    ax.axhline(0.05, color="red", ls="--", label="alpha = 0.05")
    ax.set_yscale("log"); ax.set_xlabel("Users per arm"); ax.set_ylabel("Always-valid p-value")
    ax.set_title("Sequential monitoring: CUPED reaches significance sooner"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sequential_monitoring.png"), dpi=150)

    # 3) Peeking simulation
    res = peeking_false_positive_rate()
    print("\nFalse positive rate in A/A tests (target 5%):")
    for k, v in res.items():
        print(f"  {k:32s}: {v:.1%}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(res.keys()), [v * 100 for v in res.values()],
           color=["#4c78a8", "#e45756", "#54a24b"])
    ax.axhline(5, color="k", ls="--"); ax.set_ylabel("False positive rate (%)")
    ax.set_title("Peeking inflates false positives; mSPRT controls them")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "peeking_fpr.png"), dpi=150)


if __name__ == "__main__":
    main()