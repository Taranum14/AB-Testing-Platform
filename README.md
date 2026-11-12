# A/B Test Analysis Toolkit: CUPED and Sequential Testing

A Python toolkit for analyzing online experiments faster and more safely than a standard t-test: it cuts the sample size needed using pre-experiment data (CUPED) and lets you monitor a test continuously without inflating false positives (mSPRT).

![Fixed horizon view](docs/fixed_horizon.png)

## Results

Evaluated on simulated experiments, where the true effect is known, so every method can be checked against ground truth.

| Question | Standard approach | This toolkit |
|---|---|---|
| Users per arm needed to detect a +1.0 effect | 6,320 | **3,233** (-49%) |
| 95% interval width on the estimated effect | 1.11 | **0.80** |
| Users per arm until a significant result is reachable | 7,800 | **5,200** |
| False positive rate with 20 interim looks (target 5%) | 23.7% | **1.1%** |

*Simulated data, pre/post correlation of 0.7. Gains depend on how well the pre-experiment metric predicts the in-experiment metric.*

## The problem

1. **Experiments are slow.** Noisy metrics like revenue per user need large samples to detect small effects.
2. **Checking results early breaks the statistics.** Teams look at dashboards daily, but stopping at the first significant p-value produces false winners far more often than the nominal 5%.

## Approach

**CUPED** (Deng et al., 2013). Each user's metric is adjusted using their own behavior before the experiment:
`Y_adj = Y - θ(X - mean(X))`, with `θ = Cov(X, Y) / Var(X)`.
This removes variance the pre-period already explains, shrinking it by a factor of `1 - ρ²`, while leaving the treatment effect unbiased because `X` is measured before randomization.

**Sequential testing with mSPRT** (Johari et al., 2015). A mixture sequential probability ratio test produces an *always-valid* p-value that can be checked at any time. Stopping the first time it falls below α keeps the false positive rate at or below α.

**Peeking simulation.** 1,000 A/A tests (no true effect) with 20 looks each, comparing a naive repeated t-test against mSPRT.

![Peeking false positive rates](outputs/peeking_fpr.png)

## Dashboard

An interactive Streamlit app with simulated data or your own CSV (`group`, `metric`, `pre_metric`).

| Sequential monitoring | Peeking check |
|---|---|
| ![](docs/sequential.png) | ![](docs/peeking.png) |

## Quick start

```bash
git clone https://github.com/<your-username>/ab-testing-platform.git
cd ab-testing-platform
pip install -r requirements.txt

python -m tests.test_core        # verify the statistics
python -m src.run_pipeline       # reproduce the results table, saves plots to outputs/
python -m streamlit run app.py   # launch the dashboard
```

## Project structure

```
src/
  simulate.py            synthetic experiment with a pre-experiment covariate
  stats_tests.py         Welch t-test, sample size calculation
  cuped.py               CUPED adjustment and variance reduction
  sequential.py          mSPRT always-valid p-values
  peeking_simulation.py  A/A simulation of peeking
  run_pipeline.py        end-to-end run, produces results and plots
tests/test_core.py       checks CUPED variance reduction matches theory,
                         CUPED is unbiased, and mSPRT controls false positives
app.py                   Streamlit dashboard
```

## Testing

The tests verify properties, not just that code runs:
- variance reduction matches the theoretical `ρ²`
- the CUPED estimate is unbiased over 200 repeated experiments
- peeking inflates false positives above 10% while mSPRT stays under 7%

## Limitations

- Results use simulated data. The method gains on real data depend on the correlation between the pre-experiment and experiment metrics.
- The CUPED coefficient `θ` is estimated from the experiment data itself. Production setups usually fit it on a prior period.
- mSPRT assumes approximately normal sample means and a user-chosen prior scale `tau`.
- Only a single metric is analyzed. There is no multiple-testing correction.

## Possible extensions

- Validate on a real dataset with a pre-period metric
- Group-sequential boundaries (O'Brien-Fleming) as a comparison to mSPRT
- Segment-level and uplift analysis

## References

- Deng, Xu, Kohavi, Walker. *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (2013)
- Johari, Pekelis, Walsh. *Always Valid Inference* (2015)