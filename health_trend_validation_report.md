# Health Trend Validation Report

Date: 2026-11-11
Project: Load-cell R-DAE + FCM health-trend pipeline

## 1. Purpose

This report documents the validation of the health-trend stage added after temporal R-DAE feature extraction and FCM clustering. It records the measured results and defines the limitations that must be resolved before using the trend as an input to Remaining Useful Life (RUL) prediction.

This is a pipeline-validation report. It is not evidence that the current data contains a complete physical degradation lifecycle.

## 2. Validation scope

The test used:

- All 20 `.txt` recordings in `adc_data/`
- The existing ADC parser for lines formatted as `ADC:<value>`
- Chronological 70% training, 15% validation, and 15% test partitions per file
- MinMax scaling over the pseudo-clean and raw values for each file
- The existing 32-sample temporal R-DAE window
- The existing LSTM R-DAE architecture and Adam optimizer
- 25 training epochs per recording
- Bottleneck features plus reconstruction-error statistics
- Three-cluster FCM grouping
- A 31-window moving-average smoother

The executable batch test is [test_health_trend_all_adc.py](test_health_trend_all_adc.py). The per-file measurements are stored in [health_trend_multi_file_results.csv](health_trend_multi_file_results.csv).

## 3. Health-trend calculation

The cluster IDs produced by FCM are arbitrary, so cluster number alone is not interpreted as a health state. For each file, clusters are ordered by their mean reconstruction error:

- lowest mean error: healthy-side cluster
- intermediate mean error: warning-side cluster
- highest mean error: degraded-side cluster

The raw degradation score is then calculated as:

```text
raw_degradation = 0.6 * fuzzy_cluster_severity
                 + 0.4 * normalized_mean_reconstruction_error
```

The score is smoothed with a centered moving average. The report also records:

- Trend slope: linear-fit slope of the smoothed score over test-window order
- Monotonicity ratio: fraction of consecutive smoothed-score differences that are non-negative
- State counts: threshold-based Healthy, Warning, and Degraded labels

These quantities are indicators of model output behavior, not calibrated physical health or remaining life.

## 4. Validation results

| Measure | Result |
| --- | ---: |
| ADC files tested | 20 / 20 |
| Files with positive fitted slope | 20 / 20 |
| Mean slope per test window | 0.00002609 |
| Minimum slope | 0.00000049 |
| Maximum slope | 0.00012230 |
| Mean monotonicity ratio | 0.5095 |
| Smoothing window | 31 windows |
| R-DAE epochs per file | 25 |

The single-file reference run for `535g20250502pm427ms20hz0.txt` produced:

- 4,151 samples
- 593 test windows
- Cluster order `[0, 2, 1]` from healthy-side to degraded-side
- Smoothed degradation range `[0.0000, 0.6673]`
- Trend slope `0.00012234`
- Monotonicity ratio `0.5541`
- State counts: Healthy `582`, Warning `10`, Degraded `1`

The detailed trend plot from that run is [loadcell_health_trend_results.png](loadcell_health_trend_results.png).

## 5. Interpretation

The implementation is operational and reproducible across the available recordings:

- Every file completed parsing, model training, feature extraction, FCM clustering, and trend calculation.
- Every fitted slope was positive, showing a small positive linear tendency under the current score definition.
- The monotonicity average of `0.5095` is close to an even split between upward and downward local changes.
- State counts are strongly imbalanced, with most windows classified as Healthy and usually only one classified as Degraded.

The correct conclusion is therefore:

> The health-trend stage is validated as a software and data-processing pipeline, but it is not yet validated as a physical degradation estimator.

A positive slope by itself is insufficient evidence of deterioration. It can result from signal drift, operating-condition differences, pseudo-clean target construction, cluster geometry, or the chosen score formula.

## 6. Limitations

### 6.1 No failure or RUL labels

The ADC recordings do not provide known failure times, degradation stages, or remaining-life labels. Consequently, the current experiment cannot calculate RUL accuracy, early-warning accuracy, or time-to-failure error.

### 6.2 Fixed load condition

The available recordings use the same nominal 535 g load. The model has not been tested for generalization to other loads, so load variation cannot be separated from health variation.

### 6.3 Pseudo-clean training target

For the ADC-only experiment, the clean target is a moving-average signal generated from the same raw recording. This is useful for a pipeline test, but it is not an independently measured clean reference and may suppress or distort real signal changes.

### 6.4 Operating-condition confounding

The recordings represent different frequency conditions and different lengths. A trend caused by frequency, measurement duration, or acquisition behavior may be incorrectly interpreted as degradation.

### 6.5 Temporal split limitation

Each file is split chronologically, but the model is trained and tested within the same recording. This is weaker than holding out complete files or complete operating conditions for final generalization testing.

### 6.6 Arbitrary state thresholds

The current Healthy, Warning, and Degraded boundaries are fixed normalized-score thresholds. They are not derived from expert labels, failure events, quantiles across a representative fleet, or validated operating-condition baselines. This explains the highly imbalanced state counts.

### 6.7 Cluster stability is not yet measured

FCM is rerun independently for each file. Although cluster IDs are reordered by reconstruction error, the feature distributions and cluster centers are not yet compared across files with a formal stability metric.

### 6.8 Reconstruction quality remains a risk

The earlier end-to-end reconstruction evaluation showed weak absolute reconstruction quality in parts of the pipeline. Health features derived from reconstruction error should not be treated as trustworthy degradation features until window-level and full-sequence denoising quality are separately calibrated.

## 7. Required validation gates before RUL

The next implementation should not train RUL directly from the current state labels. Before RUL modeling, the project should complete these gates:

1. Add multiple known loads and preserve load metadata.
2. Define independent operating-condition baselines.
3. Replace or qualify the pseudo-clean target with a calibrated reference where possible.
4. Test complete-file and complete-condition holdout splits.
5. Calibrate state thresholds from labels, quantiles, or validated baseline distributions.
6. Measure cluster stability across files.
7. Check whether the health score remains consistent after controlling for load and frequency.
8. Obtain degradation or failure-time labels before reporting RUL accuracy.

## 8. Current decision

The health-trend update is accepted as the next engineering milestone because the implementation runs across all 20 files and produces reproducible output. The RUL stage remains blocked for scientific validation, not for software implementation: there is currently no trustworthy target representing remaining useful life.

The next defensible task is operating-condition calibration and trend validation. After that, an RUL prototype can be added as an explicitly synthetic or weakly supervised experiment, clearly separated from validated real-data results.
