# Evaluation Report: 535 g Load Cell R-DAE Noise Reduction

## 1. Purpose

This report evaluates the recurrent denoising autoencoder (R-DAE) developed for the 535 g load cell ADC experiment. The purpose of the model is to reduce frequency-related ADC variation while retaining the stable ADC level measured at a known 535 g load.

## 2. Dataset Interpretation

Each ADC filename follows this format:

```text
535g20250502pm427ms20hz50.txt
```

| Filename part | Meaning |
| --- | --- |
| `535g` | Applied load: 535 grams |
| `20250502` | Date: 2 May 2025 |
| `pm427` | Time: 4:27 PM |
| `ms20` | Sampling interval: 20 ms (50 samples per second) |
| `hz50` | Frequency test condition: 50 Hz |

All available recordings represent the same 535 g load under different frequency conditions. The `hz0` recording is treated as the low-interference reference for this fixed load.

## 3. Reference Value

The reference ADC value was calculated as the median of the `hz0` recording.

| Item | Value |
| --- | ---: |
| Reference condition | 535 g at 0 Hz |
| Reference ADC value | 44,341,571.0 ADC counts |

The median was selected because it is less affected by occasional spikes than the arithmetic mean.

## 4. Model and Evaluation Procedure

The model uses a paper-inspired R-DAE architecture with LSTM encoder and decoder layers. The input consists of 41 consecutive ADC samples, equivalent to 0.82 seconds at the stated 20 ms sampling interval.

The nonzero-frequency recordings were divided in chronological order:

| Partition | Proportion | Purpose |
| --- | ---: | --- |
| Training | 70% | Model fitting |
| Validation | 15% | Training control and learning-rate adjustment |
| Test | 15% | Final held-out evaluation |

The final 15% of each recording was not used during training. Model predictions were compared with the `hz0` reference ADC level.

The following metrics were calculated:

```text
RMSE = sqrt(mean((predicted ADC - reference ADC)^2))

RMSE reduction (%) = (1 - R-DAE RMSE / raw RMSE) x 100

MSE reduction (%) = (1 - R-DAE MSE / raw MSE) x 100
```

## 5. Overall Held-Out Results

The mean results across the 19 nonzero-frequency recordings are shown below.

| Measure | Raw ADC | R-DAE Output |
| --- | ---: | ---: |
| Mean RMSE from the 535 g `hz0` reference | 186,229.6 ADC counts | 56.7 ADC counts |

| Improvement measure | Result |
| --- | ---: |
| Mean RMSE reduction | 99.97% |
| Mean MSE reduction | 99.99994% |

The R-DAE output is substantially closer to the stable 535 g reference than the raw ADC readings for this experiment.

## 6. Frequency-Specific Results

### 6.1 Test at 50 Hz

| Measure | Value |
| --- | ---: |
| Raw RMSE from `hz0` reference | 57,452.7 ADC counts |
| R-DAE RMSE from `hz0` reference | 81.1 ADC counts |
| RMSE reduction | 99.86% |
| MSE reduction | 99.9998% |

The R-DAE strongly reduced the ADC deviation in the held-out final portion of the 50 Hz recording.

### 6.2 Test at 97 Hz

| Measure | Value |
| --- | ---: |
| Raw RMSE from `hz0` reference | 57,850.3 ADC counts |
| R-DAE RMSE from `hz0` reference | 60.1 ADC counts |
| RMSE reduction | 99.90% |
| MSE reduction | 99.9999% |

The R-DAE also strongly reduced the ADC deviation in the held-out final portion of the 97 Hz recording.

## 7. Interpretation

For the known 535 g condition, the model performs better than the raw ADC measurement. It maps the frequency-affected recordings toward the stable ADC value measured under the 0 Hz reference condition.

The result should be interpreted carefully. The model was trained using only data from the 535 g experiment and its target is the stable 535 g reference value. Therefore, it is a denoising model for this specific calibration condition, not a general model for measuring unknown weights.

The 50 Hz and 97 Hz evaluations are chronological held-out tests. Their final 15% samples were not used during training; however, the earlier part of each respective recording was included in training. A stricter generalization test would exclude an entire frequency recording from training and use that complete recording only for testing.

## 8. Files Produced

| File | Description |
| --- | --- |
| `loadcell_adc_535g_rdae.py` | Training and evaluation script for the 535 g experiment |
| `adc_535g_rdae_results.csv` | Per-frequency held-out results |
| `adc_535g_rdae_results.png` | Plot of raw ADC, R-DAE output, and the `hz0` reference |
| `adc_535g_rdae.keras` | Saved trained R-DAE model |
| `adc_535g_rdae_scaler.pkl` | Saved input scaler |

## 9. Conclusion

The R-DAE significantly reduced frequency-related ADC deviations for the fixed 535 g load cell experiment. The average held-out RMSE decreased from 186,229.6 ADC counts to 56.7 ADC counts, giving a 99.97% RMSE reduction. The tested 50 Hz and 97 Hz recordings also showed RMSE reductions above 99.8%.

To use the same method for general load measurement, future data collection must include multiple known weights and independent recordings for each load and frequency condition.
