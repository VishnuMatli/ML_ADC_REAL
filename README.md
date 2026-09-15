# Load Cell ADC Noise Reduction

This project investigates noise reduction for load cell ADC signals using recurrent denoising autoencoders (R-DAE) and conventional filtering. The primary completed experiment uses recordings made with a stable 535 g applied load under different frequency conditions.

## Project Result

For the 535 g experiment, the `hz0` recording is used as the low-interference reference. The supervised R-DAE reduced the mean held-out RMSE from 186,229.6 ADC counts in the raw signal to 56.7 ADC counts.

| Measure | Result |
| --- | ---: |
| Mean RMSE reduction | 99.97% |
| Mean MSE reduction | 99.99994% |

This result is valid for the fixed 535 g calibration condition. It must not be interpreted as validation for unknown load values.

## Dataset

The raw data is in [`adc_data`](adc_data). Files use this naming pattern:

```text
535g20250502pm427ms20hz50.txt
```

| Part | Meaning |
| --- | --- |
| `535g` | Applied load: 535 grams |
| `20250502` | Date: 2 May 2025 |
| `pm427` | Time: 4:27 PM |
| `ms20` | Sampling interval: 20 ms (50 samples per second) |
| `hz50` | Frequency test condition: 50 Hz |

The data format is one sample per line:

```text
ADC:44374172
```

## Main Script

[`loadcell_adc_535g_rdae.py`](loadcell_adc_535g_rdae.py) is the main script for the current experiment.

It performs the following operations:

1. Reads every ADC text file from `adc_data`.
2. Uses the median ADC value of the `hz0` file as the stable 535 g reference.
3. Uses all nonzero-frequency recordings as noisy input data.
4. Splits every nonzero-frequency recording chronologically into 70% training, 15% validation, and 15% test data.
5. Trains an LSTM encoder-decoder R-DAE using 41-sample windows.
6. Evaluates the held-out test section of every frequency condition against the `hz0` reference.
7. Saves the model, scaler, CSV measurements, and plots.

## Requirements

Use Python 3.10 or later. The project requires:

```bash
pip install numpy scipy scikit-learn matplotlib joblib tensorflow
```

## Run the 535 g Experiment

From the project root, run:

```bash
python3 loadcell_adc_535g_rdae.py
```

The script takes several minutes on a CPU-only machine because it trains LSTM layers.

## Output Files

| File | Description |
| --- | --- |
| [`adc_535g_rdae.keras`](adc_535g_rdae.keras) | Saved trained R-DAE model for the 535 g experiment |
| [`adc_535g_rdae_scaler.pkl`](adc_535g_rdae_scaler.pkl) | RobustScaler fitted to training ADC data |
| [`adc_535g_rdae_results.csv`](adc_535g_rdae_results.csv) | RMSE, MSE, MAE, and improvement values for each frequency |
| [`adc_535g_rdae_results.png`](adc_535g_rdae_results.png) | Comparison plots of raw ADC, R-DAE output, and `hz0` reference |
| [`adc_535g_rdae_evaluation_report.md`](adc_535g_rdae_evaluation_report.md) | Detailed experiment report |

## Example Frequency Results

| Frequency | Raw RMSE | R-DAE RMSE | RMSE Reduction |
| ---: | ---: | ---: | ---: |
| 50 Hz | 57,452.7 | 81.1 | 99.86% |
| 97 Hz | 57,850.3 | 60.1 | 99.90% |

RMSE values are ADC counts relative to the `hz0` 535 g reference.

## Other Scripts

| File | Purpose |
| --- | --- |
| [`loadcell_adc_rdae.py`](loadcell_adc_rdae.py) | Earlier self-supervised R-DAE experiment without a known clean reference. It is retained for comparison. |
| [`loadcell_noise_reduction_final.py`](loadcell_noise_reduction_final.py) | Synthetic load cell R-DAE demonstration. |
| [`loadcell_noise_reduction.py`](loadcell_noise_reduction.py) and [`loadcell_noise_reduction_v2.py`](loadcell_noise_reduction_v2.py) | Earlier prototype implementations. |
| [`1-s2.0-S0263224125016446-main.pdf`](1-s2.0-S0263224125016446-main.pdf) | Research paper used as methodological background for the R-DAE architecture. |

## Important Limitations

- Every currently available recording uses the same 535 g load.
- The trained model has learned to map frequency-disturbed ADC values toward the 535 g `hz0` reference.
- The model is not a general load estimator and cannot be expected to predict other weights correctly.
- The test sections are chronologically unseen, but the earlier part of each frequency recording is included in training. A stricter test should withhold an entire frequency recording.

## Recommended Next Steps

1. Record data at multiple known loads, for example 0 g, 100 g, 250 g, 535 g, 750 g, and 1000 g.
2. Record each load under multiple frequency conditions.
3. Keep complete frequency and load combinations separate for final testing.
4. Build an ADC-to-grams calibration model using the multi-load dataset.
5. Evaluate denoising and weight-estimation errors in grams against a calibrated reference.

## Reports

- [`adc_535g_rdae_evaluation_report.md`](adc_535g_rdae_evaluation_report.md): Final report for the 535 g supervised R-DAE experiment.
- [`adc_noise_reduction_progress_report.md`](adc_noise_reduction_progress_report.md): Progress report covering the earlier and revised experiments.
