# Progress Report: Load Cell ADC Noise Reduction

## 1. Introduction

This report describes the progress of a load cell signal noise-reduction study. The objective was to improve the quality of raw Analog-to-Digital Converter (ADC) readings using a machine-learning-based recurrent denoising autoencoder (R-DAE). The approach was informed by the supplied research paper, which uses Long Short-Term Memory (LSTM) layers to learn temporal characteristics from noisy time-series data.

## 2. Dataset Description

The dataset is stored in the `adc_data` folder. It contains 20 text files with ADC readings sampled at approximately 20 Hz. Each record is stored in the format:

```text
ADC:44374172
```

The complete dataset contains 269,668 ADC samples. Individual recordings range from approximately 4,000 to 63,000 samples. The file names contain labels such as `0`, `50`, `51`, and `100`, but the available files do not contain a separately recorded clean load value for every ADC sample.

## 3. Methodology

Two signal-processing approaches were evaluated.

### 3.1 Median Filter Baseline

A rolling median filter was used as the baseline denoising method. The median filter is effective for reducing isolated spikes and high-frequency fluctuations while remaining simple and computationally efficient.

### 3.2 Recurrent Denoising Autoencoder

An R-DAE model was created using an LSTM encoder-decoder structure. The model used the following features:

- Input windows of 41 samples, representing approximately 2.05 seconds of data at 20 Hz.
- LSTM encoder and decoder layers to learn temporal patterns.
- Dropout layers to reduce overfitting.
- Residual reconstruction, where the model learns a noise correction instead of reconstructing the full ADC signal from the beginning.
- Robust scaling fitted only on training data.
- A chronological train-validation-test split for each recording to avoid overlap leakage between the data partitions.

Because a clean reference signal was unavailable, a rolling median estimate was used as a pseudo-clean training target. Therefore, the R-DAE evaluation measures signal smoothness and agreement with the robust estimate, rather than true denoising accuracy.

## 4. Experimental Setup

The R-DAE was trained using 12,000 randomly selected training windows and 4,000 validation windows. Training was performed for 20 epochs with an Adam optimizer, mean squared error loss, early stopping, and learning-rate reduction.

The evaluation was performed on the final chronological portion of every ADC recording. This prevents the model from being evaluated on the same time range used for training.

## 5. Results

The following table presents the mean sample-to-sample variation across all recordings. Lower values indicate a smoother signal.

| Method | Mean ADC Difference Standard Deviation | Reduction Compared with Raw ADC |
| --- | ---: | ---: |
| Raw ADC signal | 244,177.5 | 0.0% |
| Median filter | 48,739.0 | 80.0% |
| R-DAE model | 241,420.5 | 1.1% |

The median filter produced a substantial reduction in short-term ADC variation. The trained R-DAE produced only a small reduction and did not outperform the median-filter baseline.

The recordings also show periodic variations in several files. These variations may be caused by sensor behaviour, electrical interference, vibration, or real physical load changes. Without a clean reference measurement, it is not possible to classify all periodic variation as noise.

## 6. Discussion

The R-DAE approach was implemented successfully and trained on the available ADC data. However, the current dataset is not sufficient for proving that the neural network removes noise without also altering meaningful load information.

The main limitation is the lack of sample-level ground truth. The model was trained using a median-filtered version of the same noisy signal as its pseudo-clean target. As a result, the neural network cannot be expected to reliably exceed the method used to create its training target.

The median filter is currently the preferred method for this dataset because it provides a much larger reduction in short-term noise and is easier to validate.

## 7. Conclusion

The project successfully processed the real-time ADC files and compared a conventional median filter with an R-DAE machine learning model. The median filter achieved an 80.0% reduction in sample-to-sample variation, while the R-DAE achieved only a 1.1% reduction.

At the current stage, the R-DAE model should not be used as the primary noise-reduction method. The median filter should be retained as the practical baseline.

## 8. Recommendations for Future Work

1. Record stable ADC data at known calibrated weights, such as 0 g, 50 g, 100 g, and additional intermediate values.
2. Record a clean reference signal using a calibrated instrument or a long-duration stable average.
3. Label each recording with the applied weight, sampling rate, sensor configuration, and operating conditions.
4. Test whether the periodic components are caused by electrical interference, mechanical vibration, or genuine load changes.
5. Retrain the R-DAE using paired noisy and clean calibration signals.
6. Compare the final trained model with the median filter using MAE, RMSE, and calibration error against the known true weights.

## 9. Generated Project Files

- `loadcell_adc_rdae.py`: Script for training and evaluating the ADC R-DAE model.
- `adc_rdae_results.csv`: Per-recording evaluation measurements.
- `adc_rdae_results.png`: Visual comparison of raw ADC, median-filtered, and R-DAE signals.
- `adc_rdae_denoiser.keras`: Saved trained R-DAE model.
- `adc_rdae_scaler.pkl`: Saved robust scaler used by the model.
