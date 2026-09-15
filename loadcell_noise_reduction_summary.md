# LoadCell Noise Reduction using Recurrent Denoising Autoencoder (R-DAE)

## Overview

Implementation of noise reduction for LoadCell sensor data using a Recurrent Denoising Autoencoder (R-DAE) approach inspired by the bearing RUL prediction paper. The goal was to reconstruct clean LoadCell signals from noisy measurements using an LSTM-based autoencoder architecture.

## Implementation Details

### 1. Synthetic Dataset Generation

- Created 4000-sample LoadCell signal simulating weight measurements
- Base signal components:
  - Constant base weight (30 units)
  - Slow oscillation (8*sin(0.2*t))
  - Faster oscillation (3*sin(1.5*t))
  - Gradual drift (0.01*t²)
  - Peak load (1.5*exp(-0.1*(t-25)²))
- Added four noise types:
  - Gaussian white noise (std=0.1)
  - Low-frequency drift noise (temperature effects simulation)
  - Spike noise (1% of samples, std=0.3)
  - Periodic interference (50Hz hum simulation)

### 2. R-DAE Architecture

- **Window Size**: 15 samples
- **Encoder**:
  - LSTM(30, return_sequences=True)
  - LSTM(20)
- **Bottleneck**: RepeatVector(15)
- **Decoder**:
  - LSTM(20, return_sequences=True)
  - LSTM(30, return_sequences=True)
  - TimeDistributed(Dense(1))
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning_rate=0.001)

### 3. Training Process

- Normalized noisy and clean signals separately using MinMaxScaler
- Created sliding window sequences from normalized data
- Trained for 25 epochs with batch size 16 and 20% validation split
- Used teacher forcing approach: noisy sequences → clean sequences

### 4. Noise Reduction Application

- Trained model predicted denoised sequences from noisy inputs
- Reconstructed full signal using overlapping window averaging
- Applied inverse transformation using clean signal scaler (critical fix)
- Evaluated performance using MSE, MAE, RMSE metrics

## Results

### Performance Metrics

```
Noisy signal vs Clean signal:
  MSE: 0.010560
  MAE: 0.080941
  RMSE: 0.102763

Denoised signal vs Clean signal:
  MSE: 0.018990
  MAE: 0.124691
  RMSE: 0.137804

❌ Performance degraded by: 79.83%
```

### Training Progress

- Final training loss: ~4.3e-5
- Final validation loss: ~2.1e-4
- Model converged successfully showing ability to learn the mapping

### Key Observations

1. The R-DAE successfully learned to map noisy sequences to cleaner representations (low training loss)
2. However, when evaluated against the ground truth clean signal, the denoised output showed higher error than the original noisy signal
3. This suggests the model may be:
   - Over-smoothing the signal
   - Learning noise characteristics instead of suppressing them
   - Not adequately capturing the temporal dependencies for this specific noise type

## Files Generated

- `loadcell_noise_reduction_final.py` - Complete implementation script
- `loadcell_rdae_denoiser_final.h5` - Trained R-DAE model (HDF5 format)
- `loadcell_noisy_scaler_final.pkl` - Scaler for noisy input data
- `loadcell_clean_scaler_final.pkl` - Scaler for clean target data
- `loadcell_noise_reduction_final_results.png` - Visualization comparing signals
- `loadcell_noise_reduction_summary.md` - This summary document

## Conclusion

While the R-DAE architecture was successfully implemented and trained, it did not achieve effective noise reduction for the generated LoadCell dataset with the current parameters. The model learned a mapping but resulted in increased error compared to the original noisy signal.

## Potential Improvements

1. **Architecture Tuning**: Experiment with different LSTM unit sizes, window lengths, or add dropout layers
2. **Training Adjustments**: Increase epochs, adjust learning rate, or use different batch sizes
3. **Noise Characteristics**: Test with different noise profiles or simpler noise types initially
4. **Evaluation Approach**: Consider alternative metrics or validation strategies
5. **Data Preparation**: Explore different normalization techniques or feature engineering

The implementation provides a solid foundation for further experimentation with R-DAE based noise reduction for sensor signals.
