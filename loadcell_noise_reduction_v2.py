import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("LoadCell Noise Reduction using Recurrent Denoising Autoencoder (R-DAE) - Version 2")
print("=" * 80)

# 1. Create synthetic LoadCell dataset with realistic noise patterns
def generate_loadcell_data(n_samples=10000):
    """
    Generate synthetic LoadCell data simulating weight measurements with various noise types
    """
    # Time steps
    t = np.linspace(0, 100, n_samples)
    
    # Base signal: simulated weight with drift, gradual changes, and sudden loads
    base_signal = (
        50 +  # Base weight (50 units)
        10 * np.sin(0.1 * t) +  # Slow oscillation
        5 * np.sin(0.5 * t) +   # Faster oscillation
        0.02 * t**2 +           # Gradual drift (creep)
        2 * np.exp(-0.05 * (t-50)**2)  # Peak load around middle
    )
    
    # Add various types of noise
    noise = np.zeros_like(t)
    
    # 1. Gaussian white noise
    noise += np.random.normal(0, 0.15, n_samples)
    
    # 2. Low-frequency drift noise (simulating temperature effects)
    noise += 0.3 * np.sin(0.02 * t) * np.random.normal(0, 0.08, n_samples)
    
    # 3. Spike noise (simulating electrical interference)
    spike_indices = np.random.choice(n_samples, size=int(0.005 * n_samples), replace=False)
    noise[spike_indices] += np.random.normal(0, 0.5, len(spike_indices))
    
    # 4. Periodic interference (60Hz hum simulation)
    noise += 0.2 * np.sin(2 * np.pi * 60 * t / 1000) * np.random.normal(0, 0.06, n_samples)
    
    # Combined signal
    noisy_signal = base_signal + noise
    
    # Ensure positive values (weights can't be negative)
    noisy_signal = np.maximum(noisy_signal, 0.1)
    
    return t, base_signal, noisy_signal, noise

# Generate the data
print("Generating synthetic LoadCell dataset...")
time_steps, clean_signal, noisy_signal, noise_component = generate_loadcell_data(n_samples=6000)

print(f"Generated {len(time_steps)} data points")
print(f"Clean signal range: [{np.min(clean_signal):.2f}, {np.max(clean_signal):.2f}]")
print(f"Noisy signal range: [{np.min(noisy_signal):.2f}, {np.max(noisy_signal):.2f}]")
print(f"Noise level (std): {np.std(noise_component):.4f}")
print(f"SNR: {20*np.log10(np.std(clean_signal)/np.std(noise_component)):.2f} dB")

# 2. Prepare data for R-DAE as a Denoising Autoencoder
# We'll use sliding windows where input is noisy sequence and target is clean sequence
def create_sequences(data, window_size=20):
    """Create sliding window sequences for RNN input"""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

WINDOW_SIZE = 20
print(f"\nCreating sequences with window size: {WINDOW_SIZE}")

# Normalize the data separately to preserve noise characteristics
noisy_scaler = MinMaxScaler(feature_range=(0, 1))
clean_scaler = MinMaxScaler(feature_range=(0, 1))

noisy_signal_normalized = noisy_scaler.fit_transform(noisy_signal.reshape(-1, 1)).flatten()
clean_signal_normalized = clean_scaler.fit_transform(clean_signal.reshape(-1, 1)).flatten()

# Create sequences
X_noisy = create_sequences(noisy_signal_normalized, WINDOW_SIZE)  # Noisy inputs
Y_clean = create_sequences(clean_signal_normalized, WINDOW_SIZE)  # Clean targets

print(f"Training data shape: {X_noisy.shape}")
print(f"Target data shape: {Y_clean.shape}")

# 3. Implement Recurrent Denoising Autoencoder (R-DAE)
def build_rdae_denoiser(window_size, lstm_units=[50, 30]):
    """
    Build Recurrent Denoising Autoencoder for noise reduction
    Input: noisy sequence, Output: clean sequence
    """
    # Encoder
    inputs = Input(shape=(window_size, 1))
    encoded = LSTM(lstm_units[0], activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(lstm_units[1], activation='relu')(encoded)
    
    # Bottleneck representation
    # Repeat vector to go back to sequence
    repeated = RepeatVector(window_size)(encoded)
    
    # Decoder
    decoded = LSTM(lstm_units[1], activation='relu', return_sequences=True)(repeated)
    decoded = LSTM(lstm_units[0], activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(1))(decoded)
    
    # Model
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# Build and summarize the model
print("\nBuilding R-DAE denoiser model...")
rdae_model = build_rdae_denoiser(WINDOW_SIZE)
rdae_model.summary()

# 4. Train the R-DAE model as a denoising autoencoder
print("\nTraining R-DAE model as denoising autoencoder...")
history = rdae_model.fit(
    X_noisy.reshape(-1, WINDOW_SIZE, 1),
    Y_clean.reshape(-1, WINDOW_SIZE, 1),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5. Use the trained model for denoising
print("\nApplying noise reduction...")
# Predict on the entire dataset using sliding window approach
denoised_sequences = rdae_model.predict(X_noisy.reshape(-1, WINDOW_SIZE, 1), verbose=0)

# Reconstruct the full signal from overlapping windows
# Method: average the predictions for overlapping windows
denoised_signal = np.zeros_like(noisy_signal_normalized)
window_count = np.zeros_like(noisy_signal_normalized)

for i in range(len(denoised_sequences)):
    start_idx = i
    end_idx = i + WINDOW_SIZE
    denoised_signal[start_idx:end_idx] += denoised_sequences[i, :, 0]
    window_count[start_idx:end_idx] += 1

# Avoid division by zero
window_count = np.maximum(window_count, 1)
denoised_signal = denoised_signal / window_count

# For the beginning and end that don't have full windows, copy edge values
denoised_signal[:WINDOW_SIZE//2] = denoised_signal[WINDOW_SIZE//2]
denoised_signal[-WINDOW_SIZE//2:] = denoised_signal[-WINDOW_SIZE//2-1]

# Inverse transform to get original scale
denoised_signal_original = noisy_scaler.inverse_transform(denoised_signal.reshape(-1, 1)).flatten()

# 6. Evaluate noise reduction performance
mse_noisy = mean_squared_error(clean_signal, noisy_signal)
mae_noisy = mean_absolute_error(clean_signal, noisy_signal)

mse_denoised = mean_squared_error(clean_signal, denoised_signal_original)
mae_denoised = mean_absolute_error(clean_signal, denoised_signal_original)

print("\n" + "=" * 50)
print("NOISE REDUCTION PERFORMANCE")
print("=" * 50)
print(f"Noisy signal vs Clean signal:")
print(f"  MSE: {mse_noisy:.6f}")
print(f"  MAE: {mae_noisy:.6f}")
print(f"  RMSE: {np.sqrt(mse_noisy):.6f}")
print()
print(f"Denoised signal vs Clean signal:")
print(f"  MSE: {mse_denoised:.6f}")
print(f"  MAE: {mae_denoised:.6f}")
print(f"  RMSE: {np.sqrt(mse_denoised):.6f}")
print()
if mse_denoised < mse_noisy:
    improvement = (1 - mse_denoised/mse_noisy) * 100
    print(f"✅ Noise reduction: {improvement:.2f}% improvement in MSE")
    print(f"   Signal-to-noise ratio improved by: {10*np.log10(mse_noisy/mse_denoised):.2f} dB")
else:
    degradation = (mse_denoised/mse_noisy - 1) * 100
    print(f"❌ Performance degraded by: {degradation:.2f}%")

# 7. Visualize results
plt.figure(figsize=(16, 12))

# Plot 1: Full signal comparison (first 1500 samples)
plt.subplot(3, 3, 1)
plt.plot(time_steps[:1500], clean_signal[:1500], 'g-', label='Clean Signal', linewidth=2, alpha=0.8)
plt.plot(time_steps[:1500], noisy_signal[:1500], 'r-', label='Noisy Signal', alpha=0.6)
plt.plot(time_steps[:1500], denoised_signal_original[:1500], 'b-', label='Denoised Signal', linewidth=2)
plt.xlabel('Time')
plt.ylabel('LoadCell Reading')
plt.title('LoadCell Signal Comparison (First 1500 samples)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Plot 2: Zoomed section showing detail
plt.subplot(3, 3, 2)
zoom_start, zoom_end = 500, 800
plt.plot(time_steps[zoom_start:zoom_end], clean_signal[zoom_start:zoom_end], 'g-', label='Clean Signal', linewidth=2)
plt.plot(time_steps[zoom_start:zoom_end], noisy_signal[zoom_start:zoom_end], 'r-', label='Noisy Signal', alpha=0.7)
plt.plot(time_steps[zoom_start:zoom_end], denoised_signal_original[zoom_start:zoom_end], 'b-', label='Denoised Signal', linewidth=2)
plt.xlabel('Time')
plt.ylabel('LoadCell Reading')
plt.title('Zoomed Detail View (500-800 samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Noise comparison
plt.subplot(3, 3, 3)
noise_original = noisy_signal - clean_signal
noise_remaining = denoised_signal_original - clean_signal
plt.plot(time_steps[:1500], noise_original[:1500], 'r-', label='Original Noise', alpha=0.7)
plt.plot(time_steps[:1500], noise_remaining[:1500], 'b-', label='Remaining Noise', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Noise Magnitude')
plt.title('Noise Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Error distribution
plt.subplot(3, 3, 4)
plt.hist(noise_original, bins=50, alpha=0.7, label='Original Noise', color='red', density=True)
plt.hist(noise_remaining, bins=50, alpha=0.7, label='Remaining Noise', color='blue', density=True)
plt.xlabel('Error Magnitude')
plt.ylabel('Density')
plt.title('Error Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Training history
plt.subplot(3, 3, 5)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Model Training History')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Before/after scatter
plt.subplot(3, 3, 6)
plt.scatter(clean_signal, noisy_signal, alpha=0.5, s=1, c='red', label='Noisy')
plt.scatter(clean_signal, denoised_signal_original, alpha=0.5, s=1, c='blue', label='Denoised')
plt.plot([clean_signal.min(), clean_signal.max()], [clean_signal.min(), clean_signal.max()], 'k--', label='Perfect')
plt.xlabel('Clean Signal')
plt.ylabel('Measured Signal')
plt.title('Before/After Comparison')
plt.legend()
plt.grid(True, alpha=0.3)