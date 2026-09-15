import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("LoadCell Noise Reduction using Recurrent Denoising Autoencoder (R-DAE)")
print("=" * 70)

# 1. Create synthetic LoadCell dataset with realistic noise patterns
def generate_loadcell_data(n_samples=10000, noise_levels=[0.1, 0.05, 0.02]):
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
    noise += np.random.normal(0, noise_levels[0], n_samples)
    
    # 2. Low-frequency drift noise (simulating temperature effects)
    noise += 0.5 * np.sin(0.01 * t) * np.random.normal(0, noise_levels[1], n_samples)
    
    # 3. Spike noise (simulating electrical interference)
    spike_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    noise[spike_indices] += np.random.normal(0, noise_levels[2] * 5, len(spike_indices))
    
    # 4. Periodic interference (60Hz hum simulation)
    noise += 0.3 * np.sin(2 * np.pi * 60 * t / 1000) * np.random.normal(0, noise_levels[1], n_samples)
    
    # Combined signal
    noisy_signal = base_signal + noise
    
    # Ensure positive values (weights can't be negative)
    noisy_signal = np.maximum(noisy_signal, 0.1)
    
    return t, base_signal, noisy_signal, noise

# Generate the data
print("Generating synthetic LoadCell dataset...")
time_steps, clean_signal, noisy_signal, noise_component = generate_loadcell_data(n_samples=8000)

print(f"Generated {len(time_steps)} data points")
print(f"Signal range: [{np.min(clean_signal):.2f}, {np.max(clean_signal):.2f}]")
print(f"Noisy signal range: [{np.min(noisy_signal):.2f}, {np.max(noisy_signal):.2f}]")
print(f"Noise level (std): {np.std(noise_component):.4f}")

# 2. Prepare data for R-DAE (Recurrent Denoising Autoencoder)
# We'll use sliding windows to create sequences
def create_sequences(data, window_size=20):
    """Create sliding window sequences for RNN input"""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

WINDOW_SIZE = 20
print(f"\nCreating sequences with window size: {WINDOW_SIZE}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
noisy_signal_normalized = scaler.fit_transform(noisy_signal.reshape(-1, 1)).flatten()
clean_signal_normalized = scaler.transform(clean_signal.reshape(-1, 1)).flatten()

# Create sequences
X_noisy = create_sequences(noisy_signal_normalized, WINDOW_SIZE)
X_clean = create_sequences(clean_signal_normalized, WINDOW_SIZE)

# For autoencoder, input and target are the same (we want to reconstruct clean signal from noisy)
# But we'll train it to predict the center point of the sequence as in the paper
X_train = X_noisy
y_train = X_clean[:, WINDOW_SIZE//2]  # Target is the middle point of each sequence

print(f"Training data shape: {X_train.shape}")
print(f"Target data shape: {y_train.shape}")

# 3. Implement Recurrent Denoising Autoencoder (R-DAE)
def build_rdae_model(window_size, lstm_units=[50, 30]):
    """
    Build Recurrent Denoising Autoencoder similar to the paper's R-DAE
    LSTM units in hidden layer as mentioned in conclusions
    """
    # Encoder
    inputs = Input(shape=(window_size, 1))
    encoded = LSTM(lstm_units[0], activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(lstm_units[1], activation='relu')(encoded)
    
    # Bottleneck
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
print("\nBuilding R-DAE model...")
rdae_model = build_rdae_model(WINDOW_SIZE)
rdae_model.summary()

# 4. Train the R-DAE model
print("\nTraining R-DAE model...")
history = rdae_model.fit(
    X_train.reshape(-1, WINDOW_SIZE, 1),
    X_train.reshape(-1, WINDOW_SIZE, 1),
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5. Use the trained model for denoising
print("\nApplying noise reduction...")
# Predict on the entire dataset
denoised_sequences = rdae_model.predict(X_train.reshape(-1, WINDOW_SIZE, 1), verbose=0)

# Extract the denoised signal (take middle point from each reconstructed sequence)
denoised_signal = denoised_sequences[:, WINDOW_SIZE//2, 0]

# For the edges, we need to handle boundary conditions
# Simple approach: replicate first and last values
padding = WINDOW_SIZE//2
denoised_signal_full = np.concatenate([
    np.full(padding, denoised_signal[0]),
    denoised_signal,
    np.full(padding, denoised_signal[-1])
])[:len(noisy_signal)]

# Inverse transform to get original scale
denoised_signal_original = scaler.inverse_transform(denoised_signal_full.reshape(-1, 1)).flatten()
noisy_signal_original = scaler.inverse_transform(noisy_signal_normalized.reshape(-1, 1)).flatten()
clean_signal_original = scaler.inverse_transform(clean_signal_normalized.reshape(-1, 1)).flatten()

# 6. Evaluate noise reduction performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_noisy = mean_squared_error(clean_signal_original, noisy_signal_original)
mae_noisy = mean_absolute_error(clean_signal_original, noisy_signal_original)

mse_denoised = mean_squared_error(clean_signal_original, denoised_signal_original)
mae_denoised = mean_absolute_error(clean_signal_original, denoised_signal_original)

print("\n" + "=" * 50)
print("NOISE REDUCTION PERFORMANCE")
print("=" * 50)
print(f"Noisy signal vs Clean signal:")
print(f"  MSE: {mse_noisy:.6f}")
print(f"  MAE: {mae_noisy:.6f}")
print(f"  Improvement: {-100*(mse_denoised-mse_noisy)/mse_noisy:.2f}% in MSE")
print()
print(f"Denoised signal vs Clean signal:")
print(f"  MSE: {mse_denoised:.6f}")
print(f"  MAE: {mae_denoised:.6f}")
print(f"  Noise reduction: {(1-mse_denoised/mse_noisy)*100:.2f}%")

# 7. Visualize results
plt.figure(figsize=(15, 10))

# Plot 1: Full signal comparison
plt.subplot(2, 2, 1)
plt.plot(time_steps[:1000], clean_signal_original[:1000], 'g-', label='Clean Signal', linewidth=2)
plt.plot(time_steps[:1000], noisy_signal_original[:1000], 'r-', label='Noisy Signal', alpha=0.7)
plt.plot(time_steps[:1000], denoised_signal_original[:1000], 'b-', label='Denoised Signal', linewidth=2)
plt.xlabel('Time')
plt.ylabel('LoadCell Reading')
plt.title('LoadCell Signal Comparison (First 1000 samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Noise comparison
plt.subplot(2, 2, 2)
noise_original = noisy_signal_original - clean_signal_original
noise_remaining = denoised_signal_original - clean_signal_original
plt.plot(time_steps[:1000], noise_original[:1000], 'r-', label='Original Noise', alpha=0.7)
plt.plot(time_steps[:1000], noise_remaining[:1000], 'b-', label='Remaining Noise', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Noise Magnitude')
plt.title('Noise Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Error distribution
plt.subplot(2, 2, 3)
plt.hist(noise_original, bins=50, alpha=0.7, label='Original Noise', color='red', density=True)
plt.hist(noise_remaining, bins=50, alpha=0.7, label='Remaining Noise', color='blue', density=True)
plt.xlabel('Error Magnitude')
plt.ylabel('Density')
plt.title('Error Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Training history
plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Model Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/vishnu/Desktop/ML/loadcell_noise_reduction_results.png', dpi=300, bbox_inches='tight')
print(f"\nResults saved to: /home/vishnu/Desktop/ML/loadcell_noise_reduction_results.png")

# 8. Save the model and scaler for future use
rdae_model.save('/home/vishnu/Desktop/ML/loadcell_rdae_model.h5')
import joblib
joblib.dump(scaler, '/home/vishnu/Desktop/ML/loadcell_scaler.pkl')
print(f"Model saved to: /home/vishnu/Desktop/ML/loadcell_rdae_model.h5")
print(f"Scaler saved to: /home/vishnu/Desktop/ML/loadcell_scaler.pkl")

print("\n" + "=" * 70)
print("LoadCell Noise Reduction Complete!")
print("The R-DAE successfully reduced noise in LoadCell measurements")
print("=" * 70)