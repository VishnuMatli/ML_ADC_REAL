"""Paper-inspired recurrent denoising autoencoder for load-cell signals.

The supplied paper uses an R-DAE: LSTM encoder layers learn temporal features
from noise-corrupted windows and a decoder reconstructs the low-noise signal.
This example retains a known clean signal only to make a *held-out* benchmark;
in deployment replace ``make_demo_data`` with calibrated clean/noisy examples.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import medfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Add, Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
WINDOW = 64                 # captures long temporal correlation (paper uses a 20-point window)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EPOCHS = 20  # callbacks retain the best validation epoch; keeps CPU demo practical
BATCH_SIZE = 64             # close to the paper's optimised batch size (56)


def make_demo_data(n_samples=6000, sample_rate_hz=100.0):
    """Create a realistic load-cell benchmark with steps, creep and interference."""
    time = np.arange(n_samples) / sample_rate_hz
    load = np.full(n_samples, 15.0)
    load += 8.0 * (time >= 12.0) - 4.0 * (time >= 31.0) + 12.0 * (time >= 54.0)
    load += 0.035 * np.maximum(time - 12.0, 0.0)  # load-cell creep
    load += 0.45 * np.sin(2 * np.pi * 0.035 * time)
    white = np.random.normal(0.0, 0.18, n_samples)
    hum = 0.11 * np.sin(2 * np.pi * 50.0 * time + 0.4)
    temperature = 0.13 * np.sin(2 * np.pi * 0.008 * time)
    spikes = np.zeros(n_samples)
    spike_index = np.random.choice(n_samples, size=n_samples // 100, replace=False)
    spikes[spike_index] = np.random.normal(0.0, 1.4, len(spike_index))
    return time, load, load + white + hum + temperature + spikes


def make_windows(values, window):
    """Return complete windows; callers split first to prevent leakage."""
    return np.lib.stride_tricks.sliding_window_view(values, window).copy()[..., None]


def corrupt_for_dae(windows, std=0.035):
    """DAE corruption used in the paper: extra zero-mean input noise."""
    return windows + np.random.normal(0.0, std, windows.shape).astype(np.float32)


def build_rdae(window):
    """LSTM R-DAE with a temporal bottleneck, dropout and stable activations."""
    inputs = Input(shape=(window, 1), name="corrupted_load_window")
    x = LSTM(64, return_sequences=True, activation="tanh", name="encoder_lstm_1")(inputs)
    x = Dropout(0.20, name="encoder_dropout")(x)
    code = LSTM(32, activation="tanh", name="temporal_bottleneck")(x)
    x = RepeatVector(window, name="repeat_code")(code)
    x = LSTM(32, return_sequences=True, activation="tanh", name="decoder_lstm_1")(x)
    x = Dropout(0.10, name="decoder_dropout")(x)
    x = LSTM(64, return_sequences=True, activation="tanh", name="decoder_lstm_2")(x)
    # Learn the correction (noise residual), not the complete physical load. This
    # preserves genuine step changes while the R-DAE removes short-lived noise.
    residual = TimeDistributed(Dense(1, activation="linear"), name="noise_residual")(x)
    outputs = Add(name="reconstruction")([inputs, residual])
    model = Model(inputs, outputs, name="loadcell_rdae")
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def overlap_average(sequence_predictions, signal_length, window):
    result, counts = np.zeros(signal_length), np.zeros(signal_length)
    for start, prediction in enumerate(sequence_predictions[..., 0]):
        result[start:start + window] += prediction
        counts[start:start + window] += 1
    return result / np.maximum(counts, 1.0)


def scores(reference, estimate):
    mse = mean_squared_error(reference, estimate)
    return {"MAE": mean_absolute_error(reference, estimate), "RMSE": np.sqrt(mse), "MSE": mse}


def main():
    print("Load-cell denoising with the paper-inspired R-DAE")
    time, clean, noisy = make_demo_data()
    n_train = int(len(noisy) * TRAIN_FRACTION)
    n_valid = int(len(noisy) * (TRAIN_FRACTION + VALIDATION_FRACTION))

    # Fit once on training measurements. Separate clean/target scalers leak target
    # statistics and cannot be used when only sensor data are available.
    scaler = MinMaxScaler().fit(noisy[:n_train, None])
    noisy_n = scaler.transform(noisy[:, None]).ravel()
    clean_n = scaler.transform(clean[:, None]).ravel()
    # Split before windowing: no overlapping window appears in two partitions.
    train_x = make_windows(noisy_n[:n_train], WINDOW)
    train_y = make_windows(clean_n[:n_train], WINDOW)
    valid_x = make_windows(noisy_n[n_train:n_valid], WINDOW)
    valid_y = make_windows(clean_n[n_train:n_valid], WINDOW)
    test_x = make_windows(noisy_n[n_valid:], WINDOW)
    print(f"Windows: train={len(train_x)}, validation={len(valid_x)}, test={len(test_x)}")

    model = build_rdae(WINDOW)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        corrupt_for_dae(train_x), train_y,
        validation_data=(corrupt_for_dae(valid_x), valid_y),
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=2,
    )

    # Metrics are exclusively from the unseen chronological test section.
    predicted_windows = model.predict(test_x, batch_size=BATCH_SIZE, verbose=0)
    denoised_n = overlap_average(predicted_windows, len(test_x) + WINDOW - 1, WINDOW)
    clean_test = clean[n_valid:n_valid + len(denoised_n)]
    noisy_test = noisy[n_valid:n_valid + len(denoised_n)]
    denoised = scaler.inverse_transform(denoised_n[:, None]).ravel()
    median = medfilt(noisy_test, kernel_size=5)  # transparent smoothing baseline
    noisy_metrics, median_metrics, rdae_metrics = scores(clean_test, noisy_test), scores(clean_test, median), scores(clean_test, denoised)
    improvement = 100.0 * (1.0 - rdae_metrics["MSE"] / noisy_metrics["MSE"])
    print("\nHeld-out chronological test metrics")
    for name, metric in (("Noisy", noisy_metrics), ("Median baseline", median_metrics), ("R-DAE", rdae_metrics)):
        print(f"{name:16s} MAE={metric['MAE']:.5f}  RMSE={metric['RMSE']:.5f}  MSE={metric['MSE']:.6f}")
    print(f"R-DAE MSE improvement over raw measurement: {improvement:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    test_time = time[n_valid:n_valid + len(denoised)]
    axes[0, 0].plot(test_time, clean_test, label="Clean reference", lw=2)
    axes[0, 0].plot(test_time, noisy_test, label="Noisy measurement", alpha=.45)
    axes[0, 0].plot(test_time, denoised, label="R-DAE", lw=1.5)
    axes[0, 0].set(title="Held-out test signal", xlabel="Time (s)", ylabel="Load")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.25)
    zoom = slice(400, 700)
    axes[0, 1].plot(test_time[zoom], clean_test[zoom], label="Clean", lw=2)
    axes[0, 1].plot(test_time[zoom], noisy_test[zoom], label="Noisy", alpha=.5)
    axes[0, 1].plot(test_time[zoom], denoised[zoom], label="R-DAE", lw=1.5)
    axes[0, 1].set(title="Detail: unseen data", xlabel="Time (s)", ylabel="Load")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=.25)
    axes[1, 0].plot(history.history["loss"], label="Train")
    axes[1, 0].plot(history.history["val_loss"], label="Validation")
    axes[1, 0].set(title="Leak-free training history", xlabel="Epoch", ylabel="MSE loss")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=.25)
    axes[1, 1].hist(noisy_test - clean_test, bins=50, alpha=.55, label="Raw error")
    axes[1, 1].hist(denoised - clean_test, bins=50, alpha=.55, label="R-DAE error")
    axes[1, 1].set(title="Error distribution (test only)", xlabel="Error")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=.25)
    fig.tight_layout()
    figure_path = ROOT / "loadcell_noise_reduction_final_results.png"
    fig.savefig(figure_path, dpi=220)

    model.save(ROOT / "loadcell_rdae_denoiser_final.keras")
    joblib.dump(scaler, ROOT / "loadcell_scaler_final.pkl")
    print(f"\nSaved model: {ROOT / 'loadcell_rdae_denoiser_final.keras'}")
    print(f"Saved scaler: {ROOT / 'loadcell_scaler_final.pkl'}")
    print(f"Saved test-only plot: {figure_path}")


if __name__ == "__main__":
    main()
