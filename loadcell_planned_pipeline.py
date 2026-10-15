import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)
tf.random.set_seed(42)

WINDOW_SIZE = 32
EPOCHS = 25
BATCH_SIZE = 32


def generate_loadcell_signal(n_samples=3000):
    """Synthetic load cell signal with drift, periodic components, and random bursts."""
    t = np.linspace(0, 100, n_samples)
    clean = (
        50.0
        + 10.0 * np.sin(0.12 * t)
        + 6.0 * np.sin(0.55 * t)
        + 0.03 * (t ** 2)
        + 8.0 * np.exp(-0.08 * (t - 45.0) ** 2)
        + 11.0 * (t >= 20)
        + 9.0 * (t >= 45)
        - 5.0 * (t >= 70)
    )

    noise = (
        np.random.normal(0.0, 0.18, n_samples)
        + 0.25 * np.sin(0.03 * t) * np.random.normal(0.0, 0.10, n_samples)
        + 0.20 * np.sin(2 * np.pi * 60.0 * t / 1000.0)
        + 0.22 * np.sin(2 * np.pi * 0.4 * t)
    )

    spike_idx = np.random.choice(n_samples, size=max(10, int(0.006 * n_samples)), replace=False)
    noise[spike_idx] += np.random.normal(0.0, 0.8, len(spike_idx))

    noisy = np.maximum(clean + noise, 0.1)
    return t, clean, noisy


def create_sequences(data, window_size):
    return np.stack([data[i:i + window_size] for i in range(len(data) - window_size + 1)])


def split_signal(noisy, clean, train_frac=0.70, val_frac=0.15):
    n_total = len(noisy)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train_noisy = noisy[:n_train]
    train_clean = clean[:n_train]
    val_noisy = noisy[n_train:n_train + n_val]
    val_clean = clean[n_train:n_train + n_val]
    test_noisy = noisy[n_train + n_val:n_train + n_val + n_test]
    test_clean = clean[n_train + n_val:n_train + n_val + n_test]

    return train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean


def build_rdae(window_size):
    inputs = Input(shape=(window_size, 1), name='noisy_window')
    x = LSTM(40, activation='tanh', return_sequences=True, name='encoder_lstm_1')(inputs)
    x = LSTM(24, activation='tanh', name='bottleneck')(x)
    x = RepeatVector(window_size, name='repeat_bottleneck')(x)
    x = LSTM(24, activation='tanh', return_sequences=True, name='decoder_lstm_1')(x)
    x = LSTM(40, activation='tanh', return_sequences=True, name='decoder_lstm_2')(x)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'), name='reconstruction_head')(x)
    model = Model(inputs, outputs, name='loadcell_rdae')
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model


def extract_bottleneck_features(model, noisy_windows, clean_windows):
    encoder = Model(model.input, model.get_layer('bottleneck').output, name='encoder_only')
    latent = encoder.predict(noisy_windows, verbose=0)
    pred = model.predict(noisy_windows, verbose=0)
    residual = np.abs(pred - clean_windows)
    error_mean = residual.mean(axis=1).reshape(-1, 1)
    error_std = residual.std(axis=1).reshape(-1, 1)
    error_max = residual.max(axis=1).reshape(-1, 1)
    feature_matrix = np.concatenate([latent, error_mean, error_std, error_max], axis=1)
    return latent, pred, residual, feature_matrix


def improved_fcm(features, n_clusters=3, m=2.0, tol=1e-5, max_iter=200):
    rng = np.random.default_rng(42)
    n_samples = features.shape[0]
    selected = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = features[selected].copy()
    membership = np.ones((n_samples, n_clusters), dtype=float) / n_clusters

    for _ in range(max_iter):
        dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        dist = np.maximum(dist, 1e-12)
        new_membership = np.empty_like(membership)

        for i in range(n_samples):
            for j in range(n_clusters):
                denom = 0.0
                for k in range(n_clusters):
                    denom += (dist[i, j] / dist[i, k]) ** (2.0 / (m - 1.0))
                new_membership[i, j] = 1.0 / denom if denom > 0 else 1.0 / n_clusters

        new_centers = np.empty_like(centers)
        for j in range(n_clusters):
            weights = new_membership[:, j] ** m
            total_weight = weights.sum()
            if total_weight > 0:
                new_centers[j] = (weights[:, None] * features).sum(axis=0) / total_weight
            else:
                new_centers[j] = centers[j]

        if np.max(np.abs(new_membership - membership)) < tol:
            membership = new_membership
            centers = new_centers
            break

        membership = new_membership
        centers = new_centers

    labels = np.argmax(membership, axis=1)
    health_indicator = 1.0 - membership.max(axis=1)
    return centers, membership, labels, health_indicator


def reconstruct_signal(predicted_windows, original_length, window_size):
    result = np.zeros(original_length)
    counts = np.zeros(original_length)
    for i, window in enumerate(predicted_windows):
        start = i
        end = i + window_size
        result[start:end] += window[:, 0]
        counts[start:end] += 1.0
    return result / np.maximum(counts, 1.0)


def main():
    print('=== Planned methodology pipeline ===')
    print('Step 1: synthetic load-cell dataset generation')
    t, clean, noisy = generate_loadcell_signal(n_samples=3000)

    train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean = split_signal(noisy, clean)

    signal_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    all_train_values = np.concatenate([train_noisy.reshape(-1, 1), train_clean.reshape(-1, 1)])
    signal_scaler.fit(all_train_values)

    train_noisy_norm = signal_scaler.transform(train_noisy.reshape(-1, 1)).ravel()
    val_noisy_norm = signal_scaler.transform(val_noisy.reshape(-1, 1)).ravel()
    test_noisy_norm = signal_scaler.transform(test_noisy.reshape(-1, 1)).ravel()

    train_clean_norm = signal_scaler.transform(train_clean.reshape(-1, 1)).ravel()
    val_clean_norm = signal_scaler.transform(val_clean.reshape(-1, 1)).ravel()
    test_clean_norm = signal_scaler.transform(test_clean.reshape(-1, 1)).ravel()

    X_train = create_sequences(train_noisy_norm, WINDOW_SIZE)
    y_train = create_sequences(train_clean_norm, WINDOW_SIZE)
    X_val = create_sequences(val_noisy_norm, WINDOW_SIZE)
    y_val = create_sequences(val_clean_norm, WINDOW_SIZE)
    X_test = create_sequences(test_noisy_norm, WINDOW_SIZE)
    y_test = create_sequences(test_clean_norm, WINDOW_SIZE)

    print(f'Train windows: {X_train.shape[0]}')
    print(f'Validation windows: {X_val.shape[0]}')
    print(f'Test windows: {X_test.shape[0]}')

    print('\nStep 2: temporal R-DAE model training')
    model = build_rdae(WINDOW_SIZE)
    history = model.fit(
        X_train.reshape(-1, WINDOW_SIZE, 1),
        y_train.reshape(-1, WINDOW_SIZE, 1),
        validation_data=(X_val.reshape(-1, WINDOW_SIZE, 1), y_val.reshape(-1, WINDOW_SIZE, 1)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print('\nStep 3: feature extraction from bottleneck')
    latent_train, pred_train, residual_train, feature_train = extract_bottleneck_features(
        model, X_train.reshape(-1, WINDOW_SIZE, 1), y_train.reshape(-1, WINDOW_SIZE, 1)
    )
    latent_test, pred_test, residual_test, feature_test = extract_bottleneck_features(
        model, X_test.reshape(-1, WINDOW_SIZE, 1), y_test.reshape(-1, WINDOW_SIZE, 1)
    )

    print(f'Latent feature shape: {feature_test.shape}')

    print('\nStep 4: fuzzy clustering for health-state grouping')
    centers, membership, labels, health_indicator = improved_fcm(feature_test, n_clusters=3)
    print(f'Cluster labels first 20: {labels[:20]}')
    print(f'Health indicator range: [{health_indicator.min():.4f}, {health_indicator.max():.4f}]')

    reconstructed_windows = model.predict(X_test.reshape(-1, WINDOW_SIZE, 1), verbose=0)
    reconstructed_windows = np.clip(reconstructed_windows, 0.0, 1.0)
    reconstructed_seq = reconstruct_signal(reconstructed_windows, len(test_noisy_norm), WINDOW_SIZE)
    reconstructed_seq = np.clip(reconstructed_seq, 0.0, 1.0)
    reconstructed_orig = signal_scaler.inverse_transform(reconstructed_seq.reshape(-1, 1)).ravel()

    mse_noisy = mean_squared_error(test_clean, test_noisy)
    mae_noisy = mean_absolute_error(test_clean, test_noisy)
    mse_recon = mean_squared_error(test_clean, reconstructed_orig)
    mae_recon = mean_absolute_error(test_clean, reconstructed_orig)

    print('\n==== Evaluation summary ====')
    print(f'Noisy test MSE: {mse_noisy:.6f} | MAE: {mae_noisy:.6f}')
    print(f'Reconstructed test MSE: {mse_recon:.6f} | MAE: {mae_recon:.6f}')
    print(f'Raw-to-reconstruction improvement: {(1 - mse_recon / mse_noisy) * 100:.2f}%')
    print(f'Cluster count distribution: {np.bincount(labels).tolist()}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(t[:500], clean[:500], label='Clean signal', linewidth=2)
    axes[0, 0].plot(t[:500], noisy[:500], label='Noisy signal', alpha=0.7)
    axes[0, 0].set_title('Step 1: synthetic load-cell signal')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(history.history['loss'], label='Train loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation loss')
    axes[0, 1].set_title('Step 2: R-DAE training loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(test_clean[:300], label='Reference', linewidth=2)
    axes[1, 0].plot(reconstructed_orig[:300], label='Reconstruction', linewidth=1.5)
    axes[1, 0].set_title('Step 3: reconstruction output')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(feature_test[:, 0], feature_test[:, 1], c=labels, cmap='viridis', s=20, alpha=0.8)
    axes[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', s=80, marker='X')
    axes[1, 1].set_title('Step 4: fuzzy clustering on latent features')
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig('/home/vishnu/Desktop/ML/loadcell_planned_pipeline_results.png', dpi=220)
    print('\nSaved figure: /home/vishnu/Desktop/ML/loadcell_planned_pipeline_results.png')

    print('\nThis script implements the first four steps of the planned pipeline:')
    print('1. Data generation and preprocessing')
    print('2. Temporal R-DAE model training')
    print('3. Bottleneck feature extraction')
    print('4. FCM-based health-state grouping')


if __name__ == '__main__':
    main()
