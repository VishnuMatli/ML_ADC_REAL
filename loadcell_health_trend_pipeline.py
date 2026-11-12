import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from loadcell_planned_pipeline_adc import (
    BATCH_SIZE,
    EPOCHS,
    WINDOW_SIZE,
    build_rdae,
    create_sequences,
    extract_bottleneck_features,
    improved_fcm,
    load_adc_file,
    split_signal,
)
from sklearn.preprocessing import MinMaxScaler


ADC_FILE = Path('/home/vishnu/Desktop/ML/adc_data/535g20250502pm427ms20hz0.txt')
OUTPUT_PLOT = Path('/home/vishnu/Desktop/ML/loadcell_health_trend_results.png')
OUTPUT_CSV = Path('/home/vishnu/Desktop/ML/loadcell_health_trend.csv')
SMOOTHING_WINDOW = 31


def moving_average(values, window_size):
    """Smooth a time-ordered health signal without changing its length."""
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if len(values) < window_size:
        return values.copy()

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    padded = np.pad(values, (window_size // 2,), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def minmax_scale(values):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def build_health_trend(feature_matrix, membership, labels, smoothing_window):
    """Convert unordered FCM output into an ordered, time-series degradation score."""
    error_mean = feature_matrix[:, -3]
    cluster_error = np.array(
        [error_mean[labels == cluster_id].mean() if np.any(labels == cluster_id) else 0.0
         for cluster_id in range(membership.shape[1])]
    )
    severity_order = np.argsort(cluster_error)
    severity_rank = np.empty_like(severity_order)
    severity_rank[severity_order] = np.linspace(0.0, 1.0, len(severity_order))

    # Fuzzy state severity is independent of arbitrary FCM cluster numbering.
    fuzzy_severity = membership @ severity_rank
    reconstruction_severity = minmax_scale(error_mean)
    raw_degradation = 0.6 * fuzzy_severity + 0.4 * reconstruction_severity
    smoothed_degradation = moving_average(raw_degradation, smoothing_window)

    time_axis = np.arange(len(smoothed_degradation), dtype=np.float64)
    slope = float(np.polyfit(time_axis, smoothed_degradation, 1)[0]) if len(time_axis) > 1 else 0.0
    differences = np.diff(smoothed_degradation)
    monotonicity = float(np.mean(differences >= 0.0)) if len(differences) else 1.0

    state_edges = np.array([0.0, 0.33, 0.66, 1.0])
    state_names = np.array(['Healthy', 'Warning', 'Degraded'])
    state_ids = np.clip(np.digitize(smoothed_degradation, state_edges[1:-1]), 0, 2)
    state_names_for_windows = state_names[state_ids]

    return {
        'cluster_error': cluster_error,
        'severity_order': severity_order,
        'severity_rank': severity_rank,
        'fuzzy_severity': fuzzy_severity,
        'reconstruction_severity': reconstruction_severity,
        'raw_degradation': raw_degradation,
        'smoothed_degradation': smoothed_degradation,
        'slope': slope,
        'monotonicity': monotonicity,
        'state_ids': state_ids,
        'state_names': state_names_for_windows,
    }


def prepare_test_windows(noisy, clean, signal_scaler):
    train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean = split_signal(noisy, clean)

    def normalize(values):
        return signal_scaler.transform(values.reshape(-1, 1)).ravel()

    X_train = create_sequences(normalize(train_noisy), WINDOW_SIZE)
    y_train = create_sequences(normalize(train_clean), WINDOW_SIZE)
    X_val = create_sequences(normalize(val_noisy), WINDOW_SIZE)
    y_val = create_sequences(normalize(val_clean), WINDOW_SIZE)
    X_test = create_sequences(normalize(test_noisy), WINDOW_SIZE)
    y_test = create_sequences(normalize(test_clean), WINDOW_SIZE)
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print('=== Health trend update: ADC -> R-DAE -> FCM -> degradation trend ===')
    t, clean, noisy = load_adc_file(ADC_FILE)
    print(f'Using ADC file: {ADC_FILE.name}')
    print(f'Loaded samples: {len(noisy)}')

    signal_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    signal_scaler.fit(np.concatenate([noisy.reshape(-1, 1), clean.reshape(-1, 1)]))
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_test_windows(noisy, clean, signal_scaler)
    model_input = lambda values: values.reshape(-1, WINDOW_SIZE, 1)

    print(f'Train windows: {len(X_train)} | Validation windows: {len(X_val)} | Test windows: {len(X_test)}')
    print('\nTraining temporal R-DAE prerequisite')
    model = build_rdae(WINDOW_SIZE)
    model.fit(
        model_input(X_train),
        model_input(y_train),
        validation_data=(model_input(X_val), model_input(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print('\nExtracting features and clustering test windows')
    _, _, _, feature_test = extract_bottleneck_features(model, model_input(X_test), model_input(y_test))
    _, membership, labels, uncertainty_indicator = improved_fcm(feature_test, n_clusters=3)
    trend = build_health_trend(feature_test, membership, labels, SMOOTHING_WINDOW)

    test_start = int(len(noisy) * 0.85)
    window_centers = test_start + np.arange(len(labels)) + (WINDOW_SIZE - 1) / 2.0
    np.savetxt(
        OUTPUT_CSV,
        np.column_stack([
            window_centers,
            labels,
            uncertainty_indicator,
            trend['fuzzy_severity'],
            trend['reconstruction_severity'],
            trend['raw_degradation'],
            trend['smoothed_degradation'],
            trend['state_ids'],
        ]),
        delimiter=',',
        header='window_center,cluster,uncertainty_indicator,fuzzy_severity,reconstruction_severity,raw_degradation,smoothed_degradation,state_id',
        comments='',
    )

    print('\n==== Health trend summary ====')
    print(f'Cluster mean reconstruction error: {trend["cluster_error"].round(6).tolist()}')
    print(f'Cluster severity order (healthy -> degraded): {trend["severity_order"].tolist()}')
    print(f'Degradation score range: [{trend["smoothed_degradation"].min():.4f}, {trend["smoothed_degradation"].max():.4f}]')
    print(f'Trend slope per test window: {trend["slope"]:.8f}')
    print(f'Monotonicity ratio: {trend["monotonicity"]:.4f}')
    print(f'State distribution [Healthy, Warning, Degraded]: {np.bincount(trend["state_ids"], minlength=3).tolist()}')

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(window_centers, trend['fuzzy_severity'], label='Fuzzy cluster severity', alpha=0.65)
    axes[0].plot(window_centers, trend['reconstruction_severity'], label='Reconstruction-error severity', alpha=0.65)
    axes[0].plot(window_centers, trend['smoothed_degradation'], label='Smoothed degradation index', linewidth=2.2, color='black')
    axes[0].set_title('Step 5: health indicator and degradation trend')
    axes[0].set_ylabel('Normalized health/degradation score')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].step(window_centers, trend['state_ids'], where='mid', color='#C62828', linewidth=1.8)
    axes[1].set_yticks([0, 1, 2], ['Healthy', 'Warning', 'Degraded'])
    axes[1].set_title('Step 6: degradation-state evolution')
    axes[1].set_xlabel('Original ADC sample index')
    axes[1].set_ylabel('State')
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=220)
    print(f'\nSaved plot: {OUTPUT_PLOT}')
    print(f'Saved trend data: {OUTPUT_CSV}')
    print('\nRUL prediction is intentionally not included in this update; this trend output is its input.')


if __name__ == '__main__':
    main()
