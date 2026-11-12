import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from loadcell_health_trend_pipeline import (
    BATCH_SIZE,
    EPOCHS,
    SMOOTHING_WINDOW,
    WINDOW_SIZE,
    build_health_trend,
    build_rdae,
    create_sequences,
    extract_bottleneck_features,
    improved_fcm,
    load_adc_file,
    split_signal,
)


DATA_DIR = Path('/home/vishnu/Desktop/ML/adc_data')
OUTPUT_CSV = Path('/home/vishnu/Desktop/ML/health_trend_multi_file_results.csv')


def make_windows(noisy, clean):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.concatenate([noisy.reshape(-1, 1), clean.reshape(-1, 1)]))
    train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean = split_signal(noisy, clean)

    def normalize(values):
        return scaler.transform(values.reshape(-1, 1)).ravel()

    return (
        create_sequences(normalize(train_noisy), WINDOW_SIZE),
        create_sequences(normalize(train_clean), WINDOW_SIZE),
        create_sequences(normalize(val_noisy), WINDOW_SIZE),
        create_sequences(normalize(val_clean), WINDOW_SIZE),
        create_sequences(normalize(test_noisy), WINDOW_SIZE),
        create_sequences(normalize(test_clean), WINDOW_SIZE),
    )


def run_file(file_path):
    _, clean, noisy = load_adc_file(file_path)
    X_train, y_train, X_val, y_val, X_test, y_test = make_windows(noisy, clean)

    def model_input(values):
        return values.reshape(-1, WINDOW_SIZE, 1)

    model = build_rdae(WINDOW_SIZE)
    model.fit(
        model_input(X_train),
        model_input(y_train),
        validation_data=(model_input(X_val), model_input(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    feature_matrix = extract_bottleneck_features(
        model, model_input(X_test), model_input(y_test)
    )[3]
    _, membership, labels, uncertainty = improved_fcm(feature_matrix, n_clusters=3)
    trend = build_health_trend(
        feature_matrix,
        membership,
        labels,
        SMOOTHING_WINDOW,
    )
    states = np.bincount(trend['state_ids'], minlength=3)
    return {
        'file': file_path.name,
        'samples': len(noisy),
        'test_windows': len(labels),
        'slope': trend['slope'],
        'monotonicity': trend['monotonicity'],
        'min_degradation': trend['smoothed_degradation'].min(),
        'max_degradation': trend['smoothed_degradation'].max(),
        'healthy_count': states[0],
        'warning_count': states[1],
        'degraded_count': states[2],
        'cluster_order': '-'.join(str(value) for value in trend['severity_order']),
    }


def main():
    files = sorted(DATA_DIR.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f'No ADC files found in {DATA_DIR}')

    results = []
    for index, file_path in enumerate(files, start=1):
        print(f'[{index}/{len(files)}] Testing {file_path.name}', flush=True)
        results.append(run_file(file_path))
        result = results[-1]
        print(
            f"  slope={result['slope']:.8f}, monotonicity={result['monotonicity']:.4f}, "
            f"states=[{result['healthy_count']}, {result['warning_count']}, {result['degraded_count']}]",
            flush=True,
        )

    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    slopes = np.array([result['slope'] for result in results])
    monotonicities = np.array([result['monotonicity'] for result in results])
    print('\n==== Multi-file validation summary ====')
    print(f'Files tested: {len(results)}')
    print(f'Positive trend slopes: {int(np.sum(slopes > 0))}/{len(slopes)}')
    print(f'Mean slope: {slopes.mean():.8f}')
    print(f'Mean monotonicity: {monotonicities.mean():.4f}')
    print(f'Slope range: [{slopes.min():.8f}, {slopes.max():.8f}]')
    print(f'Saved results: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
