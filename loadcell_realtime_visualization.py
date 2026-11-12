import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from loadcell_planned_pipeline_adc import (
    BATCH_SIZE,
    EPOCHS,
    WINDOW_SIZE,
    build_rdae,
    create_sequences,
    extract_bottleneck_features,
    improved_fcm,
    load_adc_file,
    reconstruct_signal,
    split_signal,
)
from loadcell_health_trend_pipeline import SMOOTHING_WINDOW, build_health_trend


DEFAULT_ADC_FILE = Path('/home/vishnu/Desktop/ML/adc_data/535g20250502pm427ms20hz50.txt')
DEFAULT_OUTPUT = Path('/home/vishnu/Desktop/ML/loadcell_realtime_visualization.png')
DEFAULT_CSV = Path('/home/vishnu/Desktop/ML/loadcell_realtime_visualization.csv')


def format_adc_axis(axis):
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e6:.2f}'))
    axis.set_ylabel('ADC value (millions of counts)')


class LiveTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.train_loss = []
        self.validation_loss = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(float(logs.get('loss', np.nan)))
        self.validation_loss.append(float(logs.get('val_loss', np.nan)))
        self.dashboard.update_training(
            epoch + 1,
            self.params.get('epochs', len(self.train_loss)),
            self.train_loss,
            self.validation_loss,
        )


class RealtimeDashboard:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.figure, self.axes = plt.subplots(3, 2, figsize=(15, 11))
        self.figure.canvas.manager.set_window_title('Load-cell R-DAE realtime pipeline')
        self.status = 'Waiting for input file'
        self.stage = 0
        self.stage_names = [
            'Input loading',
            'Preprocessing and splitting',
            'R-DAE training',
            'Bottleneck extraction',
            'FCM clustering',
            'Health trend and outputs',
        ]
        self.training_line = None
        self.validation_line = None
        self._reset_axes()
        self.show_status('Ready')
        self.refresh()

    def _reset_axes(self):
        for axis in self.axes.flat:
            axis.clear()
            axis.grid(alpha=0.25)

        self.axes[0, 0].set_title('1. Raw ADC input and pseudo-clean reference')
        self.axes[0, 1].set_title('2. Processing split and window counts')
        self.axes[1, 0].set_title('3. R-DAE training progress')
        self.axes[1, 1].set_title('4. Reconstruction output')
        self.axes[2, 0].set_title('5. Bottleneck/error feature clusters')
        self.axes[2, 1].set_title('6. Health indicator and degradation trend')

    def refresh(self):
        if not self.enabled:
            return
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.03)

    def show_status(self, status):
        self.status = status
        self.figure.suptitle(
            f'Live pipeline monitor | Stage {self.stage}/6: {status}',
            fontsize=15,
            fontweight='bold',
        )
        self.refresh()

    def update_input(self, t, clean, noisy, file_name):
        self.stage = 1
        axis = self.axes[0, 0]
        axis.clear()
        axis.plot(t, noisy, label='Raw ADC', alpha=0.55, linewidth=0.8)
        axis.plot(t, clean, label='Moving-average reference', linewidth=1.5)
        axis.set_title(f'1. Input loaded: {file_name} | Applied load metadata: 535 g')
        axis.set_xlabel('Sample index')
        format_adc_axis(axis)
        axis.legend()
        axis.text(
            0.02,
            0.04,
            'Raw ADC counts are displayed in millions.\nNo ADC-to-grams calibration is applied.',
            transform=axis.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.85},
        )
        axis.grid(alpha=0.25)
        self.show_status(f'Loaded {len(noisy):,} ADC samples')

    def update_split(self, noisy, train_count, val_count, test_count, window_counts):
        self.stage = 2
        axis = self.axes[0, 1]
        axis.clear()
        split_end = [train_count, train_count + val_count, len(noisy)]
        axis.plot(np.arange(len(noisy)), noisy, color='#455A64', linewidth=0.7)
        axis.axvspan(0, split_end[0], color='#2E7D32', alpha=0.18, label='Train')
        axis.axvspan(split_end[0], split_end[1], color='#EF6C00', alpha=0.18, label='Validation')
        axis.axvspan(split_end[1], split_end[2], color='#1565C0', alpha=0.18, label='Test')
        axis.set_title('2. Chronological split and temporal windows')
        axis.set_xlabel('Sample index')
        format_adc_axis(axis)
        axis.legend(loc='upper right')
        axis.text(
            0.02,
            0.03,
            f"Windows\nTrain: {window_counts[0]:,}\nValidation: {window_counts[1]:,}\nTest: {window_counts[2]:,}",
            transform=axis.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.85},
            verticalalignment='bottom',
        )
        axis.grid(alpha=0.25)
        self.show_status('Preprocessing and window creation complete')

    def update_training(self, epoch, total_epochs, train_loss, validation_loss):
        self.stage = 3
        axis = self.axes[1, 0]
        axis.clear()
        axis.plot(np.arange(1, len(train_loss) + 1), train_loss, label='Train loss', marker='o', markersize=3)
        axis.plot(np.arange(1, len(validation_loss) + 1), validation_loss, label='Validation loss', marker='o', markersize=3)
        axis.set_title(f'3. R-DAE training: epoch {epoch}/{total_epochs}')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('MSE loss')
        axis.legend()
        axis.grid(alpha=0.25)
        self.show_status(f'Training epoch {epoch}/{total_epochs}')

    def update_reconstruction(self, reference, noisy, reconstructed):
        self.stage = 4
        axis = self.axes[1, 1]
        axis.clear()
        view_end = min(300, len(reference))
        x_values = np.arange(view_end)
        axis.plot(x_values, reference[:view_end], label='Reference', linewidth=2)
        axis.plot(x_values, noisy[:view_end], label='Noisy', alpha=0.6)
        axis.plot(x_values, reconstructed[:view_end], label='R-DAE reconstruction', linewidth=1.5)
        axis.set_title('4. Window reconstruction on test segment')
        axis.set_xlabel('Test-segment sample index')
        format_adc_axis(axis)
        axis.legend()
        axis.grid(alpha=0.25)
        self.show_status('Reconstruction generated')

    def update_clusters(self, feature_matrix, labels, centers):
        self.stage = 5
        axis = self.axes[2, 0]
        axis.clear()
        mean_error = feature_matrix[:, -3]
        std_error = feature_matrix[:, -2]
        scatter = axis.scatter(mean_error, std_error, c=labels, cmap='viridis', s=16, alpha=0.75)
        axis.scatter(centers[:, -3], centers[:, -2], c='red', marker='X', s=85, label='Centroid')
        handles = [
            Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=scatter.cmap(scatter.norm(cluster_id)),
                   markeredgecolor='none', markersize=7, label=f'Cluster {cluster_id}')
            for cluster_id in np.unique(labels)
        ]
        handles.append(Line2D([0], [0], marker='X', linestyle='None', color='red', markersize=8, label='Centroid'))
        axis.legend(handles=handles, title='FCM output')
        axis.set_xlabel('Mean absolute reconstruction error')
        axis.set_ylabel('Std. reconstruction error')
        axis.grid(alpha=0.25)
        self.show_status('Bottleneck features clustered with FCM')

    def update_trend(self, window_centers, trend):
        self.stage = 6
        axis = self.axes[2, 1]
        axis.clear()
        axis.plot(window_centers, trend['fuzzy_severity'], label='Fuzzy severity', alpha=0.65)
        axis.plot(window_centers, trend['reconstruction_severity'], label='Error severity', alpha=0.65)
        axis.plot(window_centers, trend['smoothed_degradation'], label='Smoothed degradation', color='black', linewidth=2)
        axis.set_title(f"6. Health trend | slope={trend['slope']:.6f}, monotonicity={trend['monotonicity']:.3f}")
        axis.set_xlabel('Original ADC sample index')
        axis.set_ylabel('Normalized score')
        axis.set_ylim(-0.05, 1.05)
        axis.legend()
        axis.grid(alpha=0.25)
        self.show_status('Health trend complete; outputs saved')

    def finalize(self):
        self.figure.tight_layout()
        self.refresh()


def prepare_windows(noisy, clean):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.concatenate([noisy.reshape(-1, 1), clean.reshape(-1, 1)]))
    train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean = split_signal(noisy, clean)

    def normalize(values):
        return scaler.transform(values.reshape(-1, 1)).ravel()

    windows = (
        create_sequences(normalize(train_noisy), WINDOW_SIZE),
        create_sequences(normalize(train_clean), WINDOW_SIZE),
        create_sequences(normalize(val_noisy), WINDOW_SIZE),
        create_sequences(normalize(val_clean), WINDOW_SIZE),
        create_sequences(normalize(test_noisy), WINDOW_SIZE),
        create_sequences(normalize(test_clean), WINDOW_SIZE),
    )
    return scaler, windows, (len(train_noisy), len(val_noisy), len(test_noisy))


def save_outputs(output_csv, window_centers, labels, uncertainty, trend):
    np.savetxt(
        output_csv,
        np.column_stack([
            window_centers,
            labels,
            uncertainty,
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


def run(adc_file, output_plot, output_csv, epochs, show):
    dashboard = RealtimeDashboard(enabled=show)
    dashboard.show_status('Reading one ADC file')
    t, clean, noisy = load_adc_file(adc_file)
    dashboard.update_input(t, clean, noisy, adc_file.name)

    scaler, windows, raw_counts = prepare_windows(noisy, clean)
    X_train, y_train, X_val, y_val, X_test, y_test = windows
    window_counts = (len(X_train), len(X_val), len(X_test))
    dashboard.update_split(noisy, raw_counts[0], raw_counts[1], raw_counts[2], window_counts)

    def model_input(values):
        return values.reshape(-1, WINDOW_SIZE, 1)

    model = build_rdae(WINDOW_SIZE)
    training_callback = LiveTrainingCallback(dashboard)
    model.fit(
        model_input(X_train),
        model_input(y_train),
        validation_data=(model_input(X_val), model_input(y_val)),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[training_callback],
    )

    dashboard.show_status('Extracting bottleneck features')
    _, _, _, feature_test = extract_bottleneck_features(model, model_input(X_test), model_input(y_test))
    centers, membership, labels, uncertainty = improved_fcm(feature_test, n_clusters=3)
    dashboard.update_clusters(feature_test, labels, centers)

    trend = build_health_trend(feature_test, membership, labels, SMOOTHING_WINDOW)
    reconstructed_windows = np.clip(model.predict(model_input(X_test), verbose=0), 0.0, 1.0)
    test_signal_length = raw_counts[2]
    reconstructed_seq = np.clip(reconstruct_signal(reconstructed_windows, test_signal_length, WINDOW_SIZE), 0.0, 1.0)
    reference_seq = reconstruct_signal(model_input(y_test), test_signal_length, WINDOW_SIZE)
    noisy_seq = reconstruct_signal(model_input(X_test), test_signal_length, WINDOW_SIZE)
    reconstructed = scaler.inverse_transform(reconstructed_seq.reshape(-1, 1)).ravel()
    reference = scaler.inverse_transform(reference_seq.reshape(-1, 1)).ravel()
    noisy_segment = scaler.inverse_transform(noisy_seq.reshape(-1, 1)).ravel()
    dashboard.update_reconstruction(reference, noisy_segment, reconstructed)

    test_start = int(len(noisy) * 0.85)
    window_centers = test_start + np.arange(len(labels)) + (WINDOW_SIZE - 1) / 2.0
    dashboard.update_trend(window_centers, trend)
    save_outputs(output_csv, window_centers, labels, uncertainty, trend)
    dashboard.finalize()
    dashboard.figure.savefig(output_plot, dpi=220)

    print('==== Realtime visualization complete ====')
    print(f'Input file: {adc_file}')
    print(f'Samples: {len(noisy)}')
    print(f'Windows [train, validation, test]: {window_counts}')
    print(f'Health trend slope: {trend["slope"]:.8f}')
    print(f'Monotonicity: {trend["monotonicity"]:.4f}')
    print(f'States [healthy, warning, degraded]: {np.bincount(trend["state_ids"], minlength=3).tolist()}')
    print(f'Saved dashboard: {output_plot}')
    print(f'Saved trend data: {output_csv}')

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Realtime visualization for one ADC file.')
    parser.add_argument('--file', type=Path, default=DEFAULT_ADC_FILE, help='One ADC input file.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs for the live run.')
    parser.add_argument('--output-plot', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output-csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--no-show', action='store_true', help='Run and save outputs without opening a GUI window.')
    args = parser.parse_args()

    run(args.file, args.output_plot, args.output_csv, max(1, args.epochs), not args.no_show)


if __name__ == '__main__':
    main()
