import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Model

from loadcell_planned_pipeline_adc import (
    BATCH_SIZE,
    EPOCHS,
    WINDOW_SIZE,
    build_rdae,
    create_sequences,
    improved_fcm,
    load_adc_file,
    reconstruct_signal,
    split_signal,
)
from loadcell_health_trend_pipeline import SMOOTHING_WINDOW, build_health_trend


DEFAULT_ADC_FILE = Path('/home/vishnu/Desktop/ML/adc_data/535g20250502pm427ms20hz50.txt')
DEFAULT_OUTPUT_DIR = Path('/home/vishnu/Desktop/ML/sequential_visualizations')
NOMINAL_LOAD_GRAMS = 535.0


def format_adc_axis(axis):
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e6:.2f}'))
    axis.set_ylabel('ADC value (millions of counts)')


def style_axis(axis, title, xlabel=None, ylabel=None):
    axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def animate_axis_lines(figure, axis, lines, x_values, y_values, show):
    if not show:
        return
    plt.show(block=False)
    for line in lines:
        line.set_data([], [])
    points = max(len(values) for values in y_values)
    chunk = max(1, points // 40)
    for end in range(chunk, points + chunk, chunk):
        for line, x_data, y_data in zip(lines, x_values, y_values):
            visible_end = min(end, len(y_data))
            line.set_data(x_data[:visible_end], y_data[:visible_end])
        axis.relim()
        axis.autoscale_view()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.03)


def animate_image(figure, image, data, show):
    if not show:
        return
    plt.show(block=False)
    visible = np.full_like(data, np.nan, dtype=float)
    image.set_data(visible)
    rows = max(1, data.shape[0] // 30)
    for end in range(rows, data.shape[0] + rows, rows):
        visible[:min(end, data.shape[0])] = data[:min(end, data.shape[0])]
        image.set_data(visible)
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.04)


def show_stage(figure, output_dir, stage_number, stage_name, show):
    output_path = output_dir / f'{stage_number:02d}_{stage_name}.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    if show:
        was_interactive = plt.isinteractive()
        plt.ioff()
        figure.canvas.draw()
        figure.show()
        plt.show(block=True)
        if was_interactive:
            plt.ion()
    plt.close(figure)
    print(f'Stage {stage_number}: {stage_name} -> {output_path}')
    return output_path


def adc_to_relative_weight(adc_values, nominal_load=NOMINAL_LOAD_GRAMS):
    """Show a provisional ratio-based weight view, not a calibrated conversion."""
    reference_adc = float(np.median(adc_values))
    return adc_values / reference_adc * nominal_load


def display_adc_to_weight(t, noisy, file_name, output_dir, show):
    provisional_weight = adc_to_relative_weight(noisy)
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t, noisy, color='#455A64', linewidth=0.8)
    format_adc_axis(axes[0])
    style_axis(axes[0], f'1A. Raw ADC input: {file_name}', 'Sample index')
    axes[0].text(
        0.02,
        0.04,
        'Input values are raw ADC counts, shown in millions.',
        transform=axes[0].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.85},
    )

    axes[1].plot(t, provisional_weight, color='#1565C0', linewidth=1.0)
    axes[1].axhline(NOMINAL_LOAD_GRAMS, color='#C62828', linestyle='--', label='535 g filename metadata')
    style_axis(
        axes[1],
        '1B. ADC-to-weight interpretation',
        'Sample index',
        'Provisional relative weight (g)',
    )
    axes[1].legend()
    axes[1].text(
        0.02,
        0.04,
        'Ratio view only: ADC / median(ADC) x 535 g.\nA true calibration curve is not available yet.',
        transform=axes[1].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.85},
    )
    animate_axis_lines(figure, axes[0], axes[0].lines, [t], [noisy], show)
    animate_axis_lines(figure, axes[1], axes[1].lines[:1], [t], [provisional_weight], show)
    return show_stage(figure, output_dir, 1, 'adc_to_weight', show)


def display_split(t, noisy, counts, window_counts, output_dir, show):
    train_count, val_count, test_count = counts
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(t, noisy, color='#455A64', linewidth=0.8, label='Raw ADC')
    axis.axvspan(0, train_count, color='#2E7D32', alpha=0.18, label=f'Train ({train_count:,} samples)')
    axis.axvspan(train_count, train_count + val_count, color='#EF6C00', alpha=0.18, label=f'Validation ({val_count:,} samples)')
    axis.axvspan(train_count + val_count, len(noisy), color='#1565C0', alpha=0.18, label=f'Test ({test_count:,} samples)')
    format_adc_axis(axis)
    style_axis(axis, '2. Chronological split and temporal window preparation', 'Sample index')
    axis.legend(loc='upper right')
    axis.text(
        0.02,
        0.04,
        f'Window size: {WINDOW_SIZE}\nTrain windows: {window_counts[0]:,}\nValidation windows: {window_counts[1]:,}\nTest windows: {window_counts[2]:,}',
        transform=axis.transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.88},
    )
    animate_axis_lines(figure, axis, axis.lines[:1], [t], [noisy], show)
    return show_stage(figure, output_dir, 2, 'chronological_split', show)


def display_filtering(t, noisy, clean, output_dir, show):
    residual = noisy - clean
    figure, axes = plt.subplots(2, 1, figsize=(14, 9))
    view_end = min(1000, len(noisy))
    axes[0].plot(t[:view_end], noisy[:view_end], label='Raw ADC', alpha=0.65, linewidth=0.8)
    axes[0].plot(t[:view_end], clean[:view_end], label='31-sample moving-average reference', linewidth=1.5)
    format_adc_axis(axes[0])
    style_axis(axes[0], '3A. Filtering/reference generation', 'Sample index')
    axes[0].legend()

    axes[1].plot(t[:view_end], residual[:view_end], color='#C62828', linewidth=0.8)
    axes[1].axhline(0.0, color='black', linewidth=0.8)
    style_axis(axes[1], '3B. Filtering residual: raw ADC minus moving-average reference', 'Sample index', 'Residual ADC counts')
    axes[1].text(
        0.02,
        0.04,
        f'Mean absolute residual: {np.mean(np.abs(residual)):.2f} counts\nStd. residual: {np.std(residual):.2f} counts\nThis is a pseudo-clean filtering reference, not ground truth.',
        transform=axes[1].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.88},
    )
    animate_axis_lines(figure, axes[0], axes[0].lines, [t[:view_end], t[:view_end]], [noisy[:view_end], clean[:view_end]], show)
    animate_axis_lines(figure, axes[1], axes[1].lines[:1], [t[:view_end]], [residual[:view_end]], show)
    return show_stage(figure, output_dir, 3, 'filtering_total', show)


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
    return scaler, windows


def display_normalization(windows, output_dir, show):
    X_train, y_train, _, _, X_test, y_test = windows
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(X_train[0], label='Normalized noisy input')
    axes[0, 0].plot(y_train[0], label='Normalized target')
    style_axis(axes[0, 0], '4A. One normalized training window', 'Window time step', 'Normalized value [0, 1]')
    axes[0, 0].legend()

    window_image = axes[0, 1].imshow(X_train[:80].T, aspect='auto', cmap='viridis')
    style_axis(axes[0, 1], '4B. Training windows entering the LSTM', 'Window index', 'Time step')

    axes[1, 0].plot(X_test[0], label='Normalized test input')
    axes[1, 0].plot(y_test[0], label='Normalized test target')
    style_axis(axes[1, 0], '4C. One normalized test window', 'Window time step', 'Normalized value [0, 1]')
    axes[1, 0].legend()

    axes[1, 1].axis('off')
    axes[1, 1].text(
        0.05,
        0.8,
        f'Input shape: {X_train.shape}\nModel shape: ({X_train.shape[0]}, {WINDOW_SIZE}, 1)\n\nTarget of this stage:\nConvert large ADC counts into a stable [0, 1] sequence representation for LSTM training.',
        fontsize=12,
        va='top',
        bbox={'facecolor': '#E3F2FD', 'alpha': 0.9},
    )
    animate_image(figure, window_image, X_train[:80].T, show)
    return show_stage(figure, output_dir, 4, 'normalization_and_windows', show)


class LiveTrainingWindow(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, show):
        super().__init__()
        self.output_dir = output_dir
        self.show = show
        self.figure = None
        self.axis = None
        self.train_loss = []
        self.val_loss = []

    def on_train_begin(self, logs=None):
        if not self.show:
            return
        self.figure, self.axis = plt.subplots(figsize=(12, 6))
        self.figure.canvas.manager.set_window_title('Stage 5 - Live R-DAE training')
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(float(logs.get('loss', np.nan)))
        self.val_loss.append(float(logs.get('val_loss', np.nan)))
        if not self.show:
            return
        self.axis.clear()
        self.axis.plot(self.train_loss, label='Train MSE', marker='o', markersize=3)
        self.axis.plot(self.val_loss, label='Validation MSE', marker='o', markersize=3)
        self.axis.set_title(f'5. Live R-DAE training: epoch {epoch + 1}/{self.params.get("epochs", "?")}')
        self.axis.set_xlabel('Epoch')
        self.axis.set_ylabel('MSE loss')
        self.axis.legend()
        self.axis.grid(alpha=0.25)
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.03)

    def on_train_end(self, logs=None):
        if not self.show:
            return
        self.figure.tight_layout()
        self.figure.savefig(self.output_dir / '05_rdae_training.png', dpi=220)
        was_interactive = plt.isinteractive()
        plt.ioff()
        self.figure.show()
        plt.show(block=True)
        if was_interactive:
            plt.ion()
        plt.close(self.figure)


def display_latency(model, test_inputs, output_dir, show):
    encoder = Model(model.input, model.get_layer('bottleneck').output, name='latency_encoder')
    batch_count = (len(test_inputs) + BATCH_SIZE - 1) // BATCH_SIZE
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].bar(np.arange(batch_count), np.zeros(batch_count), color='#1565C0')
    style_axis(axes[0], '6A. Live encoder latency by test batch', 'Batch index', 'Latency (ms)')
    latent_lines = axes[1].plot([], [], label='Bottleneck feature 1')
    latent_lines += axes[1].plot([], [], label='Bottleneck feature 2', alpha=0.75)
    style_axis(axes[1], '6B. Live bottleneck output', 'Test-window index', 'Latent value')
    axes[1].legend()
    if show:
        plt.show(block=False)

    batch_latencies = []
    encoded_batches = []
    for batch_start in range(0, len(test_inputs), BATCH_SIZE):
        batch = test_inputs[batch_start:batch_start + BATCH_SIZE]
        start = time.perf_counter()
        encoded = encoder.predict(batch, verbose=0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        batch_latencies.append(elapsed_ms)
        encoded_batches.append(encoded)
        if show:
            batch_index = len(batch_latencies) - 1
            bars[batch_index].set_height(elapsed_ms)
            visible_latent = np.concatenate(encoded_batches, axis=0)
            x_values = np.arange(len(visible_latent))
            latent_lines[0].set_data(x_values, visible_latent[:, 0])
            latent_lines[1].set_data(x_values, visible_latent[:, 1])
            axes[1].relim()
            axes[1].autoscale_view()
            figure.suptitle(f'6. Encoding live | completed batch {batch_index + 1}/{batch_count}', fontsize=14)
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.05)

    latent = np.concatenate(encoded_batches, axis=0)
    total_ms = float(np.sum(batch_latencies))
    per_window_ms = total_ms / len(test_inputs)
    figure.suptitle(
        f'6. Encoding latency | total={total_ms:.2f} ms | per window={per_window_ms:.4f} ms | features={latent.shape[1]}',
        fontsize=14,
    )
    result = {'latent': latent, 'total_ms': total_ms, 'per_window_ms': per_window_ms, 'batch_latencies': batch_latencies}
    show_stage(figure, output_dir, 6, 'encoding_latency', show)
    return result


def display_filtering_output(model, scaler, X_test, y_test, output_dir, show):
    model_inputs = X_test.reshape(-1, WINDOW_SIZE, 1)
    test_length = len(X_test) + WINDOW_SIZE - 1
    target_norm = reconstruct_signal(y_test.reshape(-1, WINDOW_SIZE, 1), test_length, WINDOW_SIZE)
    input_norm = reconstruct_signal(X_test.reshape(-1, WINDOW_SIZE, 1), test_length, WINDOW_SIZE)

    target = scaler.inverse_transform(target_norm.reshape(-1, 1)).ravel()
    noisy = scaler.inverse_transform(input_norm.reshape(-1, 1)).ravel()
    view_end = min(500, test_length - WINDOW_SIZE // 2)
    figure, axes = plt.subplots(2, 1, figsize=(14, 9))
    x_values = np.arange(view_end - WINDOW_SIZE // 2)
    noisy_line = axes[0].plot(x_values, noisy[WINDOW_SIZE // 2:view_end], label='Noisy test input', alpha=0.65)[0]
    target_line = axes[0].plot(x_values, target[WINDOW_SIZE // 2:view_end], label='Target / pseudo-clean reference', linewidth=2)[0]
    output_line = axes[0].plot([], [], label='R-DAE output', linewidth=1.5)[0]
    format_adc_axis(axes[0])
    style_axis(axes[0], '7A. Live total filtering: input -> R-DAE output', 'Test-segment sample index')
    axes[0].legend()
    if show:
        plt.show(block=False)

    prediction_batches = []
    for batch_start in range(0, len(model_inputs), BATCH_SIZE):
        prediction_batch = np.clip(
            model.predict(model_inputs[batch_start:batch_start + BATCH_SIZE], verbose=0),
            0.0,
            1.0,
        )
        prediction_batches.append(prediction_batch)
        if show:
            partial_predictions = np.concatenate(prediction_batches, axis=0)
            partial_length = len(partial_predictions) + WINDOW_SIZE - 1
            partial_norm = reconstruct_signal(partial_predictions, partial_length, WINDOW_SIZE)
            partial_values = scaler.inverse_transform(partial_norm.reshape(-1, 1)).ravel()
            partial_end = min(view_end, len(partial_values))
            output_line.set_data(
                np.arange(partial_end - WINDOW_SIZE // 2),
                partial_values[WINDOW_SIZE // 2:partial_end],
            )
            axes[0].relim()
            axes[0].autoscale_view()
            figure.suptitle(
                f'7. Filtering live | completed windows {min(batch_start + BATCH_SIZE, len(model_inputs))}/{len(model_inputs)}',
                fontsize=14,
            )
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.05)

    predictions = np.concatenate(prediction_batches, axis=0)
    reconstructed_norm = np.clip(reconstruct_signal(predictions, test_length, WINDOW_SIZE), 0.0, 1.0)
    reconstructed = scaler.inverse_transform(reconstructed_norm.reshape(-1, 1)).ravel()
    valid_start = WINDOW_SIZE // 2
    valid_end = len(target) - WINDOW_SIZE // 2
    mse_noisy = mean_squared_error(target[valid_start:valid_end], noisy[valid_start:valid_end])
    mae_noisy = mean_absolute_error(target[valid_start:valid_end], noisy[valid_start:valid_end])
    mse_reconstructed = mean_squared_error(target[valid_start:valid_end], reconstructed[valid_start:valid_end])
    mae_reconstructed = mean_absolute_error(target[valid_start:valid_end], reconstructed[valid_start:valid_end])

    output_line.set_data(x_values, reconstructed[valid_start:view_end])
    axes[0].relim()
    axes[0].autoscale_view()

    axes[1].axis('off')
    axes[1].text(
        0.04,
        0.85,
        f'Expected target:\nReconstructed signal should follow the pseudo-clean reference while reducing raw fluctuations.\n\nNoisy vs target:       MSE={mse_noisy:.3f}, MAE={mae_noisy:.3f}\nR-DAE output vs target: MSE={mse_reconstructed:.3f}, MAE={mae_reconstructed:.3f}\n\nThese metrics are relative to a moving-average target, not an independent physical ground truth.',
        fontsize=12,
        va='top',
        bbox={'facecolor': '#E8F5E9', 'alpha': 0.9},
    )
    animate_axis_lines(
        figure,
        axes[0],
        axes[0].lines,
        [x_values, x_values, x_values],
        [noisy[valid_start:view_end], target[valid_start:view_end], reconstructed[valid_start:view_end]],
        show,
    )
    show_stage(figure, output_dir, 7, 'total_filtering_and_reconstruction', show)
    return {
        'predictions': predictions,
        'reconstructed': reconstructed,
        'target': target,
        'noisy': noisy,
        'mse_noisy': mse_noisy,
        'mae_noisy': mae_noisy,
        'mse_reconstructed': mse_reconstructed,
        'mae_reconstructed': mae_reconstructed,
    }


def display_clusters(feature_matrix, labels, centers, output_dir, show):
    mean_error = feature_matrix[:, -3]
    std_error = feature_matrix[:, -2]
    figure, axis = plt.subplots(figsize=(10, 7))
    scatter = axis.scatter(mean_error, std_error, c=labels, cmap='viridis', s=20, alpha=0.8)
    centroid_scatter = axis.scatter(centers[:, -3], centers[:, -2], c='red', marker='X', s=90)
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=scatter.cmap(scatter.norm(cluster_id)),
               markeredgecolor='none', markersize=7, label=f'Cluster {cluster_id}')
        for cluster_id in np.unique(labels)
    ]
    handles.append(Line2D([0], [0], marker='X', linestyle='None', color='red', markersize=8, label='Centroid'))
    axis.legend(handles=handles, title='FCM output')
    style_axis(axis, '8. FCM clustering of reconstruction-error features', 'Mean absolute reconstruction error', 'Std. reconstruction error')
    if show:
        plt.show(block=False)
        scatter.set_offsets(np.empty((0, 2)))
        for end in range(max(1, len(mean_error) // 30), len(mean_error) + max(1, len(mean_error) // 30), max(1, len(mean_error) // 30)):
            scatter.set_offsets(np.column_stack([mean_error[:end], std_error[:end]]))
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.04)
        centroid_scatter.set_offsets(np.column_stack([centers[:, -3], centers[:, -2]]))
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.08)
    return show_stage(figure, output_dir, 8, 'fcm_clusters', show)


def display_health_trend(feature_matrix, membership, labels, output_dir, show):
    trend = build_health_trend(feature_matrix, membership, labels, SMOOTHING_WINDOW)
    window_centers = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(window_centers, trend['fuzzy_severity'], label='Fuzzy cluster severity', alpha=0.65)
    axes[0].plot(window_centers, trend['reconstruction_severity'], label='Reconstruction-error severity', alpha=0.65)
    axes[0].plot(window_centers, trend['smoothed_degradation'], label='Smoothed degradation index', color='black', linewidth=2)
    style_axis(axes[0], f'9A. Health trend | slope={trend["slope"]:.6f}, monotonicity={trend["monotonicity"]:.3f}', 'Test-window index', 'Normalized score')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[1].step(window_centers, trend['state_ids'], where='mid', color='#C62828')
    axes[1].set_yticks([0, 1, 2], ['Healthy', 'Warning', 'Degraded'])
    style_axis(axes[1], '9B. Health-state output', 'Test-window index', 'State')
    animate_axis_lines(
        figure,
        axes[0],
        axes[0].lines,
        [window_centers, window_centers, window_centers],
        [trend['fuzzy_severity'], trend['reconstruction_severity'], trend['smoothed_degradation']],
        show,
    )
    animate_axis_lines(figure, axes[1], axes[1].lines, [window_centers], [trend['state_ids']], show)
    show_stage(figure, output_dir, 9, 'health_trend', show)
    return trend


def display_summary(adc_file, outputs, latency, filter_result, trend, output_dir, show):
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.axis('off')
    state_counts = np.bincount(trend['state_ids'], minlength=3).tolist()
    text = (
        '10. FINAL PIPELINE OUTPUTS\n\n'
        f'Input file: {adc_file.name}\n'
        f'ADC samples: {outputs["sample_count"]:,}\n'
        f'Windows [train, validation, test]: {outputs["window_counts"]}\n'
        f'Encoding total latency: {latency["total_ms"]:.2f} ms\n'
        f'Encoding latency per test window: {latency["per_window_ms"]:.4f} ms\n'
        f'Noisy test MAE: {filter_result["mae_noisy"]:.3f}\n'
        f'R-DAE test MAE: {filter_result["mae_reconstructed"]:.3f}\n'
        f'Health trend slope: {trend["slope"]:.8f}\n'
        f'Monotonicity: {trend["monotonicity"]:.4f}\n'
        f'States [Healthy, Warning, Degraded]: {state_counts}\n\n'
        'Expected final output:\n'
        '- A denoised/reconstructed ADC-domain test signal\n'
        '- Bottleneck features and FCM clusters\n'
        '- A health/degradation indicator for later validation\n\n'
        'Saved intermediate windows and outputs are listed below.'
    )
    axis.text(0.05, 0.95, text, va='top', fontsize=13, family='monospace')
    output_text = '\n'.join(str(path) for path in output_dir.glob('*.png'))
    axis.text(0.05, 0.05, f'Visualization files:\n{output_text}', fontsize=9, va='bottom')
    return show_stage(figure, output_dir, 10, 'final_summary', show)


def run(adc_file, output_dir, epochs, show):
    output_dir.mkdir(parents=True, exist_ok=True)
    if show:
        plt.ion()

    t, clean, noisy = load_adc_file(adc_file)
    display_adc_to_weight(t, noisy, adc_file.name, output_dir, show)

    train_noisy, train_clean, val_noisy, val_clean, test_noisy, test_clean = split_signal(noisy, clean)
    counts = (len(train_noisy), len(val_noisy), len(test_noisy))
    scaler, windows = prepare_windows(noisy, clean)
    X_train, y_train, X_val, y_val, X_test, y_test = windows
    window_counts = (len(X_train), len(X_val), len(X_test))
    display_split(t, noisy, counts, window_counts, output_dir, show)
    display_filtering(t, noisy, clean, output_dir, show)
    display_normalization(windows, output_dir, show)

    model = build_rdae(WINDOW_SIZE)
    training_window = LiveTrainingWindow(output_dir, show)
    model.fit(
        X_train.reshape(-1, WINDOW_SIZE, 1),
        y_train.reshape(-1, WINDOW_SIZE, 1),
        validation_data=(X_val.reshape(-1, WINDOW_SIZE, 1), y_val.reshape(-1, WINDOW_SIZE, 1)),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[training_window],
    )
    if not show:
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(training_window.train_loss, label='Train MSE')
        axis.plot(training_window.val_loss, label='Validation MSE')
        style_axis(axis, '5. R-DAE training result', 'Epoch', 'MSE loss')
        axis.legend()
        show_stage(figure, output_dir, 5, 'rdae_training', show)

    model_inputs = X_test.reshape(-1, WINDOW_SIZE, 1)
    encoder_latency = display_latency(model, model_inputs, output_dir, show)
    latent = encoder_latency['latent']
    predictions = np.clip(model.predict(model_inputs, verbose=0), 0.0, 1.0)
    target_windows = y_test.reshape(-1, WINDOW_SIZE, 1)
    residual = np.abs(predictions - target_windows)
    feature_matrix = np.concatenate([
        latent,
        residual.mean(axis=1).reshape(-1, 1),
        residual.std(axis=1).reshape(-1, 1),
        residual.max(axis=1).reshape(-1, 1),
    ], axis=1)
    centers, membership, labels, _ = improved_fcm(feature_matrix, n_clusters=3)
    filter_result = display_filtering_output(model, scaler, X_test, y_test, output_dir, show)
    display_clusters(feature_matrix, labels, centers, output_dir, show)
    trend = display_health_trend(feature_matrix, membership, labels, output_dir, show)

    output_csv = output_dir / 'per_window_results.csv'
    np.savetxt(
        output_csv,
        np.column_stack([
            np.arange(len(labels)),
            labels,
            trend['fuzzy_severity'],
            trend['reconstruction_severity'],
            trend['smoothed_degradation'],
            trend['state_ids'],
        ]),
        delimiter=',',
        header='test_window,cluster,fuzzy_severity,reconstruction_severity,smoothed_degradation,state_id',
        comments='',
    )
    outputs = {
        'sample_count': len(noisy),
        'window_counts': window_counts,
    }
    display_summary(adc_file, outputs, encoder_latency, filter_result, trend, output_dir, show)
    print(f'Per-window data saved: {output_csv}')
    print('Sequential visualization completed.')


def main():
    parser = argparse.ArgumentParser(description='Sequential realtime visualization for one ADC file.')
    parser.add_argument('--file', type=Path, default=DEFAULT_ADC_FILE, help='One ADC input file.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--no-show', action='store_true', help='Save each stage without opening separate windows.')
    args = parser.parse_args()
    run(args.file, args.output_dir, max(1, args.epochs), not args.no_show)


if __name__ == '__main__':
    main()
