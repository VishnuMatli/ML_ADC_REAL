import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import Button
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from loadcell_planned_pipeline_adc import (
    BATCH_SIZE,
    EPOCHS,
    WINDOW_SIZE,
    build_rdae,
    create_sequences,
    load_adc_file,
    reconstruct_signal,
    split_signal,
)


DEFAULT_ADC_FILE = Path('/home/vishnu/Desktop/ML/adc_data/535g20250502pm427ms20hz50.txt')
DEFAULT_OUTPUT_DIR = Path('/home/vishnu/Desktop/ML/sequential_visualizations')
NOMINAL_LOAD_GRAMS = 535.0


def style_axis(axis, title, xlabel=None, ylabel=None):
    axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def animate_lines(figure, axis, lines, x_values, y_values, show):
    if not show:
        return
    plt.show(block=False)
    for line in lines:
        line.set_data([], [])

    points = max(len(values) for values in y_values)
    chunk = max(1, points // 50)
    for end in range(chunk, points + chunk, chunk):
        for line, x_data, y_data in zip(lines, x_values, y_values):
            visible = min(end, len(y_data))
            line.set_data(x_data[:visible], y_data[:visible])
        axis.relim()
        axis.autoscale_view()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.03)


def show_stage(figure, output_dir, stage_number, stage_name, show):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{stage_number:02d}_{stage_name}.png'
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)

    if show:
        was_interactive = plt.isinteractive()
        plt.ioff()

        advanced = {"value": False}
        button_ax = figure.add_axes([0.84, 0.015, 0.13, 0.055])
        next_button = Button(button_ax, "Next", color="#1565C0", hovercolor="#0D47A1")

        def advance(event=None):
            if advanced["value"]:
                return
            advanced["value"] = True
            if plt.fignum_exists(figure.number):
                plt.close(figure)

        next_button.on_clicked(advance)
        figure.canvas.mpl_connect("close_event", advance)

        figure.show()
        while not advanced["value"] and plt.fignum_exists(figure.number):
            figure.canvas.start_event_loop(timeout=0.1)

        if plt.fignum_exists(figure.number):
            plt.close(figure)
        if was_interactive:
            plt.ion()

    else:
        plt.close(figure)

    print(f'Stage {stage_number}: {stage_name} -> {output_path}')
    return output_path


def adc_to_weight(adc_values, nominal_load=NOMINAL_LOAD_GRAMS):
    """Provisional ratio conversion until a calibrated ADC-to-grams map is available."""
    reference_adc = float(np.median(adc_values))
    if reference_adc == 0:
        raise ValueError('Median ADC value is zero; ADC-to-weight conversion cannot be calculated.')
    return np.clip(adc_values / reference_adc * nominal_load, 0.0, None)


def generate_weight_target(raw_weight, smooth_window=31):
    """Generate the target in weight units from the raw weight signal."""
    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    padded = np.pad(raw_weight, (smooth_window // 2,), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def display_adc_conversion(t, raw_adc, weight, file_name, output_dir, show):
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    lines = axes[0].plot(t, raw_adc / 1e6, color='#455A64', linewidth=0.8, label='Raw ADC')
    style_axis(axes[0], f'1A. Raw ADC input: {file_name}', 'Sample index', 'ADC value (millions of counts)')
    axes[0].legend()

    lines += axes[1].plot(t, weight, color='#1565C0', linewidth=1.0, label='Converted weight')
    axes[1].axhline(NOMINAL_LOAD_GRAMS, color='#C62828', linestyle='--', label='Nominal 535 g condition')
    style_axis(axes[1], '1B. ADC converted to weight', 'Sample index', 'Weight (g)')
    axes[1].legend()
    axes[1].text(
        0.02,
        0.04,
        'Provisional conversion: median ADC = 535 g.\nThis uses weight units after conversion; it is not yet a calibrated load sensor model.',
        transform=axes[1].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.88},
    )

    animate_lines(figure, axes[0], [lines[0]], [t], [raw_adc / 1e6], show)
    animate_lines(figure, axes[1], [lines[1]], [t], [weight], show)
    return show_stage(figure, output_dir, 1, 'adc_to_weight', show)


def display_split(t, weight, counts, window_counts, output_dir, show):
    train_count, val_count, test_count = counts
    figure, axis = plt.subplots(figsize=(14, 6))

    line = axis.plot(t, weight, color='#455A64', linewidth=0.8, label='Weight (g)')[0]
    axis.axvspan(0, train_count, color='#2E7D32', alpha=0.18, label=f'Train: {train_count:,}')
    axis.axvspan(train_count, train_count + val_count, color='#EF6C00', alpha=0.18, label=f'Validation: {val_count:,}')
    axis.axvspan(train_count + val_count, train_count + val_count + test_count, color='#1565C0', alpha=0.18, label=f'Test: {test_count:,}')
    style_axis(axis, '2. Chronological split of weight signal', 'Sample index', 'Weight (g)')
    axis.legend(loc='upper right')
    axis.text(
        0.02,
        0.04,
        f'Window size: {WINDOW_SIZE}\nTrain windows: {window_counts[0]:,}\nValidation windows: {window_counts[1]:,}\nTest windows: {window_counts[2]:,}',
        transform=axis.transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.88},
    )

    animate_lines(figure, axis, [line], [t], [weight], show)
    return show_stage(figure, output_dir, 2, 'weight_chronological_split', show)


def display_target_generation(t, raw_weight, target_weight, output_dir, show):
    residual = raw_weight - target_weight
    figure, axes = plt.subplots(3, 1, figsize=(14, 11))

    view_end = min(1500, len(raw_weight))
    raw_line = axes[0].plot(t[:view_end], raw_weight[:view_end], color='#455A64', alpha=0.65, linewidth=0.8, label='Raw converted weight')[0]
    target_line = axes[0].plot(t[:view_end], target_weight[:view_end], color='#1565C0', linewidth=1.8, label='31-sample generated target')[0]
    style_axis(axes[0], '3A. Target generation from raw weight', 'Sample index', 'Weight (g)')
    axes[0].legend()

    residual_line = axes[1].plot(t[:view_end], residual[:view_end], color='#C62828', linewidth=0.8)[0]
    axes[1].axhline(0.0, color='black', linewidth=0.8)
    style_axis(axes[1], '3B. Raw weight minus generated target', 'Sample index', 'Residual weight (g)')

    kernel = np.ones(31, dtype=np.float64) / 31
    cumulative_removal = np.convolve(np.ones_like(raw_weight), kernel, mode='same')
    axes[2].plot(t[:view_end], raw_weight[:view_end] - target_weight[:view_end], color='#C62828', linewidth=0.8)
    axes[2].fill_between(t[:view_end], 0.0, (raw_weight[:view_end] - target_weight[:view_end]), color='#EF9A9A', alpha=0.35)
    style_axis(axes[2], '3C. Filtering effect retained in the residual', 'Sample index', 'Removed weight variation (g)')
    axes[2].text(
        0.02,
        0.04,
        f'Mean absolute residual: {np.mean(np.abs(residual)):.3f} g\nResidual standard deviation: {np.std(residual):.3f} g\nThe target is generated from the raw weight signal, not measured independently.',
        transform=axes[2].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.88},
    )

    animate_lines(figure, axes[0], [raw_line, target_line], [t[:view_end], t[:view_end]], [raw_weight[:view_end], target_weight[:view_end]], show)
    animate_lines(figure, axes[1], [residual_line], [t[:view_end]], [residual[:view_end]], show)
    animate_lines(figure, axes[2], axes[2].lines[:1], [t[:view_end]], [raw_weight[:view_end] - target_weight[:view_end]], show)
    return show_stage(figure, output_dir, 3, 'target_generation_from_weight', show)


def prepare_windows(weight, target_weight):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.concatenate([weight.reshape(-1, 1), target_weight.reshape(-1, 1)]))
    train_weight, train_target, val_weight, val_target, test_weight, test_target = split_signal(weight, target_weight)

    def normalize(values):
        return scaler.transform(values.reshape(-1, 1)).ravel()

    train_input = create_sequences(normalize(train_weight), WINDOW_SIZE)
    train_target = create_sequences(normalize(train_target), WINDOW_SIZE)
    val_input = create_sequences(normalize(val_weight), WINDOW_SIZE)
    val_target = create_sequences(normalize(val_target), WINDOW_SIZE)
    test_input = create_sequences(normalize(test_weight), WINDOW_SIZE)
    test_target = create_sequences(normalize(test_target), WINDOW_SIZE)
    raw_counts = (len(train_weight), len(val_weight), len(test_weight))
    return scaler, train_input, train_target, val_input, val_target, test_input, test_target, raw_counts


def display_windows(train_input, train_target, test_input, test_target, output_dir, show):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(train_input[0], label='Normalized weight input')
    axes[0, 0].plot(train_target[0], label='Normalized generated target')
    style_axis(axes[0, 0], '4A. One training window', 'Time step', 'Normalized weight [0, 1]')
    axes[0, 0].legend()

    image = axes[0, 1].imshow(train_input[:80].T, aspect='auto', cmap='viridis')
    style_axis(axes[0, 1], '4B. Weight windows entering R-DAE', 'Window index', 'Time step')

    axes[1, 0].plot(test_input[0], label='Normalized test weight')
    axes[1, 0].plot(test_target[0], label='Normalized test target')
    style_axis(axes[1, 0], '4C. One test window', 'Time step', 'Normalized weight [0, 1]')
    axes[1, 0].legend()

    axes[1, 1].axis('off')
    axes[1, 1].text(
        0.05,
        0.82,
        f'Input shape: {train_input.shape}\nModel input: ({WINDOW_SIZE}, 1)\n\nExpected output:\nFor each noisy weight window, output the corresponding smoothed weight window in grams after inverse scaling.',
        fontsize=12,
        va='top',
        bbox={'facecolor': '#E3F2FD', 'alpha': 0.9},
    )

    visible = np.full_like(train_input[:80].T, np.nan, dtype=float)
    image.set_data(visible)
    if show:
        rows = max(1, visible.shape[0] // 30)
        for end in range(rows, visible.shape[0] + rows, rows):
            visible[:min(end, visible.shape[0])] = train_input[:80].T[:min(end, visible.shape[0])]
            image.set_data(visible)
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.04)

    return show_stage(figure, output_dir, 4, 'weight_normalization_windows', show)


class LiveTrainingWindow(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, show):
        super().__init__()
        self.output_dir = output_dir
        self.show = show
        self.figure = None
        self.axis = None
        self.train_loss = []
        self.validation_loss = []

    def on_train_begin(self, logs=None):
        if not self.show:
            return
        self.figure, self.axis = plt.subplots(figsize=(12, 6))
        self.figure.canvas.manager.set_window_title('Stage 5 - R-DAE Training Loss')
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(float(logs.get('loss', np.nan)))
        self.validation_loss.append(float(logs.get('val_loss', np.nan)))
        if not self.show:
            return
        self.axis.clear()
        self.axis.plot(np.arange(1, len(self.train_loss) + 1), self.train_loss, label='Training MSE loss', marker='o', markersize=3)
        self.axis.plot(np.arange(1, len(self.validation_loss) + 1), self.validation_loss, label='Validation MSE loss', marker='o', markersize=3)
        style_axis(self.axis, f'5. R-DAE training loss: epoch {epoch + 1}/{self.params.get("epochs", "?")}', 'Epoch', 'MSE')
        self.axis.legend()
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.03)

    def on_train_end(self, logs=None):
        if not self.show:
            return
        self.figure.tight_layout()
        self.figure.savefig(self.output_dir / '05_rdae_training_loss.png', dpi=220)
        was_interactive = plt.isinteractive()
        plt.ioff()

        advanced = {"value": False}
        button_ax = self.figure.add_axes([0.84, 0.015, 0.13, 0.055])
        next_button = Button(button_ax, "Next", color="#1565C0", hovercolor="#0D47A1")

        def advance(event=None):
            if advanced["value"]:
                return
            advanced["value"] = True
            if plt.fignum_exists(self.figure.number):
                plt.close(self.figure)

        next_button.on_clicked(advance)
        self.figure.canvas.mpl_connect("close_event", advance)
        self.figure.show()
        while not advanced["value"] and plt.fignum_exists(self.figure.number):
            self.figure.canvas.start_event_loop(timeout=0.1)

        if plt.fignum_exists(self.figure.number):
            plt.close(self.figure)
        if was_interactive:
            plt.ion()


def display_weight_reconstruction(model, scaler, test_input, test_target, raw_counts, output_dir, show):
    model_input = test_input.reshape(-1, WINDOW_SIZE, 1)
    prediction_batches = []

    figure, axes = plt.subplots(2, 1, figsize=(14, 9))
    test_length = len(test_input) + WINDOW_SIZE - 1
    target_norm = reconstruct_signal(test_target.reshape(-1, WINDOW_SIZE, 1), test_length, WINDOW_SIZE)
    noisy_norm = reconstruct_signal(test_input.reshape(-1, WINDOW_SIZE, 1), test_length, WINDOW_SIZE)
    target_weight = scaler.inverse_transform(target_norm.reshape(-1, 1)).ravel()
    noisy_weight = scaler.inverse_transform(noisy_norm.reshape(-1, 1)).ravel()

    view_end = min(500, len(target_weight) - WINDOW_SIZE // 2)
    x_values = np.arange(view_end - WINDOW_SIZE // 2)
    noisy_line = axes[0].plot(x_values, noisy_weight[WINDOW_SIZE // 2:view_end], label='Raw converted weight', alpha=0.65)[0]
    target_line = axes[0].plot(x_values, target_weight[WINDOW_SIZE // 2:view_end], label='Generated target weight', linewidth=2)[0]
    output_line = axes[0].plot([], [], label='R-DAE output weight', linewidth=1.5)[0]
    style_axis(axes[0], '6A. R-DAE filtering in weight units', 'Test-sample index', 'Weight (g)')
    axes[0].legend()
    if show:
        plt.show(block=False)

    for batch_start in range(0, len(model_input), BATCH_SIZE):
        prediction = np.clip(model.predict(model_input[batch_start:batch_start + BATCH_SIZE], verbose=0), 0.0, 1.0)
        prediction_batches.append(prediction)

        if show:
            partial = np.concatenate(prediction_batches, axis=0)
            partial_length = len(partial) + WINDOW_SIZE - 1
            partial_norm = reconstruct_signal(partial, partial_length, WINDOW_SIZE)
            partial_weight = scaler.inverse_transform(partial_norm.reshape(-1, 1)).ravel()
            partial_end = min(view_end, len(partial_weight))
            visible_x = np.arange(partial_end - WINDOW_SIZE // 2)
            output_line.set_data(visible_x, partial_weight[WINDOW_SIZE // 2:partial_end])
            axes[0].relim()
            axes[0].autoscale_view()
            figure.suptitle(f'6. R-DAE reconstruction live | windows {min(batch_start + BATCH_SIZE, len(model_input))}/{len(model_input)}', fontsize=14)
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.05)

    predictions = np.concatenate(prediction_batches, axis=0)
    reconstructed_norm = np.clip(reconstruct_signal(predictions, test_length, WINDOW_SIZE), 0.0, 1.0)
    reconstructed_weight = scaler.inverse_transform(reconstructed_norm.reshape(-1, 1)).ravel()

    valid_start = WINDOW_SIZE // 2
    valid_end = len(target_weight) - WINDOW_SIZE // 2
    mae_noisy = mean_absolute_error(target_weight[valid_start:valid_end], noisy_weight[valid_start:valid_end])
    mae_output = mean_absolute_error(target_weight[valid_start:valid_end], reconstructed_weight[valid_start:valid_end])

    output_line.set_data(x_values, reconstructed_weight[valid_start:view_end])
    axes[0].relim()
    axes[0].autoscale_view()

    residual = noisy_weight - reconstructed_weight
    axes[1].plot(np.arange(valid_start, valid_end), residual[valid_start:valid_end], color='#C62828', linewidth=0.8)
    axes[1].axhline(0.0, color='black', linewidth=0.8)
    style_axis(axes[1], '6B. Reconstruction residual in weight units', 'Test-sample index', 'Weight residual (g)')
    axes[1].text(
        0.02,
        0.05,
        f'Raw weight vs target MAE: {mae_noisy:.3f} g\nR-DAE output vs target MAE: {mae_output:.3f} g\nNegative residual: output above target | Positive residual: output below target',
        transform=axes[1].transAxes,
        bbox={'facecolor': '#E8F5E9', 'alpha': 0.9},
    )

    show_stage(figure, output_dir, 6, 'weight_reconstruction', show)
    return {
        'raw_weight': noisy_weight,
        'target_weight': target_weight,
        'output_weight': reconstructed_weight,
        'mae_noisy': mae_noisy,
        'mae_output': mae_output,
        'valid_start': valid_start,
        'valid_end': valid_end,
    }


def display_degradation_reason(reconstruction, raw_weight, target_weight, output_dir, show):
    valid_start = reconstruction['valid_start']
    valid_end = reconstruction['valid_end']
    raw_segment = reconstruction['raw_weight'][valid_start:valid_end]
    target_segment = reconstruction['target_weight'][valid_start:valid_end]
    output_segment = reconstruction['output_weight'][valid_start:valid_end]
    residual_segment = raw_segment - target_segment

    x = np.arange(len(output_segment), dtype=np.float64)
    raw_slope = float(np.polyfit(x, raw_segment, 1)[0]) if len(x) > 1 else 0.0
    target_slope = float(np.polyfit(x, target_segment, 1)[0]) if len(x) > 1 else 0.0
    residual_slope = float(np.polyfit(x, residual_segment, 1)[0]) if len(x) > 1 else 0.0

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(x, raw_segment, label='Raw weight', alpha=0.7)
    axes[0, 0].plot(x, target_segment, label='Generated target weight', linewidth=2)
    axes[0, 0].plot(x, output_segment, label='R-DAE output weight', linewidth=1.5)
    style_axis(axes[0, 0], '7A. Weight-domain signal trend', 'Test-sample index', 'Weight (g)')
    axes[0, 0].legend()

    axes[0, 1].plot(x, residual_segment, color='#C62828', linewidth=0.8)
    axes[0, 1].axhline(0.0, color='black', linewidth=0.8)
    style_axis(axes[0, 1], '7B. Residual trend', 'Test-sample index', 'Raw weight minus target (g)')

    rolling = max(31, len(residual_segment) // 30)
    residual_variation = np.convolve(residual_segment, np.ones(rolling) / rolling, mode='same')
    axes[1, 0].plot(x, residual_variation, color='#EF6C00', linewidth=2)
    style_axis(axes[1, 0], '7C. Smoothed residual variation', 'Test-sample index', 'Residual weight (g)')

    mean_weight = float(np.mean(raw_segment))
    range_weight = float(np.ptp(raw_segment))
    residual_std = float(np.std(residual_segment))
    reasons = []
    if abs(residual_slope) > 1e-6:
        direction = 'increasing' if residual_slope > 0 else 'decreasing'
        reasons.append(f'Residual trend is {direction} ({residual_slope:.6f} g/sample).')
    else:
        reasons.append('Residual trend is nearly flat.')
    if abs(raw_slope) > 1e-6:
        direction = 'upward' if raw_slope > 0 else 'downward'
        reasons.append(f'Raw weight has a {direction} drift ({raw_slope:.6f} g/sample).')
    else:
        reasons.append('Raw weight drift is negligible.')
    reasons.append(f'Raw weight range is {range_weight:.3f} g around {mean_weight:.3f} g mean.')
    reasons.append(f'Residual variability is {residual_std:.3f} g standard deviation.')

    axes[1, 1].axis('off')
    axes[1, 1].text(
        0.05,
        0.92,
        '7D. Why degradation appears or does not appear\n\n' + '\n'.join(reasons),
        fontsize=12,
        va='top',
        bbox={'facecolor': '#FFF3E0', 'alpha': 0.95},
    )

    animate_lines(figure, axes[0, 0], axes[0, 0].lines, [x, x, x], [raw_segment, target_segment, output_segment], show)
    animate_lines(figure, axes[0, 1], axes[0, 1].lines[:1], [x], [residual_segment], show)
    animate_lines(figure, axes[1, 0], axes[1, 0].lines[:1], [x], [residual_variation], show)
    show_stage(figure, output_dir, 7, 'weight_degradation_reason', show)


def display_final_summary(adc_file, weight, reconstruction, training, raw_counts, window_counts, output_dir, show):
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.axis('off')
    final_text = (
        '8. Final weight-domain pipeline summary\n\n'
        f'Input file: {adc_file.name}\n'
        f'ADC samples: {raw_counts[0] + raw_counts[1] + raw_counts[2]:,}\n'
        f'Mean converted weight: {np.mean(weight):.3f} g\n'
        f'Weight range: {np.ptp(weight):.3f} g\n'
        f'Windows [train, validation, test]: {window_counts}\n'
        f'Training final MSE: {training.train_loss[-1]:.6f}\n'
        f'Validation final MSE: {training.validation_loss[-1]:.6f}\n'
        f'Raw weight vs target MAE: {reconstruction["mae_noisy"]:.3f} g\n'
        f'R-DAE output vs target MAE: {reconstruction["mae_output"]:.3f} g\n\n'
        'Final output: reconstructed weight in grams, residual weight, and degradation evidence.\n'
        'FCM and encoder bottleneck stages are not used in this weight-domain visualization.'
    )
    axis.text(0.05, 0.95, final_text, va='top', fontsize=13, family='monospace')
    return show_stage(figure, output_dir, 8, 'final_weight_summary', show)


def run(adc_file, output_dir, epochs, show):
    t, adc_clean, raw_adc = load_adc_file(adc_file)
    raw_weight = adc_to_weight(raw_adc)
    target_weight = generate_weight_target(raw_weight)

    display_adc_conversion(t, raw_adc, raw_weight, adc_file.name, output_dir, show)

    train_weight, train_target, val_weight, val_target, test_weight, test_target = split_signal(raw_weight, target_weight)
    counts = (len(train_weight), len(val_weight), len(test_weight))
    scaler, train_input, train_target, val_input, val_target, test_input, test_target, raw_counts = prepare_windows(raw_weight, target_weight)
    window_counts = (len(train_input), len(val_input), len(test_input))

    display_split(t, raw_weight, counts, window_counts, output_dir, show)
    display_target_generation(t, raw_weight, target_weight, output_dir, show)
    display_windows(train_input, train_target, test_input, test_target, output_dir, show)

    model = build_rdae(WINDOW_SIZE)
    training = LiveTrainingWindow(output_dir, show)
    model.fit(
        train_input.reshape(-1, WINDOW_SIZE, 1),
        train_target.reshape(-1, WINDOW_SIZE, 1),
        validation_data=(val_input.reshape(-1, WINDOW_SIZE, 1), val_target.reshape(-1, WINDOW_SIZE, 1)),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[training],
    )

    if not show:
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(np.arange(1, len(training.train_loss) + 1), training.train_loss, label='Training MSE loss', marker='o')
        axis.plot(np.arange(1, len(training.validation_loss) + 1), training.validation_loss, label='Validation MSE loss', marker='o')
        style_axis(axis, '5. R-DAE training loss', 'Epoch', 'MSE')
        axis.legend()
        show_stage(figure, output_dir, 5, 'rdae_training_loss', show)

    reconstruction = display_weight_reconstruction(model, scaler, test_input, test_target, raw_counts, output_dir, show)
    display_degradation_reason(reconstruction, raw_weight, target_weight, output_dir, show)
    display_final_summary(adc_file, raw_weight, reconstruction, training, raw_counts, window_counts, output_dir, show)


def main():
    parser = argparse.ArgumentParser(description='Sequential weight-domain load-cell visualization.')
    parser.add_argument('--file', type=Path, default=DEFAULT_ADC_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    run(args.file, args.output_dir, max(1, args.epochs), not args.no_show)


if __name__ == '__main__':
    main()
