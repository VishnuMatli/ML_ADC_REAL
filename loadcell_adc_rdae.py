"""Denoise the real 20 Hz ADC recordings in ``adc_data``.

The paper's R-DAE learns temporal structure from corrupted windows.  These
recordings have no clean ground truth, so this program uses a robust rolling
median as a *self-supervised pseudo-target*.  Consequently it reports residual
noise, not fictional MAE/RMSE accuracy.  The median result is always saved too:
do not deploy the neural output unless it is at least as stable as that baseline.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import random
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import medfilt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Add, Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam


SEED = 42
WINDOW = 41                 # 2.05 seconds at the 20 Hz sampling rate in file names
MEDIAN_KERNEL = 11          # 0.55 seconds; odd and shorter than WINDOW
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MAX_TRAIN_WINDOWS = 12_000  # avoids creating hundreds of thousands of overlapping samples
MAX_VALIDATION_WINDOWS = 4_000
EPOCHS = 20
BATCH_SIZE = 256
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "adc_data"

np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def read_adc(path):
    values = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            match = re.search(r"ADC\s*:\s*(-?\d+(?:\.\d+)?)", line)
            if match:
                values.append(float(match.group(1)))
            elif line.strip():
                raise ValueError(f"{path.name}:{line_number}: expected ADC:<number>")
    if len(values) < WINDOW:
        raise ValueError(f"{path.name}: needs at least {WINDOW} ADC readings")
    return np.asarray(values, dtype=np.float64)


def windows(values):
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW).copy()[..., None]


def split_record(values):
    first = int(len(values) * TRAIN_FRACTION)
    second = int(len(values) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return values[:first], values[first:second], values[second:]


def pseudo_clean(values):
    """Robust target: removes isolated ADC spikes without assuming a load level."""
    return medfilt(values, kernel_size=MEDIAN_KERNEL)


def build_rdae():
    inputs = Input(shape=(WINDOW, 1), name="corrupted_adc_window")
    x = LSTM(64, return_sequences=True, activation="tanh", name="encoder_lstm")(inputs)
    x = Dropout(0.15, name="encoder_dropout")(x)
    code = LSTM(32, activation="tanh", name="temporal_code")(x)
    x = RepeatVector(WINDOW)(code)
    x = LSTM(32, return_sequences=True, activation="tanh", name="decoder_lstm_1")(x)
    x = Dropout(0.10, name="decoder_dropout")(x)
    x = LSTM(64, return_sequences=True, activation="tanh", name="decoder_lstm_2")(x)
    correction = TimeDistributed(Dense(1), name="noise_correction")(x)
    output = Add(name="denoised_adc")([inputs, correction])
    model = Model(inputs, output, name="adc_rdae")
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def average_windows(prediction, signal_length):
    total, count = np.zeros(signal_length), np.zeros(signal_length)
    for start, values in enumerate(prediction[..., 0]):
        total[start:start + WINDOW] += values
        count[start:start + WINDOW] += 1
    return total / np.maximum(count, 1)


def residual_std(raw, filtered):
    """RMS of removed component; a noise indicator, not accuracy."""
    return float(np.std(raw - filtered))


def label_from_name(path):
    match = re.search(r"hz(\d+)", path.stem)
    return match.group(1) if match else path.stem


def main():
    files = sorted(DATA_DIR.glob("*.txt"), key=lambda p: int(label_from_name(p)))
    if not files:
        raise FileNotFoundError(f"No .txt ADC recordings found in {DATA_DIR}")
    records = [(path, read_adc(path)) for path in files]
    print(f"Loaded {len(records)} ADC recordings ({sum(len(x) for _, x in records):,} readings).")

    # Fit scaler from training time ranges only. Each file is split before windowing.
    training_values = np.concatenate([split_record(values)[0] for _, values in records])
    scaler = RobustScaler(quantile_range=(5, 95)).fit(training_values[:, None])
    train_x, train_y, valid_x, valid_y = [], [], [], []
    held_out = []
    for path, raw in records:
        train, valid, test = split_record(raw)
        for raw_part, x_bucket, y_bucket in ((train, train_x, train_y), (valid, valid_x, valid_y)):
            scaled_raw = scaler.transform(raw_part[:, None]).ravel()
            scaled_target = scaler.transform(pseudo_clean(raw_part)[:, None]).ravel()
            x_bucket.append(windows(scaled_raw))
            y_bucket.append(windows(scaled_target))
        held_out.append((path, test))
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    valid_x, valid_y = np.concatenate(valid_x), np.concatenate(valid_y)
    if len(train_x) > MAX_TRAIN_WINDOWS:
        chosen = np.random.default_rng(SEED).choice(len(train_x), MAX_TRAIN_WINDOWS, replace=False)
        train_x, train_y = train_x[chosen], train_y[chosen]
    if len(valid_x) > MAX_VALIDATION_WINDOWS:
        chosen = np.random.default_rng(SEED + 1).choice(len(valid_x), MAX_VALIDATION_WINDOWS, replace=False)
        valid_x, valid_y = valid_x[chosen], valid_y[chosen]
    print(f"Training windows={len(train_x):,}; validation windows={len(valid_x):,}")

    model = build_rdae()
    history = model.fit(
        train_x, train_y, validation_data=(valid_x, valid_y), shuffle=True,
        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=.5, min_lr=1e-5, verbose=1),
        ],
    )

    rows, examples = [], []
    for path, raw in held_out:
        raw_scaled = scaler.transform(raw[:, None]).ravel()
        prediction = model.predict(windows(raw_scaled), batch_size=BATCH_SIZE, verbose=0)
        rdae = scaler.inverse_transform(average_windows(prediction, len(raw))[:, None]).ravel()
        median = pseudo_clean(raw)
        raw_diff = float(np.std(np.diff(raw)))
        rdae_diff = float(np.std(np.diff(rdae)))
        row = {
            "recording": path.name, "label": label_from_name(path), "samples_test": len(raw),
            "raw_std": float(np.std(raw)), "median_residual_std": residual_std(raw, median),
            "rdae_residual_std": residual_std(raw, rdae), "raw_diff_std": raw_diff,
            "median_diff_std": float(np.std(np.diff(median))), "rdae_diff_std": rdae_diff,
            "rdae_vs_median_mae": float(np.mean(np.abs(rdae - median))),
        }
        rows.append(row)
        examples.append((path, raw, median, rdae))
        print(f"{path.name}: raw diff SD={raw_diff:.1f}, median={row['median_diff_std']:.1f}, R-DAE={rdae_diff:.1f}")

    csv_path = ROOT / "adc_rdae_results.csv"
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    model.save(ROOT / "adc_rdae_denoiser.keras")
    joblib.dump(scaler, ROOT / "adc_rdae_scaler.pkl")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for axis, (path, raw, median, rdae) in zip(axes.flat[:3], examples[:3]):
        sample = min(len(raw), 600)
        t = np.arange(sample) / 20.0
        axis.plot(t, raw[:sample], color="tab:red", alpha=.45, label="Raw ADC")
        axis.plot(t, median[:sample], color="black", lw=1.4, label="Median baseline")
        axis.plot(t, rdae[:sample], color="tab:blue", lw=1.2, label="R-DAE")
        axis.set(title=f"{path.name} (held-out tail)", xlabel="Time (s)", ylabel="ADC")
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows)); width = .38
    axes[1, 1].bar(x - width / 2, [r["median_diff_std"] for r in rows], width, label="Median")
    axes[1, 1].bar(x + width / 2, [r["rdae_diff_std"] for r in rows], width, label="R-DAE")
    axes[1, 1].set(title="Sample-to-sample variation (lower = smoother)", xlabel="File label", ylabel="ADC difference SD")
    axes[1, 1].set_xticks(x, labels, rotation=45); axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    plot_path = ROOT / "adc_rdae_results.png"
    fig.savefig(plot_path, dpi=220)
    print(f"\nSaved metrics: {csv_path}\nSaved plot: {plot_path}")
    print("Important: lower residual/difference variation indicates smoothing only; it is not a clean-signal accuracy score.")


if __name__ == "__main__":
    main()
