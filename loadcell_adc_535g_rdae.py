"""Supervised R-DAE for the 535 g ADC calibration recordings.

All supplied filenames begin with ``535g`` and therefore represent the same
applied load.  The ``hz0`` recording is used as the low-interference reference.
This model is valid for denoising the 535 g experiment only; it is not a general
load-to-grams calibration model.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "adc_data"
SEED = 42
WINDOW = 41
TRAIN_FRACTION = .70
VALIDATION_FRACTION = .15
MAX_TRAIN_WINDOWS = 12_000
MAX_VALIDATION_WINDOWS = 4_000
EPOCHS = 20
BATCH_SIZE = 256

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def frequency(path):
    match = re.search(r"hz(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not read frequency from {path.name}")
    return int(match.group(1))


def read_adc(path):
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*ADC\s*:\s*(-?\d+(?:\.\d+)?)\s*", line)
        if not match:
            raise ValueError(f"{path.name}:{line_number}: expected ADC:<number>")
        output.append(float(match.group(1)))
    return np.asarray(output)


def split(values):
    first = int(len(values) * TRAIN_FRACTION)
    second = int(len(values) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return values[:first], values[first:second], values[second:]


def windows(values):
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW).copy()[..., None]


def build_rdae():
    """Paper-style LSTM encoder/decoder; output is the estimated clean ADC trace."""
    source = Input(shape=(WINDOW, 1), name="noisy_adc_window")
    x = LSTM(64, return_sequences=True, activation="tanh", name="encoder_lstm_1")(source)
    x = Dropout(.15)(x)
    code = LSTM(32, activation="tanh", name="temporal_code")(x)
    x = RepeatVector(WINDOW)(code)
    x = LSTM(32, return_sequences=True, activation="tanh", name="decoder_lstm_1")(x)
    x = Dropout(.10)(x)
    x = LSTM(64, return_sequences=True, activation="tanh", name="decoder_lstm_2")(x)
    target = TimeDistributed(Dense(1), name="clean_adc")(x)
    model = Model(source, target, name="rdae_535g")
    model.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def merge_windows(predictions, length):
    value, count = np.zeros(length), np.zeros(length)
    for start, sequence in enumerate(predictions[..., 0]):
        value[start:start + WINDOW] += sequence
        count[start:start + WINDOW] += 1
    return value / np.maximum(count, 1)


def metrics(reference, estimate):
    mse = mean_squared_error(reference, estimate)
    return mean_absolute_error(reference, estimate), np.sqrt(mse), mse


def main():
    files = sorted(DATA_DIR.glob("*.txt"), key=frequency)
    records = [(path, read_adc(path)) for path in files]
    reference_records = [values for path, values in records if frequency(path) == 0]
    if len(reference_records) != 1:
        raise ValueError("Exactly one hz0 reference recording is required.")
    # The robust centre of the low-interference recording is the known stable
    # ADC level for this experiment's 535 g load.
    reference_adc = float(np.median(reference_records[0]))
    noisy_records = [(path, values) for path, values in records if frequency(path) != 0]
    print(f"535 g reference ADC (median of hz0): {reference_adc:.2f}")
    print(f"Training on {len(noisy_records)} nonzero-frequency recordings.")

    train_values = np.concatenate([split(values)[0] for _, values in noisy_records])
    scaler = RobustScaler(quantile_range=(5, 95)).fit(train_values[:, None])
    reference_scaled = float(scaler.transform([[reference_adc]])[0, 0])
    train_x, train_y, valid_x, valid_y = [], [], [], []
    test_records = []
    for path, values in noisy_records:
        train, valid, test = split(values)
        for part, xs, ys in ((train, train_x, train_y), (valid, valid_x, valid_y)):
            scaled = scaler.transform(part[:, None]).ravel()
            xs.append(windows(scaled))
            ys.append(np.full((len(scaled) - WINDOW + 1, WINDOW, 1), reference_scaled, dtype=np.float32))
        test_records.append((path, test))
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    valid_x, valid_y = np.concatenate(valid_x), np.concatenate(valid_y)
    rng = np.random.default_rng(SEED)
    if len(train_x) > MAX_TRAIN_WINDOWS:
        index = rng.choice(len(train_x), MAX_TRAIN_WINDOWS, replace=False)
        train_x, train_y = train_x[index], train_y[index]
    if len(valid_x) > MAX_VALIDATION_WINDOWS:
        index = rng.choice(len(valid_x), MAX_VALIDATION_WINDOWS, replace=False)
        valid_x, valid_y = valid_x[index], valid_y[index]
    print(f"Windows: train={len(train_x):,}, validation={len(valid_x):,}")

    model = build_rdae()
    history = model.fit(
        train_x, train_y, validation_data=(valid_x, valid_y), batch_size=BATCH_SIZE,
        epochs=EPOCHS, shuffle=True, verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=.5, min_lr=1e-5, verbose=1),
        ],
    )

    rows, plot_records = [], []
    for path, raw in test_records:
        scaled = scaler.transform(raw[:, None]).ravel()
        predicted = model.predict(windows(scaled), batch_size=BATCH_SIZE, verbose=0)
        denoised = scaler.inverse_transform(merge_windows(predicted, len(raw))[:, None]).ravel()
        reference = np.full(len(raw), reference_adc)
        raw_mae, raw_rmse, raw_mse = metrics(reference, raw)
        dae_mae, dae_rmse, dae_mse = metrics(reference, denoised)
        rows.append({
            "recording": path.name, "frequency_hz": frequency(path), "test_samples": len(raw),
            "raw_mae_to_hz0_adc": raw_mae, "raw_rmse_to_hz0_adc": raw_rmse, "raw_mse_to_hz0_adc": raw_mse,
            "rdae_mae_to_hz0_adc": dae_mae, "rdae_rmse_to_hz0_adc": dae_rmse, "rdae_mse_to_hz0_adc": dae_mse,
            "mse_improvement_percent": 100 * (1 - dae_mse / raw_mse),
        })
        plot_records.append((path, raw, denoised))
        print(f"{frequency(path):>3} Hz: RMSE raw={raw_rmse:.1f}, R-DAE={dae_rmse:.1f}, improvement={rows[-1]['mse_improvement_percent']:.2f}%")

    with (ROOT / "adc_535g_rdae_results.csv").open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    model.save(ROOT / "adc_535g_rdae.keras")
    joblib.dump(scaler, ROOT / "adc_535g_rdae_scaler.pkl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, (path, raw, denoised) in zip(axes.flat[:3], plot_records[:3]):
        view = min(600, len(raw)); time = np.arange(view) * .020
        axis.plot(time, raw[:view], label="Raw ADC", alpha=.45, color="tab:red")
        axis.plot(time, denoised[:view], label="R-DAE estimate", color="tab:blue")
        axis.axhline(reference_adc, label="hz0 reference", color="black", ls="--")
        axis.set(title=f"535 g, {frequency(path)} Hz", xlabel="Time (s)", ylabel="ADC")
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    hz = [row["frequency_hz"] for row in rows]
    axes[1, 1].plot(hz, [row["raw_rmse_to_hz0_adc"] for row in rows], "o-", label="Raw")
    axes[1, 1].plot(hz, [row["rdae_rmse_to_hz0_adc"] for row in rows], "o-", label="R-DAE")
    axes[1, 1].set(title="Error from stable 535 g hz0 reference", xlabel="Interference frequency (Hz)", ylabel="RMSE (ADC)")
    axes[1, 1].grid(alpha=.25); axes[1, 1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "adc_535g_rdae_results.png", dpi=220)
    print("Saved adc_535g_rdae_results.csv, adc_535g_rdae_results.png, and adc_535g_rdae.keras")


if __name__ == "__main__":
    main()
