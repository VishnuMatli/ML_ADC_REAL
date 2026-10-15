## 1. Purpose of the pipeline

This pipeline is designed to convert a noisy load-cell signal into a structured health-state representation using a sequential denoising model followed by fuzzy clustering. The first four completed stages are:

1. Synthetic signal generation
2. Data split and normalization
3. Temporal R-DAE training and bottleneck feature extraction
4. Fuzzy C-Means (FCM) clustering for health-state grouping

The goal is not yet final residual life prediction or full bearing-fault diagnosis. The current stage is to stabilize the denoising foundation and health-state representation before advancing to the next predictive layer.

---

## 2. Core variables used in the implementation

### 2.1 Global configuration variables

- WINDOW_SIZE = 32

  - Number of time steps used in each temporal window for the recurrent model.
  - This defines the sequential context window fed into the LSTM network.
- EPOCHS = 25

  - Number of full training iterations used to optimize the denoising network.
- BATCH_SIZE = 32

  - Number of window samples processed per update step in gradient descent.
- n_samples = 3000

  - Total synthetic signal length generated in the main experiment.

### 2.2 Signal generation variables

- t

  - Time axis generated with np.linspace(0, 100, n_samples).
  - Used to create the temporal structure of the synthetic load-cell signal.
- clean

  - The target denoised signal representing the underlying physical load-cell pattern.
  - It contains deterministic drift, periodic oscillations, and gradual degradation effects.
- noisy

  - The corrupted observed signal formed by clean + noise.
  - This is the signal the model learns to denoise.
- noise

  - The injected corruption term, consisting of:
    - Gaussian white noise
    - low-frequency drift noise
    - spike interference
    - periodic interference

### 2.3 Data-splitting variables

- train_noisy, train_clean

  - Training portion of noisy and clean signals.
- val_noisy, val_clean

  - Validation portion used for model selection and early learning monitoring.
- test_noisy, test_clean

  - Final unseen test set used to estimate the generalization performance.
- train_frac = 0.70

  - 70% of the signal used for training.
- val_frac = 0.15

  - 15% reserved for validation.
- n_train, n_val, n_test

  - Exact counts determined from the total length of the signal.

### 2.4 Scaling variables

- signal_scaler

  - An instance of MinMaxScaler with range (0, 1).
  - Used to normalize both noisy and clean signals into a common bounded domain for stable LSTM training.
- train_noisy_norm, train_clean_norm
- val_noisy_norm, val_clean_norm
- test_noisy_norm, test_clean_norm

  - Normalized versions of the split datasets in the same scale.

### 2.5 Sequence variables

- X_train, y_train

  - Sliding-window training inputs and targets.
- X_val, y_val

  - Validation-window inputs and targets.
- X_test, y_test

  - Test-window inputs and targets.

These are all 3D arrays shaped as:

- (number_of_windows, WINDOW_SIZE, 1)

This shape is required because each sample is a temporal window and each time step is a single scalar measurement.

### 2.6 Model variables

- inputs

  - TensorFlow Input object with shape (WINDOW_SIZE, 1)
- encoder_lstm_1

  - First LSTM encoder layer with 40 units, activation tanh
  - Reads noisy temporal context and compresses it.
- bottleneck

  - Second LSTM layer with 24 units
  - This is the compressed latent representation that captures the important temporal structure.
- repeat_bottleneck

  - RepeatVector(WINDOW_SIZE)
  - Expands the bottleneck back into a sequence so the decoder has time-aligned context.
- decoder_lstm_1

  - LSTM decoder with 24 units
- decoder_lstm_2

  - LSTM decoder with 40 units
- reconstruction_head

  - TimeDistributed(Dense(1, activation='sigmoid'))
  - Produces the denoised estimate for each time step in the sequence.
- model

  - Full sequential denoising autoencoder with encoder-decoder structure.

### 2.7 Feature extraction variables

- latent_train, latent_test

  - Bottleneck latent embeddings extracted from the encoder.
- pred_train, pred_test

  - Reconstructed outputs from the model on noisy windows.
- residual

  - Absolute difference between prediction and target clean signal within the same window.
- error_mean, error_std, error_max

  - Statistical descriptors of reconstruction error per window.
- feature_matrix

  - Concatenated representation built as:
    - latent vector + error_mean + error_std + error_max

This is the representation used for health-state clustering.

### 2.8 Fuzzy-clustering variables

- centers

  - Cluster centers computed by FCM.
- membership

  - Fuzzy membership matrix showing how strongly each sample belongs to each cluster.
- labels

  - Hard cluster assignment from argmax(membership).
- health_indicator

  - A scalar confidence-like measure derived from the strongest cluster membership:
    - health_indicator = 1.0 - max_membership

This is useful as a degradation or uncertainty indicator.

---

## 3. Process flow in technical terms

### Stage 1: Synthetic signal generation

The function generate_loadcell_signal creates a realistic time-series signal with multiple components:

- slow drift: 0.03 * t^2
- periodic oscillation: 10 sin(0.12t)
- higher-frequency oscillation: 6 sin(0.55t)
- localized degradation component: Gaussian bump near t = 45
- abrupt shifts using threshold logic on time

Then a noise term is added:

- Gaussian noise with standard deviation 0.18
- amplitude-modulated periodic noise
- 60 Hz-like interference
- random spike disturbances

Finally the signal is constrained to positive values with np.maximum(..., 0.1), reflecting the physical requirement that load-cell readings cannot be negative.

This stage creates a controlled and repeatable dataset where the clean ground truth is known, so denoising performance can be measured objectively.

### Stage 2: Data splitting and normalization

The function split_signal divides the data into train, validation, and test partitions using fixed proportions:

- 70% training
- 15% validation
- 15% test

The split is applied to the full signal before window extraction, which ensures the model is evaluated on unseen temporal regions.

After splitting, MinMaxScaler transforms all values to the [0, 1] range. This matters because the LSTM architecture is sensitive to scale and unstable gradients are more likely when the signal magnitude is large and unbounded.

### Stage 3: Temporal sequence construction

The function create_sequences transforms 1D time-series data into overlapping temporal windows:

- For each start index i, a window is formed as data[i : i + WINDOW_SIZE]
- This creates multiple short sequences for temporal learning
- The shape becomes (N_windows, WINDOW_SIZE, 1)

This is important because the denoising model is learning temporal dependencies rather than pointwise filtering.

### Stage 4: Recurrent denoising autoencoder (R-DAE)

The function build_rdae creates an encoder-decoder architecture:

- Input: noisy window of shape (32, 1)
- Encoder LSTM(40): captures temporal structure and compresses sequence context
- Bottleneck LSTM(24): produces a latent representation of the signal state
- RepeatVector(32): broadcasts the compressed state over time
- Decoder LSTM(24) -> LSTM(40): reconstructs the temporal sequence
- TimeDistributed(Dense(1, activation='sigmoid')): outputs one value per time step

The model is trained using mean squared error (MSE) loss and MAE metric. The target is the clean window, and the input is the noisy one. This makes the model a denoising autoencoder in the sequence domain.

### Stage 5: Bottleneck feature extraction

The function extract_bottleneck_features does two things simultaneously:

1. Builds an encoder-only model from the trained network
2. Extracts bottleneck features from the noisy windows

For each window, it also computes:

- residual = abs(predicted_clean - actual_clean)
- average reconstruction error
- standard deviation of reconstruction error
- maximum reconstruction error

The final feature vector is:

- latent feature vector
- error_mean
- error_std
- error_max

These features capture not only the compressed signal behavior but also the uncertainty or reconstruction difficulty of each temporal segment.

### Stage 6: Fuzzy clustering for health-state grouping

The function improved_fcm implements fuzzy c-means clustering on extracted feature vectors. Compared with hard K-means, FCM assigns a probability-like degree of membership to each cluster rather than a single label.

Technically, the algorithm:

- initializes cluster centers from randomly selected samples
- computes Euclidean distances between each sample and each center
- computes fuzzy memberships using the standard FCM update rule
- updates cluster centroids with weighted averaging across all samples
- repeats until convergence or max iteration limit is reached

The resulting cluster labels represent different health states in the latent space, and health_indicator highlights weakly assigned or uncertain samples.

---

## 4. What each method does exactly

### 4.1 generate_loadcell_signal

Purpose:

- create a synthetic load-cell signal with realistic drift, periodicity, and noisy disturbances

Technical effect:

- gives controlled ground truth for denoising experiments
- provides known reference signal for evaluation

### 4.2 create_sequences

Purpose:

- convert the 1D signal into overlapping temporal windows

Technical effect:

- prepares data for sequence-based LSTM learning
- preserves local temporal continuity that pointwise models would lose

### 4.3 split_signal

Purpose:

- separate the data into train/validation/test sets without leakage

Technical effect:

- prevents the model from seeing the test distribution during training
- ensures fair evaluation of denoising and clustering performance

### 4.4 build_rdae

Purpose:

- build the recurrent denoising autoencoder model

Technical effect:

- learns to map noisy temporal windows to cleaner versions
- compresses the signal into a latent representation and reconstructs it

### 4.5 extract_bottleneck_features

Purpose:

- extract latent signal descriptors and reconstruction error statistics

Technical effect:

- converts raw sequence predictions into a compact feature space for clustering
- adds error-based context showing which temporal windows are difficult to reconstruct

### 4.6 improved_fcm

Purpose:

- group latent feature vectors into health states

Technical effect:

- generates soft cluster assignments instead of rigid labels
- produces a health indicator reflecting how confidently each sample belongs to its dominant cluster

### 4.7 reconstruct_signal

Purpose:

- rebuild a full-length signal from overlapping window predictions

Technical effect:

- converts the windowed output of the model back into a 1D time series for comparison with the original signal

### 4.8 main

Purpose:

- coordinate the full pipeline end-to-end

Technical effect:

- runs generation, split, normalization, training, extraction, clustering, and plotting
- saves the result figure and prints the training summary

---

## 5. Mathematical interpretation of the pipeline

The flow can be described as:

1. Signal model:

   - noisy(t) = clean(t) + noise(t)
2. Sequence representation:

   - X = {x_1, x_2, ..., x_N}, where each x_i is a window of length 32
3. Denoising objective:

   - minimize MSE between model output and clean sequence:
     - L = (1 / N) Σ ||f(x_i) - y_i||^2
4. Latent representation:

   - z_i = Encoder(x_i)
5. Reconstruction error features:

   - r_i = |f(x_i) - y_i|
   - feature_i = [z_i, mean(r_i), std(r_i), max(r_i)]
6. FCM clustering objective:

   - minimize the weighted sum of squared distances between each feature vector and cluster centers under fuzzy membership constraints

This means the algorithm is not just denoising raw values; it is converting noisy temporal segments into feature descriptors that summarize both signal behavior and reconstruction uncertainty.

---

## 6. Why this pipeline is meaningful

This method is useful because it combines two complementary ideas:

- Temporal denoising model learns local signal structure
- Fuzzy clustering interprets the latent representation as health-state behavior

Instead of clustering raw noisy data, the algorithm clusters the latent and error-aware representation. This makes the health-state grouping more robust and physically meaningful, because it is built from the model’s own understanding of the signal quality, not just from raw numerical amplitude.

---

## 7. Current status and interpretation of results

The pipeline has successfully completed the first four planned steps:

- synthetic signal generation
- normalization and splitting
- temporal R-DAE training
- FCM clustering

The model shows a clear training loss decrease over the 25 epochs, which confirms learning. The latent features were successfully extracted and grouped into three clusters. This confirms that the pipeline is structurally working and can be extended to the next phases.
