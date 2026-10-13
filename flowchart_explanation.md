# Paper Methodology Flowcharts Explanation

This document explains the Mermaid flowcharts created for the bearing RUL prediction paper methodology (referenced as "1-s2.0-S0*.pdf").

## Overview
The paper describes a methodology for bearing Remaining Useful Life (RUL) prediction using:
1. **Recurrent Denoising Autoencoder (R-DAE)** for feature extraction and noise reduction
2. **Improved Fuzzy C-Means (FCM) clustering** for health state classification
3. **Trend analysis** for RUL prediction

## Files Created

### 1. `paper_methodology_flowchart.mmd`
A high-level flowchart showing the main processing pipeline:
- Raw signals → Preprocessing → R-DAE Feature Extraction → Improved FCM → RUL Prediction

### 2. `detailed_paper_methodology.mmd`
A detailed flowchart showing:
- R-DAE architecture specifics (encoder, bottleneck, decoder)
- Improved FCM algorithm iterative process
- Feedback loops and key innovation points

## Key Components Explained

### R-DAE (Recurrent Denoising Autoencoder)
- **Input**: Noisy signal windows
- **Encoder**: LSTM layers that extract temporal features
- **Bottleneck**: Compressed feature representation (used for health monitoring)
- **Decoder**: LSTM layers that reconstruct the signal
- **Innovation**: Uses reconstruction error as feedback for improving denoising

### Improved FCM Clustering
- Enhances standard FCM with better initialization or constraints
- Outputs membership degrees to different health states
- Enables soft classification of bearing degradation stages

### RUL Prediction
- Uses extracted features or health indicators
- Applies trend extrapolation (polynomial fitting, etc.)
- Predicts remaining useful life

## How to Use These Flowcharts
1. Copy the `.mmd` content into any Mermaid-compatible markdown viewer or tool
2. Or paste into online Mermaid live editor: https://mermaid.live/edit
3. The flowcharts illustrate the methodology that was adapted for LoadCell noise reduction in our implementation

## Relationship to LoadCell Implementation
Our LoadCell noise reduction implementation focused on the R-DAE component:
- Used similar LSTM-based encoder-decoder architecture
- Applied sliding window approach for sequence processing
- Trained as denoising autoencoder (noisy input → clean target)
- The paper's R-DAE was used for feature extraction, while we used it for direct noise reduction

These flowcharts provide the theoretical foundation that inspired our LoadCell noise reduction approach.