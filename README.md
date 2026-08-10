# DICE-KAN-Net

**A hybrid CNN-Transformer-KAN architecture for EEG classification — unifying real-time BCI motor-imagery decoding and clinical neurodegenerative diagnosis in a single model.**

---

## Overview

Most EEG deep learning architectures are built for exactly one job. Convolution-Transformer models like EEG Conformer decode motor imagery for brain-computer interfaces. Others, like DICE-net, diagnose Alzheimer's and Frontotemporal dementia. Neither generalizes to the other's task without a structural redesign — and both rely on flattening feature maps into a standard fully-connected classifier, which discards spatial-temporal structure and inflates parameter count.

**DICE-KAN-Net** is one architecture that does both, without modification:

- **BCI motor-imagery decoding** — intra-subject, session-transfer calibration
- **Clinical diagnosis** — inter-subject, Leave-One-Subject-Out (LOSO) classification of AD/FTD vs. healthy controls

It replaces the standard MLP classifier head with a **Kolmogorov-Arnold Network (KAN)** — learnable B-spline edge functions instead of static weighted sums — fed by **Global Average Pooling** instead of tensor flattening, on top of a multi-scale convolutional + Transformer encoder.

## Key Features

| # | Feature | Why it matters |
|---|---|---|
| 1 | **KAN classification head** | Learnable, bendable per-edge functions instead of one fixed activation — adapts to complex non-linear EEG manifolds |
| 2 | **GAP instead of flattening** | Avoids parameter explosion; preserves spatial-temporal structure into the classifier |
| 3 | **Dual-paradigm architecture** | One model, zero structural changes, spans both BCI decoding and clinical diagnosis |
| 4 | **Dynamic per-fold class weighting** | Recomputed every LOSO fold — corrects for majority-class bias common in clinical EEG datasets |
| 5 | **Causal exponential moving standardization** | Real-time-safe normalization — no look-ahead, suitable for live BCI streaming |

## Architecture

```
Raw EEG (C × T)
      │
      ▼
Causal EMA Standardization        ← real-time-safe running z-score
      │
      ▼
Multi-Scale Temporal Convolution  ← parallel kernels, fast + slow dynamics
      │
      ▼
Depthwise + Pointwise Spatial Conv ← cross-electrode relationships
      │
      ▼
Squeeze-and-Excitation Attention  ← channel-wise re-weighting
      │
      ▼
Transformer Encoder                ← global temporal context, multi-head attention
      │
      ▼
Global Average Pooling             ← structure-preserving compression
      │
      ▼
KAN Classifier Head                ← learnable B-spline edge functions
      │
      ▼
Class Probabilities
```

## Results

### BCI Motor-Imagery Decoding — BCI Competition IV-2a (9 subjects, 4-class)

Universal model pre-trained on all subjects, then fine-tuned and evaluated per subject on the held-out `E` session:

| Subject | Best Epoch | Test Accuracy (%) | Kappa | N Test Trials |
|---|---|---|---|---|
| A01 | 73 | 89.58 | 0.8611 | 288 |
| A02 | 60 | 70.49 | 0.6065 | 288 |
| A03 | 27 | 95.83 | 0.9444 | 288 |
| A04 | 62 | 71.53 | 0.6204 | 288 |
| A05 | 36 | 68.40 | 0.5787 | 288 |
| A06 | 82 | 62.50 | 0.5000 | 288 |
| A07 | 120 | 92.71 | 0.9028 | 288 |
| A08 | 29 | 88.19 | 0.8426 | 288 |
| A09 | 112 | 85.76 | 0.8102 | 288 |
| **AVERAGE** | — | **80.55** | **0.7407** | 2592 |

### Clinical Diagnosis — EEG dataset (88 subjects, strict LOSO)

| Metric | Result |
|---|---|
| AD vs. CN | AUC-ROC **0.9548**, Recall **87.81%** |
| FTD vs. CN | AUC-ROC **0.9177**, Recall **85.11%** |

## Repository Structure

```
dice-kan-net/
├── preprocess.py                  # BCI IV-2a preprocessing: GDF → per-subject tensors
├── main.py                        # universal pre-training + per-subject fine-tuning + evaluation
└── dicenet_bciciv2a_results/      # per-subject metrics, fold plots, and results CSV
    ├── fold_plots/                 # loss/accuracy curves per subject
    ├── universal_model.pth         # pre-trained universal model checkpoint
    └── dicenet_bciciv2a_results.csv
```

## Installation

```bash
git clone https://github.com/<your-username>/dice-kan-net.git
cd dice-kan-net
pip install torch mne scipy numpy pandas scikit-learn tqdm matplotlib efficient-kan
```

## Usage

**1. Preprocess the raw BCI Competition IV-2a data**

Download the `.gdf` files and the official `true_labels` `.mat` files (`A0xE.mat`) from the BCI Competition IV-2a distribution, and place them together in a `BCICIV_2a_gdf/` folder. Then run:

```bash
python preprocess.py
```

This band-pass filters (4–38 Hz), drops EOG channels, extracts the 4.0 s post-cue window (1000 timepoints @ 250 Hz), applies causal exponential moving standardization, and writes per-subject `{subject}_train.pt` / `{subject}_test.pt` tensors to `bciciv2a_processed/`.

**2. Train the universal model and fine-tune per subject**

```bash
python main.py
```

This first pre-trains a **universal model** on all subjects' pooled training data (200 epochs), then fine-tunes a copy of it per subject (200 epochs each, with data augmentation — time-shift, noise, temporal/channel masking) and evaluates on that subject's held-out `E` (eval) session. Per-subject fold plots and a summary CSV are written to `dicenet_bciciv2a_results/`.

## Datasets

- **BCI Competition IV-2a** — [bbci.de/competition/iv](https://www.bbci.de/competition/iv/) (4-class motor imagery, 9 subjects)
- **Clinical EEG dataset** — OpenNeuro ds004504 (AD/FTD/CN EEG, 88 subjects)

## Tech Stack

`PyTorch` · `MNE` · `efficient-kan` · `scikit-learn` · `pandas` · `NumPy`

## License

MIT
