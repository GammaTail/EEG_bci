"""
==========================================================
BCI Competition IV-2a — Preprocessing Pipeline
==========================================================
Goal: produce per-subject train (T) / test (E) tensors that match
the exact protocol used by EEG Conformer and most BCI-IV-2a papers,
so results in Table II are a fair comparison point.

Protocol used here (standard, not invented):
  - 22 EEG channels only (3 EOG channels dropped)
  - Band-pass filter 4-38 Hz (motor imagery band, matches Conformer)
  - Trial window: 0.5s to 4.5s after the cue (768 -> cue is at t=0
    for our purposes; we anchor on cue-onset codes 769-772 directly)
    -> 4.0 s window @ 250 Hz = 1000 timepoints
  - Exponential moving standardization per channel (causal running
    z-score), the standard trick from Schirrmeister et al. / Braindecode,
    consistently reported to help over static z-scoring on this dataset.
  - Train = A0xT.gdf (labels are embedded directly as events 769-772)
  - Test  = A0xE.gdf (labels are NOT embedded -> must load the official
    true-label .mat files released by the competition, see note below)

REQUIRED EXTRA FILES FOR THE E (EVAL) SESSIONS
------------------------------------------------
The raw A0xE.gdf files only contain a generic "unknown" marker (783) at
every cue, by design of the original competition (labels were withheld
for the competition itself). To get ground truth for E sessions you need
the official label files, one per subject:

    A01E.mat, A02E.mat, ... A09E.mat

These are distributed by the BCI Competition IV organizers as
"true labels" alongside the GDF files (look for a folder/zip named
something like "true_labels" on the same page you got the .gdf files
from, e.g. the BNCI Horizon 2020 / bbci.de mirrors). Each .mat file
contains a variable called `classlabel`, a (288,) vector of ints 1-4
in trial order, matching the trial order in the corresponding GDF file.

Place them in the SAME folder as the GDF files, named exactly like the
GDF file but with .mat extension (A01E.mat next to A01E.gdf, etc).
If a label file is missing for a subject, that subject's E session
will be skipped with a warning (T-session-only fallback is NOT used,
since that would leak train data into test).

OUTPUT
------
For each subject (e.g. "A01"), this script writes:
    OUT_DIR/A01_train.pt   -> dict with 'X' (N,22,1000) float32, 'y' (N,) int64
    OUT_DIR/A01_test.pt    -> same structure, from the E session

Run:
    python preprocess_bciciv2a.py
"""

import os
import glob
import warnings
import numpy as np
import mne
from scipy.io import loadmat
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ==========================================
# Configuration — EDIT THESE PATHS
# ==========================================
GDF_DIR = "BCICIV_2a_gdf"   # folder with A0xT.gdf / A0xE.gdf (+ A0xE.mat labels)
OUT_DIR = "bciciv2a_processed"

SFREQ_TARGET = 250.0          # native sampling rate of the dataset
LOW_FREQ, HIGH_FREQ = 4.0, 38.0   # band-pass range (motor-imagery band)
TMIN, TMAX = 0.5, 4.5         # seconds relative to cue onset -> 4.0s window
EOG_CHANNEL_KEYWORDS = ["eog"]    # used to drop EOG channels by name match

# Event codes (as strings, since MNE annotation descriptions from GDF are strings)
CUE_EVENT_IDS = {"769": 0, "770": 1, "771": 2, "772": 3}
# class 0=left hand, 1=right hand, 2=foot, 3=tongue

SUBJECTS = [f"A0{i}" for i in range(1, 10)]  # A01..A09


# ==========================================
# Exponential moving standardization
# ==========================================
def exponential_moving_standardize(data, factor_new=1e-3, init_block_size=1000, eps=1e-4):
    """
    Causal running z-score per channel, as used in Braindecode /
    Schirrmeister et al. (2017). Operates on a single trial's worth of
    data, shape (n_channels, n_times).

    data: np.ndarray (n_channels, n_times)
    """
    data = data.T  # -> (n_times, n_channels) for easier running stats
    df = np.zeros_like(data)
    running_mean = data[0]
    running_var = np.ones_like(running_mean)

    standardized = np.zeros_like(data)
    for t in range(data.shape[0]):
        if t == 0:
            running_mean = data[t]
            running_var = np.ones_like(running_mean)
        else:
            running_mean = factor_new * data[t] + (1 - factor_new) * running_mean
            diff = data[t] - running_mean
            running_var = factor_new * (diff ** 2) + (1 - factor_new) * running_var

        if t < init_block_size:
            # During the warmup block, use block statistics instead of the
            # (still noisy) running estimate, as in the reference implementation.
            block = data[: init_block_size if init_block_size <= data.shape[0] else data.shape[0]]
            block_mean = block.mean(axis=0)
            block_std = block.std(axis=0)
            standardized[t] = (data[t] - block_mean) / np.maximum(block_std, eps)
        else:
            standardized[t] = (data[t] - running_mean) / np.maximum(np.sqrt(running_var), eps)

    return standardized.T  # back to (n_channels, n_times)


# ==========================================
# GDF loading helpers
# ==========================================
def load_raw_gdf(filepath):
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose="ERROR")

    # Drop EOG channels, keep only the 22 EEG channels
    eog_chs = [ch for ch in raw.ch_names if any(k in ch.lower() for k in EOG_CHANNEL_KEYWORDS)]
    if eog_chs:
        raw.drop_channels(eog_chs)

    # Band-pass filter
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose="ERROR")

    return raw


def epoch_from_train_session(raw):
    """
    T-session files have the real cue codes (769-772) embedded directly.
    Returns X (n_trials, n_channels, n_times), y (n_trials,)
    """
    events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")

    # Build the subset of event_id_map that corresponds to our 4 cue codes
    wanted_event_id = {
        desc: code for desc, code in event_id_map.items() if desc in CUE_EVENT_IDS
    }
    if len(wanted_event_id) == 0:
        raise RuntimeError(
            "No cue events (769-772) found in this T-session file. "
            "Check that this is really a training-session GDF."
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data()  # (n_trials, n_channels, n_times)

    # Map MNE's internal event codes back to our 0-3 class labels via description
    code_to_class = {
        code: CUE_EVENT_IDS[desc] for desc, code in wanted_event_id.items()
    }
    y = np.array([code_to_class[e] for e in epochs.events[:, 2]])

    return X, y


def epoch_from_eval_session(raw, mat_label_path):
    """
    E-session files do NOT carry true labels in the GDF. We anchor epochs
    on the generic 'start of trial' marker (768) which IS present in both
    T and E sessions, then attach true labels from the official .mat file
    in trial order.
    """
    if not os.path.exists(mat_label_path):
        raise FileNotFoundError(
            f"Missing true-label file: {mat_label_path}\n"
            f"Download the official 'true labels' .mat files for the E "
            f"sessions from the BCI Competition IV-2a distribution and "
            f"place them next to the .gdf files (see header docstring)."
        )

    events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")

    # '768' = start of trial, present in both T and E sessions
    start_trial_desc = next((d for d in event_id_map if d == "768"), None)
    if start_trial_desc is None:
        raise RuntimeError("Could not find trial-start marker '768' in this E-session file.")

    trial_start_event_id = {start_trial_desc: event_id_map[start_trial_desc]}

    # The cue happens at the same relative time after 768 as in the T files
    # (BCI IV-2a protocol: 768 -> 769/770/771/772 follow ~2s later, fixed timing),
    # but to keep this robust we instead epoch directly off 768 and shift our
    # tmin/tmax to account for the ~2s gap between trial-start and cue-onset.
    CUE_OFFSET_FROM_TRIAL_START = 2.0  # seconds; fixed by the BCI IV-2a paradigm

    epochs = mne.Epochs(
        raw,
        events,
        event_id=trial_start_event_id,
        tmin=TMIN + CUE_OFFSET_FROM_TRIAL_START,
        tmax=TMAX + CUE_OFFSET_FROM_TRIAL_START,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data()  # (n_trials, n_channels, n_times)

    mat = loadmat(mat_label_path)
    if "classlabel" not in mat:
        raise KeyError(
            f"'classlabel' variable not found in {mat_label_path}. "
            f"Keys present: {list(mat.keys())}"
        )
    y_raw = mat["classlabel"].squeeze().astype(int)  # values in {1,2,3,4}

    if len(y_raw) != X.shape[0]:
        raise ValueError(
            f"Trial count mismatch for {mat_label_path}: "
            f"{X.shape[0]} epochs from GDF vs {len(y_raw)} labels in .mat file."
        )

    y = y_raw - 1  # -> {0,1,2,3} to match our convention

    return X, y


def apply_ems_to_all_trials(X):
    """X: (n_trials, n_channels, n_times) -> same shape, standardized per trial."""
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = exponential_moving_standardize(X[i])
    return out


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for sub in SUBJECTS:
        print(f"\n=== Subject {sub} ===")

        train_path = os.path.join(GDF_DIR, f"{sub}T.gdf")
        eval_path = os.path.join(GDF_DIR, f"{sub}E.gdf")
        eval_label_path = os.path.join(GDF_DIR, f"{sub}E.mat")

        if not os.path.exists(train_path):
            print(f"  ⚠ Skipping {sub}: missing {train_path}")
            continue

        # ---- Train session ----
        print(f"  Loading train session: {train_path}")
        raw_train = load_raw_gdf(train_path)
        X_train, y_train = epoch_from_train_session(raw_train)
        print(f"  Train epochs: {X_train.shape}, labels: {np.bincount(y_train)}")

        X_train = apply_ems_to_all_trials(X_train)

        torch.save(
            {
                "X": torch.tensor(X_train, dtype=torch.float32),
                "y": torch.tensor(y_train, dtype=torch.long),
            },
            os.path.join(OUT_DIR, f"{sub}_train.pt"),
        )

        # ---- Eval session ----
        if not os.path.exists(eval_path):
            print(f"  ⚠ No eval session found for {sub}, skipping test set.")
            continue

        if not os.path.exists(eval_label_path):
            print(
                f"  ⚠ No true-label .mat file found for {sub} "
                f"(expected {eval_label_path}). Skipping test set for this subject.\n"
                f"    Download the official true labels and place them alongside the GDFs."
            )
            continue

        print(f"  Loading eval session: {eval_path}")
        raw_eval = load_raw_gdf(eval_path)
        X_eval, y_eval = epoch_from_eval_session(raw_eval, eval_label_path)
        print(f"  Eval epochs: {X_eval.shape}, labels: {np.bincount(y_eval)}")

        X_eval = apply_ems_to_all_trials(X_eval)

        torch.save(
            {
                "X": torch.tensor(X_eval, dtype=torch.float32),
                "y": torch.tensor(y_eval, dtype=torch.long),
            },
            os.path.join(OUT_DIR, f"{sub}_test.pt"),
        )

    print("\n✅ Preprocessing complete. Tensors saved to:", OUT_DIR)


if __name__ == "__main__":
    main()