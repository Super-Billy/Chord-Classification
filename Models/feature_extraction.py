#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log‑mel feature extraction for frame‑level chord classification (POP909 dataset).

Main changes compared with the previous version:
1) Features are first written to **features_audio_rnn.h5** (compressed, disk‑friendly).
2) When extraction completes, the script converts that file into an *uncompressed*
   copy named **features_audio_rnn_fast.h5** and deletes the original file.
   ‑ Faster random access at the cost of larger file size.

Run:
    python feature_extraction_fast.py --meta POP909_metadata.csv
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ────────── Global paths (relative to the Models/ folder) ──────────
BASE_DIR      = os.path.dirname(__file__)          # …/project/Models
ROOT_DIR      = os.path.abspath(os.path.join(BASE_DIR, ".."))


RAW_AUDIO_DIR: str = os.path.join(ROOT_DIR)
DEFAULT_META  = os.path.join(ROOT_DIR, "POP909_metadata.csv")

ORIG_H5_PATH: str = os.path.join(BASE_DIR,"features_audio_rnn.h5")
FAST_H5_PATH: str = os.path.join(BASE_DIR,"features_audio_rnn_fast.h5")
RNN_META_PATH: str = os.path.join(BASE_DIR,"POP909_metadata_rnn.csv")

# ────────── Audio / feature parameters ──────────
SAMPLE_RATE: int = 22_050
N_MELS: int = 64
N_FFT: int = 2_048
HOP: int = 512
MAX_FRAMES: int = 128  # every segment is padded / truncated to exactly this many frames

# ──────────────────────── Helper workers ───────────────────────

def _song_worker(args: Tuple[str, List[Tuple[int, float, float]]]):
    """Worker that extracts log-mel segments from a single song."""
    wav_path, seg_rows = args
    full_path = wav_path if os.path.isabs(wav_path) else os.path.join(RAW_AUDIO_DIR, wav_path)

    y, _ = librosa.load(full_path, sr=SAMPLE_RATE, mono=True)
    idx_buf, feat_buf = [], []

    for row_idx, start_s, end_s in seg_rows:
        seg = y[int(start_s * SAMPLE_RATE): int(end_s * SAMPLE_RATE)]
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            fmax=8_000,
        )
        logmel = librosa.power_to_db(mel, ref=np.max).T  # (frames, mels)

        # pad / truncate to MAX_FRAMES so every example has equal length
        if logmel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - logmel.shape[0]
            logmel = np.pad(logmel, ((0, pad), (0, 0)))
        else:
            logmel = logmel[:MAX_FRAMES]

        idx_buf.append(row_idx)
        feat_buf.append(logmel.astype(np.float32))

    return np.asarray(idx_buf, dtype=np.int32), np.stack(feat_buf, axis=0)

# ───────────────────────── Builder class ────────────────────────

class FeatureBuilderRNN:
    """Extracts log-mel features from wav segments described in a CSV file."""

    def __init__(self, meta_csv: str):
        self.df = pd.read_csv(meta_csv)
        # Normalise path separators so Windows/Unix are handled the same way
        self.df["wav_path"] = self.df["wav_path"].str.replace(r"[\\/]+", "/", regex=True)
        if "wav_feature_idx" not in self.df.columns or self.df["wav_feature_idx"].isnull().any():
            self.df["wav_feature_idx"] = self.df.index  # contiguous index

    # ------------------------------------------------------------------
    def build(self, h5_path: str = ORIG_H5_PATH, meta_out: str = RNN_META_PATH, n_jobs: int = 4) -> None:
        """Extracts all features and writes them into *h5_path* (compressed)."""
        os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
        n_rows = len(self.df)

        # Pack tasks per song to avoid reading the same wav many times
        tasks = []
        for wav, sub in self.df.groupby("wav_path"):
            seg_rows = list(zip(sub.index.to_numpy(), sub["start_s"].to_numpy(), sub["end_s"].to_numpy()))
            tasks.append((wav, seg_rows))

        # Allocate HDF5 and fill in parallel
        with h5py.File(h5_path, "w") as h5f:
            dset = h5f.create_dataset(
                "logmel",
                shape=(n_rows, MAX_FRAMES, N_MELS),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                futures = {exe.submit(_song_worker, t): t[0] for t in tasks}
                pbar = tqdm(total=n_rows, desc="Extracting log-mel")
                for fut in as_completed(futures):
                    idx_arr, feat_arr = fut.result()
                    dset[idx_arr] = feat_arr
                    pbar.update(len(idx_arr))
                pbar.close()

        # Save refreshed metadata
        self.df.to_csv(meta_out, index=False)

# ──────────────────────── Post‑processing ───────────────────────

def convert_to_fast(src_path: str = ORIG_H5_PATH, dst_path: str = FAST_H5_PATH) -> None:
    """Create an uncompressed copy of *src_path* at *dst_path* and delete src."""
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"{src_path} not found - feature extraction must succeed first.")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name in src.keys():
            dst.create_dataset(name, data=src[name][:], dtype="float32")

    os.remove(src_path)

# ─────────────────────────── CLI entry ──────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract log-mel features then convert to *_fast.h5*.")
    parser.add_argument("--meta",
    default=os.path.join(ROOT_DIR, "POP909_metadata.csv"),
    help="Input metadata CSV in project root."
    )
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel worker processes.")
    args = parser.parse_args()

    print("[Stage 1] Feature extraction →", ORIG_H5_PATH)
    FeatureBuilderRNN(args.meta).build(n_jobs=args.jobs)

    print("[Stage 2] Converting to fast HDF5 →", FAST_H5_PATH)
    convert_to_fast()

    print("Done - fast file ready.")
