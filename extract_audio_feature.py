#!/usr/bin/env python3
"""
Render POP909 MIDI to WAV, split by chord annotations, extract fixed-size
audio embeddings and update metadata.

This version incorporates the following fixes:

* graceful SoundFont check and error handling
* index reset to avoid HDF5 out-of-range writes
* HDF5 chunking + compression
* safer padding rule (≥ 4 STFT frames)
* clipping guard after resampling
* higher-resolution STFT (n_fft = 4096, hop_length = 512)
* removal of dead code
* English-only comments
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pretty_midi as pm
import soundfile as sf
import librosa
import h5py
from tqdm import tqdm

# ---------- configuration ----------
POP909_ROOT = Path("POP909")
SF2_PATH = Path("TimGM6mb.sf2")        # change if needed
METADATA_CSV = "POP909_metadata.csv"
WAV_DIR = Path("wav_22k_mono")
H5_OUT = "features_audio.h5"

SR_ORIG = 44_100
SR_TARGET = 22_050

MEL_BINS = 64                         # 64 mel bins × (mean+std) = 128 dims
N_FFT = 4096                          # ≈186 ms
HOP = 512                             # ≈23 ms at 22 050 Hz
MIN_FRAMES = 4                        # require at least 4 frames
MIN_SAMPLES = N_FFT + HOP * (MIN_FRAMES - 1)

# -----------------------------------

WAV_DIR.mkdir(exist_ok=True)


def midi_to_wav(midi_path: Path, wav_path: Path) -> bool:
    """
    Render a MIDI file to mono 22 050 Hz WAV.
    Returns True on success, False if rendering failed.
    """
    try:
        midi = pm.PrettyMIDI(str(midi_path))
        if not SF2_PATH.exists():
            raise FileNotFoundError(f"SoundFont not found: {SF2_PATH}")

        audio = midi.fluidsynth(fs=SR_ORIG, sf2_path=str(SF2_PATH))  # stereo float32 [-1,1]
    except Exception as exc:
        print(f"[WARN]   Could not render '{midi_path.name}': {exc}")
        return False

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # convert to mono

    # down-sample
    audio = librosa.resample(audio, orig_sr=SR_ORIG, target_sr=SR_TARGET)
    audio = np.clip(audio, -1.0, 1.0)  # avoid potential clipping
    sf.write(wav_path, audio, SR_TARGET, subtype="FLOAT")
    return True


def choose_chord_file(song_dir: Path) -> Optional[Path]:
    """Return the preferred annotation file, or None if neither exists."""
    audio_ann = song_dir / "chord_audio.txt"
    midi_ann = song_dir / "chord_midi.txt"
    if audio_ann.exists():
        return audio_ann
    if midi_ann.exists():
        return midi_ann
    return None


def extract_fixed_mel_segment(
    y: np.ndarray,
    start_s: float,
    end_s: float,
    sr: int,
    mbins: int = MEL_BINS,
) -> np.ndarray:
    """
    Return a 2 x n_mels embedding (mean + std) for the specified segment.
    Pads with zeros until ≥ MIN_FRAMES STFT frames are available.
    """
    start = int(start_s * sr)
    end = int(end_s * sr)
    seg = y[start:end]

    # pad if the segment is shorter than required
    if len(seg) < MIN_SAMPLES:
        seg = np.pad(seg, (0, MIN_SAMPLES - len(seg)))

    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=mbins,
        fmax=8_000,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    mu = log_mel.mean(axis=1)
    sigma = log_mel.std(axis=1)
    return np.concatenate([mu, sigma]).astype(np.float32)  # (2 * mbins,)


def main() -> None:
    # 1. load metadata and reset index for contiguous rows
    df = pd.read_csv(METADATA_CSV).reset_index(drop=True)
    total_segments = len(df)
    feat_dim = 2 * MEL_BINS

    # 2. prepare HDF5 dataset
    with h5py.File(H5_OUT, "w") as h5f:
        dset = h5f.create_dataset(
            "mel128",
            shape=(total_segments, feat_dim),
            dtype="float32",
            chunks=(1024, feat_dim),
            compression="gzip",
        )

        # 3. iterate through songs
        grouped = df.groupby("song_id")
        for song_id, group in tqdm(grouped, desc="Songs"):
            song_id_str = str(song_id).zfill(3)
            song_dir = POP909_ROOT / song_id_str
            midi_path = song_dir / f"{song_id_str}.mid"
            wav_path = WAV_DIR / f"{song_id_str}.wav"

            if not wav_path.exists():
                if not midi_to_wav(midi_path, wav_path):
                    # skip this song if rendering failed
                    continue

            # load WAV once per song
            y, _ = sf.read(wav_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)

            # (optional) verify annotation file exists
            chord_file = choose_chord_file(song_dir)
            if chord_file is None:
                print(f"[WARN]   No chord annotation found for {song_id_str}")
                continue

            # 4. process each metadata row
            for idx in group.index:
                start_s = df.at[idx, "start_s"]
                end_s = df.at[idx, "end_s"]
                feat = extract_fixed_mel_segment(y, start_s, end_s, sr=SR_TARGET)
                dset[idx] = feat
                df.at[idx, "wav_feature_idx"] = idx  # index == row id

    # 5. save updated CSV
    if "wav_feature_idx" not in df.columns:
        df["wav_feature_idx"] = -1
    df.to_csv(METADATA_CSV, index=False)

    print(
        f"Done. Saved {total_segments} embeddings "
        f"({feat_dim} dims each) to '{H5_OUT}'."
    )


if __name__ == "__main__":
    main()
