#!/usr/bin/env python3
"""
Extract fixed-size chroma features (32x12) from POP909 MIDI segments
and update metadata with HDF5 row indices.

Author : Arman Cohan
Date   : 2025-04-28
"""

import pathlib
import numpy as np
import pandas as pd
import pretty_midi as pm
import h5py
from tqdm import tqdm

# ---------- config ----------
METADATA_CSV = "POP909_metadata.csv"
POP909_ROOT  = pathlib.Path("POP909")
H5_OUT       = "features.h5"
FS           = 100      # frames per second for chroma
F_FIXED      = 32       # fixed time frames after resampling
# --------------------------------


def resample_chroma(chroma: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Linearly resample (T, 12) chroma matrix to (target_frames, 12).
    """
    if chroma.size == 0:
        return np.zeros((target_frames, 12), dtype=np.float32)

    t_original = np.linspace(0, 1, num=chroma.shape[0], endpoint=False)
    t_target   = np.linspace(0, 1, num=target_frames,   endpoint=False)
    resampled  = np.empty((target_frames, 12), dtype=np.float32)

    for k in range(12):
        resampled[:, k] = np.interp(t_target, t_original, chroma[:, k])

    return resampled


def segment_chroma(midi_path: pathlib.Path,
                   start_s: float,
                   end_s: float,
                   fs: int,
                   f_fixed: int) -> np.ndarray:
    """
    Extract chroma on [start_s, end_s) then resample to (f_fixed, 12).
    """
    midi = pm.PrettyMIDI(str(midi_path))
    chroma = midi.get_chroma(fs=fs).T          # (T, 12)
    s_idx = int(start_s * fs)
    e_idx = int(end_s   * fs)
    chroma_seg = chroma[s_idx:e_idx, :]        # (T_seg, 12)

    return resample_chroma(chroma_seg, f_fixed).flatten()  # (f_fixed*12,)


def main():
    df = pd.read_csv(METADATA_CSV)
    total_segments = len(df)
    feature_dim = F_FIXED * 12

    # create / overwrite HDF5
    with h5py.File(H5_OUT, "w") as h5f:
        dset = h5f.create_dataset(
            "chroma_fixed",
            shape=(total_segments, feature_dim),
            dtype="float32"
        )

        for idx, row in tqdm(df.iterrows(), total=total_segments, desc="Extracting"):
            rel_path = pathlib.Path(row["midi_path"])

            if not rel_path.is_absolute():
                if rel_path.parts[0] == POP909_ROOT.name:
                    rel_path = POP909_ROOT / pathlib.Path(*rel_path.parts[1:])
                else:
                    rel_path = POP909_ROOT / rel_path
            if not rel_path.exists():         
                raise FileNotFoundError(rel_path)

            feat = segment_chroma(
                midi_path=rel_path,
                start_s=row["start_s"],
                end_s=row["end_s"],
                fs=FS,
                f_fixed=F_FIXED,
            )
            dset[idx] = feat.astype(np.float32)
            df.at[idx, "feature_idx"] = idx  # row number == HDF5 index

    # save updated CSV
    df.to_csv(METADATA_CSV, index=False)
    print(f"Done. Wrote {total_segments} segments x {feature_dim} dims to {H5_OUT}")


if __name__ == "__main__":
    main()
