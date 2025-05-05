#!/usr/bin/env python3
"""
Build POP909 metadata CSV for chord-segment classification.

Author : Iris Xia
Date   : 2025-04-28
"""

import pathlib
import pandas as pd
from tqdm import tqdm
import re

# ---------- config ----------
POP909_ROOT = pathlib.Path("POP909")   # change to your dataset path
CSV_OUT     = "POP909_metadata.csv"
FS          = 100                     # Hz, if you later want frame indices
# --------------------------------


def parse_chord_symbol(sym: str):
    """
    Split a chord symbol like 'Ab:maj7' into root='Ab', quality='maj7'.
    Symbols with slash (e.g. 'G:min/b3') -> root='G', quality='min/b3'.
    """
    if sym == "N":
        return "N", "N"
    root, *quality = sym.split(":")
    quality = quality[0] if quality else ""
    return root, quality


def read_chord_file(txt_path: pathlib.Path):
    """
    Yield dict(start_s, end_s, label, root, quality) for each line in chord_midi.txt.
    """
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) != 3:
                continue
            start, end, label = parts
            root, qual = parse_chord_symbol(label)
            yield {
                "start_s": float(start),
                "end_s":   float(end),
                "label":   label,
                "root":    root,
                "quality": qual
            }


def build_metadata(pop909_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for song_dir in tqdm(sorted(pop909_root.iterdir()), desc="Scanning songs"):
        if not song_dir.is_dir():
            continue
        song_id = song_dir.name                  # e.g. "001"
        chord_file = song_dir / "chord_midi.txt"
        midi_file  = song_dir / f"{song_id}.mid"

        if not chord_file.exists():
            print(f"[WARN] {chord_file} not found, skip.")
            continue
        if not midi_file.exists():
            print(f"[WARN] {midi_file} not found, skip.")
            continue

        wav_rel_path = f"wav_22k_mono/{song_id}.wav"

        for seg_idx, seg in enumerate(read_chord_file(chord_file)):
            rows.append({
                "song_id":      song_id,
                "segment_id":   f"{seg_idx:03d}",
                "split":        "unsplit",           # fill later
                "start_s":      seg["start_s"],
                "end_s":        seg["end_s"],
                "label":        seg["label"],
                "root":         seg["root"],
                "quality":      seg["quality"],
                "midi_path":    str(midi_file.as_posix()),
                "wav_path":     wav_rel_path,     
                "feature_idx":  -1                   # fill when feature saved
            })
    return pd.DataFrame(rows)


def main():
    df = build_metadata(POP909_ROOT)
    df["song_id"] = df["song_id"].astype(str).str.zfill(3)
    print(f"Total segments: {len(df)}")
    df.to_csv(CSV_OUT, index=False)
    print(f"CSV written to {CSV_OUT}")


if __name__ == "__main__":
    main()
