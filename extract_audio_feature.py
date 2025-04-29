#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, pandas as pd, soundfile as sf, librosa, h5py
from tqdm import tqdm

# ---- config ----
POP909_ROOT = Path("POP909")
WAV_DIR     = Path("wav_22k_mono")
META_CSV    = "POP909_metadata.csv"
H5_OUT      = "features_audio.h5"
SR, MEL_BINS, N_FFT, HOP = 22_050, 64, 4096, 512
MIN_SAMPLES = N_FFT + HOP * 3
# -----------------

def choose_chord_file(d: Path)->Optional[Path]:
    p1, p2 = d/"chord_audio.txt", d/"chord_midi.txt"
    return p1 if p1.exists() else p2 if p2.exists() else None

def segment_feature(audio: np.ndarray, ss: float, es: float)->Tuple[bool,np.ndarray]:
    s, e = int(ss*SR), int(es*SR)
    if e<=s: e=s+HOP
    seg = audio[s:e]
    if len(seg)<MIN_SAMPLES:
        seg = np.pad(seg,(0,MIN_SAMPLES-len(seg)))
    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=MEL_BINS,
        fmax=8_000,
    )
    lg  = librosa.power_to_db(mel, ref=np.max)
    vec = np.concatenate([lg.mean(1), lg.std(1)]).astype(np.float32)
    return np.isfinite(vec).all() and vec.any(), vec

def main():
    # reset wav_feature_idx
    df = pd.read_csv(META_CSV)
    df["wav_feature_idx"] = -1
    df = df.reset_index(drop=True)

    feat_dim = MEL_BINS*2
    h5f = h5py.File(H5_OUT,"w")
    dset = h5f.create_dataset("mel128",(len(df),feat_dim),"float32",
                              chunks=(1024,feat_dim),compression="gzip")

    written = miss_wav = miss_ann = bad_feat = 0
    for sid, grp in tqdm(df.groupby("song_id"),desc="Songs"):
        sid_str = str(sid).zfill(3)
        wav = WAV_DIR/f"{sid_str}.wav"
        if not wav.exists():
            miss_wav += len(grp); continue
        audio, sr_read = sf.read(wav,dtype="float32")
        if sr_read!=SR: miss_wav += len(grp); continue
        if audio.ndim>1: audio=audio.mean(1)
        if choose_chord_file(POP909_ROOT/sid_str) is None:
            miss_ann += len(grp); continue
        for idx in grp.index:
            ok, vec = segment_feature(audio, df.at[idx,"start_s"], df.at[idx,"end_s"])
            if not ok: bad_feat +=1; continue
            dset[written]=vec
            df.at[idx,"wav_feature_idx"]=written
            written+=1

    dset.resize((written,feat_dim)); h5f.close()

    df = df[df.wav_feature_idx>=0].reset_index(drop=True)
    df["wav_feature_idx"]=np.arange(len(df),dtype=int)
    assert len(df)==written, "metadata and HDF5 size mismatch!"
    df.to_csv(META_CSV,index=False)

    print("\n=== SUMMARY ===")
    print(f"written rows      : {written}")
    print(f"missing WAV rows  : {miss_wav}")
    print(f"missing annot rows: {miss_ann}")
    print(f"bad feature rows  : {bad_feat}")
    print(f"HDF5 shape        : {written} × {feat_dim}")
    print("✓ all indices consistent, extraction complete.")

if __name__ == "__main__":
    main()
