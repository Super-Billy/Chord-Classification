import librosa
import numpy as np
import os
from typing import List, Tuple
import mido
from midi_to_chords import midi_to_chords
import soundfile as sf  # to read sample rate without loading full audio
from tqdm import tqdm

DATASET_ROOT = "./dataset/"

def setup_dataset(
    train_path: str,
    test_path: str,
    dataset_root: str
) -> None:
    """
    Build train/test splits of chord‐aligned 8‑bit mel spectrograms
    and save them as .npz container files.

    Parameters
    ----------
    train_path : str
        Filepath to write the training data (e.g. 'train_data.npz').
    test_path : str
        Filepath to write the test data (e.g. 'test_data.npz').
    dataset_root : str
        Root directory containing paired WAV+MIDI subfolders.

    Outputs
    -------
    Creates two .npz files, each with keys:
      - 'mels': object array of (n_mels × n_frames) uint8 arrays
      - 'labels': object array of str chord‐pattern labels
    """
    train_mels: List[np.ndarray] = []
    train_labels: List[str] = []
    test_mels:  List[np.ndarray] = []
    test_labels:  List[str] = []

    pairs: List[Tuple[str, str]] = collect_wav_midi_pairs(dataset_root)

    for wav_path, midi_path in tqdm(pairs, desc="Processing pieces"):
        # 1) 8‑bit mel (uses librosa internally)
        mel = wav_to_8bit_mel(
            wav_path,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmin=0,
            fmax=None,
            min_db=-80.0
        )
        breakpoint()

        # 2) get sample rate for slicing
        sr = sf.info(wav_path).samplerate
        hop = 512

        # 3) chords & slicing
        chords = midi_to_chords(midi_path)            # -> List[dict]
        breakpoint()
        slices = slice_mel_by_chords(
            mel_spectrogram=mel,
            sample_rate=sr,
            hop_length=hop,
            chords=chords,
            midi_path=midi_path
        )
        labels = [c["pattern"] for c in chords]

        # 4) split 80/20 for *this* piece
        n_total = len(slices)
        n_train = int(n_total * 0.8)

        for i, seg in enumerate(slices):
            if i < n_train:
                train_mels.append(seg)
                train_labels.append(labels[i])
            else:
                test_mels.append(seg)
                test_labels.append(labels[i])

    # 5) save out
    np.savez_compressed(train_path,
                        mels=np.array(train_mels, dtype=object),
                        labels=np.array(train_labels, dtype=object))
    np.savez_compressed(test_path,
                        mels=np.array(test_mels, dtype=object),
                        labels=np.array(test_labels, dtype=object))

    print(f"✔ Saved {len(train_mels)} train slices to {train_path}")
    print(f"✔ Saved {len(test_mels)}  test slices to {test_path}")


def collect_wav_midi_pairs(root_dir: str = '.') -> List[Tuple[str, str]]:
    """
    Recursively search for WAV and MIDI files in a directory tree and return paired paths.

    Given a directory structure like:

        root/
        ├── ...
        ├── N_piece_instrument/
        │   ├── ...
        │   ├── AuMix_N_piece_instrument.wav
        │   └── Sco_N_piece_instrument.mid
        └── ...

    This function will traverse every subdirectory under `root_dir`, collect files
    ending with `.wav` and `.mid` (or `.midi`), and pair them by matching base filenames.

    Parameters
    ----------
    root_dir : str, optional
        Path to the root directory where the search begins (default: current directory `.`).

    Returns
    -------
    List[Tuple[str, str]]
        Sorted list of tuples `(wav_path, midi_path)` for each matching pair found.
    """
    pairs: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        # gather audio and midi files in this folder
        wavs = [f for f in filenames if f.lower().endswith('.wav')]
        mids = [f for f in filenames if f.lower().endswith('.mid') or f.lower().endswith('.midi')]

        # pick the AuMix wav file
        aumix = [w for w in wavs if os.path.splitext(w)[0].startswith('AuMix_')]
        if not aumix:
            continue
        wav_file = aumix[0]
        wav_base = os.path.splitext(wav_file)[0]

        # extract suffix after first underscore
        parts = wav_base.split('_', 1)
        if len(parts) != 2:
            continue
        suffix = parts[1]

        # find the corresponding midi by matching suffix
        for mid in mids:
            mid_base = os.path.splitext(mid)[0]
            mid_parts = mid_base.split('_', 1)
            if len(mid_parts) == 2 and mid_parts[1] == suffix:
                pairs.append((os.path.join(dirpath, wav_file),
                              os.path.join(dirpath, mid)))
                break

    return sorted(pairs)

def wav_to_8bit_mel(wav_path,
                    n_mels=128,
                    n_fft=2048,
                    hop_length=512,
                    fmin=0,
                    fmax=None,
                    min_db=-80.0):
    """
    Load a WAV file and convert it to an 8‑bit mel‑spectrogram.

    Parameters
    ----------
    wav_path : str
        Path to input WAV file.
    n_mels : int
        Number of mel bands.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop (stride) length between frames.
    fmin : float
        Lowest frequency (Hz).
    fmax : float or None
        Highest frequency (Hz). If None, uses sr/2.
    min_db : float (<0)
        Floor for the dB scaling (e.g. -80 dB).

    Returns
    -------
    np.ndarray of uint8, shape (n_mels, n_frames)
        Each value in [0, 255], where 0 corresponds to `min_db` and
        255 to 0 dB.
    """

    # 1. Load audio (mono) at native sampling rate
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # 2. Compute mel-scaled power spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr/2,
        power=2.0
    )

    # 3. Convert power to decibels (max at 0 dB, floor at `min_db`)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=abs(min_db))

    # 4. Normalize to [0, 1]
    #    S_db ranges in [min_db, 0], so (S_db - min_db)/(0 - min_db)
    S_norm = (S_db - min_db) / abs(min_db)

    # 5. Scale to [0, 255] and convert to uint8
    S_8bit = (S_norm * 255).clip(0, 255).astype(np.uint8)

    return S_8bit

def slice_mel_by_chords(
    mel_spectrogram: np.ndarray,
    sample_rate: int,
    hop_length: int,
    chords: list[dict],
    midi_path: str
) -> list[np.ndarray]:
    """
    Slice a mel‑spectrogram into segments aligned to MIDI chord timings.

    Parameters
    ----------
    mel_spectrogram : np.ndarray, shape (n_mels, n_frames)
        Your precomputed mel‑spectrogram.
    sample_rate : int
        Sampling rate used when computing that spectrogram.
    hop_length : int
        Hop length (in samples) between frames in the spectrogram.
    chords : List[Dict[str, Any]]
        Each dict must have integer 'start' and 'end' fields (MIDI ticks).
    midi_path : str
        Path to the MIDI file whose ticks correspond to those in `chords`.

    Returns
    -------
    List[np.ndarray]
        A list of mel‑spectrogram slices (each shape: n_mels × n_chord_frames),
        in the same order as `chords`.
    """
    # 1. Load MIDI and grab its resolution & tempo (µs per beat)
    midi = mido.MidiFile(midi_path)
    tpq = midi.ticks_per_beat
    # default to 120 BPM if no tempo event found
    tempo = next((msg.tempo for track in midi.tracks 
                           for msg in track 
                           if msg.type == 'set_tempo'),
                 500_000)

    segments = []
    for chord in chords:
        # 2. Convert tick counts to seconds
        t_start = mido.tick2second(chord['start'], tpq, tempo)
        t_end   = mido.tick2second(chord['end'],   tpq, tempo)

        # 3. Map seconds → frame indices
        f_start = int(round(t_start * sample_rate / hop_length))
        f_end   = int(round(t_end   * sample_rate / hop_length))

        # 4. Clamp to valid range
        f_start = max(0, f_start)
        f_end   = min(mel_spectrogram.shape[1], f_end)

        # 5. Slice out that time window
        segments.append(mel_spectrogram[:, f_start:f_end])

    return 


if __name__ == "__main__":

    pairs = collect_wav_midi_pairs(root_dir=DATASET_ROOT)

    total_chords = []

    for wav, midi in pairs:
        chords = midi_to_chords(midi)
        total_chords += chords
    
    unknown_count = 0
    for chord in total_chords:
        if chord["pattern"] == "unknown":
            unknown_count += 1
    
    print(f'Percentage of unknown chords: {unknown_count/len(total_chords)*100}')