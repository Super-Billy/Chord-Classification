


## CPSC 581 Introduction to Machine Learning 
### Final Project - Chord Classification

---
## Repository structure

| Path | Type | Description |
|------|------|-------------|
| **`POP909/`** | folder | Original POP909 dataset (909 sub-folders `001/…/909/`). Each sub-folder contains `###.mid` and annotation files such as `chord_midi.txt`, `chord_audio.txt`. |
| **`TimGM6mb.sf2`** | file | General-MIDI SoundFont used by `pretty_midi.fluidsynth()` for rendering audio. |
| **`POP909_metadata.csv`** | file | Master metadata table: one row *per chord segment*. Added/updated by the scripts. |
| **`features_audio.h5`** | file | HDF5 store created by the audio pipeline (`extract_audio_feature.py`). Dataset name: `mel128`  (`N_segments × 128`). |
| **`build_metadata.py`** | script | One-off script that scans `POP909/`, reads `chord_midi.txt`, and builds the initial `POP909_metadata.csv`. |
| **`feature_extraction.py`** | script | **Symbolic pipeline** – reads `.mid`, slices by time-stamps, extracts 12-dim chroma, resamples to 32 frames, flattens to 384-dim, writes `features.h5` (dataset `chroma384`) and updates `feature_idx`. |
| **`extract_audio_feature.py`** | script | **Audio pipeline** – load `.wav`, computes whole-song log-Mel once, slices each chord segment, saves 128-dim embedding to `features_audio.h5` (dataset `mel128`) and updates `wav_feature_idx`. |
| **`features.h5`** | file | (Created by `feature_extraction.py`) 384-dim chroma features. |
| **`.gitignore` / `README.md`** | misc | House-keeping. |

Note: You must download the **POP909** Dataset to run **feature_extraction.py**. You must dowload **all .wav files** to run **extract_audio_feature.py**

### Link for all .wav file (18GB)
https://yaleedu-my.sharepoint.com/:f:/g/personal/benlu_wang_yale_edu/EmZ_Fz6CddZLmRDYMvRGdl4Baz-KNIQ-k8tNfxQ_qxVMEw?e=UKuSN5

### Link for the POP909 Dataset (60MB)
https://github.com/music-x-lab/POP909-Dataset/tree/master/POP909

---

## Metadata schema 

| Column | Example | Meaning |
|--------|---------|---------|
| `song_id` | `001` | Three-digit POP909 index |
| `segment_id` | `042` | Index within song |
| `start_s` / `end_s` | `12.345` | Segment boundaries in seconds |
| `label` | `C:min7` | Original chord symbol (`N` = no-chord) |
| `feature_idx` | `12345` | Row in **`features.h5/chroma384`** *(symbolic pipeline)* |
| `wav_feature_idx` | `12345` | Row in **`features_audio.h5/mel128`** *(audio pipeline)* |

---

## Example – load embeddings in PyTorch

```python
import h5py, pandas as pd, torch

meta = pd.read_csv("POP909_metadata.csv")
h5   = h5py.File("features_audio.h5", "r")     # or features.h5
X    = h5["mel128"]                            # memory-mapped dataset

# get first 1024 training samples
sel   = meta.query("split == 'train'").head(1024)
feats = torch.tensor(X[sel["wav_feature_idx"]])   # (1024, 128)
labels = sel["label"].values
```

---


## Citation

If you use this repository, please cite the original POP909 paper:

> Ziyu Wang et al. **POP909: A Pop-Song Dataset for Music Arrangement
> Generation.** *ISMIR 2020*.

