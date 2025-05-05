


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


### 📦 Dataset Requirement

To **reproduce or run** our project, you **must manually download** the **POP909 dataset**, including **all `.wav` audio files**, and perform **local feature extraction**.

🧪 If you have any problems on running our code, please contact benlu.wang@yale.edu for help!

---

### 🔗 Download Links

* **All `.wav` audio files (\~18GB)**
  [Download Link](https://yaleedu-my.sharepoint.com/:f:/g/personal/benlu_wang_yale_edu/EvW9mAXUU9xBqiFLTcPD3V4BhRAx5YaLsrze7NgBBlYpkA?e=rBCbyn)

* **POP909 metadata (\~60MB)**
  [POP909 GitHub Repository](https://github.com/music-x-lab/POP909-Dataset/tree/master/POP909)

---

### ⚙️ Setup & Feature Extraction (⏱ \~20 hours due to dataset size)

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Extract Global Embedding Features (`mel128`) for Simple Models**

   ```bash
   python build_metadata.py
   python extract_audio_feature.py
   ```

3. **Extract Frame-Level Log-Mel Features (`logmel`) for Complex Models**

   ```bash
   cd Models
   python feature_extraction.py
   ```

---

### 🧪 Run Model Training (Example: CNN)

1. Navigate to the CNN folder:

   ```bash
   cd Models/CNN
   ```

2. Make the script executable and run it:

   ```bash
   chmod +x hyper_search_cnn.sh
   ./hyper_search_cnn.sh
   ```

3. Logs and results will be saved in:

   ```
   Models/CNN/logs/
   ```


---


## Citation

If you use this repository, please cite the original POP909 paper:

> Ziyu Wang et al. **POP909: A Pop-Song Dataset for Music Arrangement
> Generation.** *ISMIR 2020*.

