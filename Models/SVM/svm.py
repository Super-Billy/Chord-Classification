#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM chord classifier using original POP909 features.
- Inputs: mel128 log-mel spectrograms from features_audio.h5
- Labels: chord labels from POP909_metadata.csv
- Stratified 90/10 train/test split (if possible)
- Outputs accuracy, weighted F1, classification report, and confusion matrix
"""

import numpy as np
import pandas as pd
import h5py
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -------- File paths and config --------
META_CSV = "/home/yaxin/My_Files/ml_final/Chord-Classification/POP909_metadata.csv"
H5_FILE  = "/home/yaxin/My_Files/ml_final/Chord-Classification/features_audio.h5"
DS_NAME  = "mel128"
TEST_SIZE = 0.1
SUBSAMPLE = None  # ✅ 使用全部数据

# ✅ 最优超参数
BEST_C = 10.0
BEST_GAMMA = "scale"

def main():
    # 1. Load metadata
    df = pd.read_csv(META_CSV)
    df["wav_feature_idx"] = df["wav_feature_idx"].astype(int)
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"✅ Total usable samples: {len(df)}")

    # 2. Optional: Subsample data
    if SUBSAMPLE is not None and SUBSAMPLE < len(df):
        df = df.sample(n=SUBSAMPLE, random_state=42).reset_index(drop=True)
        print(f"🔍 Using a random subset of {SUBSAMPLE} samples.")
    else:
        print(f"🔍 Using all {len(df)} samples.")

    # 3. Load features
    with h5py.File(H5_FILE, "r") as h5:
        X_all = h5[DS_NAME][:]
    X = X_all[df["wav_feature_idx"].to_numpy()]
    print(f"✅ Feature matrix shape: {X.shape}")

    # 4. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"✅ Number of classes: {len(le.classes_)}")

    # 5. Stratified train/test split
    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Some classes occur only once — using random split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=stratify
    )

    # 6. Pipeline: scale + SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=BEST_C, gamma=BEST_GAMMA, decision_function_shape="ovr"))
    ])

    # 7. Train
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    pipe.fit(X_train_flat, y_train)

    # 8. Evaluate
    y_pred = pipe.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"\n✅ Accuracy: {acc:.3f} | Macro F1-score: {f1:.3f}\n")

    # 9. Report
    labels_in_test = np.unique(y_test)
    print("Classification report:")
    print(classification_report(
        y_test, y_pred,
        labels=labels_in_test,
        target_names=le.inverse_transform(labels_in_test),
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 10. Optional: save summary
    with open("results_summary.txt", "a") as f:
        f.write(f"C={BEST_C}, gamma={BEST_GAMMA}, subsample={SUBSAMPLE}, acc={acc:.4f}, macro_f1={f1:.4f}\n")

if __name__ == "__main__":
    main()
