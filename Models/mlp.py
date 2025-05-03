
"""
MLP chord classifier using original POP909 features.
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -------- File paths and configuration --------
META_CSV = "POP909_metadata.csv"        # Metadata with labels and feature indices
H5_FILE  = "features_audio.h5"          # HDF5 file with mel128 features
DS_NAME  = "mel128"                     # Dataset name inside HDF5 file
TEST_SIZE = 0.1                         # 10% test set

# -------- MLP hyperparameters --------
HIDDEN_LAYERS = (256, 128)
ACTIVATION    = "relu"
ALPHA         = 1e-4                    # L2 regularization strength
MAX_ITER      = 200

def main():
    # 1. Load metadata CSV
    df = pd.read_csv(META_CSV)
    df["wav_feature_idx"] = df["wav_feature_idx"].astype(int)
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"✅ Total usable samples: {len(df)}")

    # 2. Load features from HDF5
    with h5py.File(H5_FILE, "r") as h5:
        X_all = h5[DS_NAME][:]
    X = X_all[df["wav_feature_idx"].to_numpy()]
    print(f"✅ Feature matrix shape: {X.shape}")  # (N, 128, 128)

    # 3. Encode chord labels as integers
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"✅ Number of classes: {len(le.classes_)}")

    # 4. Stratified train/test split if possible
    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Some classes occur only once — using random split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=stratify
    )

    # 5. Build pipeline (flatten → scale → classify)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation=ACTIVATION,
            alpha=ALPHA,
            max_iter=MAX_ITER,
            solver="adam",
            random_state=42,
            verbose=True
        ))
    ])

    # 6. Train the model (flatten 128×128 into vector)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    pipe.fit(X_train_flat, y_train)

    # 7. Evaluate on test set
    y_pred = pipe.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\n✅ Accuracy: {acc:.3f} | Weighted F1-score: {f1:.3f}\n")

    # 8. Report metrics
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

if __name__ == "__main__":
    main()
