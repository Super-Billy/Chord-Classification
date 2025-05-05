#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final PyTorch Logistic Regression with best hyperparameters for chord classification.
"""

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings

# === Constants ===
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

META_CSV = "/home/yaxin/My_Files/ml_final/Chord-Classification/POP909_metadata.csv"
H5_FILE = "/home/yaxin/My_Files/ml_final/Chord-Classification/features_audio.h5"

# === Model ===
class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def main():
    print(f"✅ Using device: {DEVICE}")
    print(f"✅ Best Hyperparameters: lr={LR}, batch_size={BATCH_SIZE}, epochs={EPOCHS}")

    # === Load data ===
    df = pd.read_csv(META_CSV)
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"✅ Usable samples: {len(df)}")

    with h5py.File(H5_FILE, "r") as h5:
        X_all = h5["mel128"][:]
    X = X_all[df["wav_feature_idx"].astype(int)]
    print(f"✅ Feature shape: {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"✅ Number of classes: {len(le.classes_)}")

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Some classes occur only once — using random split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=stratify, random_state=42
    )

    # === Preprocess ===
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # === Train ===
    model = TorchLogisticRegression(X_train_tensor.shape[1], len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0.0
        for i in range(0, X_train_tensor.size(0), BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch_x = X_train_tensor[idx]
            batch_y = y_train_tensor[idx]

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f}")

    # === Evaluate ===
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\n✅ Final Accuracy: {acc:.3f} | Macro F1-score: {f1:.3f}\n")
        print("Classification Report:")
        print(classification_report(
            y_true, y_pred,
            labels=np.unique(y_true),
            target_names=le.inverse_transform(np.unique(y_true)),
            zero_division=0
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        with open("results_summary.txt", "a") as f:
            f.write(f"[BEST] lr={LR}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, acc={acc:.4f}, macro_f1={f1:.4f}\n")

if __name__ == "__main__":
    main()
