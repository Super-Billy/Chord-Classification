#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch-based Logistic Regression for chord classification from POP909.
Supports CLI arguments for hyperparameter search.

- Inputs: mel128 log-mel spectrograms from features_audio.h5
- Labels: chord labels from POP909_metadata.csv
- CLI arguments: --lr, --batch-size, --epochs, --device
- Output: Accuracy, F1, classification report, confusion matrix
"""

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -------- Parse CLI arguments --------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Logistic Regression for POP909")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    return parser.parse_args()

# -------- Model definition --------
class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# -------- Main training pipeline --------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    print(f"✅ Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}")

    # Load metadata and features
    df = pd.read_csv("POP909_metadata.csv")
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"✅ Total usable samples: {len(df)}")

    with h5py.File("features_audio.h5", "r") as h5:
        X_all = h5["mel128"][:]
    X = X_all[df["wav_feature_idx"].astype(int)]
    print(f"✅ Feature matrix shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"✅ Number of classes: {len(le.classes_)}")

    # Stratified split
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Some classes occur only once — using random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=stratify, random_state=42)

    # Flatten + scale
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Build model
    model = TorchLogisticRegression(X_train_tensor.shape[1], len(le.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0.0
        for i in range(0, X_train_tensor.size(0), args.batch_size):
            idx = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train_tensor[idx], y_train_tensor[idx]
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        print(f"\n✅ Final Accuracy: {acc:.3f} | Weighted F1-score: {f1:.3f}\n")
        print("Classification report:")
        print(classification_report(
            y_true, y_pred,
            labels=np.unique(y_true),
            target_names=le.inverse_transform(np.unique(y_true)),
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
