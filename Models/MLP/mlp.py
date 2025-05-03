#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch MLP chord classifier using mel128 features from POP909.
- Supports GPU (e.g. RTX 6000 Ada)
- CLI arguments: --lr, --batch-size, --epochs, --hidden
"""

import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings

# -------- CLI arguments --------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=str, default="256,128")  # comma-separated
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

# -------- MLP Model --------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------- Main --------
def main():
    args = parse_args()
    hidden_sizes = list(map(int, args.hidden.split(",")))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"✅ Using {device} | hidden={hidden_sizes} | lr={args.lr} | batch_size={args.batch_size} | epochs={args.epochs}")

    # Load metadata and features
    df = pd.read_csv("POP909_metadata.csv")
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"✅ Usable samples: {len(df)}")

    with h5py.File("features_audio.h5", "r") as h5:
        X_all = h5["mel128"][:]
    X = X_all[df["wav_feature_idx"].astype(int)]
    y = LabelEncoder().fit_transform(df["label"])
    print(f"✅ Feature shape: {X.shape} | Classes: {len(np.unique(y))}")

    # Stratified split
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Some classes occur only once — using random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=stratify, random_state=42)

    # Flatten & scale
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Model setup
    model = MLP(X_train_tensor.shape[1], hidden_sizes, len(np.unique(y))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0
        for i in range(0, X_train_tensor.size(0), args.batch_size):
            idx = permutation[i:i + args.batch_size]
            xb, yb = X_train_tensor[idx], y_train_tensor[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test_tensor), dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"\n✅ Accuracy: {acc:.3f} | Weighted F1: {f1:.3f}\n")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
