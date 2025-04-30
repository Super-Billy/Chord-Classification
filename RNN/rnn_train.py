#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM‑Attention chord classifier (concise logging).

This version removes progress bars and verbose prints; it outputs **only** one
line per epoch in the form:

    Epoch 01  val_acc=0.792

All hyper‑parameters remain configurable via CLI arguments as before (see
--help). Place this file in the RNN/ directory alongside
features_audio_rnn_fast.h5 and POP909_metadata_rnn.csv.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import h5py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset

# ────────── Paths ──────────
BASE_DIR = os.path.dirname(__file__)
RNN_H5_PATH = os.path.join(BASE_DIR, "features_audio_rnn_fast.h5")
RNN_META_PATH = os.path.join(BASE_DIR, "POP909_metadata_rnn.csv")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# ────────── Dataset ──────────
N_MELS = 64

class ChordDatasetRNN(Dataset):
    """Lazy HDF5 loader; avoids pickling HDF handles."""

    def __init__(self, df: pd.DataFrame, h5_path: str, label_enc: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.le = label_enc
        self._h5: Optional[h5py.File] = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        feat_idx = self.df.at[idx, "wav_feature_idx"]
        x = torch.from_numpy(self._h5["logmel"][feat_idx][()])
        y = torch.tensor(self.le.transform([self.df.at[idx, "label"]])[0])
        return x, y

# ────────── Model ──────────
class BiLSTMAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_MELS,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.attn = nn.Linear(hidden * 2, heads)
        self.fc = nn.Linear(hidden * 2 * heads + hidden * 2, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.norm(h)
        w = torch.softmax(self.attn(h), dim=1)
        context = (h.unsqueeze(-1) * w.unsqueeze(2)).sum(1)
        context = context.permute(0, 2, 1).flatten(1)
        global_avg = h.mean(dim=1)
        return self.fc(torch.cat([context, global_avg], dim=1))

# ────────── Training ──────────

def train_rnn(args: argparse.Namespace):
    df = pd.read_csv(RNN_META_PATH)
    label_enc = LabelEncoder().fit(df["label"])

    class_counts = df["label"].value_counts()
    stratify = df["label"] if class_counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Rare classes (<2) — random split used.")

    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify)

    tr_ds = ChordDatasetRNN(tr_df, RNN_H5_PATH, label_enc)
    va_ds = ChordDatasetRNN(va_df, RNN_H5_PATH, label_enc)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttn(args.hidden, args.heads, args.dropout, len(label_enc.classes_)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01) if args.scheduler == "cosine" else StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # validate
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                tot += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        val_acc = correct / tot

        scheduler.step()
        print(f"Epoch {epoch:02d}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "bilstm_attn.pth"))

# ────────── CLI ──────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--label-smoothing", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()

if __name__ == "__main__":
    train_rnn(get_args())