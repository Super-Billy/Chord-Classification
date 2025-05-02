#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNN chord classifier – CNN front‑end + BiLSTM + multi‑head attention.

Fix 2025‑04‑30:
* Corrected LSTM `input_size` calculation (mismatch caused runtime error).
* Now derives feature dimension analytically: `flat_dim = (base_channels*4) * (n_mels//4)`.
* Minor refactor for clarity; no change to external CLI.
"""
from __future__ import annotations

import argparse
import math
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
N_MELS = 64  # log‑mel bins


class ChordDatasetCRNN(Dataset):
    """HDF5-backed dataset returning (1, T, M) float32 tensors."""

    def __init__(self, df: pd.DataFrame, h5_path: str, le: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.le = le
        self._h5: Optional[h5py.File] = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        feat_idx = self.df.at[idx, "wav_feature_idx"]
        x = torch.from_numpy(self._h5["logmel"][feat_idx][()]).unsqueeze(0).float()
        y = torch.tensor(self.le.transform([self.df.at[idx, "label"]])[0], dtype=torch.long)
        return x, y


# ────────── Model ──────────
class CRNNAttn(nn.Module):
    """CRNN with additive attention."""

    def __init__(
        self,
        *,
        n_mels: int = N_MELS,
        base_channels: int = 64,
        cnn_k: int = 3,
        lstm_hidden: int = 512,
        heads: int = 4,
        n_classes: int = 320,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        C0 = base_channels
        pad = cnn_k // 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, C0, cnn_k, padding=pad), nn.BatchNorm2d(C0), nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(C0, C0 * 2, cnn_k, padding=pad), nn.BatchNorm2d(C0 * 2), nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(C0 * 2, C0 * 4, cnn_k, padding=pad), nn.BatchNorm2d(C0 * 4), nn.ReLU(True),
        )
        self.dropout = nn.Dropout(dropout)

        # ---- derive flattened feature dimension ----
        mel_down = n_mels // 4  # two 2×2 pools along frequency
        self.flat_dim = C0 * 4 * mel_down

        self.lstm = nn.LSTM(
            input_size=self.flat_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = nn.Linear(lstm_hidden * 2, heads)
        self.fc = nn.Linear(lstm_hidden * 2 * heads + lstm_hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, 1, T, M)
        z = self.cnn(x)                         # (B, C', T', M')
        z = self.dropout(z)
        B, C, T_p, M_p = z.shape               # C == base_channels*4, M_p == N_MELS/4
        z = z.permute(0, 2, 1, 3).contiguous().view(B, T_p, C * M_p)
        assert z.size(-1) == self.flat_dim, "LSTM input dim mis-match"
        h, _ = self.lstm(z)
        h = self.norm(h)
        logits = self.attn(h)
        w = torch.softmax(logits, dim=1)
        context = (h.unsqueeze(-1) * w.unsqueeze(2)).sum(1).permute(0, 2, 1).flatten(1)
        global_avg = h.mean(dim=1)
        return self.fc(torch.cat([context, global_avg], 1))


# ────────── Training ──────────

def train_crnn(args: argparse.Namespace):
    df = pd.read_csv(RNN_META_PATH)
    le = LabelEncoder().fit(df["label"])

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Rare classes (<2) – random split applied.")

    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify)
    tr_ds = ChordDatasetCRNN(tr_df, RNN_H5_PATH, le)
    va_ds = ChordDatasetCRNN(va_df, RNN_H5_PATH, le)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CRNNAttn(
        n_mels=N_MELS,
        base_channels=args.base_channels,
        cnn_k=args.kernel_size,
        lstm_hidden=args.hidden,
        heads=args.heads,
        n_classes=len(le.classes_),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05) if args.scheduler == "cosine" else StepLR(optimizer, step_size=10, gamma=0.5)

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        tot, correct = 0, 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                tot += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        val_acc = correct / tot
        scheduler.step()
        print(f"Epoch {ep:02d}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "crnn_attn.pth"))


# ────────── CLI ──────────

def get_args():
    p = argparse.ArgumentParser("CRNN chord classifier")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    train_crnn(get_args())
