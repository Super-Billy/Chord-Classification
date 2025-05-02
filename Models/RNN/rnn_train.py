#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiRNN-Attention chord classifier with concise epoch logging.

Replaces the original BiLSTM backbone by a two-layer bi-directional vanilla RNN
(`nn.RNN`). The script prints **one line per epoch** with accuracy and F1 scores:

    Epoch 01  val_acc=0.792  f1_macro=0.710  f1_weighted=0.789

Place this file in the RNN/ directory alongside:
    - features_audio_rnn_fast.h5
    - POP909_metadata_rnn.csv

Run:
    python birnn_train.py --help   # list available CLI options
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset

# ────────── Paths ──────────
BASE_DIR = os.path.dirname(__file__)
RNN_H5_PATH = os.path.join(BASE_DIR, "../features_audio_rnn_fast.h5")
RNN_META_PATH = os.path.join(BASE_DIR, "../POP909_metadata_rnn.csv")
# WEIGHT_DIR = os.path.join(BASE_DIR, "weights")
# os.makedirs(WEIGHT_DIR, exist_ok=True)

# ────────── Dataset ──────────
N_MELS = 64  # log-mel feature dimension saved in HDF5


class ChordDatasetRNN(Dataset):
    """Lazy HDF5 loader; avoids pickling HDF handles in DataLoader workers."""

    def __init__(self, df: pd.DataFrame, h5_path: str, label_enc: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.le = label_enc
        self._h5: Optional[h5py.File] = None  # opened lazily in __getitem__

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        feat_idx = self.df.at[idx, "wav_feature_idx"]
        x = torch.from_numpy(self._h5["logmel"][feat_idx][()])  # (T, 64)
        y = torch.tensor(
            self.le.transform([self.df.at[idx, "label"]])[0], dtype=torch.long
        )
        return x, y


# ────────── Model ──────────
class BiRNNAttn(nn.Module):
    """Two-layer bi-directional RNN with multi-head additive attention."""

    def __init__(
        self,
        *,
        mel_bins: int = N_MELS,
        hidden: int = 512,
        n_classes: int = 320,
        heads: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=mel_bins,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh",
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.attn = nn.Linear(hidden * 2, heads)
        self.fc = nn.Linear(hidden * 2 * heads + hidden * 2, n_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        h, _ = self.rnn(x)  # (B, T, 2H)
        h = self.norm(h)

        logits = self.attn(h)  # (B, T, heads)
        if lengths is not None:
            T = x.size(1)
            mask = (torch.arange(T, device=x.device)[None, :] >= lengths[:, None])
            logits.masked_fill_(mask.unsqueeze(-1), -1e4)

        w = torch.softmax(logits, dim=1)  # (B, T, heads)
        context = (h.unsqueeze(-1) * w.unsqueeze(2)).sum(1)  # (B, 2H, heads)
        context = context.permute(0, 2, 1).flatten(1)  # (B, 2H*heads)
        global_avg = h.mean(dim=1)  # (B, 2H)
        out = torch.cat([context, global_avg], dim=1)
        return self.fc(out)


# ────────── Training ──────────

def train_rnn(args: argparse.Namespace) -> None:
    # ---- data ----
    df = pd.read_csv(RNN_META_PATH)
    label_enc = LabelEncoder().fit(df["label"])

    class_counts = df["label"].value_counts()
    stratify = df["label"] if class_counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Rare classes (<2 samples) – random split used.")

    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify)
    tr_ds = ChordDatasetRNN(tr_df, RNN_H5_PATH, label_enc)
    va_ds = ChordDatasetRNN(va_df, RNN_H5_PATH, label_enc)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # ---- model ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BiRNNAttn(
        mel_bins=N_MELS,
        hidden=args.hidden,
        n_classes=len(label_enc.classes_),
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    # ---- loss / optim / scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # ---- training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ---- validation ----
        model.eval()
        tot = correct = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)

                tot += y.size(0)
                correct += (preds == y).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y.cpu().tolist())

        val_acc = correct / tot
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        f1_weighted = f1_score(all_targets, all_preds, average="weighted")

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:02d}  val_acc={val_acc:.3f} f1_macro={f1_macro:.3f}  f1_weighted={f1_weighted:.3f}")

    # ---- save ----
    # torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "birnn_attn.pth"))


# ────────── CLI ──────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BiRNN-Attention chord classifier")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    train_rnn(get_args())
