#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer‑Attention chord classifier with concise epoch logging **and**
macro / weighted F1 read‑outs.

This replaces the original two‑layer bi‑directional vanilla RNN backbone with a
Transformer encoder.  Each epoch prints one line

    Epoch 01  val_acc=0.792  f1_macro=0.701  f1_weighted=0.788

Place this file in the RNN/ directory alongside:
    - features_audio_rnn_fast.h5
    - POP909_metadata_rnn.csv

Run:
    python transformer_train.py --help
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from typing import Optional

import h5py
import numpy as np
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
N_MELS = 64  # log‑mel feature dimension saved in HDF5


class ChordDatasetRNN(Dataset):
    """Lazy HDF5 loader; avoids pickling HDF handles in DataLoader workers."""

    def __init__(self, df: pd.DataFrame, h5_path: str, label_enc: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.le = label_enc
        self._h5: Optional[h5py.File] = None  # opened lazily in __getitem__

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        feat_idx = self.df.at[idx, "wav_feature_idx"]
        # (T, 64) -> float32 tensor
        x = torch.from_numpy(self._h5["logmel"][feat_idx][()]).float()
        y = torch.tensor(
            self.le.transform([self.df.at[idx, "label"]])[0], dtype=torch.long
        )
        return x, y


# ────────── Model ──────────
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no dropout for simplicity)."""

    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class ChordTransformer(nn.Module):
    """Transformer encoder for frame‑level chord classification."""

    def __init__(
        self,
        *,
        mel_bins: int = N_MELS,
        d_model: int = 512,
        nhead: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 4,
        n_classes: int = 320,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(mel_bins, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Linear(d_model, n_classes)

        # Init cls token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, mel_bins)
            lengths: optional (B,) tensor of true lengths **excluding** CLS
        """
        B = x.size(0)
        # prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls_tok, self.input_proj(x)], dim=1)  # (B,T+1,D)
        x = self.pos_enc(x)

        if lengths is not None:
            pad_mask = (
                torch.arange(x.size(1) - 1, device=x.device)[None, :] >= lengths[:, None]
            )
            pad_mask = torch.cat(
                [torch.zeros(B, 1, device=x.device, dtype=torch.bool), pad_mask], dim=1
            )
        else:
            pad_mask = None

        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,T+1,D)
        out = self.norm(h[:, 0])  # CLS representation
        return self.fc(out)


# ────────── Training ──────────
def train_transformer(args: argparse.Namespace) -> None:  # noqa: C901
    # ---- data ----
    df = pd.read_csv(RNN_META_PATH)
    label_enc = LabelEncoder().fit(df["label"])

    # stratified split only when every class has ≥2 samples
    class_counts = df["label"].value_counts()
    stratify = df["label"] if class_counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Rare classes (<2 samples) – random split used.")

    tr_df, va_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=stratify
    )

    tr_ds = ChordDatasetRNN(tr_df, RNN_H5_PATH, label_enc)
    va_ds = ChordDatasetRNN(va_df, RNN_H5_PATH, label_enc)

    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # ---- model ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ChordTransformer(
        mel_bins=N_MELS,
        d_model=args.hidden,
        nhead=args.heads,
        ff_dim=args.ff_dim,
        num_layers=args.layers,
        n_classes=len(label_enc.classes_),
        dropout=args.dropout,
    ).to(device)

    # ---- loss / optim / scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.scheduler == "cosine":
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
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

        # validation
        model.eval()
        tot, correct = 0, 0
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                tot += y.size(0)
                correct += (preds == y).sum().item()
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
        val_acc = correct / tot
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:02d}  val_acc={val_acc:.3f} f1_macro={f1_macro:.3f}  f1_weighted={f1_weighted:.3f}")

    # ---- save ----
    # torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "transformer_attn.pth"))


# ────────── CLI ──────────
def get_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser("Transformer-Attention chord classifier")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--heads", type=int, default=8, help="number of attention heads")
    p.add_argument("--hidden", type=int, default=512, help="Transformer d_model")
    p.add_argument("--ff-dim", type=int, default=2048, help="feed-forward dim")
    p.add_argument("--layers", type=int, default=4, help="encoder layers")

    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--device", default="cuda:2")
    return p.parse_args()


if __name__ == "__main__":
    train_transformer(get_args())
