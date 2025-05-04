#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast CNN chord classifier (log‑Mel input).

• Early (2×) down‑sampling in both time & freq
• AMP mixed‑precision training (new torch.amp API)
• Lighter DataLoader settings
Prints one line per epoch, e.g.:
    Epoch 01  val_acc=0.812  f1_macro=0.735  f1_weighted=0.804
"""

from __future__ import annotations

import argparse
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset

# ────────── Paths ──────────
BASE_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(BASE_DIR, "../features_audio_rnn_fast.h5")
META_PATH = os.path.join(BASE_DIR, "../POP909_metadata_rnn.csv")

N_MELS = 64  # feature dimension saved in HDF5


# ────────── Dataset ──────────
class ChordDatasetCNN(Dataset):
    """Return (1, T, 64) log‑Mel tensors."""

    def __init__(self, df: pd.DataFrame, h5_path: str, label_enc: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.le = label_enc
        self._h5: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        feat_idx = self.df.at[idx, "wav_feature_idx"]
        x = torch.from_numpy(self._h5["logmel"][feat_idx][()]).unsqueeze(0)  # (1,T,64)
        y = torch.tensor(
            self.le.transform([self.df.at[idx, "label"]])[0], dtype=torch.long
        )
        return x.float(), y


# ────────── Model ──────────
class ConvBNAct(nn.Sequential):
    """Conv2d → BN → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Optional[Tuple[int, int]] = None,
        dilation: int = 1,
    ):
        if padding is None:
            padding = (
                (kernel[0] // 2) * dilation,
                (kernel[1] // 2) * dilation,
            )
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b, c, *_ = x.shape
        s = self.avg(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, dilation=dilation)
        self.conv2 = ConvBNAct(ch, ch, dilation=dilation)
        self.se = SEBlock(ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return F.relu(out + x, inplace=True)


class CNNChord(nn.Module):
    """Light CNN with early down‑sampling & dilated residual blocks."""

    def __init__(self, n_classes: int, base_ch: int = 64, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            # ↓↓↓ stride=(2,2) → T & Mel both halved
            ConvBNAct(1, base_ch, kernel=(5, 7), stride=(2, 2)),
            ConvBNAct(base_ch, base_ch),
        )
        self.stage1 = ResidualBlock(base_ch, dilation=1)
        self.stage2 = ResidualBlock(base_ch, dilation=2)
        self.stage3 = ResidualBlock(base_ch, dilation=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_ch, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ────────── Train / Eval ──────────
def train_cnn(args: argparse.Namespace):
    # data
    df = pd.read_csv(META_PATH)
    label_enc = LabelEncoder().fit(df["label"])
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Rare classes (<2 samples) – random split applied.")
    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify)

    tr_ds = ChordDatasetCNN(tr_df, H5_PATH, label_enc)
    va_ds = ChordDatasetCNN(va_df, H5_PATH, label_enc)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"  # pin_memory only matters on CUDA

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=pin)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=pin)

    # model & optim
    model = CNNChord(
        n_classes=len(label_enc.classes_),
        base_ch=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # AMP context & scaler
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # train loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        # validation
        model.eval()
        tot = correct = 0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast_ctx:
                    preds = model(x).argmax(1)
                tot += y.size(0)
                correct += (preds == y).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_tgts.extend(y.cpu().tolist())

        val_acc = correct / tot
        f1_macro = f1_score(all_tgts, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_tgts, all_preds, average="weighted", zero_division=0)
        if scheduler:
            scheduler.step()

        # ---- epoch log ----
        print(
            f"Epoch {epoch:02d}  val_acc={val_acc:.3f} f1_macro={f1_macro:.3f}  f1_weighted={f1_weighted:.3f}",
            flush=True,  # ensure immediate output
        )


# ────────── CLI ──────────
def get_args():
    p = argparse.ArgumentParser("Fast CNN chord classifier")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # optimise conv autotune
    train_cnn(get_args())
