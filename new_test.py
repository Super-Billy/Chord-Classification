
#!/usr/bin/env python3
"""
Quick test for top-K chord-label classification using LogisticRegression.

• 读取 POP909_metadata.csv，过滤已有特征行
• 选出出现最频繁的 top_k 标签
• 随机抽样 n_sub 样本
• 标准化 + 多分类 LogisticRegression (saga)
• K-fold (cv) 评估

Author: you
Date  : 2025-04-28
"""

import numpy as np
import pandas as pd
import h5py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# -------- 配置 --------
META_CSV = "POP909_metadata.csv"
H5_FILE  = "features_audio.h5"
DS_NAME  = "mel128"
TOP_K    = 80     # 只测试最频繁的 TOP_K 类别
N_SUB    = 2000   # 最多抽取样本数
CV       = 3      # 3-fold 交叉验证
# ----------------------

def main():
    # 1. 读取并过滤已有特征的行
    df = pd.read_csv(META_CSV)
    df["wav_feature_idx"] = df["wav_feature_idx"].astype(int)
    df = df[df["wav_feature_idx"] >= 0].copy()
    print(f"Total samples with features: {len(df)}")

    # 2. 选 top-K 最频繁标签
    freq = df["label"].value_counts()
    top_labels = freq.nlargest(TOP_K).index.tolist()
    print(f"Top {TOP_K} labels:", top_labels)
    df = df[df["label"].isin(top_labels)].reset_index(drop=True)
    print(f"Samples after filtering to top {TOP_K} labels: {len(df)}")

    # 3. 随机抽样子集
    n = min(N_SUB, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"Using a random subset of {n} samples for testing.")

    # 4. 加载 HDF5 特征
    with h5py.File(H5_FILE, "r") as h5:
        X_all = h5[DS_NAME][:]
    idxs = df["wav_feature_idx"].to_numpy(dtype=int)
    X = X_all[idxs]
    print(f"Feature matrix shape: {X.shape}")

    # 5. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    print(f"Number of classes: {len(le.classes_)}")

    # 6. 构造 Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1
        ))
    ])

    # 7. K-fold 交叉验证
    print(f"Running {CV}-fold cross-validation...")
    scores = cross_val_score(
        pipe, X, y,
        cv=CV,
        scoring="accuracy",
        n_jobs=-1
    )
    print(f"Top-{TOP_K} classification accuracy ({n} samples, {CV}-fold): "
          f"{scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

if __name__ == "__main__":
    main()
