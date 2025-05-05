# --- imports ----------------------------------------------------------------
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Put your raw text block here – unchanged from what the training script prints
# ---------------------------------------------------------------------------
raw_text = """
learning_rate: 2e-4
d_model: 256
layer_number: 2
batch_size: 128
val_acc: 0.876, f1_macro: 0.586, f1_weighted: 0.869
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 4
batch_size: 512
val_acc: 0.844, f1_macro: 0.443, f1_weighted: 0.830
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 4
batch_size: 256
val_acc: 0.908, f1_macro: 0.692, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 2
batch_size: 512
val_acc: 0.884, f1_macro: 0.611, f1_weighted: 0.878
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 2
batch_size: 128
val_acc: 0.901, f1_macro: 0.683, f1_weighted: 0.898
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 4
batch_size: 64
val_acc: 0.910, f1_macro: 0.701, f1_weighted: 0.907
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 4
batch_size: 128
val_acc: 0.906, f1_macro: 0.689, f1_weighted: 0.902
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 4
batch_size: 128
val_acc: 0.883, f1_macro: 0.610, f1_weighted: 0.878
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 4
batch_size: 64
val_acc: 0.907, f1_macro: 0.682, f1_weighted: 0.904
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 2
batch_size: 256
val_acc: 0.902, f1_macro: 0.687, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 2
batch_size: 64
val_acc: 0.871, f1_macro: 0.592, f1_weighted: 0.864
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 4
batch_size: 64
val_acc: 0.895, f1_macro: 0.652, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 4
batch_size: 128
val_acc: 0.897, f1_macro: 0.678, f1_weighted: 0.894
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 4
batch_size: 512
val_acc: 0.879, f1_macro: 0.601, f1_weighted: 0.873
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 4
batch_size: 64
val_acc: 0.884, f1_macro: 0.626, f1_weighted: 0.880
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 4
batch_size: 512
val_acc: 0.905, f1_macro: 0.686, f1_weighted: 0.902
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 2
batch_size: 256
val_acc: 0.881, f1_macro: 0.602, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 2
batch_size: 512
val_acc: 0.849, f1_macro: 0.449, f1_weighted: 0.833
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 2
batch_size: 128
val_acc: 0.859, f1_macro: 0.539, f1_weighted: 0.850
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 2
batch_size: 128
val_acc: 0.900, f1_macro: 0.678, f1_weighted: 0.897
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 4
batch_size: 256
val_acc: 0.909, f1_macro: 0.693, f1_weighted: 0.906
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 2
batch_size: 64
val_acc: 0.893, f1_macro: 0.659, f1_weighted: 0.889
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 2
batch_size: 64
val_acc: 0.886, f1_macro: 0.620, f1_weighted: 0.880
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 2
batch_size: 512
val_acc: 0.859, f1_macro: 0.527, f1_weighted: 0.850
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 4
batch_size: 256
val_acc: 0.885, f1_macro: 0.607, f1_weighted: 0.879
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 4
batch_size: 64
val_acc: 0.898, f1_macro: 0.677, f1_weighted: 0.894
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 2
batch_size: 256
val_acc: 0.903, f1_macro: 0.682, f1_weighted: 0.900
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 2
batch_size: 128
val_acc: 0.898, f1_macro: 0.675, f1_weighted: 0.894
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 2
batch_size: 64
val_acc: 0.893, f1_macro: 0.656, f1_weighted: 0.889
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 2
batch_size: 512
val_acc: 0.898, f1_macro: 0.668, f1_weighted: 0.895
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 4
batch_size: 256
val_acc: 0.895, f1_macro: 0.665, f1_weighted: 0.891
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 2
batch_size: 64
val_acc: 0.894, f1_macro: 0.647, f1_weighted: 0.890
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 4
batch_size: 512
val_acc: 0.907, f1_macro: 0.699, f1_weighted: 0.903
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 4
batch_size: 256
val_acc: N/A, f1_macro: N/A, f1_weighted: N/A
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 4
batch_size: 512
val_acc: 0.892, f1_macro: 0.638, f1_weighted: 0.887
--------------------------------------------------
learning_rate: 1e-4
d_model: 512
layer_number: 2
batch_size: 128
val_acc: 0.886, f1_macro: 0.639, f1_weighted: 0.881
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 4
batch_size: 64
val_acc: 0.045, f1_macro: 0.000, f1_weighted: 0.004
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 4
batch_size: 64
val_acc: 0.904, f1_macro: 0.677, f1_weighted: 0.901
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 2
batch_size: 256
val_acc: 0.906, f1_macro: 0.691, f1_weighted: 0.903
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 4
batch_size: 128
val_acc: 0.908, f1_macro: 0.699, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 2
batch_size: 64
val_acc: 0.903, f1_macro: 0.687, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 2
batch_size: 256
val_acc: 0.845, f1_macro: 0.451, f1_weighted: 0.830
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 4
batch_size: 512
val_acc: 0.897, f1_macro: 0.642, f1_weighted: 0.893
--------------------------------------------------
learning_rate: 4e-4
d_model: 512
layer_number: 2
batch_size: 64
val_acc: 0.895, f1_macro: 0.670, f1_weighted: 0.891
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 4
batch_size: 512
val_acc: 0.900, f1_macro: 0.680, f1_weighted: 0.897
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 2
batch_size: 256
val_acc: 0.887, f1_macro: 0.635, f1_weighted: 0.881
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 2
batch_size: 256
val_acc: 0.866, f1_macro: 0.523, f1_weighted: 0.856
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 4
batch_size: 256
val_acc: 0.899, f1_macro: 0.674, f1_weighted: 0.895
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 2
batch_size: 64
val_acc: 0.904, f1_macro: 0.695, f1_weighted: 0.901
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 2
batch_size: 512
val_acc: 0.903, f1_macro: 0.679, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 2
batch_size: 512
val_acc: 0.894, f1_macro: 0.644, f1_weighted: 0.889
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 2
batch_size: 256
val_acc: 0.893, f1_macro: 0.658, f1_weighted: 0.889
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 2
batch_size: 64
val_acc: 0.895, f1_macro: 0.680, f1_weighted: 0.891
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 2
batch_size: 512
val_acc: 0.875, f1_macro: 0.589, f1_weighted: 0.868
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 4
batch_size: 128
val_acc: 0.894, f1_macro: 0.677, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 4
batch_size: 128
val_acc: 0.898, f1_macro: 0.661, f1_weighted: 0.894
--------------------------------------------------
learning_rate: 1e-4
d_model: 768
layer_number: 2
batch_size: 256
val_acc: 0.887, f1_macro: 0.649, f1_weighted: 0.882
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 2
batch_size: 128
val_acc: 0.896, f1_macro: 0.663, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 4
batch_size: 256
val_acc: 0.904, f1_macro: 0.689, f1_weighted: 0.900
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 4
batch_size: 64
val_acc: 0.907, f1_macro: 0.704, f1_weighted: 0.904
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 4
batch_size: 256
val_acc: 0.863, f1_macro: 0.532, f1_weighted: 0.854
--------------------------------------------------
learning_rate: 2e-4
d_model: 256
layer_number: 4
batch_size: 512
val_acc: 0.865, f1_macro: 0.493, f1_weighted: 0.854
--------------------------------------------------
learning_rate: 2e-4
d_model: 512
layer_number: 2
batch_size: 128
val_acc: 0.899, f1_macro: 0.679, f1_weighted: 0.896
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 4
batch_size: 64
val_acc: 0.888, f1_macro: 0.635, f1_weighted: 0.883
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 4
batch_size: 128
val_acc: 0.045, f1_macro: 0.000, f1_weighted: 0.004
--------------------------------------------------
learning_rate: 1e-4
d_model: 256
layer_number: 2
batch_size: 512
val_acc: 0.831, f1_macro: 0.366, f1_weighted: 0.809
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 4
batch_size: 128
val_acc: 0.909, f1_macro: 0.697, f1_weighted: 0.906
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 4
batch_size: 128
val_acc: 0.903, f1_macro: 0.660, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 4e-4
d_model: 768
layer_number: 4
batch_size: 256
val_acc: 0.194, f1_macro: 0.010, f1_weighted: 0.094
--------------------------------------------------
learning_rate: 2e-4
d_model: 768
layer_number: 2
batch_size: 128
val_acc: 0.906, f1_macro: 0.690, f1_weighted: 0.903
--------------------------------------------------
learning_rate: 4e-4
d_model: 256
layer_number: 2
batch_size: 512
val_acc: 0.870, f1_macro: 0.560, f1_weighted: 0.863
--------------------------------------------------

""".strip()

# --- 1. parse ---------------------------------------------------------------
rows, current = [], {}
float_or_nan = lambda x: float(x) if x not in ('N/A', 'nan', 'NaN') else math.nan

for line in raw_text.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('---'):                 # separator between blocks
        if current:
            rows.append(current)
            current = {}
        continue

    key, val = [x.strip() for x in line.split(':', 1)]

    # hyper‑parameters
    if key == 'learning_rate':
        current[key] = float(val)
    elif key == 'd_model':
        current[key] = int(val)
    elif key == 'layer_number':
        current[key] = int(val)
    elif key == 'batch_size':
        current[key] = int(val)

    # three metrics may appear on one line
    elif key in ('val_acc', 'f1_macro', 'f1_weighted'):
        for part in re.split(r',\s*', line):
            k, v = [x.strip() for x in part.split(':', 1)]
            current[k] = float_or_nan(v)

# catch the last block
if current:
    rows.append(current)

df = pd.DataFrame(rows).dropna(subset=['val_acc'])  # drop rows with N/A scores
df.sort_values(['learning_rate', 'd_model',
                'layer_number', 'batch_size'],
               inplace=True)

# --- 2. compute unique values ----------------------------------------------
unique_lrs     = sorted(df['learning_rate'].unique())
unique_dmodels = sorted(df['d_model'].unique())
unique_layers  = sorted(df['layer_number'].unique())
unique_batches = sorted(df['batch_size'].unique())

# --- 3. plotting helper -----------------------------------------------------
def plot_metric(metric: str, title: str, file_name: str) -> None:
    """
    Grid layout (2 rows × 4 cols):
        rows  = layer_number        (len == 2)
        cols  = batch_size          (len == 4)
        line  = d_model             (≤ 3 markers)
        x‑axis= learning_rate
    """
    n_rows, n_cols = len(unique_layers), len(unique_batches)   # 2 × 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey=True
    )

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*'] * 3

    for i, lyr in enumerate(unique_layers):                # row index
        for j, batch in enumerate(unique_batches):         # col index
            ax = axes[i, j] if n_rows > 1 else axes[j]
            subset = df[
                (df['layer_number'] == lyr) &
                (df['batch_size']   == batch)
            ]
            for k, d_model in enumerate(unique_dmodels):
                line_df = subset[subset['d_model'] == d_model]\
                              .sort_values('learning_rate')
                if line_df.empty:
                    continue
                ax.plot(
                    line_df['learning_rate'],
                    line_df[metric],
                    marker=markers[k],
                    label=f"d_model={d_model}"
                )

            ax.set_title(f"{title}\nlayers={lyr}, bs={batch}")
            ax.set_xticks(unique_lrs)
            ax.set_xticklabels([f"{x:.0e}" for x in unique_lrs], rotation=45)
            if j == 0:
                ax.set_ylabel(title)
            if i == n_rows - 1:
                ax.set_xlabel("learning_rate")

            # 统一图例：放在最后一列、第一行的右侧
            if (i == 0) and (j == n_cols - 1):
                ax.legend(
                    title="d_model",
                    bbox_to_anchor=(1.05, 0.0, 0.2, 1.0),
                    loc="upper left",
                    borderaxespad=0.
                )

    plt.tight_layout()
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
# --- 4. draw separate figures for each metric ------------------------------
plot_metric("val_acc",   "Validation Accuracy",   "val_acc_grid.png")
plot_metric("f1_macro",  "F1‑Macro",              "f1_macro_grid.png")
