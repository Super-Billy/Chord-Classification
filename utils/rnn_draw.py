# --- imports ----------------------------------------------------------------
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- raw text ---------------------------------------------------------------
raw_text = """
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.876, f1_macro: 0.628, f1_weighted: 0.871
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.838, f1_macro: 0.494, f1_weighted: 0.825
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.875, f1_macro: 0.630, f1_weighted: 0.868
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.868, f1_macro: 0.611, f1_weighted: 0.862
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.854, f1_macro: 0.539, f1_weighted: 0.843
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.880, f1_macro: 0.642, f1_weighted: 0.875
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.842, f1_macro: 0.523, f1_weighted: 0.832
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.861, f1_macro: 0.565, f1_weighted: 0.852
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.884, f1_macro: 0.661, f1_weighted: 0.879
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.843, f1_macro: 0.506, f1_weighted: 0.830
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.825, f1_macro: 0.440, f1_weighted: 0.806
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.861, f1_macro: 0.582, f1_weighted: 0.853
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.879, f1_macro: 0.658, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.871, f1_macro: 0.614, f1_weighted: 0.865
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.856, f1_macro: 0.556, f1_weighted: 0.847
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.887, f1_macro: 0.642, f1_weighted: 0.881
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.872, f1_macro: 0.620, f1_weighted: 0.865
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.879, f1_macro: 0.632, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.862, f1_macro: 0.579, f1_weighted: 0.853
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.867, f1_macro: 0.613, f1_weighted: 0.861
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.868, f1_macro: 0.596, f1_weighted: 0.861
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.876, f1_macro: 0.631, f1_weighted: 0.870
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.840, f1_macro: 0.481, f1_weighted: 0.825
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.874, f1_macro: 0.612, f1_weighted: 0.867
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.869, f1_macro: 0.615, f1_weighted: 0.862
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.855, f1_macro: 0.578, f1_weighted: 0.847
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.869, f1_macro: 0.624, f1_weighted: 0.862
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.816, f1_macro: 0.375, f1_weighted: 0.792
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.847, f1_macro: 0.535, f1_weighted: 0.837
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.885, f1_macro: 0.659, f1_weighted: 0.880
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.854, f1_macro: 0.555, f1_weighted: 0.845
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.871, f1_macro: 0.625, f1_weighted: 0.865
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.854, f1_macro: 0.572, f1_weighted: 0.845
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.865, f1_macro: 0.579, f1_weighted: 0.857
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.831, f1_macro: 0.459, f1_weighted: 0.815
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.851, f1_macro: 0.531, f1_weighted: 0.839
--------------------------------------------------
""".strip()

# --- 1. parse ---------------------------------------------------------------
rows = []
current = {}
for line in raw_text.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('---'):
        if current:
            rows.append(current)
            current = {}
        continue
    key, val = [x.strip() for x in line.split(':', 1)]
    if key == 'learning_rate':
        current[key] = float(val)
    elif key == 'batch_size':
        current[key] = int(val)
    elif key == 'hidden_layer_size':
        current[key] = int(val)
    elif key in ('val_acc', 'f1_macro', 'f1_weighted'):
        parts = re.split(r',\s*', line)
        for p in parts:
            k, v = [x.strip() for x in p.split(':')]
            current[k] = float(v)
# catch the last block
if current:
    rows.append(current)

df = pd.DataFrame(rows)

# --- 2. prepare unique values ----------------------------------------------
unique_batch_sizes = sorted(df['batch_size'].unique())
lr_values           = sorted(df['learning_rate'].unique())
hidden_sizes        = sorted(df['hidden_layer_size'].unique())

# --- 3. draw 2×3 line plots for val_acc and f1_macro ------------------------
fig, axes = plt.subplots(2, len(unique_batch_sizes),
                         figsize=(5 * len(unique_batch_sizes), 8),
                         sharex=True)

metrics = ['val_acc', 'f1_macro']
titles  = ['Validation Accuracy', 'F1-Macro']

for i, metric in enumerate(metrics):
    for j, bs in enumerate(unique_batch_sizes):
        ax = axes[i, j]
        df_sub = df[df['batch_size'] == bs]
        for hs in hidden_sizes:
            df_line = df_sub[df_sub['hidden_layer_size'] == hs].sort_values('learning_rate')
            ax.plot(df_line['learning_rate'],
                    df_line[metric],
                    marker='o',
                    label=f"hidden={hs}")
        ax.set_title(f"{titles[i]} (batch_size={bs})")
        if i == 1:
            ax.set_xlabel("learning_rate")
        if j == 0:
            ax.set_ylabel(titles[i])
        ax.set_xticks(lr_values)
        ax.set_xticklabels([f"{x:.0e}" for x in lr_values])
        if i == 0 and j == len(unique_batch_sizes) - 1:
            ax.legend(title="hidden_layer_size", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle("", fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig("rnn.png", dpi=600, bbox_inches='tight')
