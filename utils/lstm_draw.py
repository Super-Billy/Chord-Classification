# Updated code: parse new data and plot as line charts (learning_rate vs metrics, lines per hidden size, subplots per batch size)

import re
import pandas as pd
import matplotlib.pyplot as plt

# --- raw data ---------------------------------------------------------------
raw_text = """
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.900, f1_macro: 0.691, f1_weighted: 0.895
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.861, f1_macro: 0.547, f1_weighted: 0.851
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.897, f1_macro: 0.698, f1_weighted: 0.893
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.889, f1_macro: 0.657, f1_weighted: 0.884
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.883, f1_macro: 0.629, f1_weighted: 0.877
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.902, f1_macro: 0.690, f1_weighted: 0.898
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.874, f1_macro: 0.632, f1_weighted: 0.869
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.891, f1_macro: 0.648, f1_weighted: 0.886
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.902, f1_macro: 0.690, f1_weighted: 0.898
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.868, f1_macro: 0.582, f1_weighted: 0.860
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.845, f1_macro: 0.494, f1_weighted: 0.831
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.884, f1_macro: 0.640, f1_weighted: 0.880
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.901, f1_macro: 0.699, f1_weighted: 0.897
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.900, f1_macro: 0.676, f1_weighted: 0.895
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.881, f1_macro: 0.631, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.905, f1_macro: 0.699, f1_weighted: 0.900
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.880, f1_macro: 0.654, f1_weighted: 0.875
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.905, f1_macro: 0.704, f1_weighted: 0.901
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.880, f1_macro: 0.654, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.892, f1_macro: 0.670, f1_weighted: 0.887
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.896, f1_macro: 0.671, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.902, f1_macro: 0.688, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.863, f1_macro: 0.556, f1_weighted: 0.852
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.902, f1_macro: 0.676, f1_weighted: 0.897
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.899, f1_macro: 0.678, f1_weighted: 0.894
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.877, f1_macro: 0.633, f1_weighted: 0.871
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.896, f1_macro: 0.679, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.834, f1_macro: 0.430, f1_weighted: 0.817
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.878, f1_macro: 0.629, f1_weighted: 0.871
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.902, f1_macro: 0.703, f1_weighted: 0.898
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.880, f1_macro: 0.633, f1_weighted: 0.874
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.896, f1_macro: 0.682, f1_weighted: 0.892
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.881, f1_macro: 0.644, f1_weighted: 0.876
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.894, f1_macro: 0.666, f1_weighted: 0.889
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.856, f1_macro: 0.538, f1_weighted: 0.845
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.879, f1_macro: 0.615, f1_weighted: 0.873
--------------------------------------------------
""".strip()

# --- parse data blocks -----------------------------------------------------
rows = []
current = {}
for line in raw_text.splitlines():
    line = line.strip()
    # skip separators
    if not line or line.startswith('-'):
        continue
    # start new block on learning_rate
    if line.startswith('learning_rate:') and current:
        rows.append(current)
        current = {}
    key, val = [x.strip() for x in line.split(':', 1)]
    if key == 'learning_rate':
        current['learning_rate'] = float(val)
    elif key == 'batch_size':
        current['batch_size'] = int(val)
    elif key == 'hidden_layer_size':
        current['hidden_layer_size'] = int(val)
    elif key == 'val_acc':
        parts = re.split(r',\s*', line)
        for part in parts:
            k, v = [x.strip() for x in part.split(':', 1)]
            current[k] = float(v)
# append the last record
if current:
    rows.append(current)

df = pd.DataFrame(rows)

# --- plot line charts -------------------------------------------------------
unique_bs = sorted(df['batch_size'].unique())
unique_hl = sorted(df['hidden_layer_size'].unique())
lr_values = sorted(df['learning_rate'].unique())

fig, axes = plt.subplots(2, len(unique_bs),
                         figsize=(5 * len(unique_bs), 8),
                         sharex=True)

metrics = ['val_acc', 'f1_macro']
titles  = ['Validation Accuracy', 'F1-Macro Score']

for i, metric in enumerate(metrics):
    for j, bs in enumerate(unique_bs):
        ax = axes[i, j]
        df_sub = df[df['batch_size'] == bs]
        for hl in unique_hl:
            df_line = df_sub[df_sub['hidden_layer_size'] == hl].sort_values('learning_rate')
            ax.plot(df_line['learning_rate'],
                    df_line[metric],
                    marker='o',
                    label=f"hidden={hl}")
        ax.set_title(f"{titles[i]} (batch_size={bs})")
        if i == len(metrics) - 1:
            ax.set_xlabel("Learning Rate")
        if j == 0:
            ax.set_ylabel(titles[i])
        ax.set_xticks(lr_values)
        ax.set_xticklabels([f"{x:.0e}" for x in lr_values])
        # legend only on first row, last column
        if i == 0 and j == len(unique_bs) - 1:
            ax.legend(title="Hidden Size", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig("metrics_by_lr_hidden_per_batch.png", dpi=600, bbox_inches='tight')
