# --- imports ----------------------------------------------------------------
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- raw text ---------------------------------------------------------------
raw_text = """
learning_rate: 3e-4
base_channel: 48
dropout: 0.4
val_acc: 0.718, f1_macro: 0.074, f1_weighted: 0.622
--------------------------------------------------
learning_rate: 5e-4
base_channel: 96
dropout: 0.3
val_acc: 0.780, f1_macro: 0.163, f1_weighted: 0.720
--------------------------------------------------
learning_rate: 5e-4
base_channel: 48
dropout: 0.4
val_acc: 0.730, f1_macro: 0.077, f1_weighted: 0.633
--------------------------------------------------
learning_rate: 8e-4
base_channel: 64
dropout: 0.5
val_acc: 0.754, f1_macro: 0.113, f1_weighted: 0.677
--------------------------------------------------
learning_rate: 3e-4
base_channel: 48
dropout: 0.3
val_acc: 0.719, f1_macro: 0.074, f1_weighted: 0.623
--------------------------------------------------
learning_rate: 5e-4
base_channel: 96
dropout: 0.4
val_acc: 0.777, f1_macro: 0.153, f1_weighted: 0.715
--------------------------------------------------
learning_rate: 3e-4
base_channel: 96
dropout: 0.5
val_acc: 0.738, f1_macro: 0.083, f1_weighted: 0.644
--------------------------------------------------
learning_rate: 3e-4
base_channel: 64
dropout: 0.3
val_acc: 0.729, f1_macro: 0.075, f1_weighted: 0.631
--------------------------------------------------
learning_rate: 3e-4
base_channel: 64
dropout: 0.4
val_acc: 0.730, f1_macro: 0.075, f1_weighted: 0.632
--------------------------------------------------
learning_rate: 8e-4
base_channel: 96
dropout: 0.5
val_acc: 0.789, f1_macro: 0.178, f1_weighted: 0.733
--------------------------------------------------
learning_rate: 3e-4
base_channel: 48
dropout: 0.5
val_acc: 0.717, f1_macro: 0.074, f1_weighted: 0.621
--------------------------------------------------
learning_rate: 3e-4
base_channel: 64
dropout: 0.5
val_acc: 0.728, f1_macro: 0.075, f1_weighted: 0.630
--------------------------------------------------
learning_rate: 5e-4
base_channel: 64
dropout: 0.5
val_acc: 0.735, f1_macro: 0.079, f1_weighted: 0.638
--------------------------------------------------
learning_rate: 8e-4
base_channel: 64
dropout: 0.4
val_acc: 0.755, f1_macro: 0.121, f1_weighted: 0.681
--------------------------------------------------
learning_rate: 8e-4
base_channel: 48
dropout: 0.4
val_acc: 0.737, f1_macro: 0.086, f1_weighted: 0.646
--------------------------------------------------
learning_rate: 3e-4
base_channel: 96
dropout: 0.4
val_acc: 0.746, f1_macro: 0.096, f1_weighted: 0.660
--------------------------------------------------
learning_rate: 5e-4
base_channel: 64
dropout: 0.4
val_acc: 0.740, f1_macro: 0.087, f1_weighted: 0.649
--------------------------------------------------
learning_rate: 8e-4
base_channel: 96
dropout: 0.4
val_acc: 0.795, f1_macro: 0.203, f1_weighted: 0.748
--------------------------------------------------
learning_rate: 5e-4
base_channel: 96
dropout: 0.5
val_acc: 0.766, f1_macro: 0.131, f1_weighted: 0.694
--------------------------------------------------
learning_rate: 5e-4
base_channel: 64
dropout: 0.3
val_acc: 0.742, f1_macro: 0.093, f1_weighted: 0.654
--------------------------------------------------
learning_rate: 8e-4
base_channel: 64
dropout: 0.3
val_acc: 0.771, f1_macro: 0.147, f1_weighted: 0.708
--------------------------------------------------
learning_rate: 3e-4
base_channel: 96
dropout: 0.3
val_acc: 0.750, f1_macro: 0.104, f1_weighted: 0.669
--------------------------------------------------
learning_rate: 5e-4
base_channel: 48
dropout: 0.3
val_acc: 0.731, f1_macro: 0.078, f1_weighted: 0.635
--------------------------------------------------
learning_rate: 8e-4
base_channel: 48
dropout: 0.3
val_acc: 0.747, f1_macro: 0.104, f1_weighted: 0.665
--------------------------------------------------
learning_rate: 8e-4
base_channel: 48
dropout: 0.5
val_acc: 0.734, f1_macro: 0.080, f1_weighted: 0.638
--------------------------------------------------
learning_rate: 5e-4
base_channel: 48
dropout: 0.5
val_acc: 0.727, f1_macro: 0.075, f1_weighted: 0.629
--------------------------------------------------
learning_rate: 8e-4
base_channel: 96
dropout: 0.3
val_acc: 0.800, f1_macro: 0.216, f1_weighted: 0.755
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
    elif key == 'base_channel':
        current[key] = int(val)
    elif key == 'dropout':
        current[key] = float(val)
    elif key in ('val_acc', 'f1_macro', 'f1_weighted'):
        # split “val_acc: …, f1_macro: …, f1_weighted: …”
        parts = re.split(r',\s*', line)
        for p in parts:
            k, v = [x.strip() for x in p.split(':')]
            current[k] = float(v)
# catch the last block
if current:
    rows.append(current)

df = pd.DataFrame(rows)

# --- 2. prepare unique values ----------------------------------------------
unique_dropouts = sorted(df['dropout'].unique())
lr_values       = sorted(df['learning_rate'].unique())
bc_values       = sorted(df['base_channel'].unique())

# --- 3. draw 2×3 line plots for val_acc and f1_macro ------------------------
fig, axes = plt.subplots(2, len(unique_dropouts),
                         figsize=(5 * len(unique_dropouts), 8),
                         sharex=True)

metrics = ['val_acc', 'f1_macro']
titles  = ['Accuracy', 'F1-Macro Score']

for i, metric in enumerate(metrics):
    for j, d in enumerate(unique_dropouts):
        ax = axes[i, j]
        df_sub = df[df['dropout'] == d]
        # one line per base_channel
        for bc in bc_values:
            df_line = df_sub[df_sub['base_channel'] == bc].sort_values('learning_rate')
            ax.plot(df_line['learning_rate'],
                    df_line[metric],
                    marker='o',
                    label=f"bc={bc}")
        ax.set_title(f"{titles[i]} (dropout={d})")
        if i == 1:
            ax.set_xlabel("learning_rate")
        if j == 0:
            ax.set_ylabel(titles[i])
        ax.set_xticks(lr_values)
        ax.set_xticklabels([f"{x:.0e}" for x in lr_values])
        if i == 0 and j == len(unique_dropouts) - 1:
            ax.legend(title="base_channel", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle("", fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig("metrics_by_lr_bc.png", dpi=600, bbox_inches='tight')
