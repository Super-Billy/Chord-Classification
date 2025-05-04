# --- imports ----------------------------------------------------------------
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- raw text ---------------------------------------------------------------
raw_text = """
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.898, f1_macro: 0.658, f1_weighted: 0.893
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.910, f1_macro: 0.690, f1_weighted: 0.906
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.912, f1_macro: 0.699, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.878, f1_macro: 0.579, f1_weighted: 0.870
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.911, f1_macro: 0.719, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.913, f1_macro: 0.704, f1_weighted: 0.909
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.912, f1_macro: 0.712, f1_weighted: 0.909
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.908, f1_macro: 0.702, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.910, f1_macro: 0.703, f1_weighted: 0.907
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.911, f1_macro: 0.705, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.901, f1_macro: 0.679, f1_weighted: 0.896
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.912, f1_macro: 0.712, f1_weighted: 0.909
--------------------------------------------------
learning_rate: 1e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.910, f1_macro: 0.708, f1_weighted: 0.907
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.914, f1_macro: 0.711, f1_weighted: 0.910
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.907, f1_macro: 0.696, f1_weighted: 0.904
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.914, f1_macro: 0.710, f1_weighted: 0.911
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.911, f1_macro: 0.705, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.914, f1_macro: 0.719, f1_weighted: 0.910
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.912, f1_macro: 0.699, f1_weighted: 0.909
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.913, f1_macro: 0.717, f1_weighted: 0.910
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 256
val_acc: 0.908, f1_macro: 0.707, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 8e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.908, f1_macro: 0.706, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 4e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.911, f1_macro: 0.706, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 4e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.911, f1_macro: 0.700, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.903, f1_macro: 0.682, f1_weighted: 0.899
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.906, f1_macro: 0.692, f1_weighted: 0.902
--------------------------------------------------
learning_rate: 8e-4
batch_size: 256
hidden_layer_size: 512
val_acc: 0.913, f1_macro: 0.703, f1_weighted: 0.910
--------------------------------------------------
learning_rate: 4e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.908, f1_macro: 0.701, f1_weighted: 0.905
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 256
val_acc: 0.893, f1_macro: 0.648, f1_weighted: 0.887
--------------------------------------------------
learning_rate: 2e-4
batch_size: 256
hidden_layer_size: 256
val_acc: 0.911, f1_macro: 0.711, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 2e-4
batch_size: 1024
hidden_layer_size: 512
val_acc: 0.911, f1_macro: 0.697, f1_weighted: 0.907
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 768
val_acc: 0.914, f1_macro: 0.703, f1_weighted: 0.910
--------------------------------------------------
learning_rate: 2e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.912, f1_macro: 0.705, f1_weighted: 0.908
--------------------------------------------------
learning_rate: 1e-4
batch_size: 512
hidden_layer_size: 512
val_acc: 0.910, f1_macro: 0.713, f1_weighted: 0.906
--------------------------------------------------
learning_rate: 8e-4
batch_size: 1024
hidden_layer_size: 768
val_acc: 0.905, f1_macro: 0.701, f1_weighted: 0.902
--------------------------------------------------
learning_rate: 1e-4
batch_size: 256
hidden_layer_size: 768
val_acc: 0.911, f1_macro: 0.704, f1_weighted: 0.908
--------------------------------------------------
""".strip()

## --- parse the raw data ----------------------------------------------------
rows = []
current = {}
for line in raw_text.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    # boundary between records
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
    elif key == 'val_acc':
        # parse val_acc, f1_macro, f1_weighted all at once
        parts = re.split(r',\s*', line)
        for p in parts:
            k, v = [x.strip() for x in p.split(':', 1)]
            current[k] = float(v)
# add last record
if current:
    rows.append(current)

df = pd.DataFrame(rows)

# sanity check: 36 records and exactly three batch sizes
assert len(df) == 36 and set(df['batch_size']) == {256, 512, 1024}

# prepare unique settings
batch_sizes    = sorted(df['batch_size'].unique())
learning_rates = sorted(df['learning_rate'].unique())
hidden_sizes   = sorted(df['hidden_layer_size'].unique())

# set up subplots
fig, axes = plt.subplots(2, len(batch_sizes),
                         figsize=(5 * len(batch_sizes), 8),
                         sharex=True)

metrics = ['val_acc', 'f1_macro']
titles  = ['Validation Accuracy', 'F1-Macro']

# draw lines on each subplot
for i, metric in enumerate(metrics):
    for j, bs in enumerate(batch_sizes):
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
        ax.set_xticks(learning_rates)
        ax.set_xticklabels([f"{lr:.0e}" for lr in learning_rates])

# add a single legend on the right side of the figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels,
           title="hidden_layer_size",
           bbox_to_anchor=(0.9, 0.5),
           loc='center left')

# adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

# overall title and save/show
plt.suptitle("", fontsize=16, y=1.02)
plt.savefig("crnn.png", dpi=300, bbox_inches='tight')
plt.show()
# plt.savefig("rnn.png", dpi=600, bbox_inches='tight')