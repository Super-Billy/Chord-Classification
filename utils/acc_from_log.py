#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traverse a directory of .log files (optionally filtered by filename substring)
to find the maximum val_acc at epoch 10 and epoch 20.
Prints the epoch, value, and file path for each.
Usage:
    python find_max_val_acc.py <root_directory> <name_filter>
If <name_filter> is a non-empty string, only files whose names contain that
substring will be processed. To process all .log files, pass an empty string:
    python find_max_val_acc.py /path/to/logs ""
"""

import os
from glob import glob
import re
import sys

def find_max_val_acc(root_dir, name_filter):
    # Regex matches lines like "Epoch 10  val_acc=0.904"
    pattern = re.compile(r'Epoch\s+(\d+)\s+val_acc=(\d+\.\d+)')
    epoch10_values = []  # list of (val_acc, filepath)
    epoch20_values = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # only consider .log files
            if not filename.endswith('.log'):
                continue
            # if a filter is provided, skip files that don't contain it
            if name_filter and name_filter not in filename:
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        m = pattern.search(line)
                        if not m:
                            continue
                        epoch = int(m.group(1))
                        acc = float(m.group(2))
                        if epoch == 5:
                            epoch10_values.append((acc, filepath))
                        elif epoch == 10:
                            epoch20_values.append((acc, filepath))
            except Exception as e:
                print(f"Warning: could not read {filepath}: {e}", file=sys.stderr)

    # report the best val_acc for epoch 10
    if epoch10_values:
        best10, file10 = max(epoch10_values, key=lambda x: x[0])
        print(f"Epoch 5: max val_acc = {best10:.3f}, file = {file10}")
    else:
        print("No epoch 5 entries found.")

    # report the best val_acc for epoch 20
    if epoch20_values:
        best20, file20 = max(epoch20_values, key=lambda x: x[0])
        print(f"Epoch 10: max val_acc = {best20:.3f}, file = {file20}")
    else:
        print("No epoch 10 entries found.")



def extract_metrics_from_logs(log_dir):
    """
    For each .log file in log_dir, parse the filename for hyperparameters
    and read the file to extract val_acc, f1_macro, and f1_weighted at Epoch 15.
    Prints only non-None hyperparameters and the metrics (no filename line).
    """
    # mapping from abbreviation to full parameter name
    abbr_map = [
        ("lr", "learning_rate"),
        ("ch", "base_channel"),
        ("do", "dropout"),
        ("bs", "batch_size"),
        ("hid", "hidden_layer_size"),
        ("h",  "hidden_layer_size"),
        ("d",  "d_model"),
        ("L",  "layer_number"),
    ]

    # find all .log files
    for filepath in glob(os.path.join(log_dir, "*.log")):
        filename = os.path.basename(filepath)
        name_part = filename[:-4]  # strip ".log"

        # parse hyperparameters from filename
        hypers = {}
        for token in name_part.split("_"):
            for abbr, full in abbr_map:
                if token.startswith(abbr):
                    value = token[len(abbr):]
                    hypers[full] = value
                    break

        # ensure all keys exist
        for _, full in abbr_map:
            hypers.setdefault(full, None)

        # read file and search for Epoch 15 line
        val_acc = f1_macro = f1_weighted = "N/A"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("Epoch 15"):
                    m = re.search(
                        r"val_acc=(?P<val_acc>\S+)\s+f1_macro=(?P<f1_macro>\S+)\s+f1_weighted=(?P<f1_weighted>\S+)",
                        line
                    )
                    if m:
                        val_acc = m.group("val_acc")
                        f1_macro = m.group("f1_macro")
                        f1_weighted = m.group("f1_weighted")
                    break

        # print only non-None hyperparameters
        for param, val in hypers.items():
            if val is not None:
                print(f"{param}: {val}")
        # print metrics
        print(f"val_acc: {val_acc}, f1_macro: {f1_macro}, f1_weighted: {f1_weighted}")
        print("-" * 50)





