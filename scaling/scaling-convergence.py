#!/usr/bin/env  python3
# Matplotlib setup with latex
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import directory_names, log_file_glob_strs, path_to_directories
from npp.scaling_common import iter_scaling_nodes

metric = "cpu_time"
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"scaling"
nodes_ref = int(32)
x_ref = 'ncpus'

xlim = []
ylim = []

if __name__ == "__main__":

    params = {'text.usetex': True, 'font.size': 10}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)

    ylabel = fr"{metric}"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"Samples (time steps)")
    ax.grid(which='both', axis='both')

    df_stat = pd.DataFrame(columns=["scheme", "dt", "ncpu", "nodes", f"{metric}-mean", f"{metric}-std"])

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, process_file):
        print(f"\treading {case['nodes']}x{case['ncpu']}")

        df = pd.read_pickle(node_dir / process_file)[log_str]
        df = df.apply(pd.to_numeric)

        npoints_remove = int(0.1 * len(df))
        df = df.iloc[npoints_remove:]

        metricData = df[metric]
        mean = metricData.mean()
        rolling_mean = metricData.expanding().mean()
        rolling_std = rolling_mean.rolling(window=100).std()

        label = f"{case['label']} {case['nodes']}x{case['ncpu']}"
        ax.errorbar(rolling_mean.index, rolling_mean, yerr=rolling_std, label=label, alpha=0.8)

        tolerance = 1e-3
        window_diff = npoints_remove
        diff = abs(mean - rolling_mean.iloc[-window_diff:-1]) / mean
        max_diff = diff.max()
        if max_diff < tolerance:
            print(f"\tConverged (Δ = {100 * max_diff:2.2f} %)")
        else:
            print(f"\tNOT converged (Δ = {100 * max_diff:2.2f} %)")

        case_data = {
            "scheme": case["scheme"],
            "dt": case["dt"],
            "ncpu": case["ncpu"],
            "nodes": case["nodes"],
            f"{metric}-mean": mean,
            f"{metric}-std": metricData.std(),
        }
        df_stat = pd.concat([df_stat, pd.DataFrame([case_data])], ignore_index=True)

    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    plt.show()
