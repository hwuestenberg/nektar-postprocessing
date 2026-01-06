#!/usr/bin/env  python3
# Matplotlib setup with latex
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import directory_names, log_file_glob_strs, path_to_directories, save_directory
from npp.scaling_common import iter_scaling_nodes

metrics = [
    "iterations_p",
    "iterations_u",
    "iterations_v",
    "iterations_w",
]
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = save_directory + "scaling" + "".join(f"-{metric}" for metric in metrics)

nodes_ref = int(4)
x_ref = 'ncpus'

xlim = []

base_columns = [
    "scheme",
    "dt",
    "ncpu",
    "nodes",
    "ncpus",
    "global_dof_per_rank",
    "local_dof_per_rank",
]
metric_columns = [
    column
    for metric in metrics
    for column in (f"{metric}-mean", f"{metric}-std", f"{metric}-max", f"{metric}-min")
]

if __name__ == "__main__":

    params = {'text.usetex': True, 'font.size': 10}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ylabel = "Mean iteration counts +/- max and min"
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(which='both', axis='both')

    ax.set_xlabel(r"Number of Processors $N_{P}$")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    df_stat = pd.DataFrame(columns=[*base_columns, *metric_columns])

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, process_file):
        print(f"\treading {case['nodes']}x{case['ncpu']}")

        df = pd.read_pickle(node_dir / process_file)[log_str]
        df = df.apply(pd.to_numeric)

        npoints_remove = int(0.1 * len(df))
        df = df.iloc[npoints_remove:]

        case_data = dict(case)
        for metric in metrics:
            metric_data = df[metric]
            case_data[f"{metric}-mean"] = metric_data.mean()
            case_data[f"{metric}-std"] = metric_data.std()
            case_data[f"{metric}-max"] = metric_data.max()
            case_data[f"{metric}-min"] = metric_data.min()

        df_case = pd.DataFrame([case_data])
        df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

    nodes_min = df_stat['nodes'].min()
    if nodes_ref:
        df_stat = df_stat[df_stat['nodes'] >= nodes_ref]
    else:
        nodes_ref = nodes_min

    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        df_plot = df_stat.loc[df_stat['scheme'] == scheme]

        for metric, metric_color in zip(metrics, TABLEAU_COLORS):
            ls = 'solid'
            mean = df_plot[f'{metric}-mean']
            max_val = df_plot[f'{metric}-max']
            min_val = df_plot[f'{metric}-min']

            x_val = df_plot['ncpus'] if x_ref == 'ncpus' else df_plot['nodes'] / nodes_min
            ax.errorbar(
                x_val,
                mean,
                yerr=[mean - min_val, max_val - mean],
                capsize=4,
                label=scheme + " " + metric,
                color=metric_color,
                linestyle=ls,
            )

    if not xlim:
        xlim = ax.get_xlim()
    ax.set_xlim(xlim)

    reference_plot = df_stat[df_stat['scheme'] == df_stat['scheme'].iloc[0]]
    x_pos = reference_plot['ncpus'].to_numpy()
    x2_lab = reference_plot['global_dof_per_rank'].astype(int).astype(str)
    y_pos = reference_plot[f'{metrics[0]}-mean'] * 1.1

    for x, y, text in zip(x_pos, y_pos, x2_lab):
        ax.text(
            x,
            y,
            text,
            ha='center',
            va='bottom',
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
        )
    ax.set_title("Numbers are global DoF per rank")

    ax.legend()

    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
