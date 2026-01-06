#!/usr/bin/env  python3
# Matplotlib setup with latex
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

params = {'text.usetex': True, 'font.size': 10}
plt.rcParams.update(params)

from config import directory_names, log_file_glob_strs, path_to_directories, save_directory
from npp.scaling_common import iter_scaling_nodes



# # Choose lift [1] or drag [0]
metric = "cpu_time"
# metric = "parallelEfficiency"
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"scaling"
savename = save_directory + savename

nodes_ref = int(4)
# x_ref = 'nodes'
x_ref = 'ncpus'

xlim = []
ylim_su = []
ylim_pe = []



if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax_dt = fig.add_subplot(311)
    ax_su = fig.add_subplot(312, sharex=ax_dt)
    ax_pe = fig.add_subplot(313, sharex=ax_dt)

    ylabel = r"Comp. time per $\Delta t$"
    ax_dt.set_ylabel(ylabel)
    ax_dt.set_yscale('log')

    ylabel = r"Speed Up"# $S = T(N_P=1)/T(N_{P}=P)$"
    ax_su.set_ylabel(ylabel)
    ax_su.set_yscale('linear')

    ylabel = "Parallel Efficiency"
    ax_pe.set_ylabel(ylabel)

    ax_pe.set_xlabel(r"Number of Processors $N_{P}$")
    ax_pe.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dt.grid(which='both', axis='both')
    ax_su.grid(which='both', axis='both')
    ax_pe.grid(which='both', axis='both')


    # Dataframe for gathering statistics for each case
    df_stat = pd.DataFrame(
        columns=[
            "scheme",
            "dt",
            "ncpu",
            "nodes",
            "ncpus",
            "global_dof_per_rank",
            "local_dof_per_rank",
            f"{metric}-mean",
            f"{metric}-std",
        ]
    )

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, process_file):
        print(f"\treading {case['nodes']}x{case['ncpu']}")

        df = pd.read_pickle(node_dir / process_file)[log_str]
        df = df.apply(pd.to_numeric)

        npoints_remove = int(0.1 * len(df))
        df = df.iloc[npoints_remove:]

        metric_data = df[metric]
        case_data = dict(case)
        case_data[f"{metric}-mean"] = metric_data.mean()
        case_data[f"{metric}-std"] = metric_data.std()

        df_case = pd.DataFrame([case_data])
        df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)


    # Filter minimum nodes
    nodes_min = df_stat['nodes'].min()
    if nodes_ref:
        df_stat = df_stat[df_stat['nodes'] >= nodes_ref]
    else:
        nodes_ref = nodes_min

    # Add ideal reference lines
    ideal_su = df_stat[x_ref].unique() / df_stat[x_ref].unique().min()
    ax_su.plot(df_stat[x_ref].unique(), ideal_su, linestyle='--', color='black', label='ideal')
    ax_pe.plot(df_stat[x_ref].unique(), np.ones_like(df_stat[x_ref].unique()), linestyle='--', color='black', label='ideal')


    # Plot by scheme: speedup
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_scheme = df_stat.loc[df_stat['scheme'] == scheme]

        # Comp. time per time step
        dt = df_scheme[f'{metric}-mean']

        # Compute speed-up (strong scaling)
        su = df_scheme[f'{metric}-mean'].iloc[0] / df_scheme[f'{metric}-mean']

        # Compute parallel efficiency (strong scaling)
        pe = 1 - ((ideal_su - su) / su)

        # Choose x-reference
        x_val = df_scheme['ncpus']
        if x_ref == "nodes":
            x_val = df_scheme['nodes'] / nodes_min

        # Add ideal reference for comp. time per dt
        dt_label = 'ideal'
        if scheme_color != list(TABLEAU_COLORS)[0]:
            dt_label = ''
        ax_dt.plot(x_val, dt.iloc[0] / ideal_su, linestyle='--', color='black', label=dt_label)

        # Plot
        ax_dt.plot(x_val, dt, marker='o', label=scheme)
        ax_su.plot(x_val, su, marker='o', label=scheme)
        ax_pe.plot(x_val, pe, marker='o', label=scheme)
        # ax.errorbar(x_val, su, df_plot[f'{metric}-std'],
        #             color=scheme_color, capsize=4)

    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax_su.get_xlim()
        xlim = ax_pe.get_xlim()
    if not ylim_su:
        ylim_su = ax_su.get_ylim()
    if not ylim_pe:
        ylim_pe = ax_pe.get_ylim()
    ax_su.set_xlim(xlim)
    ax_su.set_ylim(ylim_su)
    ax_pe.set_ylim(ylim_pe)

    # # Create top axis with dof_per_rank
    # positions where you want the top labels to appear:
    x_pos = df_scheme['ncpus'].to_numpy()  # <- your primary x positions
    x2_lab = df_scheme['global_dof_per_rank'].astype(int).astype(str)
    y_pos = df_scheme[f'{metric}-mean'] * 1.1

    for x, y, text in zip(x_pos, y_pos, x2_lab):
        ax_dt.text(x, y, text, ha='center', va='bottom', fontsize=8,
                   color='black',
                   bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    ax_dt.set_title("Numbers are global DoF per rank")
        # ax_dt.annotate(text, xy=(x, y), xytext=(0, 10), textcoords='offset points',
        #                ha='center', va='bottom', fontsize=8)

    # ax2 = ax_dt.twiny()
    # ax2.set_xlim(ax_dt.get_xlim())  # keep same limits
    # ax2.xaxis.set_major_locator(FixedLocator(x_pos))  # same number as labels
    # ax2.set_xticklabels(x2_lab)  # now lengths match
    # ax2.set_xlabel('Global DoF per rank')

    ax_dt.legend()


    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
