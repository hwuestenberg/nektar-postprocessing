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

from config import directory_names, log_file_glob_strs, path_to_directories, save_directory
from npp.scaling_common import iter_scaling_nodes, read_timer_output



# metric = "Execute"
# metric = "ViscousSolve"
# metric = "PressureSolve"
metrics = ["PressureSolve", "ViscousSolve"]

log_str = log_file_glob_strs[0]

savename = save_directory + "scaling" + "".join(f"-{metric}" for metric in metrics)

nodes_ref = int(4)
# x_ref = 'nodes'
x_ref = 'ncpus'

xlim = []
ylim_su = []
ylim_pe = []


case_columns = [
    "scheme",
    "dt",
    "ncpu",
    "nodes",
    "ncpus",
    "global_dof_per_rank",
    "local_dof_per_rank",
]

timer_columns = ["function", "average", "min", "max", "count", "level"]

replacements = [
    ("Pressure Forcing", "PressureForcing"),
    ("Pressure Solve", "PressureSolve"),
    ("Viscous Forcing", "ViscousForcing"),
    ("Viscous Solve", "ViscousSolve"),
    ("Pressure BCs", "PressureBCs"),
    ("Advection Terms", "AdvectionTerms"),
    ("ExchangeCoords local", "ExchangeCoordslocal"),
]





if __name__ == "__main__":

    params = {'text.usetex': True, 'font.size': 10}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(6, 6))
    ax_dt = fig.add_subplot(311)
    ax_su = fig.add_subplot(312, sharex=ax_dt)
    ax_pe = fig.add_subplot(313, sharex=ax_dt)

    ax_dt.set_ylabel(r"Comp. time per $\Delta t$")
    ax_dt.set_yscale('log')

    ax_su.set_ylabel(r"Speed Up")
    ax_pe.set_ylabel("Parallel Efficiency")

    ax_pe.set_xlabel(r"Number of Processors $N_{P}$")
    ax_pe.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dt.grid(which='both', axis='both')
    ax_su.grid(which='both', axis='both')
    ax_pe.grid(which='both', axis='both')

    df_stat = pd.DataFrame(columns=[*timer_columns, *case_columns])

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, log_file_glob_strs[0]):
        print(f"\treading {case['nodes']}x{case['ncpu']}")

        df_func = read_timer_output(node_dir, log_file_glob_strs[0], timer_columns, replacements)

        for case_column in case_columns:
            df_func[case_column] = case[case_column]

        df_stat = pd.concat([df_stat, df_func], axis=0, ignore_index=True)


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
        for metric, ls in zip(metrics, ['solid', 'dashed', 'dotted']):
            # Extract data for this plot
            df_plot = df_stat.loc[df_stat['scheme'] == scheme].loc[df_stat['function'] == metric]

            # Comp. time per time step
            dt = df_plot['average']

            # Compute speed-up (strong scaling)
            su = df_plot['average'].iloc[0] / df_plot['average']

            # Compute parallel efficiency (strong scaling)
            pe = 1 - ((ideal_su - su) / su)

            # Choose x-reference
            x_val = df_plot['ncpus']
            if x_ref == "nodes":
                x_val = df_plot['nodes'] / nodes_min

            # Add ideal reference for comp. time per dt
            dt_label = 'ideal'
            if scheme_color != list(TABLEAU_COLORS)[0] or metric != metrics[0]:
                dt_label = ''
            ax_dt.plot(x_val, dt.iloc[0] / ideal_su, linestyle='--', color='black', label=dt_label)

            # Plot
            ax_dt.plot(x_val, dt, marker='o', color=scheme_color, label=scheme + " " + metric, linestyle=ls)
            ax_su.plot(x_val, su, marker='o', color=scheme_color, label=scheme + " " + metric, linestyle=ls)
            ax_pe.plot(x_val, pe, marker='o', color=scheme_color, label=scheme + " " + metric, linestyle=ls)
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

    ax_dt.legend()


    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
