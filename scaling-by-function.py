#!/usr/bin/env  python3
# Matplotlib setup with latex
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params)
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd

import os
from glob import glob

from utilities import get_time_step_size, mser, get_label, get_scheme, get_dof
from config import (
    directory_names,
    path_to_directories,
    dtref,
    customMetrics,
    ref_area,
    ctu_len,
    divtol,
    force_file_skip_start,
    save_directory,
)



# # Choose lift [1] or drag [0]
metric = "Execute"

savename = f"scaling-{metric}"
savename = save_directory + savename

nodes_ref = int(4)
# x_ref = 'nodes'
x_ref = 'cpus'

xlim = []
ylim_su = []
ylim_pe = []


case_columns = [
    "scheme",
    "dt",
    "ncpu",
    "nodes",
    "ncpus",
]

timer_columns = [
    "function",
    "average",
    "min",
    "max",
    "count",
    "level",
]

replace_old = [
    "Pressure Forcing",
    "Pressure Solve",
    "Viscous Forcing",
    "Viscous Solve",
    "Pressure BCs",
    "Advection Terms",
    "ExchangeCoords local",
]

replace_new = [
    "PressureForcing",
    "PressureSolve",
    "ViscousForcing",
    "ViscousSolve",
    "PressureBCs",
    "AdvectionTerms",
    "ExchangeCoordslocal",
]





if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(4, 6))
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
    df_stat = pd.DataFrame(columns=[
        *timer_columns,
        *case_columns

    ])


    # Loop all files
    for dirname in directory_names:
        # Setup paths
        full_directory_path = path_to_directories + dirname

        # use replace to change to scaling directory
        full_directory_path = full_directory_path.replace("physics", "scaling")
        node_directories = [x for x in os.walk(full_directory_path)][0][1] # get all Yx64 directories
        nodes = sorted([int(x.split("x")[0]) for x in node_directories])
        n_cpus = max([int(x.split("x")[1]) for x in node_directories])

        # Create dictionary for gathering data
        case_dict = {'scheme': get_scheme(full_directory_path)}

        # Loop each node-directory in sorted order
        for n_nodes in nodes:
            print(f"\treading {n_nodes}x{n_cpus}")

            # Get number of cpus and nodes
            case_dict['ncpu'] = n_cpus
            case_dict['nodes'] = n_nodes
            case_dict['ncpus'] = n_cpus * n_nodes

            # Add processed file name
            node_directory_path = full_directory_path + f"{n_nodes}x{n_cpus}/"
            # full_file_path = node_directory_path + process_file

            # Get number of local/global DoF
            get_dof(case_dict, node_directory_path)
            case_dict['global_dof_per_rank'] = case_dict['global_dof'] / case_dict['ncpus']
            case_dict['local_dof_per_rank'] = case_dict['local_dof'] / case_dict['ncpus']

            # Get time step size
            dt = get_time_step_size(node_directory_path)
            case_dict['dt'] = dt

            # Get plot styling
            label, marker, mfc, ls, color = get_label(node_directory_path, dt)
            print("\nProcessing {0}...".format(label))

            # Find raw log file
            log_files = glob(node_directory_path + "log*")
            log_file = [file for file in log_files if not "log_info.csv" in file][0]

            ## Preproessing to make read_csv possible
            timer_file = node_directory_path + "timer_info.txt"
            reachTimerOutput = False
            with open(log_file, 'r', encoding='utf-8') as f_in, \
                    open(timer_file, 'w', encoding='utf-8') as f_out:

                for i, line in enumerate(f_in):
                    # Detect start of timer output, skip all lines before
                    if "Execute" in line or reachTimerOutput:
                        reachTimerOutput = True
                    else:
                        continue

                    # Detect end of output
                    if "Victory!" in line:
                        break

                    for old, new in zip(replace_old, replace_new):
                        line = line.replace(old, new)
                    f_out.write(line)

            # Read timer output
            df_func = pd.read_csv(timer_file, names=timer_columns, sep="\s+|\t+|\s+\t+|\t+\s+", engine='python')
            df_func = df_func.dropna(axis=1)
            df_func.columns = timer_columns

            # Add metadata from case_dict
            for case_column in case_columns:
                df_func[case_column] = case_dict[case_column]

            # Verbose print
            # print(df_func)

            # Transform to DataFrame and concatenate
            df_stat = pd.concat([df_stat, df_func], axis=0, ignore_index=True)

        # # Verbose check concatenation
        # print(df_stat)


    # Filter minimum nodes
    nodes_min = df_stat['nodes'].min()
    if nodes_ref:
        df_stat = df_stat[df_stat['nodes'] >= nodes_ref]
    else:
        nodes_ref = nodes_min


    # Plot by scheme: speedup
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_plot = df_stat.loc[df_stat['scheme'] == scheme].loc[df_stat['function'] == metric]

        # Comp. time per time step
        dt = df_plot['average']

        # Compute speed-up (strong scaling)
        su = df_plot['average'].iloc[0] / df_plot['average']

        # Compute parallel efficiency (strong scaling)
        pe = 1 - ((2 ** np.arange(len(df_plot)) - su) / su)

        # Choose x-reference
        x_val = df_plot['ncpus']
        if x_ref == "nodes":
            x_val = df_plot['nodes'] / nodes_min

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

    ax_su.legend()


    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()