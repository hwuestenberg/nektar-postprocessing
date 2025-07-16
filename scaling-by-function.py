#!/usr/bin/env  python3
# Matplotlib setup with latex
import matplotlib.pyplot as plt

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

from utilities import get_time_step_size, mser, get_label, get_scheme
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ref_area, ctu_len, divtol, force_file_skip_start, save_directory, file_glob_strs



# # Choose lift [1] or drag [0]
metric = "Execute"

savename = f"scaling-{metric}"
savename = save_directory + savename

xlim = []
ylim_su = []
ylim_pe = []


case_columns = [
    "scheme",
    "dt",
    "ncpu",
    "nodes",
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
    fig = plt.figure(figsize=(5, 4))
    ax_su = fig.add_subplot(211)
    ax_pe = fig.add_subplot(212, sharex=ax_su)

    ylabel = r"Speed Up $S = T(N_P=1)/T(N_{P}=P)$"
    ax_su.set_ylabel(ylabel)

    ylabel = "Parallel Efficiency"
    ax_pe.set_ylabel(ylabel)

    ax_pe.set_xlabel(r"Number of nodes, each with $N_{P} = 64$ CPUs")
    ax_su.xaxis.set_major_formatter(FormatStrFormatter('%d'))

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

            # Add processed file name
            node_directory_path = full_directory_path + f"{n_nodes}x{n_cpus}/"

            # Get time step size
            dt = get_time_step_size(node_directory_path)
            case_dict['dt'] = dt

            # Get plot styling
            label, marker, mfc, ls, color = get_label(node_directory_path, dt)
            print("\nProcessing {0}...".format(label))

            # # Read file
            # df = pd.read_csv(full_file_path, sep=',')
            #
            # # Remove initial 10% of data (this includes setup)
            # npoints = len(df)
            # npoints_remove = int(0.1 * npoints)
            # df = df.iloc[npoints_remove:]
            #
            # # Extract signal
            # metricData = df[metric]
            #
            # # Do statistics
            # mean = metricData.mean()
            # std = metricData.std()
            # cv = std/mean
            #
            # # # Verbose statistics
            # # print("Mean = {0}".format(mean))
            # # print("Std  = {0}".format(std))
            # # print("CV   = {0}\n".format(cv))
            #
            # # Add statistics to dict
            # case_dict[f'{metric}-mean'] = metricData.mean()
            # case_dict[f'{metric}-std'] = metricData.std()

            # Find raw log file
            log_file = glob(node_directory_path + "log.*")[0]

            ## Preproessing to make read_csv possible
            # Read performance by function into csv
            skiprows = -1
            with open(log_file, "r") as log:
                for line in log.readlines():
                    if "Execute" in line:
                        break
                    skiprows += 1

            # Fix spaces
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                for old, new in zip(replace_old, replace_new):
                    content = content.replace(old, new)
            new_log_file = log_file.replace("log.","log_fix.")
            with open(new_log_file, 'w', encoding='utf-8') as f:
                f.write(content)

            df_func = pd.read_csv(new_log_file, skiprows=skiprows, sep="\s+|\t+|\s+\t+|\t+\s+", engine='python')
            df_func = df_func.dropna(axis=1)
            df_func.columns = timer_columns
            for case_column in case_columns:
                df_func[case_column] = case_dict[case_column]
            print(df_func)

            # Transform to DataFrame and concatenate
            df_stat = pd.concat([df_stat, df_func], axis=0, ignore_index=True)

        # Verbose check concatenation
        print(df_stat)


    # Plot by scheme: speedup
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_scheme = df_stat.loc[df_stat['scheme'] == scheme].loc[df_stat['function'] == metric]
        nodes_ref = df_scheme['nodes'].min()

        # Compute speed-up (strong scaling)
        su = df_scheme['average'].iloc[0] / df_scheme['average']

        # Compute parallel efficiency (strong scaling)
        pe = 1 - ((2 ** np.arange(len(df_scheme)) - su) / su)

        # Plot
        ax_su.plot(df_scheme['nodes'], su, marker='o', label=scheme)
        ax_pe.plot(df_scheme['nodes'], pe, marker='o', label=scheme)
        # ax.errorbar(df_plot['nodes'] / nodes_ref, su, df_plot[f'{metric}-std'],
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