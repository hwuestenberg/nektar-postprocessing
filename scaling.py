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
metric = "cpu_time"
# metric = "parallelEfficiency"
process_file = "log_info.csv"

savename = f"scaling"
savename = save_directory + savename

xlim = []
ylim_su = []
ylim_pe = []



if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(5, 4))
    ax_su = fig.add_subplot(211)
    ax_pe = fig.add_subplot(212, sharex=ax_su)

    ylabel = r"Speed Up $S = T(N_P=1)/T(N_{P}=P)$"
    ax_su.set_ylabel(ylabel)

    ylabel = "Parallel Efficiency"
    ax_pe.set_ylabel(ylabel)

    ax_su.set_xlabel(r"Number of Processors $N_{P}$")
    ax_su.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_su.grid(which='both', axis='both')
    ax_pe.grid(which='both', axis='both')


    # Dataframe for gathering statistics for each case
    df_stat = pd.DataFrame(columns=[
        "scheme",
        "dt",
        "ncpu",
        "nodes",
        f"{metric}-mean",
        f"{metric}-std",
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
            full_file_path = node_directory_path + process_file

            # Get time step size
            dt = get_time_step_size(node_directory_path)
            case_dict['dt'] = dt

            # Get plot styling
            label, marker, mfc, ls, color = get_label(node_directory_path, dt)
            print("\nProcessing {0}...".format(label))

            # Read file
            df = pd.read_csv(full_file_path, sep=',')

            # Remove initial 10% of data (this includes setup)
            npoints = len(df)
            npoints_remove = int(0.1 * npoints)
            df = df.iloc[npoints_remove:]

            # Extract signal
            metricData = df[metric]

            # Do statistics
            mean = metricData.mean()
            std = metricData.std()
            cv = std/mean

            # # Verbose statistics
            # print("Mean = {0}".format(mean))
            # print("Std  = {0}".format(std))
            # print("CV   = {0}\n".format(cv))

            # Add statistics to dict
            case_dict[f'{metric}-mean'] = metricData.mean()
            case_dict[f'{metric}-std'] = metricData.std()

            # Transform to DataFrame and concatenate
            df_case = pd.DataFrame([case_dict])
            df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

        # Verbose check concatenation
        print(df_stat)


    # Plot by scheme: speedup
    nodes_ref = df_stat['nodes'].min()
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_plot = df_stat.loc[df_stat['scheme'] == scheme]

        # Compute speed-up (strong scaling)
        su = df_plot[f'{metric}-mean'].iloc[0] / df_plot[f'{metric}-mean']

        # Compute parallel efficiency (strong scaling)
        pe = 1 - ((2 ** np.arange(len(df_plot)) - su) / su)

        # Plot
        ax_su.plot(df_plot['nodes'] / nodes_ref, su, marker='o', label=scheme)
        ax_pe.plot(df_plot['nodes'] / nodes_ref, pe, marker='o', label=scheme)
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