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
    save_directory, log_file_glob_strs,
)



# # Choose lift [1] or drag [0]
metric = "cpu_time"
# metric = "parallelEfficiency"
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"scaling"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'ncpus'

xlim = []
ylim = []



if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)

    ylabel = fr"{metric}"
    ax.set_ylabel(ylabel)
    # ax.set_yscale('log')

    ax.set_xlabel(r"Samples (time steps)")

    ax.grid(which='both', axis='both')


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
        node_directories = [x for x in os.walk(full_directory_path)][0][1]
        # Remove that have not been preprocessed yet
        node_directories = [x for x in node_directories if len(glob(full_directory_path + x + "/" + process_file)) == 1]
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
            full_file_path = node_directory_path + process_file

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

            # Read file
            # df = pd.read_csv(full_file_path, sep=',')
            df = pd.read_pickle(full_file_path)
            df = df[log_str]

            # Safe conversion to numeric
            df = df.apply(pd.to_numeric)


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

            # Compute cumulative mean
            cumulative_mean = metricData.expanding().mean()
            window_std = 100  # rolling window for smoothness
            rolling_std = cumulative_mean.rolling(window=window_std).std()

            # ax.plot(cumulative_mean, label=label + f" {n_nodes}x{n_cpus}")
            ax.errorbar(cumulative_mean.index, cumulative_mean, yerr=rolling_std, label=label + f" {n_nodes}x{n_cpus}", alpha=0.8)

            # Check convergence of cumulative mean
            tolerance = 1e-3  # acceptable change (0.1 %)
            window_diff = npoints_remove  # number of last samples to check

            # Compare the mean at the end vs. "window" samples earlier
            diff = abs(mean - cumulative_mean.iloc[-window_diff:-1]) / mean
            max_diff = diff.max()

            if max_diff < tolerance:
                print(f"\tConverged (Δ = {100 * max_diff:2.2f} %)")
            else:
                print(f"\tNOT converged (Δ = {100 * max_diff:2.2f} %)")

            # # Verbose statistics
            # print("Mean = {0}".format(mean))
            # print("Std  = {0}".format(std))
            # print("CV   = {0}\n".format(cv))

    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    plt.show()

    # # Save data
    # plt.savefig(savename + ".pdf", bbox_inches='tight')
    # df_stat.to_csv(savename + ".csv", sep=',')
    # print(f"Wrote files {savename} as pdf and csv")
