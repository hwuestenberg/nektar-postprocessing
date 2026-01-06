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
    save_directory, log_file_glob_strs, ctu_len,
)



# # Choose lift [1] or drag [0]
plot_metric = "ctu_time"
# metric = "parallelEfficiency"
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"implicit-scaling"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'dt'

xlim = []
ylim_su = []



if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(6, 2.7))
    ax_dt = fig.add_subplot(211)
    ax_su = fig.add_subplot(212, sharex=ax_dt)
    # ax_it = fig.add_subplot(313, sharex=ax_dt)

    ylabel = r"Comp. time per CTU [h]"
    ax_dt.set_ylabel(ylabel)
    ax_dt.set_yscale('log')

    ylabel = r"Speed Up"# $S = T(N_P=1)/T(N_{P}=P)$"
    ax_su.set_ylabel(ylabel)
    ax_su.set_yscale('log')
    ax_su.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ylabel = r"Sum of velocity iterations"  # $S = T(N_P=1)/T(N_{P}=P)$"
    # ax_it.set_ylabel(ylabel)
    # ax_it.set_yscale('linear')

    ax_su.set_xlabel(r"Time step increase $\times \Delta t_{CFL}$")
    ax_su.set_xscale("log")
    ax_su.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dt.grid(which='both', axis='both')
    ax_su.grid(which='both', axis='both')
    # ax_it.grid(which='both', axis='both')


    # Dataframe for gathering statistics for each case
    df_stat = pd.DataFrame(columns=[
        "scheme",
        "dt",
        "ncpu",
        "nodes",
        "ncpus",
        "global_dof_per_rank",
        "local_dof_per_rank",
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

            # Process all metrics and append to case_dict
            additional_metrics = ["ctu_time", "iterations_uvw"]
            metrics = df.columns.to_list() + additional_metrics
            for metric in metrics:
                # Extract signal (check for special cases)
                ref_metric = metric
                if metric == "ctu_time":
                    ref_metric = "cpu_time"
                elif metric == "iterations_uvw":
                    ref_metric = "iterations_u"
                metricData = df[ref_metric]

                # Compute time-to-solution (aka per CTU)
                if metric == "ctu_time":
                    num_time_steps = ctu_len / dt
                    metricData = metricData * num_time_steps
                elif metric == "iterations_uvw":
                    metricData = df["iterations_u"] + df["iterations_v"] + df["iterations_w"]

                # Add statistics to dict
                case_dict[f'{metric}-mean'] = metricData.mean()
                case_dict[f'{metric}-std'] = metricData.std()

            # Transform to DataFrame and concatenate
            df_case = pd.DataFrame([case_dict])
            df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

    # Verbose check concatenation
    print(df_stat)


    # Filter minimum nodes
    nodes_min = df_stat['nodes'].min()
    dt_min = df_stat['dt'].min()
    ref_scheme = 'semi-implicit'
    df_ref = df_stat.loc[df_stat['scheme'] == ref_scheme].loc[df_stat['nodes'] == nodes_ref]
    if len(df_ref) == 0:
        raise ValueError(f"No reference data found for scheme: {ref_scheme} using nodes {nodes_ref}")

    # Add ideal reference lines
    x_unique = df_stat[x_ref].round(7).unique()


    # Plot by scheme: speedup
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_node = df_stat.loc[df_stat['nodes'] == nodes_ref]
        df_scheme = df_node.loc[df_stat['scheme'] == scheme]

        # Comp. time per CTU
        dt = df_scheme[f'{plot_metric}-mean'] / 3600

        # Compute speed-up relative to semi-implicit
        su = df_ref[f'{plot_metric}-mean'].iloc[0] / df_scheme[f'{plot_metric}-mean']

        # Compute number of iterations
        it_uvw = df_scheme['iterations_uvw-mean']
        # it_p = df_scheme['iterations_p-mean']

        # Choose x-reference
        x_val = df_scheme[x_ref]
        if x_ref == "nodes":
            x_val = df_scheme['nodes'] / nodes_min
        if x_ref == "dt":
            x_val = x_val / dt_min

        # Plot
        ax_dt.plot(x_val, dt, marker='o', label=scheme)
        ax_su.plot(x_val, su, marker='o', label=scheme)

        # ax_it.plot(x_val, it_uvw, marker='o', label=scheme + " velocity", color=scheme_color)
        # ax_it.plot(x_val, it_p, marker='o', label=scheme + " pressure", linestyle='dashed', color=scheme_color)

        # # With error bars
        # rel_err = df_scheme[f'{metric}-std']
        # yerr_lower = dt - dt / (1 + rel_err)
        # yerr_upper = dt * (1 + rel_err) - dt
        # ax_dt.errorbar(x_val, dt, yerr=[yerr_lower, yerr_upper], fmt='o', linestyle='None',
        #             color=scheme_color, capsize=4)

    ## Aesthetics
    # # Set x/y-limits
    # if not xlim:
    #     xlim = ax_su.get_xlim()
    #     xlim = ax_pe.get_xlim()
    # if not ylim_su:
    #     ylim_su = ax_su.get_ylim()
    # ax_su.set_xlim(xlim)
    # ax_su.set_ylim(ylim_su)


    ax_dt.legend()


    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()