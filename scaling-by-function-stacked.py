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
    save_directory, log_file_glob_strs,
)



metric = "Execute"
# metric = "ViscousSolve"
# metric = "PressureSolve"
metrics = [
    # "AdvectionTerms",
    "PressureSolve",
    "ViscousSolve",
    # "Interp1DScaled",
]

log_str = log_file_glob_strs[0]

savename = f"scaling-stack"
for metric in metrics:
    savename += f"-{metric}"
savename = save_directory + savename

nodes_ref = int(16)
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
    "local_dof_per_rank"
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
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    xlabel = r"Number of Processors $N_{P}$"
    ax.set_xlabel(xlabel)

    ylabel = r"Average comp. time per function"
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    ax.grid(which='both', axis='both')


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
        node_directories = [x for x in os.walk(full_directory_path)][0][1]
        # Remove that have not been preprocessed yet
        node_directories = [x for x in node_directories if len(glob(full_directory_path + x + "/" + log_file_glob_strs[0])) > 0]
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
            log_files = glob(node_directory_path + log_file_glob_strs[0])
            log_files = [l for l in log_files if not ".pkl" in l]
            log_file = [file for file in log_files if not "log_info.csv" in file][0]

            ## Preproessing to make read_csv possible
            timer_file = node_directory_path + "timer_info.txt"
            reachTimerOutput = False
            with open(log_file, 'r', encoding='utf-8') as f_in, \
                    open(timer_file, 'w', encoding='utf-8') as f_out:

                for nbars, line in enumerate(f_in):
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
    nbars = 0
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        # Extract data for this scheme
        df_scheme = df_stat.loc[df_stat['scheme'] == scheme]

        # Extract level 0 functions only
        df_scheme_lvl0 = df_scheme.loc[df_scheme['level'] == 0]

        # Extract node
        for node in df_scheme_lvl0['nodes'].unique():
            df_node = df_scheme_lvl0.loc[df_stat['nodes'] == node]
            ncpus = df_node['ncpus'].max()
            df_node = df_node.copy() # avoid manipulating original df

            # compute relative average time
            total_time = df_node[df_node["function"] == "Execute"]["average"].iloc[0]
            # df_node['rel_average'] = df_node['average'] / total_time

            # Sort bar plot data by threshold
            # df_bar = df_plot.loc[df_plot['rel_average'] > 0.02]
            df_bar = df_node.loc[df_node['function'].isin(metrics)]

            # Sum "other" contributions
            # new_row = df_plot[df_plot["function"] == "Execute"]
            new_row = df_node[~df_node["function"].isin(metrics + ["Execute"])].sum(numeric_only=True)
            new_row["function"] = "Other"
            new_row["average"] = total_time - df_bar["average"].sum()
            df_bar.loc[len(df_bar)] = new_row

            # Remove "Execute" from bar plot
            df_bar = df_bar[df_bar["function"] != "Execute"]

            # Get timing relative to total
            df_bar['rel_average'] = df_bar['average'] / total_time


            # Set index for plotting
            # df_bar = df_bar.set_index("function")
            num_time_steps = df_bar["count"][df_bar["function"] == "PressureSolve"].iloc[0]
            df_bar['average_per_dt'] = df_bar['average'] / num_time_steps

            # Plot as stacked bar
            # df_bar.plot(kind="bar", stacked=True)
            # ax.bar(i, df_bar['average_per_dt'], label=scheme)
            bottom = 0
            for val, rel_val, function_name, color in zip(df_bar['average_per_dt'], df_bar['rel_average'], df_bar['function'], TABLEAU_COLORS):
                # Only print label once
                label = function_name
                if nbars > 0:
                    label = None

                ax.bar(nbars, val, bottom=bottom, color=color, edgecolor='black', label=label)
                ax.text(nbars, bottom + val / 2, f"{100*rel_val:.1f}\%", ha='center', va='bottom', color='black')
                bottom += val  # stack vertically
            nbars += 1


    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax.get_xlim()
    ax.set_xlim()

    # ax.set_xticks(np.arange(nbars), df_scheme['ncpus'].unique().astype(str))

    ax.legend()

    # Get current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Reverse the order
    plt.legend(handles[::-1], labels[::-1])


    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()