#!/usr/bin/env  python3
# Matplotlib setup with latex
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

params = {'text.usetex': True,
          'font.size': 10}
plt.rcParams.update(params)
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd

import os
from glob import glob

from npp.implicit_common import iter_implicit_cases
from config import (
    directory_names,
    path_to_directories,
    save_directory,
    log_file_glob_strs,
    ctu_len,
)



# # Choose lift [1] or drag [0]
plot_metric = "substeps"
# metric = "parallelEfficiency"
process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"implicit-substeps"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'dt'

xlim = []
ylim = [0, 60]



if __name__ == "__main__":

    # Create figure and axes
    fig = plt.figure(figsize=(6, 1.5))
    ax_sub = fig.add_subplot(111)

    ylabel = r"Average $N_{\Delta \tau}$"
    ax_sub.set_ylabel(ylabel)
    ax_sub.set_yscale('linear')

    ax_sub.set_xlabel(r"Time step increase $\times \Delta t_{CFL}$")
    ax_sub.set_xscale("log")
    ax_sub.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_sub.grid(which='both', axis='both')


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
        ]
    )

    for case_data in iter_implicit_cases(
        directory_names,
        path_to_directories,
        process_file,
        log_str,
        additional_metrics=["ctu_time"],
        ctu_len=ctu_len,
        compute_ci=True,
    ):
        df_case = pd.DataFrame([case_data])
        df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

    # Verbose check concatenation
    print(df_stat)


    # Filter minimum nodes
    nodes_min = df_stat['nodes'].min()
    dt_min = df_stat['dt'].min()
    ref_scheme = 'sub-stepping'
    df_ref = df_stat.loc[df_stat['scheme'] == ref_scheme].loc[df_stat['nodes'] == nodes_ref]
    if len(df_ref) == 0:
        raise ValueError(
            f"No reference data found for scheme: {ref_scheme} using nodes {nodes_ref}"
        )


    # Add ideal reference lines
    x_unique = df_stat[x_ref].round(7).unique()
    # ideal_su = x_unique / df_ref['dt'].iloc[0]
    # ax_su.plot(x_unique, ideal_su, linestyle='--', color='black', label='ideal')
    # ax_pe.plot(x_unique, np.ones_like(x_unique), linestyle='--', color='black', label='ideal')


    # Plot by scheme
    for scheme, scheme_color in zip(df_stat["scheme"].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_node = df_stat.loc[df_stat["nodes"] == nodes_ref]
        df_scheme = df_node.loc[df_node["scheme"] == scheme]

        # Compute number of substeps
        mean = df_scheme["substeps-mean"]
        ci = df_scheme["substeps-ci95"] if "substeps-ci95" in df_scheme else None

        # Choose x-reference
        x_val = df_scheme[x_ref]
        if x_ref == "nodes":
            x_val = df_scheme["nodes"] / nodes_min
        if x_ref == "dt":
            x_val = x_val / dt_min

        # Plot
        ax_sub.plot(x_val, mean, marker='o', label=scheme + " substeps", color=scheme_color)
        if substeps_ci is not None:
            ax_sub.fill_between(
                x_val,
                substeps_mean - substeps_ci,
                substeps_mean + substeps_ci,
                color=scheme_color,
                alpha=0.2,
            )

    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax_sub.get_xlim()
    if not ylim:
        ylim = ax_sub.get_ylim()
    ax_sub.set_xlim(xlim)
    ax_sub.set_ylim(ylim)


    ax_sub.legend()

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()