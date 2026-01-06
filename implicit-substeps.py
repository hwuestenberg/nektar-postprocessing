#!/usr/bin/env  python3
# Matplotlib setup with latex
import matplotlib.pyplot as plt

params = {'text.usetex': True,
          'font.size': 10}
plt.rcParams.update(params)
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter

import pandas as pd

from npp.implicit_common import iter_implicit_cases
from config import (
    directory_names,
    path_to_directories,
    save_directory,
    log_file_glob_strs,
    ctu_len,
)

process_file = "log_info.pkl"

log_str = log_file_glob_strs[0]

savename = f"implicit-substeps"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'dt'

if __name__ == "__main__":
    # Create figure and axes
    fig = plt.figure(figsize=(6, 2.7))
    ax_substeps = fig.add_subplot(111)

    ylabel = r"$\\overline{\\mathrm{substeps}}$"
    ax_substeps.set_ylabel(ylabel)
    ax_substeps.set_yscale("linear")

    ax_substeps.set_xlabel(r"Time step increase $\\times \\Delta t_{CFL}$")
    ax_substeps.set_xscale("log")
    ax_substeps.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_substeps.grid(which="both", axis="both")

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
    nodes_min = df_stat["nodes"].min()
    dt_min = df_stat["dt"].min()
    ref_scheme = "semi-implicit"
    df_ref = df_stat.loc[df_stat["scheme"] == ref_scheme].loc[
        df_stat["nodes"] == nodes_ref
    ]
    if len(df_ref) == 0:
        raise ValueError(
            f"No reference data found for scheme: {ref_scheme} using nodes {nodes_ref}"
        )

    # Plot by scheme
    for scheme, scheme_color in zip(df_stat["scheme"].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_node = df_stat.loc[df_stat["nodes"] == nodes_ref]
        df_scheme = df_node.loc[df_node["scheme"] == scheme]

        # Compute number of substeps
        substeps_mean = df_scheme["substeps-mean"]
        substeps_ci = df_scheme["substeps-ci95"] if "substeps-ci95" in df_scheme else None

        # Choose x-reference
        x_val = df_scheme[x_ref]
        if x_ref == "nodes":
            x_val = df_scheme["nodes"] / nodes_min
        if x_ref == "dt":
            x_val = x_val / dt_min

        # Plot
        ax_substeps.plot(x_val, substeps_mean, marker="o", label=scheme, color=scheme_color)
        if substeps_ci is not None:
            ax_substeps.fill_between(
                x_val,
                substeps_mean - substeps_ci,
                substeps_mean + substeps_ci,
                color=scheme_color,
                alpha=0.2,
            )

    ax_substeps.legend()

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches="tight")
    df_stat.to_csv(savename + ".csv", sep=",")
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
