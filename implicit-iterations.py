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

savename = f"implicit-iterations"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'dt'

if __name__ == "__main__":
    # Create figure and axes
    fig = plt.figure(figsize=(6, 2.7))
    ax_it_uvw = fig.add_subplot(211)
    ax_it_p = fig.add_subplot(212, sharex=ax_it_uvw)

    ylabel = r"$\\overline{ u + v + w }$"
    ax_it_uvw.set_ylabel(ylabel)
    ax_it_uvw.set_yscale("linear")

    ylabel = r"$\\overline{p}$"
    ax_it_p.set_ylabel(ylabel)
    ax_it_p.set_yscale("linear")

    ax_it_p.set_xlabel(r"Time step increase $\\times \\Delta t_{CFL}$")
    ax_it_p.set_xscale("log")
    ax_it_p.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_it_uvw.grid(which="both", axis="both")
    ax_it_p.grid(which="both", axis="both")

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
        additional_metrics=["ctu_time", "iterations_uvw"],
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

    # Plot by scheme: speedup
    for scheme, scheme_color in zip(df_stat["scheme"].unique(), TABLEAU_COLORS):
        # Extract data for this plot
        df_node = df_stat.loc[df_stat["nodes"] == nodes_ref]
        df_scheme = df_node.loc[df_node["scheme"] == scheme]

        # Compute number of iterations
        uvw_mean = df_scheme["iterations_uvw-mean"]
        uvw_std = df_scheme["iterations_uvw-std"]
        uvw_ci = df_scheme["iterations_uvw-ci95"]

        p_mean = df_scheme["iterations_p-mean"]
        p_std = df_scheme["iterations_p-std"]
        p_ci = df_scheme["iterations_p-ci95"]

        # Choose x-reference
        x_val = df_scheme[x_ref]
        if x_ref == "nodes":
            x_val = df_scheme["nodes"] / nodes_min
        if x_ref == "dt":
            x_val = x_val / dt_min

        # Plot
        ax_it_uvw.plot(
            x_val,
            uvw_mean,
            marker="o",
            label=scheme + " velocity",
            color=scheme_color,
        )
        ax_it_p.plot(
            x_val,
            p_mean,
            marker="o",
            label=scheme + " pressure",
            linestyle="dashed",
            color=scheme_color,
        )

        # ax_it_uvw.errorbar(x_val, uvw_mean, yerr=uvw_ci, marker='o', label=scheme + " velocity", color=scheme_color, capsize=5)
        # ax_it_p.errorbar(x_val, p_mean, yerr=p_std, marker='o', label=scheme + " pressure", linestyle='dashed', color=scheme_color)

    ax_it_uvw.legend()

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches="tight")
    df_stat.to_csv(savename + ".csv", sep=",")
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
