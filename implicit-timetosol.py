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

savename = f"implicit-scaling"
savename = save_directory + savename

nodes_ref = int(32)
# x_ref = 'nodes'
x_ref = 'dt'

if __name__ == "__main__":
    # Create figure and axes
    fig = plt.figure(figsize=(6, 2.7))
    ax_dt = fig.add_subplot(211)
    ax_su = fig.add_subplot(212, sharex=ax_dt)
    # ax_it = fig.add_subplot(313, sharex=ax_dt)

    ylabel = r"Comp. time per CTU [h]"
    ax_dt.set_ylabel(ylabel)
    ax_dt.set_yscale("log")

    ylabel = r"Speed Up"  # $S = T(N_P=1)/T(N_{P}=P)$"
    ax_su.set_ylabel(ylabel)
    ax_su.set_yscale("log")
    ax_su.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ylabel = r"Sum of velocity iterations"  # $S = T(N_P=1)/T(N_{P}=P)$"
    # ax_it.set_ylabel(ylabel)
    # ax_it.set_yscale('linear')

    ax_su.set_xlabel(r"Time step increase $\\times \\Delta t_{CFL}$")
    ax_su.set_xscale("log")
    ax_su.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dt.grid(which="both", axis="both")
    ax_su.grid(which="both", axis="both")
    # ax_it.grid(which='both', axis='both')

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
        compute_ci=False,
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

        # Comp. time per CTU
        dt = df_scheme[f"{plot_metric}-mean"] / 3600

        # Compute speed-up relative to semi-implicit
        su = df_ref[f"{plot_metric}-mean"].iloc[0] / df_scheme[f"{plot_metric}-mean"]

        # Compute number of iterations
        it_uvw = df_scheme["iterations_uvw-mean"]
        # it_p = df_scheme['iterations_p-mean']

        # Choose x-reference
        x_val = df_scheme[x_ref]
        if x_ref == "nodes":
            x_val = df_scheme["nodes"] / nodes_min
        if x_ref == "dt":
            x_val = x_val / dt_min

        # Plot
        ax_dt.plot(x_val, dt, marker="o", label=scheme)
        ax_su.plot(x_val, su, marker="o", label=scheme)

        # ax_it.plot(x_val, it_uvw, marker='o', label=scheme + " velocity", color=scheme_color)
        # ax_it.plot(x_val, it_p, marker='o', label=scheme + " pressure", linestyle='dashed', color=scheme_color)

        # # With error bars
        # rel_err = df_scheme[f'{metric}-std']
        # yerr_lower = dt - dt / (1 + rel_err)
        # yerr_upper = dt * (1 + rel_err) - dt
        # ax_dt.errorbar(x_val, dt, yerr=[yerr_lower, yerr_upper], fmt='o', linestyle='None',
        #             color=scheme_color, capsize=4)

    ax_dt.legend()

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches="tight")
    df_stat.to_csv(savename + ".csv", sep=",")
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
