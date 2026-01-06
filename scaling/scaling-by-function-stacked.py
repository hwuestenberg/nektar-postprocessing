#!/usr/bin/env  python3
# Matplotlib setup with latex
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import directory_names, log_file_glob_strs, path_to_directories, save_directory
from npp.scaling_common import iter_scaling_nodes, read_timer_output

metric = "Execute"
metrics = ["PressureSolve", "ViscousSolve"]

log_str = log_file_glob_strs[0]

savename = save_directory + "scaling-stack" + "".join(f"-{metric}" for metric in metrics)

nodes_ref = int(16)
x_ref = 'ncpus'

xlim = []

case_columns = [
    "scheme",
    "dt",
    "ncpu",
    "nodes",
    "ncpus",
    "global_dof_per_rank",
    "local_dof_per_rank",
]

timer_columns = ["function", "average", "min", "max", "count", "level"]

replacements = [
    ("Pressure Forcing", "PressureForcing"),
    ("Pressure Solve", "PressureSolve"),
    ("Viscous Forcing", "ViscousForcing"),
    ("Viscous Solve", "ViscousSolve"),
    ("Pressure BCs", "PressureBCs"),
    ("Advection Terms", "AdvectionTerms"),
    ("ExchangeCoords local", "ExchangeCoordslocal"),
]

if __name__ == "__main__":

    params = {'text.usetex': True, 'font.size': 10}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.set_xlabel(r"Number of Processors $N_{P}$")
    ax.set_ylabel(r"Average comp. time per function")
    ax.set_yscale('log')
    ax.grid(which='both', axis='both')

    df_stat = pd.DataFrame(columns=[*timer_columns, *case_columns])

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, log_file_glob_strs[0]):
        print(f"\treading {case['nodes']}x{case['ncpu']}")

        df_func = read_timer_output(node_dir, log_file_glob_strs[0], timer_columns, replacements)
        for case_column in case_columns:
            df_func[case_column] = case[case_column]

        df_stat = pd.concat([df_stat, df_func], axis=0, ignore_index=True)

    nodes_min = df_stat['nodes'].min()
    if nodes_ref:
        df_stat = df_stat[df_stat['nodes'] >= nodes_ref]
    else:
        nodes_ref = nodes_min

    nbars = 0
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        df_scheme = df_stat.loc[df_stat['scheme'] == scheme]
        df_scheme_lvl0 = df_scheme.loc[df_scheme['level'] == 0]

        for node in df_scheme_lvl0['nodes'].unique():
            df_node = df_scheme_lvl0.loc[df_scheme_lvl0['nodes'] == node]
            ncpus = df_node['ncpus'].max()
            df_node = df_node.copy()

            total_time = df_node[df_node["function"] == metric]["average"].iloc[0]
            df_bar = df_node.loc[df_node['function'].isin(metrics)]

            new_row = df_node[~df_node["function"].isin(metrics + [metric])].sum(numeric_only=True)
            new_row["function"] = "Other"
            new_row["average"] = total_time - df_bar["average"].sum()
            df_bar.loc[len(df_bar)] = new_row

            df_bar = df_bar[df_bar["function"] != metric]

            df_bar['rel_average'] = df_bar['average'] / total_time
            num_time_steps = df_bar["count"][df_bar["function"] == "PressureSolve"].iloc[0]
            df_bar['average_per_dt'] = df_bar['average'] / num_time_steps

            bottom = 0
            for val, rel_val, function_name, color in zip(
                df_bar['average_per_dt'],
                df_bar['rel_average'],
                df_bar['function'],
                TABLEAU_COLORS,
            ):
                label = function_name if nbars == 0 else None
                ax.bar(nbars, val, bottom=bottom, color=color, edgecolor='black', label=label)
                ax.text(nbars, bottom + val / 2, f"{100*rel_val:.1f}\%", ha='center', va='bottom', color='black')
                bottom += val
            nbars += 1

    if not xlim:
        xlim = ax.get_xlim()
    ax.set_xlim(xlim)

    ax.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
