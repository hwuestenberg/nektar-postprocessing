#!/bin/python3
import os

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

from case_processing import iter_force_cases
from utilities import mser, get_scheme
from config import (
    directory_names,
    path_to_directories,
    dtref,
    customMetrics,
    ref_area,
    ctu_len,
    divtol,
    force_file_skip_start,
    save_directory,
    force_file_glob_strs,
)


############################
####### SCRIPT USER INPUTS
############################
# Choose lift [1] or drag [0]
metric = customMetrics[1]

forces_file = force_file_glob_strs[0]
forces_file_noext = forces_file.split('.')[0]
# forces_file = "DragLift.fce"
ctu_skip = 1e10 # sort of redundant with MSER
use_mser = False

averaging_len = 1000 # [CTU] redundant due to MSER, just use large number
n_downsample = 2

savename = f"mean-{metric}-{forces_file_noext}"
savename = save_directory + savename
############################


xlim = [8e-6 / dtref, 1.2e-3 / dtref]
ylim = []


# Verbose prints
print("Using forces_file:", forces_file)
# print("Averaging over {0} CTUs".format(averaging_len))




if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ylabel = r"$\overline{C}_l$"
    if metric == customMetrics[0]:
        ylabel = r"$\overline{C}_d$"
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(r"Time step increase $\Delta t_{CFL}$")
    ax.set_xlabel(r"Time step increase $\times \Delta t_{CFL}$")
    # ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.grid(True, which='both', axis='both')


    # Dataframe for gathering statistics for each case
    df_stat = pd.DataFrame(columns=[
        "scheme",
        "dt",
        f"{metric}-mean",
        f"{metric}-std",
    ])


    # Loop all files
    for force_case in iter_force_cases(
        directory_names=directory_names,
        path_to_directories=path_to_directories,
        forces_file_noext=forces_file_noext,
        metric=metric,
        ctu_skip=ctu_skip,
        n_downsample=n_downsample,
        ref_area=ref_area,
        ctu_len=ctu_len,
    ):
        dt = force_case.metadata.dt
        label = force_case.metadata.label
        print("\nProcessing {0}...".format(label))

        signal = force_case.signal
        physTime = force_case.phys_time

        # Determine end of transient via mser
        if use_mser:
            mser_stride_length = 10 if dt < 5e-5 else 1
            intTransient = mser(signal, physTime, debug_plot = False, stride_length=mser_stride_length)
            timeTransient = physTime.iloc[intTransient]
            print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

        # # Remove end of transient from signal
        # signal = signal.iloc[intTransient:]

        # Do statistics
        mean = signal.mean()
        # mean = np.sqrt(np.pow(signal, 2).sum() / len(signal)) # compute RMS (lift for cylinder case)
        std = signal.std()
        cv = std/mean

        # Ignore if diverged
        if np.abs(signal.iloc[-1]) > divtol or np.abs(mean) > divtol:
            print("Last datapoint = {0:.1e} or mean = {1:.1e} is larger than {2:.1e}. Assuming divergence and skipping.".format(np.abs(signal.iloc[-1]), np.abs(mean), divtol))
            break

        # Add statistics to dict
        case_dict = {
            'scheme': get_scheme(force_case.metadata.file_path),
            'dt': dt,
            f'{metric}-mean': mean,
            f'{metric}-std': std,
        }

        # Transform to DataFrame and concatenate
        df_case = pd.DataFrame([case_dict])
        df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

    # Verbose check concatenation
    print(df_stat)

    # Plot by scheme: dt vs mean force
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        df_plot = df_stat.loc[df_stat['scheme'] == scheme]
        ax.plot(df_plot['dt'] / dtref, df_plot[f'{metric}-mean'], marker='o', label=scheme)
        ax.errorbar(df_plot['dt'] / dtref, df_plot[f'{metric}-mean'], df_plot[f'{metric}-std'], color=scheme_color, capsize=4)

    # # Add +/- 1% error of semi-implicit
    # ref_scheme = 'semi-implicit'
    # if ref_scheme in df_stat['scheme'].unique():
    #     df_plot = df_stat.loc[df_stat['scheme'] == ref_scheme]
    #     dt_max_min = [df_stat['dt'].max() / dtref, df_stat['dt'].min() / dtref]
    #     ax.fill_between(dt_max_min, y1=df_plot[f'{metric}-mean']*1.01, y2=df_plot[f'{metric}-mean']*0.99,
    #                     alpha=0.2, label=rf"+/- 1\% error", color=list(TABLEAU_COLORS)[0])

    # ax.legend(loc='best')
    # Move the legend above the plot
    ax.legend(
        loc='lower center',  # position legend at the bottom center of the bbox
        bbox_to_anchor=(0.5, 1.02),  # 0.5 = center horizontally, 1.02 = slightly above the axes
        ncol=2,  # number of columns (optional)
        frameon=True  # remove the box (optional)
    )
    # plt.tight_layout()  # adjust layout to fit legend

    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
