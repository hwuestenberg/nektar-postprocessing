#!/bin/python3
import os

# Matplotlib setup with latex
import matplotlib.pyplot as plt

params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params)
from matplotlib.colors import TABLEAU_COLORS

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from case_processing import iter_force_cases
from utilities import mser
from config import (
    directory_names,
    path_to_directories,
    customMetrics,
    ref_area,
    ctu_len,
    save_directory,
    force_file_skip_start,
    force_file_glob_strs,
)


############################
####### SCRIPT USER INPUTS
############################
# Choose lift [1] or drag [0]
metric = customMetrics[1]

forces_file = force_file_glob_strs[0]
forces_file_noext = forces_file.split('.')[0]
ctu_skip = 1e10 # sort of redundant with MSER
use_mser = False

n_downsample = 2


# Prefix for any saved figures
savename = f"{metric}-{forces_file_noext}-end-of-transient"
savename = save_directory + savename
############################



# Verbose prints
print("Using forces_file:", forces_file)



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"$C_l$"
    if metric == customMetrics[0]:
        ylabel = r"$C_d$"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"CTU")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.grid(True, which='both', axis='both')

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
        physTime = force_case.phys_time
        signal = force_case.signal

        dt = force_case.metadata.dt
        label = force_case.metadata.label
        color = force_case.metadata.color
        print("\nProcessing {0}...".format(label))

        # Determine end of transient via mser
        if use_mser:
            mser_stride_length = 10 if dt < 5e-5 else 1
            intTransient = mser(signal, physTime, debug_plot = False, stride_length=mser_stride_length)
            timeTransient = physTime.iloc[intTransient]
            print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

            # Plot end of transient
            ax.plot([timeTransient for i in range(2)], [signal.min(), signal.max()],
                    linestyle='dashed', color='black', label="End of transient")

        # Check averages w/o transient
        print(f"Average {metric} with transient is:\t{signal.mean()}")

        # Plot
        ax.plot(physTime, signal, color=color, alpha=0.3, label=label)
        ax.plot(
            [physTime.iloc[0], physTime.iloc[-1]],
            [signal.mean() for i in range(2)],
            color=color,
            alpha=1.0,
            label="Mean " + label,
        )

        # # # Uncomment to plot reverse cumulative mean and std
        # plot_cumulative_mean_std(signal, physTime, ax, color=dir_color, label=label)
        #
        # # Add averaging/mean
        # stepsPerCtu = ctu_len / dt
        # filter_width=int(len(signal)/50)
        #
        # ## Savitzky-golay filter
        # metric_smooth = savgol_filter(signal, window_length=filter_width, polyorder=1, mode="interp")
        # #ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color='black', linestyle='dotted', alpha=0.8)
        # ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color=color, linestyle='solid', alpha=1.0)#, label='Moving average')

        # Handle legend outside
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best')


    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    plt.show()


