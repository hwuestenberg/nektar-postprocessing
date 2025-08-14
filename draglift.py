#!/bin/python3

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

from utilities import get_time_step_size, get_label, mser, plot_cumulative_mean_std, filter_time_interval
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
# forces_file = "DragLift.fce"
ctu_skip = 100 # sort of redundant with MSER
use_mser = True

n_downsample = 2


# Prefix for any saved figures
savename = f"{metric}-{forces_file.split('.')[0]}-end-of-transient"
savename = save_directory + savename
############################



# Verbose prints
print("Using forces_file:", forces_file)
print("Skipping {0} CTUs from the start".format(ctu_skip))



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"$C_l$"
    if metric == customMetrics[0]:
        ylabel = r"$C_d$"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$t^\star$")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.grid(True, which='both', axis='both')

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname
        filename = forces_file.replace(".fce", f"-process-overlap-{force_file_skip_start}.fce")
        full_file_path = full_directory_path + filename

        # Get time step size
        # Note that we cannot detect 4e-6 from force file
        # because the sampling rate is set to 4e-5
        if "quasi3d" in full_directory_path:
            dt = 4e-6
        else:
            dt = get_time_step_size(full_directory_path)

        # Get plot styling
        label, marker, mfc, ls, color = get_label(full_file_path, dt, raw_label=False)
        print("\nProcessing {0}...".format(label))

        # Read file
        df = pd.read_csv(full_file_path, sep=',')

        # Extract time and data
        physTime = df["Time"]
        physTime = physTime / ctu_len # Normalise to CTUs
        signal = df[metric]

        # Build mask based on time interval
        physTime, signal = filter_time_interval(physTime, signal, ctu_skip)

        # Correct data (coeff = 2 * Force)
        signal = 2 * signal

        # Normalise by area
        # Note quasi-3d is averaged along spanwise
        if "quasi3d" in full_file_path:
            signal = signal / ctu_len
        else:
            signal = signal / ref_area

        # Downsample
        # Note: do this before MSER
        if n_downsample > 1:
            signal = signal[::n_downsample]
            physTime = physTime[::n_downsample]
            # label += f" downsample {n_downsample}"

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
        # ax.plot([physTime.iloc[0], physTime.iloc[-1]], [signal.mean() for i in range(2)], color=dir_color, alpha=1.0, label="Mean " + label)

        # # Uncomment to plot reverse cumulative mean and std
        # plot_cumulative_mean_std(signal, physTime, ax, dir_color, label)

        # Add averaging/mean
        stepsPerCtu = ctu_len / dt
        filter_width=int(len(signal)/50)

        ## Savitzky-golay filter
        metric_smooth = savgol_filter(signal, window_length=filter_width, polyorder=1, mode="interp")
        #ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color='black', linestyle='dotted', alpha=0.8)
        ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color=color, linestyle='solid', alpha=1.0)#, label='Moving average')

        # Handle legend outside
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best')


    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    plt.show()


