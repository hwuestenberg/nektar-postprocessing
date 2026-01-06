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
from scipy import signal

from utilities import get_time_step_size, get_label, mser, plot_cumulative_mean_std, filter_time_interval
from config import directory_names, path_to_directories, ref_area, ctu_len, save_directory, log_file_glob_strs

####### SCRIPT USER INPUTS
# Choose lift [1] or drag [0]
metric = 'cfl'

log_file = "log_info.pkl"
signal_len_from_end = 40.0 # in CTUs
use_mser = False

n_downsample = 1


# Prefix for any saved figures
savename = f"{metric}"
savename = save_directory + savename




# Verbose prints
print("Using {0} CTUs from the end".format(signal_len_from_end))



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = rf"{metric.upper()}"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$t^\star$")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("linear")
    if metric == "cfl":
        ax.set_yscale("log")
    ax.grid(True, which='both', axis='both')

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname
        full_file_path = full_directory_path + log_file

        # Get time step size
        dt = get_time_step_size(full_directory_path)

        # Get plot styling
        label, marker, mfc, ls, dir_color = get_label(full_directory_path, dt, raw_label=False)
        print("\nProcessing {0}...".format(label))

        # Read file
        # df = pd.read_csv(full_file_path, sep=',')
        df = pd.read_pickle(full_file_path)
        df = df[log_file_glob_strs[0]]

        print("Possible metrics found in file:", df.keys())

        # Extract time and data
        physTime = df["phys_time"]
        physTime = physTime / ctu_len # Normalise to CTUs
        signal = df[metric]

        # Build mask based on time interval
        physTime, signal = filter_time_interval(physTime, signal, signal_len_from_end)

        # Downsample
        # Note: do this before MSER
        if n_downsample > 1:
            signal = signal[::n_downsample]
            physTime = physTime[::n_downsample]
            label += f" downsample {n_downsample}"

        # Determine end of transient via mser
        if use_mser:
            mser_stride_length = 10 if dt < 5e-5 else 1
            intTransient = mser(signal, physTime, debug_plot = False, stride_length=mser_stride_length)
            timeTransient = physTime.iloc[intTransient]
            print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

            # Plot end of transient
            ax.plot([timeTransient for i in range(2)], [signal.mean() * 1.2, signal.mean() * 0.8],
                    linestyle='dashed', color=dir_color)

        # Check averages w/o transient
        print(f"Average {metric} with transient is:\t{signal.mean()}")


        # Plot
        ax.plot(physTime, signal, color=dir_color, alpha=0.3, label=label)
        mean = signal.mean()
        ax.plot([physTime.iloc[0], physTime.iloc[-1]], [signal.mean() for i in range(2)], color=dir_color,
                alpha=1.0)  # , label="Mean " + label)

        # # Uncomment to plot reverse cumulative mean and std
        # plot_cumulative_mean_std(signal, physTime, ax, dir_color, label)

        # # Add averaging/mean
        # stepsPerCtu = ctu_len / dt
        # filter_width=int(len(signal)/20)
        #
        # ## Savitzky-golay filter
        # metric_smooth = signal.savgol_filter(signal, window_length=filter_width, polyorder=1, mode="interp")
        # #ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color='black', linestyle='dotted', alpha=0.8)
        # ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color=dir_color, linestyle='solid', alpha=0.8, label=label)

        # Handle legend outside
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best')


    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    plt.show()


