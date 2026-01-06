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
from config import directory_names, path_to_directories, ref_area, ctu_len, save_directory, dtref, log_file_glob_strs

####### SCRIPT USER INPUTS
# Choose lift [1] or drag [0]
metric = 'cfl'

log_file = "log_info.pkl"
signal_len_from_end = 10.0 # in CTUs


# Prefix for any saved figures
savename = f"{metric}-bar"
savename = save_directory + savename




# Verbose prints
print("Using {0} CTUs from the end".format(signal_len_from_end))



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ylabel = r"$\overline{" + f"{metric.upper()}" + r"}$"
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(r"")
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(style='plain',axis='y', scilimits=(0,0), useMathText=True)
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', axis='y')

    unit = 1.0
    bar_position = 0.0
    dtstrs = []
    mult = 1
    width = unit / 2

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

        # Extract data
        signal = df[metric]
        mean = signal.mean()
        std = signal.std()

        dtstrs.append(str(int(round(dt / dtref, 0))))

        # Reset position for reference dt
        if dt <= 2 * dtref:
            bar_position = unit
            mult = 1

        # Compute bar positions
        if "linear-implicit" in label:
            bar_offset = width
        elif "sub-stepping" in label:
            bar_offset = 2 * width
        else:
            bar_offset = 0.0
        bar_position = mult * unit

        rects = ax.bar(bar_position + bar_offset, mean, width=width, color=dir_color, label=label)
        # ax.errorbar(bar_position + bar_offset, mean, yerr=2 * std, fmt="o", color="r") # std is almost invisible
        ax.bar_label(rects, padding=5, fmt="%.1f")

        mult += 2


    # Set xticks
    # most ridiculous and inefficient sorting operation. Thanks python <3
    dtstrs = [float(dt) for dt in np.unique(dtstrs)]
    dtstrs = sorted(dtstrs)
    dtstrs = [str(int(dt))  + r"$\times \Delta t_{CFL}$" for dt in dtstrs]
    print(f"dtstrs: {dtstrs}")
    ax.set_xticks(np.arange(1, len(dtstrs) + 1) * 2 - width, dtstrs)

    lower, upper = ax.get_ylim()
    ax.set_ylim(1.0, upper*1.2)

    # Custom legend
    import matplotlib.patches as mpatches

    # Create dummy bar plot handles
    bars = []
    bars.append(mpatches.Patch(color='tab:blue', label='semi-implicit'))
    bars.append(mpatches.Patch(color='tab:orange', label='linear-implicit'))
    bars.append(mpatches.Patch(color='tab:green', label='sub-stepping'))
    ax.legend(handles=bars)


    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    plt.show()


