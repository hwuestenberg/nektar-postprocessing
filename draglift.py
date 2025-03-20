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

from utilities import get_time_step_size, get_label, mser, plot_cumulative_mean_std
from config import directory_names, path_to_directories, \
    customMetrics, ref_area, ctu_len, save_directory

# Choose lift [1] or drag [0]
metric = customMetrics[1]

forces_file = "FWING_TOTAL_forces-process.fce"
ctu_skip = 0.0 # sort of redundant with MSER

# Prefix for any saved figures
savename = f"{metric}-{forces_file.split('.')[0]}.pdf"
savename = save_directory + savename

# Plot adjusting
xlim  = []
ylim  = []



# Verbose prints
print("Using forces_file:", forces_file)
print("Skipping {0} CTUs from the start".format(ctu_skip))



if __name__ == "__main__":
    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"PSD($C_l$)"
    if metric == customMetrics[0]:
        ylabel = r"PSD($C_d$)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$t^\star$")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.grid(True, which='both', axis='both')

    # Loop all files
    for dirname in directory_names:
        # Setup paths
        full_directory_path = path_to_directories + dirname

        filename = forces_file.replace("-process", f"-process-overlap-5") # add overlap

        n_downsamples = [i for i in [1]]#, 2, 5, 10, 20, 100]]

        for n_downsample, downsample_color in zip(n_downsamples, TABLEAU_COLORS):
            print(f"\nProcessing {filename} with downsampling {n_downsample}")
            full_file_path = full_directory_path + filename

            # Get time step size
            # Note that we cannot detect 4e-6 from force file
            # because the sampling rate is set to 4e-5
            if "quasi3d" in full_directory_path:
                dt = 4e-6
            else:
                dt = get_time_step_size(full_directory_path)

            # Get plot styling
            label, marker, mfc, ls, color = get_label(full_file_path, dt)
            print("Processing {0}...".format(label))

            # Read file
            df = pd.read_csv(full_file_path, sep=',')

            # Extract time and data
            physTime = df["Time"]
            physTime = physTime / ctu_len # Normalise to CTUs

            # Build mask based on time interval
            tmin = physTime.min()
            tmax = physTime.max()
            lowerMask = physTime >= tmin + ctu_skip
            upperMask = physTime <= tmax
            mask = (lowerMask == 1) & (upperMask == 1)
            if not np.any(mask):
                print("No data for interval = [{0}, {1}]".format(tmin + ctu_skip, tmax))
                continue
            else:
                print("Using data on interval = [{0}, {1}]".format(physTime[mask].iloc[0], physTime[mask].iloc[-1]))
            physTime = physTime[mask]

            # Extract signal
            metricData = df[metric]

            # Reduce using time-interval mask
            metricData = metricData[mask]

            # Correct data (coeff = 2 * Force)
            metricData = 2 * metricData

            # Normalise by area
            # Note quasi-3d is averaged along spanwise
            if "quasi3d" in full_file_path:
                metricData = metricData / ctu_len
            else:
                metricData = metricData / ref_area

            # Downsample
            # Note: do this before MSER
            if n_downsample > 1:
                metricData = metricData[::n_downsample]
                physTime = physTime[::n_downsample]
                label += f" downsample {n_downsample}"

            # Determine end of transient via mser
            intTransient = mser(metricData, physTime, debug_plot = False)
            timeTransient = physTime.iloc[intTransient]
            print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

            # Check averages w/o transient
            print(f"Average {metric} with transient is:\t{metricData.mean()}")
            print(f"Average {metric} without transient is:\t{metricData.iloc[intTransient:].mean()}")

            # Show end of transient
            ax.plot([timeTransient for i in range(2)], [metricData.mean() * 1.2, metricData.mean() * 0.8],
                    linestyle='dashed', color=downsample_color)

            # Plot
            ax.plot(physTime, metricData, color=downsample_color, alpha=0.3, label=label)


            # # Uncomment to plot reverse cumulative mean and std
            # plot_cumulative_mean_std(metricData, physTime, ax, dir_color, label)

            # # Add averaging/mean
            # stepsPerCtu = ctu_len / dt
            # filter_width=int(len(metricData)/20)
            #
            # ## Savitzky-golay filter
            # metric_smooth = signal.savgol_filter(metricData, window_length=filter_width, polyorder=1, mode="interp")
            # #ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color='black', linestyle='dotted', alpha=0.8)
            # ax.plot(physTime[filter_width:-filter_width], metric_smooth[filter_width:-filter_width], color=dir_color, linestyle='solid', alpha=0.8, label=label)

            ## Aesthetics
            # # Set x/y-limits
            # if xlim:
            #     ax.set_xlim(xlim)
            # if ylim:
            #     ax.set_ylim(ylim)

            # Handle legend outside
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='best')


    if savename:
        thisSavename = savename + "-" + metric
        # Check if it contains file extension
        if ".pdf" not in savename:
            thisSavename += ".pdf"
        fig.savefig(thisSavename, bbox_inches="tight")
        fig.savefig(thisSavename.replace("pdf","png"), bbox_inches="tight")


    plt.show()


