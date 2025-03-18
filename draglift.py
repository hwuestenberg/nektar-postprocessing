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

from utilities import get_time_step_size, get_label, mser
from config import directory_names, path_to_directories, \
    customMetrics, ylabels, ynames, ref_area, ctu_len





# Prefix for any saved figures
savename = ""



# Parse command line arguments                          
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(                                    
    "forces_file", help="Nektar .fce file", type=str, default='DragLift-process.fce', nargs='?')
parser.add_argument(
    "ctuSkip", help="Skip time from the start, in CTUs", type=float, default=0.0, nargs='?')
args = parser.parse_args()                              


# Verbose prints
print("Using forces_file:", args.forces_file)
print("Skipping {0} CTUs from the start".format(args.ctuSkip))



if __name__ == "__main__":
    # Create figures and axes
    figs = list()
    axes = list()
    for metric, ylabel in zip(customMetrics, ylabels):
        figs.append(plt.figure(figsize=(8,4)))
        axes.append(figs[-1].add_subplot(111))
        axes[-1].set_ylabel(ylabel)
        axes[-1].ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
        axes[-1].set_xlabel(r"$t^\star$")
        axes[-1].grid(True, which='both')

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        forces_file = args.forces_file
        overlap_names = [forces_file.replace("-process", f"-process-overlap-{i}") for i in [0]]#[0, 1, 2, 3, 4, 5, 10]]

        for filename, file_color in zip(overlap_names, TABLEAU_COLORS):
            print(f"Processing {filename}")
            full_file_path = full_directory_path + filename

            # Get time step size
            # Note that we cannot detect 4e-6 from force file because the sampling rate is set to 4e-5
            if "quasi3d" in full_directory_path:
                dt = 4e-6
            else:
                dt = get_time_step_size(full_directory_path)

            # Get plot styling
            label, marker, mfc, ls, color = get_label(full_file_path, dt)
            print("\nProcessing {0}...".format(label))

            # Read file
            df = pd.read_csv(full_file_path, sep=',')

            # Extract time and data
            physTime = df["Time"]
            physTime = physTime / ctu_len # Normalise to CTUs

            # Build mask based on time interval
            tmin = physTime.min()
            tmax = physTime.max()
            lowerMask = physTime >= tmin + args.ctuSkip
            upperMask = physTime <= tmax
            mask = (lowerMask == 1) & (upperMask == 1)
            if not np.any(mask):
                print("No data for interval = [{0}, {1}]".format(tmin + args.ctuSkip, tmax))
                continue
            else:
                print("Using data on interval = [{0}, {1}]".format(physTime[mask].iloc[0], physTime[mask].iloc[-1]))
            physTime = physTime[mask]

            for metric, fig, ax in zip(customMetrics, figs, axes):
                print(f"metric: {metric}")
                if metric not in df.columns:
                    print("Could not find {0} in {1}".format(metric, df.columns))
                    continue

                # Extract data
                metricData = df[metric]

                # Reduce with time-interval mask
                metricData = metricData[mask]

                # Correct data (coeff = 2 * Force)
                metricData = 2 * metricData

                # Normalise by area
                if "quasi3d" in full_file_path:
                    metricData = metricData / ctu_len # quasi-3d is averaged along spanwise
                else:
                    metricData = metricData / ref_area

                # Determine end of transient via mser
                # mse(df[['Time', metric]].to_numpy())
                intTransient = mser(df, metric, debug_plot = False)
                timeTransient = physTime.iloc[intTransient]
                print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

                # if "overlap" in filename:
                #     label += " overlap {0}".format(filename.split('overlap-')[1].split('.')[0])

                ax.plot([timeTransient for i in range(2)], [metricData.mean() * 1.2, metricData.mean() * 0.8],
                        linestyle='dashed', color=dir_color, label="End of transient" + " overlap {0}".format(filename.split('overlap-')[1].split('.')[0]))

                print(f"Average {metric} without transient is: {metricData.iloc[intTransient:].mean()}")

                # Plot
                #ax.plot(physTime, metricData, label=label, color=color)
                ax.plot(physTime, metricData, color=dir_color, alpha=0.3, label=label)

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
        for metric, fig, ax in zip(ynames, figs, axes):
            thisSavename = savename + "-" + metric 
            # Check if it contains file extension
            if ".pdf" not in savename:
                thisSavename += ".pdf"
            fig.savefig(thisSavename, bbox_inches="tight")
            fig.savefig(thisSavename.replace("pdf","png"), bbox_inches="tight")


    plt.show()


