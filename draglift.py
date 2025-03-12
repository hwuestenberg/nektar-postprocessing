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

from utilities import get_time_step_size, mser, mse
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ylabels, ynames, ref_area, ctu_len




# Please-work data
savename = ""




# Parse command line arguments                          
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(                                    
    "forces_file", help="Nektar .fce file", type=str, default='FWING_TOTAL_forces-process.fce', nargs='?')
parser.add_argument(
    "ctuSkip", help="Skip time from the start, in CTUs", type=float, default=0.0, nargs='?')
args = parser.parse_args()                              

print("Using forces_file:", args.forces_file)

print("Skipping {0} CTUs from the start".format(
    args.ctuSkip))





def get_label(full_file_path, dt=0, fsample=0):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'

    # Add time step size
    label += "{0: >3d}".format(
            int(round(dt/dtref))
            )
    label += r"$ \Delta t_{CFL}$"

    # Add sampling frequency
    if fsample:
        label += " $f_{sample} =$"
        label += "${0:.1e}$".format(fsample)
        label += " "

    if "linear" in full_file_path:
        label += " linear-implicit"
    elif "semi" in full_file_path:
        label += " semi-implicit"
    elif "substepping" in full_file_path:
        label += " substepping"
    if "quasi3d" in full_file_path:
        label += " Slaughter et al. 2023"
        mfc='None'
        marker='o'

    if "5bl" in full_file_path:
        label += " Mesh A"
    elif "8bl" in full_file_path:
        label += " Mesh B"
    elif "refined" in full_file_path:
        label += " Mesh C"
    elif "please-work" in full_file_path:
        label += " Mesh D"

    if "advfreeze100" in full_file_path:
        label += " afreeze 100"
    elif "advfreeze50" in full_file_path:
        label += " afreeze 50"
    elif "advfreeze20" in full_file_path:
        label += " afreeze 20"
    elif "advfreeze10" in full_file_path:
        label += " afreeze 10"
    elif "advfreeze5" in full_file_path:
        label += " afreeze 5"
    elif "advfreeze2" in full_file_path:
        label += " afreeze 2"
    elif "advfreeze1" in full_file_path:
        label += " afreeze 1"

    return label, marker, mfc


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
        overlap_names = [forces_file.replace("-process", f"-process-overlap-{i}") for i in [5]]#[0, 1, 2, 3, 4, 5, 10]]

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
            label, marker, mfc = get_label(full_file_path, dt)
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
            # Build mask based on time interval
            #lowerMask = time > args.beginAverage
            #upperMask = time < args.endAverage
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
                intTransient = mser(df, metric, debug_plot = True)
                timeTransient = physTime.iloc[intTransient]
                print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))
                ax.plot([timeTransient for i in range(2)], [metricData.mean() * 1.2, metricData.mean() * 0.8],
                        linestyle='dashed', color=file_color, label="End of transient")

                # Plot
                #ax.plot(physTime, metricData, label=label, color=color)
                ax.plot(physTime, metricData, color=file_color, alpha=0.3)

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


