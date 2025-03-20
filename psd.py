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
from scipy.signal import welch

from utilities import get_time_step_size, get_label, mser
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ref_area, ctu_len, freestream_velocity, save_directory

####### SCRIPT USER INPUTS
# Choose lift or drag
metric = customMetrics[1]

forces_file = "FWING_TOTAL_forces.fce"
averaging_len = 10 # [CTU] redundant due to MSER, just use large number

savename = f"PSD-{metric}-{forces_file.split('.')[0]}.pdf"
savename = save_directory + savename

# Welch parameters
use_welch = True
overlap = 2
windows = 4

xlim = []
ylim = [1e-10, 1e-2]



# Verbose prints
print("Using forces_file:", forces_file)
# print("Averaging over {0} CTUs".format(averaging_len))


if __name__ == "__main__":
    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"PSD($C_l$)"
    if metric == customMetrics[0]:
        ylabel = r"PSD($C_d$)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Strouhal number $St$")
    #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', axis='both')

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        forces_file = forces_file
        filename = forces_file.replace(".fce", f"-process-overlap-5.fce") # add overlap

        n_downsamples = [i for i in [2]]#, 2, 5, 10]]

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
            label, marker, mfc, ls, color = get_label(full_file_path, dt, color=downsample_color)
            print("Processing {0}...".format(label))

            # Read file
            df = pd.read_csv(full_file_path, sep=',')

            # Extract time and data
            physTime = df["Time"]
            physTime = physTime / ctu_len # Normalise to CTUs

            # Mask for given length from final time
            tmax = physTime.max()
            lowerMask = physTime >= tmax - averaging_len
            upperMask = physTime <= tmax
            mask = (lowerMask == 1) & (upperMask == 1)
            if not np.any(mask):
                print("No data for interval = [{0}, {1}]".format(tmax - averaging_len, tmax))
                continue
            else:
                print("Using data on interval = [{0}, {1}]".format(physTime[mask].iloc[0], physTime[mask].iloc[-1]))

            # Reduce data set based on mask
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

            # # Determine end of transient via mser
            # intTransient = mser(metriData, physTime, debug_plot=False)
            # timeTransient = physTime.iloc[intTransient]
            # print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))


            # Get raw sample frequency
            f_sample = 1 / (physTime.iloc[1] - physTime.iloc[0])
            print(f"Raw sample frequency: {f_sample}")


            ## Downsample
            # However, not for reference data
            # TODO check difference of downsampling BEFORE and AFTER normalisation
            if n_downsample > 1 and not "quasi3d" in full_directory_path:
                metricData = metricData[::n_downsample]
                physTime = physTime[::n_downsample]
                f_sample = f_sample / n_downsample
                print(f"Downsampled sample frequency: {f_sample}")
                label += f" downsample {n_downsample}"


            # Normalisation: remove mean
            # TODO check effect of normalisation on PSD
            metricData = metricData - metricData.mean()


            # Compute FFT using Welch's method
            # Note outputs PSD directly
            if use_welch:
                nperseg = round(float(metricData.shape[0]) / windows)  # Dividing the length of the sample
                freq_welch, psd_welch = welch(metricData, fs=f_sample, nperseg=nperseg)#, noverlap=overlap)
                ax.plot(freq_welch, psd_welch, label="Welch " + label, linestyle=ls, color=color)

            # Compute FFT directly
            # Note need to compute PSD via abs()**2
            else:
                fft_vals = np.fft.rfft(metricData)
                fft_freqs = np.fft.rfftfreq(len(metricData), 1 / f_sample)
                psd_fft = np.abs(fft_vals) ** 2 / len(metricData)
                ax.plot(fft_freqs, psd_fft, label="FFT " + label, color=color)


    ## Aesthetics
    # Set x/y-limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid()

    ## Print reference -5/3 slope
    #xlim = np.array([100.0, 1000])
    #points =  xlim**(-5/3) * 1e-3
    #ax.plot(xlim, points, label='Reference -5/3', linestyle='dashed', color='black')
    #xlim = np.array([30.0, 200])
    #points =  xlim**(-25/3) * 1e+5
    #ax.plot(xlim, points, label='Reference -25/3', linestyle='dotted', color='black')

    # Print reference Strouhal numbers
    stref = [20, 30, 40, 60, 140, 200]
    for st in stref:
        ax.plot([st, st], [1e5,1e-16], linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st, 10*ylim[0], str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')

    if savename:
        fig.savefig(savename, bbox_inches="tight")

    plt.show()
