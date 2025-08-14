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
from scipy.signal import welch

from utilities import get_time_step_size, get_label, mser, filter_time_interval, check_sampling_rates
from config import (
    directory_names,
    path_to_directories,
    dtref,
    customMetrics,
    ref_area,
    ctu_len,
    ref_velocity,
    save_directory,
    force_file_glob_strs,
    force_file_skip_start,
)

####### SCRIPT USER INPUTS
# Choose lift or drag
metric = customMetrics[1]

forces_file = force_file_glob_strs[0]
averaging_len = 10.1 # [CTU] redundant due to MSER, just use large number

n_downsample = 2
f_target = None#12500  # Downsample to target sample rate for all cases

# Reference frequencies
ref_freq = [21, 29, 40]#, 60, 140, 200]

# Welch parameters
use_welch = True
overlap = 2
windows = 4

xlim = []
ylim = [1e-4, 1e4]

savename = f"PSD-dt-{metric}-{forces_file.split('.')[0]}"
# savename = f"PSD-scheme-{metric}-{forces_file.split('.')[0]}"
# savename = f"PSD-james-{metric}-{forces_file.split('.')[0]}"
savename = save_directory + savename




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
    ax.set_xlabel("Frequency [Hz]")
    if ctu_len != 1.0:
        ax.set_xlabel("Strouhal number $St$")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname
        _, file_extension = os.path.splitext(forces_file)
        filename = forces_file.replace(f"{file_extension}", f"-process-overlap-{force_file_skip_start}{file_extension}")
        full_file_path = full_directory_path + filename

        # Get time step size
        # Note for James' data, we cannot detect 4e-6 from force file
        # because the sampling rate is set to 4e-5
        dt = get_time_step_size(full_directory_path)

        # Get plot styling
        label, marker, mfc, ls, color = get_label(full_file_path, dt, color=dir_color)
        print("\nProcessing {0}...".format(label))

        # Read file
        df = pd.read_csv(full_file_path, sep=',')

        # Extract time and data
        physTime = df["Time"]
        physTime = physTime / ctu_len # Normalise to CTUs
        signal = df[metric]

        # check_sampling_rates(physTime, True)

        # Build mask based on time interval
        physTime, signal = filter_time_interval(physTime, signal, averaging_len)

        # Correct data (coeff = 2 * Force)
        signal = 2 * signal

        # Normalise by area
        # Note quasi-3d is averaged along spanwise
        if "quasi3d" in full_file_path:
            signal = signal / ctu_len
        else:
            signal = signal / ref_area

        # Get raw sample frequency
        f_sample = max([
                1 / (physTime.iloc[1] - physTime.iloc[0]),
                1 / (physTime.iloc[9] - physTime.iloc[8]),
                1 / (physTime.iloc[5] - physTime.iloc[4]),
                4 / (physTime.iloc[5] - physTime.iloc[1]),
                1 / (physTime.iloc[99] - physTime.iloc[98]),
            ])
        if f_sample == np.inf:
            f_sample = 1 / dt * ctu_len
        print(f"Estimated sample frequency: {f_sample} \t vs expected frequency: {1 / dt * ctu_len}")

        ## Downsample (to target frequency)
        # However, not for reference data
        if f_target:
            n_downsample = int(round(f_sample / f_target, 0)) if int(round(f_sample / f_target, 0)) > 1 else 1

        if n_downsample > 1: # and not "quasi3d" in full_directory_path:
            signal = signal[::n_downsample]
            physTime = physTime[::n_downsample]
            f_sample = f_sample / n_downsample
            print(f"Downsampled sample frequency: {f_sample}")
            # label += f" downsample {n_downsample}"

        # # Determine end of transient via mser
        # mser_stride_length = 10 if dt < 5e-5 else 1
        # intTransient = mser(signal, physTime, stride_length=mser_stride_length)
        # timeTransient = physTime.iloc[intTransient]
        # print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))
        #
        # # Remove end of transient from signal
        # signal = signal.iloc[intTransient:]

        # Normalisation: remove mean
        # This should only affect the zero frequency (mean) mode
        signal = signal - signal.mean()


        # Compute FFT using Welch's method
        # Note outputs PSD directly
        if use_welch:
            nperseg = round(float(signal.shape[0]) / windows)  # Dividing the length of the sample
            freq_welch, psd_welch = welch(signal, fs=f_sample, nperseg=nperseg)#, noverlap=overlap)
            # label = "Welch " + label
            ax.plot(freq_welch, psd_welch / psd_welch[0], label=label, linestyle=ls, color=dir_color)

        # Compute FFT directly
        # Note need to compute PSD via abs()**2
        else:
            fft_vals = np.fft.rfft(signal)
            fft_freqs = np.fft.rfftfreq(len(signal), 1 / f_sample)
            psd_fft = np.abs(fft_vals) ** 2 / len(signal)
            # label = "FFT " + label
            ax.plot(fft_freqs, psd_fft, label=label, color=color)


    ## Aesthetics
    # Set x/y-limits

    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print reference -5/3 slope
    xlim = np.array([1e2, 1e3])
    points =  xlim**(-5/3) * 1e5
    ax.plot(xlim, points, label='Reference -5/3', linestyle='dashed', color='black')
    # xlim = np.array([1e2, 5e2])
    # points =  xlim**(-30/3) * 1e22
    # ax.plot(xlim, points, label='Reference -30/3', linestyle='dotted', color='black')

    # Print reference frequencies
    for st in ref_freq:
        ax.plot([st, st], ylim, linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st, ylim[0]*10, str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')
    ax.grid(True, which='both', axis='both')

    fig.savefig(savename + ".pdf", bbox_inches="tight")
    print(f"Wrote file {savename} as pdf")

    plt.show()
