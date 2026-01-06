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

from case_processing import iter_force_cases
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
forces_file_noext = forces_file.split('.')[0]
ctu_skip = 1e10 # sort of redundant with MSER

n_downsample = 2
f_target = None#12500  # Downsample to target sample rate for all cases

# Reference frequencies
ref_freq = [21, 29, 40]#, 60, 140, 200]

# Welch parameters
use_welch = True
overlap = 2
windows = 8

xlim = [1e0, 2e3]
ylim = [1e-1, 1e4]

savename = f"psd-dt-{metric}-{forces_file.split('.')[0]}"
# savename = f"psd-scheme-{metric}-{forces_file.split('.')[0]}"
# savename = f"psd-james-{metric}-{forces_file.split('.')[0]}"
savename = save_directory + savename




# Verbose prints
print("Using forces_file:", forces_file)



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(4,2))
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
        ls = force_case.metadata.linestyle
        print("\nProcessing {0}...".format(label))

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

        # Normalisation: remove mean
        # This should only affect the zero frequency (mean) mode
        signal = signal - signal.mean()


        # Compute FFT using Welch's method
        # Note outputs PSD directly
        if use_welch:
            nperseg = round(float(signal.shape[0]) / windows)  # Dividing the length of the sample
            print(f"nperseg = {nperseg}")
            freq_welch, psd_welch = welch(signal, fs=f_sample, nperseg=nperseg)#, noverlap=overlap)
            # label = "Welch " + label
            ax.plot(freq_welch, psd_welch / psd_welch[0], label=label, linestyle=ls, color=color)

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
    points =  xlim**(-5/3) * 1e6
    ax.plot(xlim, points, label='Reference -5/3', linestyle='dashed', color='black')
    # xlim = np.array([1e2, 5e2])
    # points =  xlim**(-30/3) * 1e22
    # ax.plot(xlim, points, label='Reference -30/3', linestyle='dotted', color='black')

    # Print reference frequencies
    i = -5 # for spacing
    for st in ref_freq:
        ax.plot([st, st], ylim, linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st + i, ylim[0]*2, str(st))
        i = i + 5

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(loc='best')

    ax.legend(
        loc='lower center',  # position legend at the bottom center of the bbox
        bbox_to_anchor=(0.5, 1.02),  # 0.5 = center horizontally, 1.02 = slightly above the axes
        ncol=2,  # number of columns (optional)
        frameon=True  # remove the box (optional)
    )

    fig.savefig(savename + ".pdf", bbox_inches="tight")
    print(f"Wrote file {savename} as pdf")

    plt.show()
