# Matplotlib setup with latex
import os
from itertools import cycle

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params) 
from matplotlib.colors import TABLEAU_COLORS

import numpy as np
import pandas as pd
from scipy.signal import welch

from case_processing import iter_case_metadata
from config import (
    directory_names,
    path_to_directories,
    dtref,
    customMetrics,
    ref_area,
    ctu_len,
    ref_velocity,
    save_directory,
    history_file_glob_strs,
    force_file_skip_start,
)


############################
####### SCRIPT USER INPUTS
# Choose lift or drag
metric = 'tke'

time = 8.0

file = history_file_glob_strs[3]
file_noext = file.split('.')[0]

n_downsample = 2
f_target = None#2500  # Downsample to target sample rate for all cases

# Reference frequencies
ref_freq = []#[21, 29, 40]


# Welch parameters
use_welch = True
overlap = 2
windows = 4

xlim = []#[1e0, 1e4]
ylim = []#[1e-8, 1e2]

savename = f"PSD-his-{metric}-" + file_noext
savename = save_directory + savename



if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"$PSD({0})$".format(metric)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Wave number $k$")
    if ctu_len != 1.0:
        ax.set_xlabel("Strouhal number $St$")
    ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', axis='both')

    # Loop all files
    for metadata in iter_case_metadata(
        file_name="historypoints.pkl",
        directory_names=directory_names,
        base_path=path_to_directories,
    ):
        dt = metadata.dt
        label = metadata.label
        print("\nProcessing {0}...".format(label))

        # Read file
        df = pd.read_pickle(metadata.file_path)
        df = df[file_noext].dropna()
        npoints = len(df.index.unique('point'))

        colors = TABLEAU_COLORS
        if npoints > 9:
            cmap = plt.get_cmap("viridis")
            colors = [cmap(i) for i in np.linspace(0, 1, npoints)]



        # Split for each location
        for i, pos_color in zip([i for i in range(0,npoints)], colors):
            # i = npoints-1
            print("Procesing history point {0}...".format(i))
            pos_label = label + " location {0}".format(i)
            if npoints > 9:
                pos_label = None
            # savename += f"-loc-{i}"

            # Extract time series for a given point
            dfi = df.xs(i, level='point')

            # Extract time and data
            physTime = dfi["Time"]
            physTime = physTime / ctu_len # Normalise to CTUs

            if metric == "tke":
                signal = 0.5 * (dfi['u']**2 + dfi['v']**2 + dfi['w']**2)
            else:
                signal = dfi[metric]

            mean = signal.mean()
            print(f"Metric {metric} mean = {mean:.4e}")

            # Normalisation: remove mean
            # This should only affect the zero frequency (mean) mode
            signal = signal - mean

            # Get raw sample frequency
            f_sample = max([
                1 / (physTime.iloc[1] - physTime.iloc[0]),
                1 / (physTime.iloc[9] - physTime.iloc[8]),
                1 / (physTime.iloc[5] - physTime.iloc[4]),
                1 / (physTime.iloc[99] - physTime.iloc[98]),
            ])
            print(f"Estimated sample frequency: {f_sample} \t vs expected frequency: {1 / dt * ctu_len}")

            ## Downsample (to target frequency)
            # However, not for reference data
            if f_target:
                n_downsample = int(round(f_sample / f_target, 0)) if int(round(f_sample / f_target, 0)) > 1 else 1

            if n_downsample > 1:  # and not "quasi3d" in full_directory_path:
                signal = signal[::n_downsample]
                physTime = physTime[::n_downsample]
                f_sample = f_sample / n_downsample
                print(f"Downsampled sample frequency: {f_sample}")
                # label += f" downsample {n_downsample}"

            if use_welch:
                nperseg = round(float(signal.shape[0]) / windows)  # Dividing the length of the sample
                freq_welch, psd_welch = welch(signal, fs=f_sample, nperseg=nperseg)  # , noverlap=overlap)
                # label = "Welch " + label
                ax.plot(freq_welch, psd_welch / psd_welch[0], label=pos_label, linestyle=ls, color=pos_color)
            else:
                # Do FFT
                coeffs = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), 1 / f_sample)

                # one-sided energy spectrum (velocity PSD-like)
                # factor 2 because we dropped negative freqs
                n = len(signal)
                Suu = (2.0 / (n ** 2)) * (np.abs(coeffs) ** 2)# / win_norm ** 2
                # psd_fft = np.abs(coeffs) ** 2 / len(signal)
                # label = "FFT " + label
                ax.plot(freqs, Suu, label=pos_label, color=pos_color)



    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print reference -5/3 slope
    xlim = np.array([3e0, 1e3])
    points =  xlim**(-5/3) * 1e1
    ax.plot(xlim, points, label='Reference -5/3', linestyle='dashed', color='black')
    # points = xlim ** (-30 / 3) * 1e3
    # ax.plot(xlim, points, label='Reference -30/3', linestyle='dashed', color='black')

    # Print reference frequencies
    if ref_freq:
        for st in ref_freq:
            ax.plot([st, st], ylim, linestyle='dashed', color='blue', alpha=0.3)
            ax.text(st, ylim[0], str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')

    fig.savefig(savename + ".pdf", bbox_inches="tight")
    print(f"Wrote file {savename} as pdf")

    plt.show()
