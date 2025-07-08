# Matplotlib setup with latex
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

from utilities import get_time_step_size, get_label, mser, filter_time_interval, check_sampling_rates
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ref_area, ctu_len, ref_velocity, save_directory, file_glob_strs, force_file_skip_start

####### SCRIPT USER INPUTS
# Choose lift or drag
metric = 'u'

time = 8.0

file = file_glob_strs[-1]
averaging_len = 4.1 # [CTU] redundant due to MSER, just use large number

n_downsample = 2
f_target = None#2500  # Downsample to target sample rate for all cases

# Reference frequencies
ref_freq = [21, 29, 40]

# Do 3D-FFT in space
# otherwise does 3D-FFT in time
fft_in_space = False

# Welch parameters
use_welch = True
overlap = 2
windows = 4

xlim = [1e0, 1e4]
ylim = [1e-8, 1e2]

savename = f"PSD-his-{metric}-dt"
savename = save_directory + savename



def compute_energy_spectrum(u, v, w, Lx, Ly, Lz, num_bins=None):
    """
    Compute the isotropic energy spectrum E(k) from 3D velocity fields.

    Parameters
    ----------
    u, v, w : ndarray
        3D arrays of velocity components of shape (Nx, Ny, Nz).
    Lx, Ly, Lz : float
        Physical domain lengths in x, y, z directions.
    num_bins : int, optional
        Number of k-shells. Defaults to max(Nx,Ny,Nz)//2.

    Returns
    -------
    k_vals : ndarray
        Midpoint wavenumbers of each bin.
    E_spectrum : ndarray
        Energy summed in each k-shell.
    """
    # Grid sizes
    Nx, Ny, Nz = u.shape

    # Forward FFT (normalized)
    norm = (Nx * Ny * Nz)
    u_hat = np.fft.fftn(u) / norm
    v_hat = np.fft.fftn(v) / norm
    w_hat = np.fft.fftn(w) / norm

    # Spectral energy density
    E_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

    # Wavenumber components
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * np.pi
    kz = np.fft.fftfreq(Nz, d=Lz / Nz) * 2.0 * np.pi
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2).ravel()
    E_flat = E_density.ravel()

    # Choose number of bins
    k_max = k_mag.max()
    if num_bins is None:
        num_bins = max(u.shape) // 2

    # Bin edges and midpoints
    k_edges = np.linspace(0.0, k_max, num_bins + 1)
    k_vals = 0.5 * (k_edges[:-1] + k_edges[1:])

    # Integrate energy in shells
    bin_indices = np.digitize(k_mag, k_edges)
    E_spectrum = np.zeros(num_bins)
    for i in range(1, num_bins + 1):
        mask = bin_indices == i
        E_spectrum[i - 1] = E_flat[mask].sum()

    return k_vals, E_spectrum





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
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname
        file_extension = "." + file.split('.')[-1]
        filename = file.replace(file_extension, f"-process-overlap-{force_file_skip_start}{file_extension}")
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
        npoints = int(len(df['Time']) / len(df['Time'].unique()))


        if fft_in_space:
            # TODO extract coordinates during preprocessor.py
            df_coord = pd.read_csv(full_file_path.replace(f"-process-overlap-{force_file_skip_start}{file_extension}", file_extension),
                                   nrows=npoints,
                                   sep=r"(?!#)\s+", engine='python',
                                   header=None, skiprows=1, usecols=[2,3,4], names=["x", "y", "z"])
            Lx = df_coord['x'].max()
            Ly = df_coord['y'].max()
            Lz = df_coord['z'].max()
            Nx = len(df_coord['x'].unique())
            Ny = len(df_coord['y'].unique())
            Nz = len(df_coord['z'].unique())
            for t in np.linspace(15, 30, 3   ):

                cmap = matplotlib.colormaps.get_cmap("Oranges")  # Use the "Oranges" colormap
                if "weak" in full_file_path:
                    cmap = matplotlib.colormaps.get_cmap("Greens")  # Use the "Oranges" colormap
                t_color = cmap(0.2 + 0.8 * t / 30)

                # Extract data at given time
                physTime = df["Time"]
                time_idx = np.argmin(abs(physTime.unique() - t))
                time_near = physTime.unique()[time_idx]
                print(f"time_near = {time_near}")

                df_time = df[df['Time'] == time_near]

                # Reshape velocity into 3D array
                u = df_time['u'].to_numpy().reshape((Nx, Ny, Nz))
                v = df_time['v'].to_numpy().reshape((Nx, Ny, Nz))
                w = df_time['w'].to_numpy().reshape((Nx, Ny, Nz))

                # Now do some sort of 3D FFt (np.fftn) stuff
                k_vals, E_spec = compute_energy_spectrum(u, v, w, Lx, Ly, Lz)

                ax.plot(k_vals, E_spec, label=f"{label}, t = {time_near:.2f}", color=t_color, linestyle=ls)

        else:
            # Split for each location
            for i, pos_color in zip([8], TABLEAU_COLORS):
                # i = npoints-1
                print("Procesing history point {0}...".format(i))
                pos_label = label + " location {0}".format(i)
                savename += f"loc-{i}"

                # Extract time series for a given point
                dfi = df.iloc[i::npoints]

                # Extract time and data
                physTime = dfi["Time"]
                physTime = physTime / ctu_len # Normalise to CTUs
                signal = dfi[metric]

                # Normalisation: remove mean
                # This should only affect the zero frequency (mean) mode
                signal = signal - signal.mean()

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
                    ax.plot(freq_welch, psd_welch / psd_welch[0], label=label, linestyle=ls, color=dir_color)
                else:
                    # Do FFT
                    fft_vals = np.fft.rfft(signal)
                    fft_freqs = np.fft.rfftfreq(len(signal), 1 / f_sample)
                    psd_fft = np.abs(fft_vals) ** 2 / len(signal)
                    # label = "FFT " + label
                    ax.plot(fft_freqs, psd_fft, label=pos_label, color=pos_color)



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
    for st in ref_freq:
        ax.plot([st, st], ylim, linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st, ylim[0], str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')

    fig.savefig(savename + ".pdf", bbox_inches="tight")
    print(f"Wrote file {savename} as pdf")

    plt.show()
