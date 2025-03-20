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
    customMetrics, ref_area, ctu_len, freestream_velocity


## SCRIPT USER INPUTS
# Please-work data
savename = ""

# Choose lift or drag
metric = customMetrics[1]

# Welch parameters
overlap = 2
windows = 4

xlim = []
ylim = [1e-16, 1e3]






# Parse command line arguments                          
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(                                    
    "avgLen", help="Length of time interval from latest time for sampling average, in CTU", type=float, default=10, nargs='?')
parser.add_argument(                                    
    "forces_file", help="File that contains the force data", type=str, default="FWING_TOTAL_forces-process.fce", nargs='?')
args = parser.parse_args()


# Verbose prints
print("Using forces_file:", args.forces_file)
print("Averaging over {0} CTUs".format(
    args.avgLen))



"""
    Parv's function
"""
def calculate_frequency(time_array, norm: bool, print_message: bool):
    """!
    Calculate the frequency of the time signal provided
    Print the message if needed

    @param time_array (np.array): numpy array with the force trace of the geometry, can be coefficient of lift, or drag, or any integral value
    @param norm (bool): Whether time_array has been normalised by CTU or not
    @param print_message (float): To print the sampling dt
   
    @return freq (np.float): Sampling frequency of the data. this is typically 1/dt
    """
    #Determine the time-step, frequency of force sampling
    dt = abs(time_array.iloc[1]-time_array.iloc[0]) #Determine the time-step of force sampling
    if print_message:
        if norm:
            print(f"Time step between sampling: {dt} CTUs")
        else:
            print(f"Time step between sampling: {dt} secs")
    freq = 1/dt
    return freq



def calculate_psd(sample_trace, frequency_sample, l_ref: float, u_ref: float, sample_division: int, psd_Noverlap: int) :
    """!
    Calculate the Strouhal number and the PSD using the input signal provided.

    @param forces_trace (np.array): numpy array with the force trace of the geometry, can be coefficient of lift, or drag, or any integral value
    @param frequency_sample (np.float): Sampling frequency of the forces data. this is typically 1/dt, where dt is timestep of sample_trace
    @param l_ref (float): Characteristic length scale of the problem. MOstly the same as value used to normalise for CTU.
    @param u_ref (float): Reference velocity for the problem. The default is 1.0 for Nektar++
    @param sample_division (float): Number of divisions in the signal. This influences nperseg, which has an effect on the PSD at low frequencies
    @param psd_Noverlap (int): Number to divide the nWindow by to calculate the noverlap
    https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
    Note: l_ref, u_ref are exposed but always pass normalised arrays
   
    @return strouhal_number, psd (np.array): Array of Strouhal number, Power spectral density of x strouhal_number
    """
    #Determine the frequency and power spectral density for each sampling point
   
    nWindow = round(float(sample_trace.shape[0])/sample_division) #Dividing the length of the sample
    N_overlap=round(float(nWindow)/psd_Noverlap)
    print(f"nperseg: {nWindow}, \tnoverlap: {N_overlap}")
    # Compute PSD using Welch's method
    frequencies, psd = welch(sample_trace,fs=frequency_sample,nperseg=nWindow)#,
                             #noverlap=N_overlap)
    #how to decide nperseg?, has an effect on energy values on low frequency
    #https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
    #https://www.osti.gov/biblio/5688766
    strouhal_number = frequencies*(l_ref/u_ref)
    return strouhal_number, psd


if __name__ == "__main__":
    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = r"PSD($C_l$)"
    if metric == customMetrics[0]:
        ylabel = r"PSD($C_d$)"
    ax.set_ylabel(ylabel)
    #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xlabel("Strouhal number $St$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        forces_file = args.forces_file
        filename = forces_file.replace("-process", f"-process-overlap-5") # add overlap

        n_downsamples = [i for i in [1, 2, 5, 10]]

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

            # Mask for given length from final time
            tmax = physTime.max()
            lowerMask = physTime >= tmax - args.avgLen
            upperMask = physTime <= tmax
            mask = (lowerMask == 1) & (upperMask == 1)
            if not np.any(mask):
                print("No data for interval = [{0}, {1}]".format(tmax - args.avgLen, tmax))
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

            ## Downsample (
            # TODO check difference of downsampling BEFORE and AFTER normalisation
            if n_downsample > 1:
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
            freq_welch, psd_welch = calculate_psd(metricData, f_sample, 1.0, freestream_velocity, windows, overlap)
            ax.plot(freq_welch, psd_welch, label="Welch " + label, linestyle=ls, color=downsample_color)

            # # Compute FFT directly
            # # Note need to compute PSD via abs()**2
            # fft_vals = np.fft.rfft(metricData)
            # fft_freqs = np.fft.rfftfreq(len(metricData), 1 / f_sample)
            # psd_fft = np.abs(fft_vals) ** 2 / len(metricData)
            # ax.plot(fft_freqs, psd_fft, label=label, color=downsample_color)


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
    stref = [20, 30, 40, 50, 60]
    for st in stref:
        ax.plot([st, st], [1e2,1e-16], linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st, 1e-16, str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')

    # if savename:
    #     if ".pdf" not in savename:
    #         savename += ".pdf"
    #     #savename = savename.replace("pdf","png") # switch to png
    #     fig.savefig(savename, bbox_inches="tight")

    plt.show()
