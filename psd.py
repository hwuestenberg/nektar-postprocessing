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

from utilities import get_time_step_size
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ref_area, ctu_len, freestream_velocity

# Please-work data
savename = ""


# Parse command line arguments                          
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(                                    
    "avgLen", help="Length of time interval from latest time for sampling average, in CTU", type=float, default=2.5, nargs='?')
parser.add_argument(                                    
    "forces_file", help="File that contains the force data", type=str, default="FWING_TOTAL_forces-process.fce", nargs='?')
parser.add_argument(                                    
    "plotDrag", help="Boolean that chooses to analyse drag or lift signal", type=int, default=0, nargs='?')
args = parser.parse_args()


# Verbose prints
print("Using forces_file:", args.forces_file)

if args.plotDrag:
    print("Analysing drag signal..")
else:
    print("Analysing lift signal..")

print("Averaging over {0} CTUs".format(
    args.avgLen / 0.25))
#print("Averaging the interval: [{0}, {1}]".format(
#    args.beginAverage, args.endAverage))


# PSD parameters
overlap = 1
windows = 1


xlim = []
ylim = [1e-16, 1e3]


"""
    Minor utility functions
"""
def getlabel(casestr, color, dt=0, fsample=0):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'
    ls='solid'
    color = color

    # Add time step size
    if dt < dtref:
        label += "${0:.1f}$".format(
                round(dt/dtref,1)
                )
    else:
        label += "${0:d}$".format(
                int(round(dt/dtref))
                )
    label += r"$ \Delta t_{CFL}$"

    # Add reynolds number
    if "/re" in casestr:
        re = casestr.split("/re")[-1].split("/")[0]
        label += " "
        label += "Re = {0:.1e}".format(float(re))

    # Add sampling frequency
    if fsample:
        label += " $f_{sample} =$"
        label += "${0:.1e}$".format(fsample)
        label += " "

    if "linearimplicit" in casestr:
        label += " linear-implicit"
    elif "semi" in casestr:
        label += " semi-implicit"
    elif "substep" in casestr:
        label += " substepping"
    elif "quasi3d" in casestr:
        label += " Slaughter et al. (2023)"
        color = 'black'

    #if "5bl" in casestr:
    #    label += " Mesh A"
    #elif "8bl" in casestr:
    #    label += " Mesh B"
    #elif "refined" in casestr:
    #    label += " Mesh C"
    #elif "please-work" in casestr:
    #    label += " Mesh D"

    return label, marker, mfc, ls, color

"""
    Parv's function
"""
def normalize_signal(sample_array):
    """!
    Normalise signal around mean, bring around 0.

    @param sample_array (np.array): numpy array with the force trace of the geometry, can be coefficient of lift, or drag, or any integral value
    @return normalized_signal (np.array)
    """
    mean_array = np.mean(sample_array)
    normalized_signal = sample_array[:] - mean_array
    return normalized_signal



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
    print("nperseg:", nWindow)
    N_overlap=round(float(nWindow)/psd_Noverlap)
    print("noverlap:", N_overlap)
    # Compute PSD using Welch's method
    frequencies, psd = welch(sample_trace,fs=frequency_sample,nperseg=nWindow)#,
                             #noverlap=N_overlap)
    #how to decide nperseg?, has an effect on energy values on low frequency
    #https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
    #https://www.osti.gov/biblio/5688766
    print(f"Normalisation values used for length={l_ref}, u_ref={u_ref}")
    strouhal_number = frequencies*(l_ref/u_ref)
    return strouhal_number, psd


if __name__ == "__main__":
    # Create figure and axis
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ylabel = "PSD"
    if args.plotDrag:
        ylabel = r"PSD($C_d$)"
    else:
        ylabel = r"PSD($C_l$)"
    ax.set_ylabel(ylabel)
    #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xlabel("Strouhal number")
    ax.set_xscale("log")
    ax.set_yscale("log")

    figForce = plt.figure(figsize=(4,3))
    axForce = figForce.add_subplot(111)
    ylabel = "Force"
    if args.plotDrag:
        ylabel = r"$C_d$"
    else:
        ylabel = r"$C_l$"
    axForce.set_ylabel(ylabel)
    #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    axForce.set_xlabel(r"$t^\star$")

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        forces_file = args.forces_file
        overlap_names = [forces_file.replace("-process", f"-process-overlap-{i}") for i in [0, 3, 5]]#[0, 1, 2, 3, 4, 5, 10]]

        for filename, file_color in zip(overlap_names, TABLEAU_COLORS):
            print(f"Processing {filename}")
            full_file_path = full_directory_path + filename

            if "quasi3d" in full_directory_path:
                dt = 4e-6
            else:
                dt = get_time_step_size(full_directory_path)
            label, marker, mfc, ls, dir_color = getlabel(full_directory_path, dir_color, dt)
            print("\nProcessing {0}...".format(label))

            # Read file
            if args.plotDrag:
                metric = customMetrics[0]
            else:
                metric = customMetrics[1]
            df = pd.read_csv(full_file_path, sep=',')

            # Extract time and data
            physTime = df["Time"]
            physTime = physTime# / ctu_len # Normalise to CTUs

            # Mask for given length from final time
            tmax = np.max(physTime)
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

            # Extract data
            metricData = df[metric]

            # Reduce with time-interval mask
            metricData = metricData[mask]

            # Correct data (coeff = 2 * Force)
            metricData = 2 * metricData

            # Normalise by area
            if "quasi3d" in full_file_path:
                metricData = metricData / ctu_len  # quasi-3d is averaged along spanwise
            else:
                metricData = metricData / ref_area

            # Apply MSER for detecting initial transient
            # intTransient = mser(df, metric, debug_plot = True)

            # Apply downsampling?

            # Pre-process
            metricData = normalize_signal(metricData)
            f_sample = calculate_frequency(physTime, False, True)
            print("f_sample: ", f_sample)

            ## Downsample
            #nsplit = int(round(f_sample / min_fsample))
            #metricData = metricData[::nsplit]
            #physTime = physTime[::nsplit]
            #f_sample = f_sample/nsplit
            #print("Downsampled f_sample: ", f_sample)

            # freq_welch, psd_welch = calculate_psd(metricData, f_sample, 1.0, freestream_velocity, windows, overlap)
            # ax.plot(freq_welch * ctu_len, psd_welch, label=label, linestyle=ls)

            fft_vals = np.fft.rfft(metricData)
            fft_freqs = np.fft.rfftfreq(len(metricData), 1/f_sample)

            # Step 3: Compute the Power Spectral Density (PSD)
            psd_fft = np.abs(fft_vals) ** 2 / len(metricData)

            if "overlap" in filename:
                label += " overlap {0}".format(filename.split('overlap-')[1].split('.')[0])

            ax.plot(fft_freqs * ctu_len, psd_fft, label=label)


            # Plot
            #ax.plot(strouhal * 0.25, psd, label=label, color=color)
            axForce.plot(physTime, metricData, label=label, color=dir_color, linestyle=ls)


    ## Aesthetics
    # Set x/y-limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid()
    axForce.grid()

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
        ax.plot([st, st], [1e-3,1e-16], linestyle='dashed', color='blue', alpha=0.3)
        ax.text(st, 1e-16, str(st))

    # Handle legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best')
    handles, labels = axForce.get_legend_handles_labels()
    axForce.legend(loc='best')


    if savename:
        if ".pdf" not in savename:
            savename += ".pdf"
        #savename = savename.replace("pdf","png") # switch to png
        fig.savefig(savename, bbox_inches="tight")
        # figForce.savefig(savename.replace("psd","psd-raw-force"), bbox_inches="tight")

    plt.show()
