#!/usr/bin/env  python3
import os, sys, subprocess, re
from glob import glob
from os.path import basename
from pathlib import Path
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt, cm

import config
# Import for charles plots
# from CharLESForces import *
# from compareForcesCharLES import *

from config import ctu_len, dtref, boundary_names


def get_ctu_names(glob_string, descending=False):

    # Find starts and mids automatically
    files = glob(glob_string)

    # Extracted numbers will be stored here
    starts = []
    mids = []

    # Regex pattern to find two numbers in the filename
    pattern = re.compile(r'ctu_(\d+)_(\d+)_')

    # Loop through the filenames and extract numbers
    for filename in files:
        match = pattern.search(filename)
        if match:
            start, end = match.group(1), match.group(2)
            starts.append(start)
            mids.append(end)

    # Sort descending
    starts = sorted(starts, reverse=descending)
    mids = sorted(mids, reverse=descending)

    # # use only unique
    # starts = list(dict.fromkeys(starts))
    # mids = list(dict.fromkeys(mids))

    return starts, mids



def get_data_frame(filename, skip_start = 0, skip_end = 0):
    """Read force or history point data into a DataFrame.

    The file type is determined from the suffix.  History point files
    (``.his``) are treated differently from force files (``.fce``) since the
    former contain multiple points per time step.  All other extensions are
    handled like force files.
    """

    # Detect file type from suffix
    file_type = Path(filename).suffix

    # Pre-read file and check for 2D/3D
    headerskip = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            #print(i, line)
            if "Time" in line:
                headerskip = i
                break
            if file_type == ".his" and not "#" in line:
                headerskip = i-1
                break

    # Read file
    df = pd.read_csv(filename, header=headerskip, sep=r"(?!#)\s+", engine='python')

    if file_type == ".his":
        n_his_points = headerskip

        # Adjust columns for 2D or 3D
        nvars = len(df.columns)
        if nvars == 5:
            df.columns = ['Time', 'u', 'v', 'w', 'p']
        else:
            df.columns = ['Time', 'u', 'v', 'p']

        # Skip points at start or end, if set
        if skip_start:
            df = df.iloc[(n_his_points*skip_start):]
        if skip_end != 0:
            df = df.iloc[:-(n_his_points*skip_end)]

    # Remove hash-column, necessary for .fce files
    else:
        temp_columns = df.columns[1:]
        df = df.iloc[:,:-1]
        df.columns = temp_columns

        # Skip points at start or end, if set
        if skip_start:
            df = df.iloc[skip_start:]
        if skip_end != 0:
            df = df.iloc[:-skip_end]

    return df


def get_time_step_size(directory_name):
    # james' case is sampled at higher frequency than 1
    if "quasi3d" in directory_name:
        return 4e-6

    # Read cputime.dat
    dftime = pd.read_csv(directory_name + "log_info.csv", sep=",")

    # Get time/step info for x-axis
    phystime = dftime["phys_time"].to_numpy()
    steps = dftime["steps"].to_numpy()

    # Use the entire available range.
    # This avoids an IndexError for short log files and yields a robust average
    # time step size.
    dt = (phystime[11] - phystime[0]) / (steps[11] - steps[0])
    return dt


def get_stabilisation(full_file_path):
    if "gjp" in full_file_path:
        str = " GJP"
    elif "dgsvv" in full_file_path:
        str = " DGSVV"
    else: # default to semi-implicit
        str = ""

    return str

def get_scheme(full_file_path):
    if "linear" in full_file_path:
        scheme = "linear-implicit"# (weak pressure)"
    elif "semi" in full_file_path:
        scheme = "semi-implicit"# (strong pressure)"
    elif "weak" in full_file_path:
        scheme = "semi-implicit"# (weak pressure)"
    elif "substepping" in full_file_path:
        scheme = "sub-stepping"
    elif "quasi3d" in full_file_path:
        scheme = "Slaughter et al. (2023)"
    else: # default to semi-implicit
        # scheme = ""
        scheme = "semi-implicit"# (strong pressure)"

    return scheme

def get_color_by_scheme(full_file_path, dtcfl):
    if "linear" in full_file_path:
        cmap = cm.get_cmap("Oranges")  # Use the "Oranges" colormap
        # color = cmap(0.2 + 0.8 * np.log(dtcfd) / np.log(100)) if dtcfd != 1 else 'tab:orange'
        color = 'tab:orange'
        if dtcfl == 1:
            color = 'tab:blue'
        elif dtcfl > 10:
            color = 'tab:green'
    elif "semi" in full_file_path:
        color = 'tab:blue'
    elif "substepping" in full_file_path:
        cmap = cm.get_cmap("Greens")
        # color = cmap(0.2 + 0.8 * np.log(dtcfd) / np.log(100)) if dtcfd != 1 else 'tab:green'
        color = 'tab:green'
    elif "quasi3d" in full_file_path:
        color = "black"
    elif "weak" in full_file_path:
        color = 'tab:purple'
    else:
        color = "red"

    return color



# Build case specific label and style for plots (define defaults here)
def get_label(full_file_path, dt = 0.0, sampling_frequency = 0, color ='tab:blue', raw_label = False):
    label = ""
    marker = "."
    mfc = 'None'
    ls = 'solid'

    # Add time step size
    dtcfl = round(dt / dtref, 1)
    if dt >= dtref - 0.1 and dt <= dtref + 0.1:
        label += "{0:d}".format(
            int(dtcfl)
        )
    else:
        label += "{0:.1f}".format(
            dtcfl
        )
    label += r"$ \Delta t_{CFL}$"

    # Add sampling frequency
    if sampling_frequency:
        label += " $f_{sample} =$"
        label += "${0:.1e}$".format(sampling_frequency)
        label += " "

    # # Add reynolds number # disabled for channel
    # if "/re" in full_file_path:
    #     re = full_file_path.split("/re")[-1].split("/")[0]
    #     label += " "
    #     label += "Re = {0:.1e}".format(float(re))

    label += f" {get_scheme(full_file_path)}"
    color = get_color_by_scheme(full_file_path, dtcfl)

    label += f" {get_stabilisation(full_file_path)}"

    # Overwrite label for Slaughter et al. (2023)
    if "quasi3d" in full_file_path:
        label = "Slaughter et al. (2023)"
        color = 'black'

    # if "weak" in full_file_path:
    #     ls = 'dashed'

    # Add mesh info
    # if "5bl" in full_file_path:
    #     label += " Mesh A"
    # elif "8bl" in full_file_path:
    #     label += " Mesh B"
    # elif "refined" in full_file_path:
    #     label += " Mesh C"
    # elif "please-work" in full_file_path:
    #     label += " Mesh D"

    if raw_label:
        label = full_file_path

    return label, marker, mfc, ls, color


# Compute and plot cumulative average (from back to front)
def plot_cumulative_mean_std(data, phys_time, axis, color, label):
    cumulative_avg = data.expanding().mean()
    cumulative_std = data.expanding().std()

    axis.errorbar(phys_time, cumulative_avg, yerr=cumulative_std, label='', color=color, marker='', linestyle='',
                  alpha=0.01)
    axis.plot(phys_time, cumulative_avg, label=label, color=color)


def get_dof(case_dictionary, node_directory_path, variable_string = "u"):
    raw_log_file = glob(node_directory_path + "log*")[0]
    start_pattern = "Assembly map statistics for field "
    end_pattern = "Number of local/global dof"
    capture = False
    with open(raw_log_file, "r") as f:
        for line in f:
            if start_pattern + variable_string in line:
                capture = True
                continue
            # Capture dof data in next line
            if capture:
                line_segments = line.strip().split(" ")
                case_dictionary['global_dof'] = int(line_segments[-1])
                case_dictionary['local_dof'] = int(line_segments[-2])
            # End after capture
            if end_pattern in line:
                break
    AssertionError(case_dictionary['global_dof'] and case_dictionary['local_dof'], f"Could not find local_dof or global_dof in log file {raw_log_file}.")
    return

def filter_time_interval(physTime : pd.Series, signal : pd.Series, signal_len_from_end : float, use_mask : bool = False):
    # Mask based criterion
    tmin = physTime.min()
    tmax = physTime.max()

    # Compute lower bound
    # lower_criterion = tmin + ctu_skip
    lower_criterion = tmax - signal_len_from_end

    # Masking approach
    if use_mask:
        lowerMask = physTime >= lower_criterion # determine start from end
        upperMask = physTime <= tmax
        mask = (lowerMask == 1) & (upperMask == 1)
        if not np.any(mask):
            print("No data for interval = [{0}, {1}]".format(tmax - signal_len_from_end, tmax))
        else:
            print("Using data on interval = [{0}, {1}]".format(physTime[mask].iloc[0], physTime[mask].iloc[-1]))

        physTime = physTime[mask]
        signal = signal[mask]
    else:
        # Find nearest based reduction
        index = abs(physTime - (lower_criterion)).idxmin()
        print(f"Filtering time. Start index: {index} at time: {physTime.iloc[index]}")
        physTime = physTime.iloc[index:]
        signal = signal.iloc[index:]

    return physTime, signal


def check_sampling_rates(physTime, debug_plot=False):
    time_diffs = physTime.diff()
    if time_diffs.nunique() > 1:
        print(f"Sample rate is CHANGING with {time_diffs}")
    else:
        print(f"Sample rate is CONSTANT with {time_diffs.unique()}")

    if debug_plot:
        plt.figure()
        plt.plot(time_diffs)
        plt.xscale('linear')
        plt.yscale('log')
        plt.show()


def extract_boundary_id(f):
    filename = basename(f).split('.')[0]
    boundary_name = filename.split('_')[-1]
    bid = boundary_names.index(boundary_name)
    return bid



"""
    Marginal Standard Error Rule (mser)

    Estimate the end of the transient based on the minimum mean squared error 
    (mse) of the signal's mean with varying truncation.

    @param dataframe Dataframe containing the signal to analyse for transient 
    influence.
    @param signalKey Key for the dataframe's column.


"""
def mser(signal : pd.Series, time : pd.Series, stride_length : int = 1, debug_plot : bool = False):
    # Determine truncation range
    # i.e. range in which we expect the transient to be
    npoints = signal.shape[0]
    truncationRange = int(npoints / 2) # For simplicity, we choose half of all data

    # Save mean-squared-error sums for each truncation
    sums = list()

    # Loop with increasing truncation
    for d in range(0, truncationRange, stride_length):
        # Extract truncated signal
        truncSignal = signal.iloc[d:]

        # Determine mean-squared-error against truncated mean
        sums.append(
                ((truncSignal - truncSignal.mean()) ** 2).sum() / (npoints - d) ** 2
                )

    # DEBUG plot truncated mse
    if debug_plot:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(time.iloc[:truncationRange:stride_length] / ctu_len, sums)

    # Multiply by stride length to account for "lower resolution"
    dstar = np.argmin(sums) * stride_length
    return dstar



def plot_charles_results():

    # Path to data
    path_to_charles_data = '/home/henrik/Documents/simulation_data/codeVerification/f1-ifw/eifw/charles/'

    # reference variables
    uref = 12.5
    lref = 0.25
    rhoref = 1.2
    spanlen = 0.04
    Aref = lref * spanlen

    simulation_names = [
            'charles_mesh_stitch_v2_5_better_STL',
            'charles_mesh_stitch_v7_5_better_STL_no_wall_model_small_dt',
            'charles_mesh_stitch_v32_3_better_STL',
            ]

    simulation_labels = [
            'CharLES mesh 2',
            'CharLES mesh 7',
            'CharLES mesh 32',
            ]


    dataForces = list()
    for simulation_name, simulation_label in zip(simulation_names, simulation_labels):
        dataForces[-1].append(CharLESForces(uref, lref, rhoref, Aref))
        dataForces[-1].import_forces(filename = filename, initial_time = 0, simulation_name = simulation_name, legend_label = simulation_label)



