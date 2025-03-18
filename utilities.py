#!/usr/bin/env  python3
import os, sys, subprocess, glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import for charles plots
# from CharLESForces import *
# from compareForcesCharLES import *

from config import ctu_len, dtref






def get_data_frame(filename, skip_start = 0, skip_end = 0):
    # Pre-read file and check for 2D/3D
    headerskip = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            #print(i, line)
            if "Time" in line:
                headerskip = i
                break

    # Read file
    df = pd.read_csv(filename, header=headerskip, sep=r"(?!#)\s+", engine='python')
    df = df.iloc[skip_start:]
    if skip_end != 0:
        df = df.iloc[:-skip_end]


    # Remove hash-column, necessary for .fce files
    temp_columns = df.columns[1:]
    df = df.iloc[:,:-1]
    df.columns = temp_columns

    return df


def get_time_step_size(directory_name):
    # Read cputime.dat
    dftime = pd.read_csv(directory_name + "log_info.csv", sep=",")

    # Get time/step info for x-axis
    phystime = dftime["phys_time"].to_numpy()
    steps = dftime["steps"].to_numpy()
    dt = (phystime[1] - phystime[0]) / (steps[1] - steps[0]) # Get time step size
    return dt


def get_label(full_file_path, color = 'tab:blue', dt = 0.0, sampling_Frequency=0):
    # Build case specific label and style for plots (define defaults here)
    label = ""
    marker = "o"
    mfc = 'None'
    ls = 'solid'
    color = color

    # Add time step size
    if dt < dtref:
        label += "${0:.1f}$".format(
            round(dt / dtref, 1)
        )
    else:
        label += "{0: >3d}".format(
            int(round(dt / dtref))
        )
    label += r"$ \Delta t_{CFL}$"

    # Add sampling frequency
    if sampling_Frequency:
        label += " $f_{sample} =$"
        label += "${0:.1e}$".format(sampling_Frequency)
        label += " "

    # Add reynolds number
    if "/re" in full_file_path:
        re = full_file_path.split("/re")[-1].split("/")[0]
        label += " "
        label += "Re = {0:.1e}".format(float(re))

    if "linear" in full_file_path:
        label += " linear-implicit"
    elif "semi" in full_file_path:
        label += " semi-implicit"
    elif "substepping" in full_file_path:
        label += " sub-stepping"
    elif "quasi3d" in full_file_path:
        label += " Slaughter et al. (2023)"
        color = 'black'

    if "5bl" in full_file_path:
        label += " Mesh A"
    elif "8bl" in full_file_path:
        label += " Mesh B"
    elif "refined" in full_file_path:
        label += " Mesh C"
    elif "please-work" in full_file_path:
        label += " Mesh D"

    return label, marker, mfc, ls, color





"""
    Marginal Standard Error Rule (mser)

    Estimate the end of the transient based on the minimum mean squared error 
    (mse) of the signal's mean with varying truncation.

    @param dataframe Dataframe containing the signal to analyse for transient 
    influence.
    @param signalKey Key for the dataframe's column.


"""
def mser(dataframe, signalKey='signal', timeKey='Time', debug_plot = False):
    # Determine truncation range
    # i.e. range in which we expect the transient to be
    npoints = dataframe.shape[0]
    truncationRange = int(npoints / 2) # For simplicity, we choose half of all data
    # print("npoints {0}, truncRange 0 to {1}".format(npoints, truncationRange))

    # Choose stride length (accuracy vs comp. efficiency)
    stride_length = 100# if npoints < 1e4 else 10

    # Save mean-squared-error sums for each truncation
    sums = list()

    # Loop with increasing truncation
    for d in range(0, truncationRange, stride_length):
        # Extract truncated signal
        truncSignal = dataframe[signalKey].iloc[d:]

        # Determine mean-squared-error against truncated mean
        sums.append(
                ((truncSignal - truncSignal.mean()) ** 2).sum() / (npoints - d) ** 2
                )

    # DEBUG plot truncated mse
    if debug_plot:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(dataframe[timeKey].iloc[:truncationRange:stride_length] / ctu_len, sums, label=signalKey)

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

