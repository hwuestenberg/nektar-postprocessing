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


def get_plot_label_style(filename, color, dt = 0, sample_frequency = 0):
    # Build case specific label for plots
    label = ""
    marker = "o"
    mfc='None'
    ls=''
    color = color

    # Add time step size
    if dt:
        dtcfl = dt/dtref
        if dtcfl >= 1.0 - 1e-10:
            dtcfl = int(round(dtcfl))
            label += "{0: >3d}".format(dtcfl)
        else:
            dtcfl = round(dtcfl)
            label += "~{0: >3.1f}".format(dtcfl)
        label += r"$ \Delta t_{CFL}$"

    # Add info for linear-implicit scheme
    if "implicit" in filename or "VCSImplicit" in filename:
        label += " Linear-implicit"
        #if "Updated" in filename:
        #    label += r" upd. $\mathbf{\tilde{u}}$"
        #elif "Extrapolated" in filename:
        #    label += r" ext. $\mathbf{\tilde{u}}$"

        #if "Skew" in filename:
        #    label += " (skew.)"
        #    marker = 'x'
        #else:
        #    label += " (conv.)"

        if "dealias" in filename:
            label += " exact quad."
            marker = 's'

    # Add info for semi-implicit scheme
    elif "semi" in filename or "VelocityCorrectionScheme" in filename:
        label += " Semi-implicit"

    ## Space info
    #if "equal-order" in filename:
    #    label += " P,P"
    #elif "taylor-hood" in filename:
    #    marker = 'x'
    #    label += " P,P-1"

    ## Time order info
    #if "IMEXOrder1" in filename:
    #    label += " $\Delta t^1$"
    #elif "IMEXOrder2" in filename:
    #    label += " $\Delta t^2$"

    # Add polynomial order info
    if "p2" in filename:
        label += " P2"
    elif "p3" in filename:
        label += " P3"
    elif "p4" in filename:
        label += " P4"
    elif "p5" in filename:
        label += " P5"
    elif "p6" in filename:
        label += " P6"

    # Add sampling frequency
    if sample_frequency:
        label += "$f_{sample} =$"
        label += "${0:.1e}$".format(sample_frequency)
        label += " "

    return label, marker, mfc, ls, color




# Adapt this function for use above
def mse(df_force):
    totalSize = np.size(content[:, 1])
    counter = 1
    mySum = content[totalSize-1, 1]
    minMSE = 10000
    index_minMSE = 0

    for i in range(totalSize-2, int(totalSize/3), -1000):
        mySum = mySum + content[i, 1]
        counter = counter + 1
        truncated_mean = mySum/(totalSize - counter)
        myMSE = 0
        for j in range(totalSize-2, 1, -1):
            difference = content[j, 1] - truncated_mean
            myMSE = myMSE + difference**2
        myMSE = myMSE/(totalSize - counter)**2
        if (myMSE < minMSE):
            minMSE = myMSE
            index_minMSE = i
            print(index_minMSE, minMSE)
        #print('index: ', i, 'MSE: ', myMSE)
        #MSE.append(myMSE)

    print('The minimum MSE is: ', minMSE)
    print('The final sample number to achive min(MSE) is: ', index_minMSE)
    print('Time-averaging can start from T : ', content[index_minMSE, 0], 'CTUs')



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
    print("npoints {0}, truncRange 0 to {1}".format(npoints, truncationRange))

    # Choose stride length (accuracy vs efficiency)
    stride_length = 1 if npoints < 1e4 else 10

    # Save mean-squared-error sums for each truncation
    sums = list()

    # Loop with increasing truncation
    for d in range(0, truncationRange, stride_length):
        # Extract truncated signal
        truncSignal = dataframe[signalKey].iloc[d:]

        # Determine mean-squared-error against truncated mean
        sums.append(
                ((truncSignal - truncSignal.mean()) ** 2).mean() / (npoints - d)
                )

    # DEBUG plot truncated mse
    if debug_plot:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(dataframe[timeKey].iloc[:truncationRange:stride_length] / ctu_len, sums, label=signalKey)

    # Multiply by stride length to account for lower resolution
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

