#!/usr/bin/env  python3
from utilities import logcrawler, get_data_frame, get_time_step_size, get_plot_label_style

# Matplotlib setup with latex
import matplotlib.pyplot as plt
#plt.rcParams['text.latex.preamble']=[r"%\usepackage{newtxtext}"]
params = {'text.usetex': True,
 'font.size' : 10,
# 'font.serif': ['Times New Roman'],
# 'font.family': 'serif'
}
plt.rcParams.update(params) 
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

import numpy as np
import pandas as pd
import os, time, sys, subprocess, getopt
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "variable", help="Variable to be plotted", type=str)
args = parser.parse_args()
print("Plotting variable:", args.variable)

# Characteristic length
CTUlen = 0.25

skip = 0

# Define to save figures
savename = ""
savename = "eifw-" + args.variable + ""

dirnames = [
        "3d/please-work/physics/semidt1e-5/",
        "3d/please-work/physics/implicitdt1e-4/",
        "3d/please-work/physics/implicitdt5e-4/",
        "3d/please-work/physics/implicitdt1e-3/",
        ]


#savename = "2difw-" + args.variable + ""
#dirnames = [
#        #"2d/hx1/physics/semidt1e-6/",
#        #"2d/hx1/physics/semidt5e-6/",
#        "2d/hx1/physics/implicitdt5e-6/",
#        "2d/hx1/physics/implicitdt1e-5/",
#        "2d/hx1/physics/implicitdt5e-5/",
#        "2d/hx1/physics/implicitdt1e-4/",
#        "2d/hx1/physics/implicitdt5e-4/",
#        ]



#savename = "perf-iterations-pressure-pfreeze"
#dirnames = [
#        "3d/5bl/timings/semidt1e-5/reference/",
#        "3d/5bl/timings/implicitdt1e-4/pfreeze1/",
#        "3d/5bl/timings/implicitdt1e-4/pfreeze10/",
#        "3d/5bl/timings/implicitdt1e-4/pfreeze100/",
#        #"3d/5bl/timings/implicitdt1e-4/pfreeze1/",
#        #"3d/5bl/timings/implicitdt1e-3/pfreeze1/",
#        #"3d/5bl/timings/semidt1e-5/vcsweakpressure/",
#        #"3d/5bl/timings/implicitdt5e-4/pfreeze1/",
#        ]



if __name__ == "__main__":
    #logcrawler(dirnames, "EnergyFile.mdl", 2)

    # Plots
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)

    lns = list()
    avgcputimes = list()
    avgiterations = list()

    sortLabels = False

    # Loop all files and plot each
    linestyles=["solid", "dashed", "dotted", "dashdot"]

    for dirname, color in zip(dirnames, TABLEAU_COLORS):
        dt = get_time_step_size(dirname)
        label, marker, mfc, ls, color = get_plot_label_style(dirname, color, dt)
        print("\nProcessing {0}...".format(label))

        # Setup path for this directory
        path = dirname 

        # Read cputime.dat
        dftime = pd.read_csv(path + "cputime.dat", sep=" ")

        # Get time/step info for x-axis
        phystime = dftime["phystime"].to_numpy()
        steps = dftime["step"].to_numpy()
        dt = (phystime[1] - phystime[0]) / (steps[1] - steps[0]) # Get time step size

        # Adjust time array for length
        # This assumes that all runs use the same IO_StepInfo
        if ("iter" in args.variable or "cfl" in args.variable) and int(steps[1] - steps[0]) != 1:
            nsteps = steps[0]*len(steps)
            phystime = np.linspace(phystime[0] - (steps[0]-1)*dt, phystime[-1], nsteps)

        Ndts = CTUlen/dt # number dts per CTU
        phystime = phystime/CTUlen # convert to CTU
        step = phystime # plot over time instead of step count

        # Read data from relevant array
        if "iter" in args.variable:
            dfiter = pd.read_csv(path + "iterations.dat", sep=" ")
            data = dfiter[args.variable].to_numpy()
        elif "cfl" in args.variable:
            dfcfl = pd.read_csv(path + "cfl.dat", sep=" ")
            data = dfcfl[args.variable].to_numpy()
        elif "energy" in args.variable:
            dfenergy = pd.read_csv(path + "EnergyFile.mdl", usecols=[0,1], sep="(?!#)\s+", engine='python')
            dfenergy.columns = ["time", "energy"] # Remove hash column
            data = dfenergy[args.variable].to_numpy()
            step = dfenergy["time"].to_numpy()
            step = step / CTUlen # convert to CTU
        else:
            data = dftime[args.variable].to_numpy()

        print("data")
        print(len(data))
        print("step")
        print(len(step))

        # Check whether data is available (detects 1 even for empty iteration count)
        if (len(data) <= 1):
            print("No data for variable {0} in directory {1}. Skipping".format(args.variable, dirname))
            continue

        # Adjust for length mismatch
        if len(data) != len(step):
            ratio = len(step) / len(data)
            if ratio > 1:
                ratio = int(ratio)
                print("Trimming time by ratio {0}".format(ratio))
                step = step[::ratio]
            else:
                ratio = int(1/ratio) # invert ratio
                print("Trimming data by ratio {0}".format(ratio))
                data = data[::ratio]

        # Plot iteration vs time
        lns += ax.plot(step, data, label=label, linestyle=linestyles[0])
        avg = np.mean(data)
        textpos = 20#step[0] + 0.8*(step[-1] - step[0])
        ax.text(textpos, avg*1.1, "avg = {0:.2f}".format(avg), fontsize=12, weight='bold')
        ax.plot([step[0], step[-1]], [avg, avg], label='', linestyle='--', color='black', alpha=1.0)
        
    ## Configure axes style
    # labels over time
    xlabel=r"$t^\star$"
    ax.set_xlabel(xlabel)#Time step index")
    ax.set_ylabel(args.variable.upper())
    ax.grid(which="both") # Enable grid
    #ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_yscale("log")

    labels = [l.get_plot_label_style() for l in lns]
    if (sortLabels):
        print("Sorting all labels based on average CPU time.")
        _, labels = sortListsBasedOnList(avgcputimes, labels)
        _, lns = sortListsBasedOnList(avgcputimes, lns)
    # Reverse order so fastest is at the bottom of legend
    ax.legend(lns, labels, loc="upper left")


    ## Save figures
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    plt.show()

