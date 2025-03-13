#!/usr/bin/env  python3

# from CharLESProbes import *
# from compareProbesCharLES import *

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

import numpy as np
import pandas as pd
import os, subprocess

import argparse

# Parse command line arguments                          
parser = argparse.ArgumentParser()                      
parser.add_argument(                                    
    "variable", help="Choose either cp or cf", type=str, default='cf', nargs='?')
args = parser.parse_args()                              


CTUlen = 0.25 # main plane chord
spanlenNpp = 0.05 # span-wise domain size

dtref = 1e-5 # Refernce time step for CFL \approx 1


from config import path_to_directories, directory_names

savename = "eifw-" + args.variable + '-'
dirnames = [
        #"",
        #"",
        #"",
        #"3d/5bl/physics/semidt1e-5/means/",
        #"3d/5bl/physics/implicitdt1e-4/means/",
        #"3d/5bl/physics/implicitdt5e-4/means/",
        #"3d/8bl/physics/semidt1e-5/means/",
        #"3d/8bl/physics/implicitdt1e-4/means/",
        #"3d/8bl/physics/implicitdt5e-4/means/",
        #"3d/refined/physics/semidt1e-5/means/",
        #"3d/refined/physics/implicitdt1e-4/means/",
        #"3d/refined/physics/implicitdt5e-4/means/",
        "3d/please-work/physics/semiimplicit/dt1e-5/means/",
        # "3d/please-work/physics/implicitdt5e-5/means/",
        # "3d/please-work/physics/implicitdt1e-4/means/",
        # "3d/please-work/physics/implicitdt5e-4/means/",
        # "3d/please-work/physics/implicitdt1e-3/means/",
        "quasi3d/james/farringdon_data/Cf/",
        ]

ctuname = "ctu_204_210"
ctuname2 = "ctu_204_223"
ctuname3 = "ctu_204_248"

if args.variable == "cf":
    fnames = [
            "mean_fields_" + ctuname + "_avg_wss_b0.csv",
            #"mean_fields_" + ctuname + "_avg_wss_b1.csv",
            #"mean_fields_" + ctuname + "_avg_wss_b2.csv",
            "mean_fields_" + ctuname2+ "_avg_wss_b0.csv",
            #"mean_fields_" + ctuname2+ "_avg_wss_b1.csv",
            #"mean_fields_" + ctuname2+ "_avg_wss_b2.csv",
            "mean_fields_" + ctuname3+ "_avg_wss_b0.csv",
            #"mean_fields_" + ctuname3+ "_avg_wss_b1.csv",
            #"mean_fields_" + ctuname3+ "_avg_wss_b2.csv",
            ]
else:
    dirnames[-1] = "quasi3d/james/farringdon_data/CP/"
    fnames = [
            "mean_fields_" + ctuname + "_avg_b0.csv",
            #"mean_fields_" + ctuname + "_avg_b1.csv",
            #"mean_fields_" + ctuname + "_avg_b2.csv",
            "mean_fields_" + ctuname2+ "_avg_b0.csv",
            #"mean_fields_" + ctuname2+ "_avg_b1.csv",
            #"mean_fields_" + ctuname2+ "_avg_b2.csv",
            "mean_fields_" + ctuname3+ "_avg_b0.csv",
            #"mean_fields_" + ctuname3+ "_avg_b1.csv",
            #"mean_fields_" + ctuname3+ "_avg_b2.csv",
            ]






def getlabel(casestr, dt):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'

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

    if "implicit" in casestr:
        label += " implicit"
    elif "semi" in casestr:
        label += " semi-implicit"
    elif "quasi3d" in casestr:
        label += " Slaughter et al. (2023)"


    #if "svv3" in casestr:
    #    label += " SVV = 0.3"
    #elif "svv4" in casestr:
    #    label += " SVV = 0.4"
    #elif "svv5" in casestr:
    #    label += " SVV = 0.5"
    #else:
    #    label += " SVV = 1.0"


    return label, marker, mfc



def getTimeStepSize(path):
    # Read cputime.dat
    dftime = pd.read_csv(path + "cputime.dat", sep=" ")

    # Get time/step info for x-axis
    phystime = dftime["phystime"].to_numpy()
    steps = dftime["step"].to_numpy()
    dt = (phystime[1] - phystime[0]) / (steps[1] - steps[0]) # Get time step size
    return dt




if __name__ == "__main__":
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)

    """
    #### Do charles
    # Path to data
    path = '/home/henrik/Documents/00_phd/03_project/simulations/codeVerification/f1-ifw/eifw/charles/'

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

    y250slices = list()
    for simulation_name, simulation_label in zip(simulation_names, simulation_labels):
        y250slices.append(CharLESProbes(uref, lref, rhoref, Aref, simulation_label))

        folder = path + simulation_name + '/probes/'
        y250slices[-1].import_probes_README(folder + 'y250_line.README')
        y250slices[-1].read_all_available_variables()

        last_timestamp = y250slices[-1].timestamps[-1]
        y250slices[-1].Data['t=' + str(last_timestamp) +':Avg Cf'] = y250slices[-1].Data['t=' + str(last_timestamp) + ':avg(tau_wall())' ] / (0.5*rhoref*uref**2)
        #y250slices[-1].Data['t=' + str(last_timestamp) +':Avg Cfx'] = y250slices[-1].Data['t=' + str(last_timestamp) + ':avg(tau_wall(0))' ] / (0.5*rhoref*uref**2)

    # Plot charles data
    for y250slice in y250slices:
        y250slice.plot_variable(fig, ax, x_axis = 'x/c', variable = 'Avg Cf', timestamp_idx = -1, xrange = [])
        #y250slice.plot_variable(fig, ax, x_axis = 'x/c', variable = 'Avg Cfx', timestamp_idx = -1, xrange = [])
    """

    skip = 1 # take only every "skip-th" point

    for dname, color in zip(directory_names, TABLEAU_COLORS):
        xorigin = 0
        for fname in fnames:
            if not os.path.exists(dname + fname):
                print("Did not find {0}".format(dname + fname))
                continue

            # Define case name
            if "quasi3d" in dname:
                dt = 4e-6
            else:
                dt = getTimeStepSize(dname)
            label, marker, mfc = getlabel(dname, dt)
            print("Processing {0}...".format(label))

            df = pd.read_csv(dname + fname)

            # Process x-coordinate
            if "quasi3d" in dname:
                x = df["x"]
            else:
                x = df.iloc[:, 0]
            x = x / CTUlen # scale with cord

            # Set origin to zero
            if xorigin == 0:
                xorigin = np.min(x)
            x = x - xorigin # Set origin to zero
            print("xorigin:", xorigin)

            # Process surface quantity
            if "quasi3d" in dname:
                sq = df[args.variable]
            else:
                if args.variable == "cf":
                    datastr = "Shear_mag"
                elif args.variable == "cp":
                    datastr = "p"
                sq = df[datastr]
                sq = 2 * sq # Normalise against dynamic pressure

            #if fname not in fnames[-1] and fname not in fnames[-4]:
            #    label = ""

            ax.plot(x, sq, marker=marker, linestyle='', markeredgewidth=1.5, color=color, label=label, markerfacecolor=mfc)#, alpha=0.8)

            if args.variable == "cf":
                ax.set_ylabel(r"Skin friction coefficient $\overline{C_f}$")
            elif args.variable == "cp":
                ax.set_ylabel(r"Pressure coefficient $\overline{C_p}$")
            #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
            ax.set_xlabel("x/c")
            #upper = 0.12
            #ax.set_ylim([ax.get_ylim()[0], upper])
            #ax.set_xlim([-0.1, 1.1])
            ax.legend()
    ax.grid()

    if savename:
        fig.savefig(savename + ctuname + ".png", bbox_inches="tight")
        #fig.savefig(savename + ctuname + ".pdf", bbox_inches="tight")

    if args.variable == "cf":
        ax.set_xlim([0.33, 0.79])
        ax.set_ylim([-0.005, 0.045])
    elif args.variable == "cp":
        ax.set_xlim([0.34, 0.63])
        ax.set_ylim([-8.6, -4.9])

    if savename:
        fig.savefig(savename + ctuname + "-zoom.png", bbox_inches="tight")
        #fig.savefig(savename + ctuname + "-zoom.pdf", bbox_inches="tight")


    plt.show()

