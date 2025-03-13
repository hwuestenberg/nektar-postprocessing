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
import os, sys, glob, subprocess

from scipy.interpolate import griddata
import alphashape

import argparse

# Parse command line arguments                          
parser = argparse.ArgumentParser()                      
parser.add_argument(                                    
    "variable", help="Choose either cp or cf", type=str, default='cf', nargs='?')
args = parser.parse_args()                              


CTUlen = 0.25 # main plane chord
spanlenNpp = 0.05 # span-wise domain size

dtref = 1e-5 # Refernce time step for CFL \approx 1

#savename = "eifw-" + args.variable + "-refine"
#savename = "eifw-" + args.variable + "-dt-influence"
#savename = "eifw-" + args.variable + "-vs-james"
# savename = "eifw-" + args.variable + "-substepping"

savename = ""
dirnames = [
        #"3d/5bl/physics/semidt1e-5/means/",
        #"3d/8bl/physics/semidt1e-5/means/",
        #"3d/refined/physics/semidt1e-5/means/",
        "3d/please-work/physics/semiimplicit/dt1e-5/means/",
        # "3d/please-work/physics/implicitdt5e-5/means/",
        # "3d/please-work/physics/substepping/dt5e-5/means/",
        #"3d/please-work/physics/implicitdt1e-4/means/",
        #"3d/please-work/physics/implicitdt5e-4/means/",
        #"3d/please-work/physics/implicitdt1e-3/means/",
        #"quasi3d/james/farringdon_data/Cf/",
        ]

#ctuname = "ctu_20_30"
#ctuname = "ctu_15_20"
ctuname = "ctu_1746_2146"
ctuname2 = "ctu_172_223"
ctuname3 = "ctu_160_182"

if args.variable == "cf":
    fnames = [
            "mean_fields_" + ctuname + "_avg_wss_b0.csv",
            "mean_fields_" + ctuname + "_avg_wss_b1.csv",
            "mean_fields_" + ctuname + "_avg_wss_b2.csv",
            "mean_fields_" + ctuname2+ "_avg_wss_b0.csv",
            "mean_fields_" + ctuname2+ "_avg_wss_b1.csv",
            "mean_fields_" + ctuname2+ "_avg_wss_b2.csv",
            "mean_fields_" + ctuname3+ "_avg_wss_b0.csv",
            "mean_fields_" + ctuname3+ "_avg_wss_b1.csv",
            "mean_fields_" + ctuname3+ "_avg_wss_b2.csv",
            ]
else:
    if "quasi3d" in dirnames[-1]:
        dirnames[-1] = "quasi3d/james/farringdon_data/CP/"
    fnames = [
            "mean_fields_" + ctuname + "_avg_b0.csv",
            "mean_fields_" + ctuname + "_avg_b1.csv",
            "mean_fields_" + ctuname + "_avg_b2.csv",
            "mean_fields_" + ctuname2+ "_avg_b0.csv",
            "mean_fields_" + ctuname2+ "_avg_b1.csv",
            "mean_fields_" + ctuname2+ "_avg_b2.csv",
            "mean_fields_" + ctuname3+ "_avg_b0.csv",
            "mean_fields_" + ctuname3+ "_avg_b1.csv",
            "mean_fields_" + ctuname3+ "_avg_b2.csv",
            ]



# Build concave hull of the original geometry 
# and sample equispaced points on the geometry
def getSampleCoordinatesOnGeometry(x_slice_coords, z_slice_coords):
    # Get xz coordinates in shape required for alphashapes
    xz = np.array((x_slice_coords, z_slice_coords)).transpose()
    
    # Find concave hull
    alpha = 10.0 # Trial-n-error
    hull = alphashape.alphashape(xz, alpha)

    # Get the exterior boundary as a LineString
    boundary = hull.exterior

    # Calculate the total length of the boundary
    boundary_length = boundary.length

    # Define the number of points you want to sample along the boundary
    num_points = 500

    # Calculate the distance between consecutive sampled points
    distance_between_points = boundary_length / num_points

    # Sample equidistant points along the boundary
    sampled_points = []
    for i in range(num_points):
        sampled_points.append(
                boundary.interpolate(i * distance_between_points)
                )

    # Extract the x and y coordinates of the sampled points
    x = [point.x for point in sampled_points]
    z = [point.y for point in sampled_points] # note this is the z-coordinate
    return x, z





def getlabel(casestr, color, dt):
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
    label += "$ \Delta t_{CFL}$"

    if "implicit" in casestr:
        label += " implicit"
    elif "semi" in casestr:
        label += " semi-implicit"
    elif "substepping" in casestr:
        label += " substepping"
    elif "quasi3d" in casestr:
        label += " Slaughter et al. (2023)"
        color = 'black'

    if "5bl" in casestr:
        label += " Mesh A"
    elif "8bl" in casestr:
        label += " Mesh B"
    elif "refined" in casestr:
        label += " Mesh C"
    elif "please-work" in casestr:
        label += " Mesh D"

    return label, marker, mfc, ls, color



def getTimeStepSize(path):
    # Read cputime.dat
    dftime = pd.read_csv(path + "cputime.dat", sep=" ")

    # Get time/step info for x-axis
    phystime = dftime["phystime"].to_numpy()
    steps = dftime["step"].to_numpy()
    dt = (phystime[1] - phystime[0]) / (steps[1] - steps[0]) # Get time step size
    return dt




if __name__ == "__main__":
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)

    # Loop all directories (cases) and files
    for dname, color in zip(dirnames, TABLEAU_COLORS):
        xorigin = 0
        for fname in fnames:
            nslices = 0
            if not os.path.exists(dname + fname):
                print("Did not find {0}".format(dname + fname))
                continue

            # Define case name
            if "quasi3d" in dname:
                dt = 4e-6
            else:
                dt = getTimeStepSize(dname)
            label, marker, mfc, ls, color = getlabel(dname, color, dt)
            print("Processing {0}...".format(label))

            # Process surface quantity
            if "quasi3d" in dname:
                df = pd.read_csv(dname + fname)
                x = df["x"] / CTUlen
                sq = df[args.variable]
            else:
                # Find all slices and sort by increasing location
                slicenames = glob.glob(dname + fname.replace(".csv","") + '-slicey*.csv')
                slicenames = sorted(slicenames)
                nslices = len(slicenames)

                # Call slicer
                if nslices == 0:
                    print("Calling slicer.py")
                    paraviewPath = "/home/henrik/Downloads/ParaView-5.11.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch "
                    slicerPath = "/home/henrik/Documents/00_phd/03_project/simulations/codeVerification/f1-ifw/eifw/3d/slicer.py "
                    slicerArgs = "--in_file " + dname + fname.replace(".csv",".vtu")
                    cmdSlice = paraviewPath + slicerPath + slicerArgs
                    msg = subprocess.check_output(cmdSlice, shell=True)

                    # Re-do finding slices
                    slicenames = glob.glob(dname + fname.replace(".csv","") + '-slicey*.csv')
                    slicenames = sorted(slicenames)
                    nslices = len(slicenames)

                dfslice = pd.read_csv(slicenames[0])
                if 'Points:0' in dfslice.columns:
                    x_slice_coords = dfslice.iloc[:, 4]
                    y_slice_coords = dfslice.iloc[:, 5]
                    z_slice_coords = dfslice.iloc[:, 6]

                # Sample x and z points on the geometry
                sampled_x, sampled_z = getSampleCoordinatesOnGeometry(x_slice_coords, z_slice_coords)

                # Process all slices and compute mean at interpolated coordinate
                sqmean = 0
                for sname in slicenames:
                    df = pd.read_csv(sname)

                    # Process surface quantity
                    if args.variable == "cf":
                        datastr = "Shear_mag"
                    elif args.variable == "cp":
                        datastr = "p"
                    data = df[datastr]
                    sq = 2 * data # Normalise against dynamic pressure

                    # Extract x, y, z coordinates
                    # Check for paraview or nektar input csv
                    if 'Points:0' in df.columns:
                        x_coords = df.iloc[:, 4]
                        y_coords = df.iloc[:, 5]
                        z_coords = df.iloc[:, 6]
                    elif 'x' in df.columns or 'y' in df.columns:
                        x_coords = df.iloc[:, 0]
                        y_coords = df.iloc[:, 1]
                        z_coords = df.iloc[:, 2]

                    # Method = 'Nearest' gives the best result
                    # that is one point for each input coordinate
                    # linear and cubic would only give a few points on suction side (interpolation error?)
                    sq_interp = griddata((x_coords, z_coords), sq, (sampled_x, sampled_z), method='nearest')

                    sqmean += sq_interp
                sq = sqmean / nslices
                x = np.array(sampled_x) / CTUlen

            # Shift x to zero, if necessary
            if xorigin == 0:
                xorigin = np.min(x)
            x = x - xorigin # Set origin to zero
            print("xorigin:", xorigin)

            if "b2" not in fname:
                label = ""

            ax.plot(x, sq, marker=marker, linestyle='', markeredgewidth=1.5, color=color, label=label, markerfacecolor=mfc)#, alpha=0.8)

            if args.variable == "cf":
                ax.set_ylabel("Skin friction coefficient $\overline{C_f}$")
            elif args.variable == "cp":
                ax.set_ylabel("Pressure coefficient $\overline{C_p}$")
            #ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
            ax.set_xlabel("x/c")
            #upper = 0.12
            #ax.set_ylim([ax.get_ylim()[0], upper])
            #ax.set_xlim([-0.1, 1.1])
            ax.legend()
    ax.grid()

    if savename:
        fig.savefig(savename + ".pdf", bbox_inches="tight")

    # zoom mp-lsb
    if args.variable == "cf":
        ax.set_xlim([0.33, 0.79])
        ax.set_ylim([-0.005, 0.045])
    elif args.variable == "cp":
        ax.set_xlim([0.34, 0.63])
        ax.set_ylim([-8.6, -4.9])

    if savename:
        fig.savefig(savename + "-mplsb.pdf", bbox_inches="tight")

    # zoom f1-lsb
    if args.variable == "cf":
        ax.set_xlim([1.08, 1.36])
        ax.set_ylim([-0.005, 0.022])
    elif args.variable == "cp":
        ax.set_xlim([1.18, 1.28])
        ax.set_ylim([-3.1, -1.5])

    if savename:
        fig.savefig(savename + "-f1lsb.pdf", bbox_inches="tight")

    # zoom f2-bypass
    if args.variable == "cf":
        ax.set_xlim([1.38, 1.64])
        ax.set_ylim([-0.005, 0.035])
    elif args.variable == "cp":
        ax.set_xlim([1.42, 1.49])
        ax.set_ylim([-0.01, 0.73])

    if savename:
        fig.savefig(savename + "-f2bypass.pdf", bbox_inches="tight")

    plt.show()

