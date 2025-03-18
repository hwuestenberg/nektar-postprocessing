#!/bin/python3

import os, glob, subprocess

# Matplotlib setup with latex
import matplotlib.pyplot as plt
params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params) 
from matplotlib.colors import TABLEAU_COLORS

import numpy as np
import pandas as pd

from scipy.interpolate import griddata
import alphashape

from utilities import get_time_step_size, get_label
from config import directory_names, path_to_directories, ctu_len, ctu_names, boundary_names

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "variable", help="Choose either cp or cf", type=str, default='cf', nargs='?')
args = parser.parse_args()                              



savename = ""

# Set variable extension for the respective average files
# Pressure data: no specific extensions
# Skin friction: use wss (wall shear stress) extension
var_extension = ''
if args.variable == "cf":
    var_extension = 'wss_'

wss_variable = 'Shear_mag'


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


def create_slices(full_file_path):
    # Find all slices and sort by increasing location
    slicenames = glob.glob(full_file_path.replace(".csv", "") + '-slicey*.csv')
    slicenames = sorted(slicenames)
    nslices = len(slicenames)

    # Call slicer if no slices found
    if nslices == 0:
        print("Calling slicer.py")
        paraviewPath = "/home/henrik/sw/ParaView-5.11.2-MPI-Linux-Python3.9-x86_64/bin/pvbatch "
        slicerPath = "/home/henrik/Documents/simulation_data/postprocess/slicer.py "
        slicerArgs = "--in_file " + full_file_path.replace(".csv", ".vtu")
        cmdSlice = paraviewPath + slicerPath + slicerArgs
        msg = subprocess.check_output(cmdSlice, shell=True)

        # Again search for slices
        slicenames = glob.glob(full_file_path.replace(".csv", "") + '-slicey*.csv')
        slicenames = sorted(slicenames)
        nslices = len(slicenames)

    # Read paraview-type csv file with coordinates x,y,z = Points:0,1,2
    dfslice = pd.read_csv(slicenames[0])
    x_slice_coords = dfslice.iloc[:, 4]
    y_slice_coords = dfslice.iloc[:, 5]
    z_slice_coords = dfslice.iloc[:, 6]

    return slicenames, x_slice_coords, z_slice_coords


def interpolate_and_average_slices(slicenames):
    # Process all slices and compute mean at interpolated coordinate
    sq_mean = 0
    for sname in slicenames:
        df = pd.read_csv(sname)

        # Process surface quantity
        if args.variable == "cf":
            datastr = wss_variable
        elif args.variable == "cp":
            datastr = "p"
        else:
            print(f"ERROR. Cannot interpret command line argument for args.variable: {args.variable}. Exiting.")
            exit()
        sq = df[datastr]
        sq = 2 * sq  # Normalise against dynamic pressure

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
        else:
            print(f"ERROR. Cannot find x/y/z coordinates in dataframe df with columns: {df.columns}. Exiting")
            exit()

        # Method = 'Nearest' gives the best result
        # that is one point for each input coordinate
        # linear and cubic would only give a few points on suction side (interpolation error?)
        sq_interp = griddata((x_coords, z_coords), sq, (sampled_x, sampled_z), method='nearest')

        sq_mean += sq_interp

    return sq_mean


if __name__ == "__main__":
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        # Reset x-origin to zero before all three wings are being processed
        xorigin = 0

        # Loop all files (this loops different ctu names and wing elements)
        for ctuname, ctu_color in zip(ctu_names, TABLEAU_COLORS):
            # Define case name
            if "quasi3d" in full_directory_path:
                dt = 4e-6
            else:
                dt = get_time_step_size(full_directory_path)

            # Get plot styling
            label, marker, mfc, ls, color = get_label(full_directory_path, dt=dt)
            print("\nProcessing {0}...".format(label))

            # Loops all wing elements (boundaries)
            for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
                filename = "means/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".csv"
                full_file_path = full_directory_path + filename
                if not os.path.exists(full_file_path):
                    print("Did not find {0}".format(full_file_path))
                    continue

                # Process x-coordinate and surface quantity
                if "quasi3d" in full_directory_path:
                    df = pd.read_csv(full_file_path)
                    x = df["x"] / ctu_len
                    sq = df[args.variable]
                else:
                    slicenames, x_slice_coords, z_slice_coords = create_slices(full_file_path)
                    nslices = len(slicenames)

                    # Sample x and z points on the geometry
                    sampled_x, sampled_z = getSampleCoordinatesOnGeometry(x_slice_coords, z_slice_coords)

                    sq_mean = interpolate_and_average_slices(slicenames)
                    sq = sq_mean / nslices
                    x = np.array(sampled_x) / ctu_len

                # Shift x to zero, only done for boundary_names[0]
                if xorigin == 0:
                    xorigin = np.min(x)
                x = x - xorigin # Set origin to zero

                # Create label for only one of the wings
                if bname != boundary_names[0]:
                   label = ""
                else:
                   label += f" {ctuname}"

                # Plot data
                ax.plot(x, sq, marker=marker, linestyle='', markeredgewidth=1.5, color=ctu_color, label=label, markerfacecolor=mfc)#, alpha=0.8)

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

    # if savename:
    #     fig.savefig(savename + ".pdf", bbox_inches="tight")
    #
    # # zoom mp-lsb
    # if args.variable == "cf":
    #     ax.set_xlim([0.33, 0.79])
    #     ax.set_ylim([-0.005, 0.045])
    # elif args.variable == "cp":
    #     ax.set_xlim([0.34, 0.63])
    #     ax.set_ylim([-8.6, -4.9])
    #
    # if savename:
    #     fig.savefig(savename + "-mplsb.pdf", bbox_inches="tight")
    #
    # # zoom f1-lsb
    # if args.variable == "cf":
    #     ax.set_xlim([1.08, 1.36])
    #     ax.set_ylim([-0.005, 0.022])
    # elif args.variable == "cp":
    #     ax.set_xlim([1.18, 1.28])
    #     ax.set_ylim([-3.1, -1.5])
    #
    # if savename:
    #     fig.savefig(savename + "-f1lsb.pdf", bbox_inches="tight")
    #
    # # zoom f2-bypass
    # if args.variable == "cf":
    #     ax.set_xlim([1.38, 1.64])
    #     ax.set_ylim([-0.005, 0.035])
    # elif args.variable == "cp":
    #     ax.set_xlim([1.42, 1.49])
    #     ax.set_ylim([-0.01, 0.73])
    #
    # if savename:
    #     fig.savefig(savename + "-f2bypass.pdf", bbox_inches="tight")

    plt.show()

