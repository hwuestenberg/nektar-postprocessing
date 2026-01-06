#!/bin/python3

import os, glob, subprocess
import pickle

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

from utilities import get_time_step_size, get_label, get_ctu_names, extract_boundary_id
from config import directory_names, path_to_directories, ctu_len, boundary_names, save_directory, path_to_mesh_boundary, \
    boundary_names_long, boundary_names_long_map

surf_variable = 'cp'
wss_variable = 'Shear_mag'

## plot only part of the surfaces
boundary_names_skip = list()
# boundary_names_skip = boundary_names[0:1]
# boundary_names_skip = boundary_names[1:]


# savename = f"surface-james-{surf_variable}"
savename = f"surface-scheme-{surf_variable}"
# savename = f"surface-dt-{surf_variable}"
savename = save_directory + savename


# Set variable extension for the respective average files
# Pressure data: no specific extensions
# Skin friction: use wss (wall shear stress) extension
var_extension = ''
if surf_variable == "cf":
    var_extension = 'wss_'



path_to_session = path_to_directories + directory_names[0] + "means/session.xml"

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
    slicenames = glob.glob(full_file_path.replace(".vtu", "") + '-slicey*.csv')
    slicenames = sorted(slicenames)
    nslices = len(slicenames)

    # Call slicer if no slices found
    if nslices == 0:
        print("Calling slicer.py")
        paraviewPath = "/home/henrik/sw/ParaView-5.13.3-MPI-Linux-Python3.10-x86_64/bin/pvbatch "
        slicerPath = "/home/henrik/Documents/simulation_data/postprocess/slicer.py "
        slicerArgs = "--in_file " + full_file_path.replace(".csv", ".vtu")
        cmdSlice = paraviewPath + slicerPath + slicerArgs
        msg = subprocess.check_output(cmdSlice, shell=True)

        # Again search for slices
        slicenames = glob.glob(full_file_path.replace(".vtu", "") + '-slicey*.csv')
        slicenames = sorted(slicenames)
        nslices = len(slicenames)

    # Read paraview-type csv file with coordinates x,y,z = Points:0,1,2
    dfslice = pd.read_csv(slicenames[0])
    x_slice_coords = dfslice.iloc[:, 4]
    y_slice_coords = dfslice.iloc[:, 5]
    z_slice_coords = dfslice.iloc[:, 6]

    return slicenames, x_slice_coords, z_slice_coords


def d_on_reference_via_griddata(x, y, d, x_ref, y_ref,
                                method_linear='linear',
                                fill_with_nearest=True):
    """
    Interpolate d(x,y) from scattered (x,y,d) and sample it at (x_ref,y_ref).

    Returns
    -------
    d_ref : (M,) interpolated values on the reference curve
    d_ref_src : (M,) where values came from: 'linear' or 'nearest' (if filled)
    """
    pts = np.column_stack([x, y])
    xi  = np.column_stack([x_ref, y_ref])

    # 1) Linear interpolation inside the convex hull
    d_ref = griddata(pts, d, xi, method=method_linear)

    # 2) Optionally fill NaNs (outside the hull) with nearest-neighbor
    # src = np.full_like(x_ref, 'linear', dtype=object)
    if fill_with_nearest and np.any(np.isnan(d_ref)):
        dn = griddata(pts, d, xi, method='nearest')
        mask = np.isnan(d_ref)
        d_ref[mask] = dn[mask]
        # src[mask] = 'nearest'

    # return d_ref, src
    return d_ref


def interpolate_and_average_slices(slicenames, ref_slice_x, ref_slice_z):
    # Process all slices and compute mean at interpolated coordinate
    sq_mean = 0
    for sname in slicenames:
        df = pd.read_csv(sname)

        # Process surface quantity
        if surf_variable == "cf":
            datastr = wss_variable
        elif surf_variable == "cp":
            datastr = "p"
        else:
            print(f"ERROR. Cannot interpret command line argument for surf_variable: {surf_variable}. Exiting.")
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
        # sq_interp = griddata((x_coords, z_coords), sq, (ref_slice_x, ref_slice_z), method='nearest')

        sq_interp = d_on_reference_via_griddata(x_coords, z_coords, sq, ref_slice_x, ref_slice_z,
                                                method_linear='linear',
                                                fill_with_nearest=True)

        # ax.plot(ref_slice_x, sq_interp, label='interp' + sname.split("/")[-1], marker='x', color='black', linestyle='')

        # ax.plot(x_coords, z_coords, label=sname.split("/")[-1], marker='o', linestyle='')

        sq_mean += sq_interp

    return sq_mean / len(slicenames)


if __name__ == "__main__":
    fig = plt.figure(figsize=(9,2))
    ax = fig.add_subplot(111)
    ylabel = r"$\overline{C_p}$"
    if surf_variable == "cf":
        ylabel = r"$\overline{C_f}$"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("$x/c$")
    # ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)


    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        # Get plot styling
        dt = get_time_step_size(full_directory_path)
        label, marker, mfc, ls, color = get_label(full_directory_path, dt=dt, color=dir_color)
        print("\nProcessing {0}...".format(label))

        # Get names of available averages
        ctu_info = get_ctu_names(f"{full_directory_path}means/mean_fields_*_avg_{var_extension}{boundary_names[0]}.fld")
        ctu_names = [f"ctu_{start}_{end}" for start, end in zip(ctu_info[0], ctu_info[1])]

        if not ctu_names and not "quasi3d" in dirname:
            print(f"No mean fields found with glob string: {full_directory_path}/means/mean_fields_*_avg_{var_extension}{boundary_names[0]}.fld ")
            continue

        if "quasi3d" in dirname:
            ctu_names = ["ctu_20_30"]
        else:
            ctu_names = [ctu_names[0]]  # take only longest average
            print(f"using average for {ctu_names}.")

        # Reset x-origin to zero before all three wings are being processed
        xorigin = 0
        label_skip = False

        # Loop all files (this loops different ctu names and wing elements)
        for ctuname, ctu_color in zip(ctu_names, TABLEAU_COLORS):

            # Loops all wing elements (boundaries)
            for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
                filename = "means/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".vtu"
                if "quasi3d" in dirname:
                    filename = f"{surf_variable.upper()}/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".csv"
                full_file_path = full_directory_path + filename

                # Convert via NekPy, TODO this currently introduces NaN into data
                # if not "quasi3d" in dirname:
                    # convert_bnd_fld(full_file_path.replace(".vtu", ".fld"), "vtu")
                    # convert_bnd_fld(full_file_path.replace(".vtu", ".fld"), "csv")

                if not os.path.exists(full_file_path):
                    print("Did not find {0}".format(full_file_path))
                    continue

                # Process x-coordinate and surface quantity
                if "quasi3d" in full_directory_path:
                    df = pd.read_csv(full_file_path)
                    x = df["x"]
                    sq = df[surf_variable]
                else:
                    slicenames, x_slice_coords, z_slice_coords = create_slices(full_file_path)

                    # ## Create reference slice using concave hull and save in pickle
                    # Sample x and z points on the geometry
                    # ref_slice_x, ref_slice_z = getSampleCoordinatesOnGeometry(x_slice_coords, z_slice_coords)
                    # Write to pickle
                    # with open(f'./reference_slice_{bname}.pkl', 'wb') as file:
                    #     pickle.dump([sampled_x, sampled_z], file)

                    # Load from pickle
                    long_bname = boundary_names_long_map[bname]
                    with open(f'./reference_slice_{long_bname}.pkl', 'rb') as file:
                        ref_slice_x, ref_slice_z = pickle.load(file)

                    x = np.array(ref_slice_x)
                    sq = interpolate_and_average_slices(slicenames, ref_slice_x, ref_slice_z)

                # plt.show()
                # exit()

                # Shift x to zero, only done for boundary_names[0]
                if xorigin == 0:
                    xorigin = np.min(x)
                x = x - xorigin # Set origin to zero

                # Scale x for ctu_len
                x = x / ctu_len

                # # Shift flap 1 and flap 2 away for clear visibility
                # if bname == 'b1':
                #     x += 0.1
                # elif bname == 'b2':
                #     x += 0.2

                if bname in boundary_names_skip:
                    continue

                # Create label for only one of the wings
                if label_skip:
                    label = ""
                else:
                    label_skip = True

                # Plot data
                ax.plot(x, sq, marker=marker, linestyle='', markeredgewidth=1.5, color=color, label=label, markerfacecolor=mfc)#, alpha=0.8)

                #upper = 0.12
                #ax.set_ylim([ax.lim()[0], upper])
                #ax.set_xlim([-0.1, 1.1])
                ax.legend()
    ax.grid()

    if savename:
        for bname in boundary_names:
            if bname not in boundary_names_skip:
                savename += "-" + boundary_names_long_map[bname]
        print(f"Writing figure to {savename}.pdf")
        fig.savefig(savename + ".pdf", bbox_inches="tight")
    #
    # # zoom mp-lsb
    # if surf_variable == "cf":
    #     ax.set_xlim([0.33, 0.79])
    #     ax.set_ylim([-0.005, 0.045])
    # elif surf_variable == "cp":
    #     ax.set_xlim([0.34, 0.63])
    #     ax.set_ylim([-8.6, -4.9])
    #
    # if savename:
    #     fig.savefig(savename + "-mplsb.pdf", bbox_inches="tight")
    #
    # # zoom f1-lsb
    # if surf_variable == "cf":
    #     ax.set_xlim([1.08, 1.36])
    #     ax.set_ylim([-0.005, 0.022])
    # elif surf_variable == "cp":
    #     ax.set_xlim([1.18, 1.28])
    #     ax.set_ylim([-3.1, -1.5])
    #
    # if savename:
    #     fig.savefig(savename + "-f1lsb.pdf", bbox_inches="tight")
    #
    # # zoom f2-bypass
    # if surf_variable == "cf":
    #     ax.set_xlim([1.38, 1.64])
    #     ax.set_ylim([-0.005, 0.035])
    # elif surf_variable == "cp":
    #     ax.set_xlim([1.42, 1.49])
    #     ax.set_ylim([-0.01, 0.73])
    #
    # if savename:
    #     fig.savefig(savename + "-f2bypass.pdf", bbox_inches="tight")

    plt.show()

