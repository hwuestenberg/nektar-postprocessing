#!/bin/python3

import os, glob, subprocess

import sys
sys.path.insert(0, "/home/henrik/code/nektar-animation/build-master/python")
from NekPy.FieldUtils import *

# Matplotlib setup with latex
import matplotlib.pyplot as plt
params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params) 
from matplotlib.colors import TABLEAU_COLORS

import numpy as np
import pandas as pd

from utilities import get_time_step_size, get_label, get_ctu_names, extract_boundary_id
from config import directory_names, path_to_directories, ctu_len, boundary_names, save_directory, path_to_mesh_boundary, \
    boundary_names_long

surf_variable = 'cf'
wss_variable = 'Shear_mag'

# plot only part of the surfaces
boundary_names = boundary_names[0:1]


savename = f"surface-james-{surf_variable}"
savename = save_directory + savename


# Set variable extension for the respective average files
# Pressure data: no specific extensions
# Skin friction: use wss (wall shear stress) extension
var_extension = ''
if surf_variable == "cf":
    var_extension = 'wss_'



path_to_session = path_to_directories + directory_names[0] + "means/session.xml"

# FieldConvert calls
def convert_bnd_fld_to_csv(f):
    bid = extract_boundary_id(f)

    field = Field(sys.argv, force_output=True)
    InputModule.Create("xml", field, path_to_mesh_boundary[bid]).Run()
    # InputModule.Create("xml", field, path_to_session).Run()
    InputModule.Create("fld", field, f).Run()
    OutputModule.Create("csv", field, f.replace(".fld", ".csv")).Run()




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

        # Loop all files (this loops different ctu names and wing elements)
        for ctuname, ctu_color in zip(ctu_names, TABLEAU_COLORS):

            # Loops all wing elements (boundaries)
            for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
                filename = "means/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".fld"
                if "quasi3d" in dirname:
                    filename = f"{surf_variable.upper()}/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".csv"
                full_file_path = full_directory_path + filename
                if not "quasi3d" in dirname:
                    convert_bnd_fld_to_csv(full_file_path)
                    full_file_path = full_file_path.replace(".fld", ".csv")

                if not os.path.exists(full_file_path):
                    print("Did not find {0}".format(full_file_path))
                    continue

                # Read file
                df = pd.read_csv(full_file_path, sep=',')

                # Process x-coordinate and surface quantity
                if "quasi3d" in full_directory_path:
                    x = df["x"] / ctu_len
                    sq = df[surf_variable]
                else:
                    x = df.iloc[:, 0] / ctu_len
                    if surf_variable == "cf":
                        datastr = wss_variable
                    elif surf_variable == "cp":
                        datastr = "p"
                    else:
                        print(f"ERROR. Cannot interpret command line argument for surf_variable: {surf_variable}. Exiting.")
                        exit()

                    sq = df[datastr]
                    sq = 2 * sq # Normalise against dynamic pressure

                # Shift x to zero, only done for boundary_names[0]
                if xorigin == 0:
                    xorigin = np.min(x)
                x = x - xorigin # Set origin to zero

                # Create label for only one of the wings
                if bname != boundary_names[0]:
                   label = ""
                else:
                   label += ""#f" {ctuname}"

                # Plot data
                ax.plot(x, sq, marker=marker, linestyle='', markeredgewidth=1.5, color=dir_color, label=label, markerfacecolor=mfc)#, alpha=0.8)

                #upper = 0.12
                #ax.set_ylim([ax.get_ylim()[0], upper])
                #ax.set_xlim([-0.1, 1.1])
                ax.legend()
    ax.grid()
    #
    # if savename:
    #     fig.savefig(savename + ".png", bbox_inches="tight")
    #     #fig.savefig(savename + ".pdf", bbox_inches="tight")
    #
    # # # Set zoom window around transition location
    # # if surf_variable == "cf":
    # #     ax.set_xlim([0.33, 0.79])
    # #     ax.set_ylim([-0.005, 0.045])
    # # elif surf_variable == "cp":
    # #     ax.set_xlim([0.34, 0.63])
    # #     ax.set_ylim([-8.6, -4.9])
    #
    # if savename:
    #     fig.savefig(savename + "-zoom.png", bbox_inches="tight")
    #     #fig.savefig(savename + "-zoom.pdf", bbox_inches="tight")


    plt.show()

