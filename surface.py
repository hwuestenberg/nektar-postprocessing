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
import os

from utilities import get_time_step_size
from config import directory_names, path_to_directories, dtref, ctu_len, ctu_names, boundary_names

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


def get_label(full_file_path, dt):
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

    if "linear" in full_file_path:
        label += " linear-implicit"
    elif "semi" in full_file_path:
        label += " semi-implicit"
    elif "substepping" in full_file_path:
        label += " sub-stepping"
    elif "quasi3d" in full_file_path:
        label += " Slaughter et al. (2023)"
        mfc='None'
        marker='o'

    return label, marker, mfc


if __name__ == "__main__":
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    # Loop all files
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname

        # Reset x-origin to zero before all three wings are being processed
        xorigin = 0

        # Loop all files (this loops different ctu names and wing elements)
        # for filename, file_color in zip(filenames, TABLEAU_COLORS):
        for ctuname, ctu_color in zip(ctu_names, TABLEAU_COLORS):
            for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
                filename = "means/mean_fields_" + ctuname + "_avg_" + var_extension + bname + ".csv"
                full_file_path = full_directory_path + filename
                if not os.path.exists(full_file_path):
                    print("Did not find {0}".format(full_file_path))
                    continue

                # Define case name
                if "quasi3d" in full_directory_path:
                    dt = 4e-6
                else:
                    dt = get_time_step_size(full_directory_path)

                # Get plot styling
                label, marker, mfc = get_label(full_file_path, dt)
                print("\nProcessing {0}...".format(label))

                # Read file
                df = pd.read_csv(full_file_path, sep=',')

                # Process x-coordinate
                if "quasi3d" in full_file_path:
                    x = df["x"]
                else:
                    x = df.iloc[:, 0]
                x = x / ctu_len # scale with cord

                # Set origin to zero
                if xorigin == 0:
                    xorigin = np.min(x)
                x = x - xorigin # Set origin to zero
                print("xorigin:", xorigin)

                # Process surface quantity
                if "quasi3d" in full_file_path:
                    sq = df[args.variable]
                else:
                    if args.variable == "cf":
                        datastr = "Shear_mag"
                    elif args.variable == "cp":
                        datastr = "p"
                    else:
                        print(f"ERROR. Cannot interpret command line argument for args.variable: {args.variable}. Exiting.")
                        exit()
                    sq = df[datastr]
                    sq = 2 * sq # Normalise against dynamic pressure

                # Create label for only one of the wings
                if bname != boundary_names[0]:
                   label = ""

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

    if savename:
        fig.savefig(savename + ".png", bbox_inches="tight")
        #fig.savefig(savename + ".pdf", bbox_inches="tight")

    # # Set zoom window around transition location
    # if args.variable == "cf":
    #     ax.set_xlim([0.33, 0.79])
    #     ax.set_ylim([-0.005, 0.045])
    # elif args.variable == "cp":
    #     ax.set_xlim([0.34, 0.63])
    #     ax.set_ylim([-8.6, -4.9])

    if savename:
        fig.savefig(savename + "-zoom.png", bbox_inches="tight")
        #fig.savefig(savename + "-zoom.pdf", bbox_inches="tight")


    plt.show()

