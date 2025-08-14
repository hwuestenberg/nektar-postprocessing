#!/usr/bin/env  python3

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pandas as pd
import os


from config import directory_names, path_to_directories, save_directory, boundary_names, boundary_map, ctu_len, kinvis, path_to_mesh, path_to_mesh_boundary
from utilities import get_label



ctuname = "ctu_20_30"
filenames = [
        "mean_fields_" + ctuname + "_avg_wallunits",
        ]

savename = f"wallunits"
savename = save_directory + savename


if __name__ == "__main__":

    # Create figures
    figs = list()
    axs = list()
    for bname in boundary_names:
        figs.append(plt.figure(figsize=(6,2)))
        axs.append(figs[-1].add_subplot(111))

    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname
        for filename in filenames:
            file_path = full_directory_path + filename
            label = get_label(file_path)
            label = label[0] # ignore all other returns

            # Loops all wing elements (boundaries)
            xorigin = 0
            for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
                # Get csv file for wss
                boundary_file_path = file_path + "_" + bname

                # Read yplus
                df = pd.read_csv(boundary_file_path, sep=' ')
    
                # Process x-coordinate
                if "yplus" in boundary_file_path:
                    x = df.iloc[:, 1].to_numpy()
                else:
                    x = df["x"].to_numpy()
                x = x / ctu_len # scale with cord
    
                # Build case specific label
                label = get_label(boundary_file_path)
                label = label[0]
    
                variables = ["xplus", "yplus", "zplus"]
                for ax, var in zip(axs, variables):
                    wallunit = np.abs(df[var]) / 4
                    ax.plot(x, wallunit, marker='o', linestyle='', markeredgewidth=1.0, color=b_color, label=label)#, alpha=0.8)

        for ax in axs:
            ax.set_xlabel("$x/c$")
            ax.set_yscale("log")

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Plot limits based on Georgiadis et al. (2010)
            if ax == axs[0]:
                ax.set_ylabel("$\Delta x^+$")
                # Set x^+ = 50 limit bar
                ax.plot([xmin, xmax], [150, 150], '-r')
                ax.plot([xmin, xmax], [50, 50], '-r')
                ax.text(xmax*0.9, 50*1.1, "$\Delta x^+ limit$")
                ax.fill_between([xmin, xmax], y1=50, y2=150, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
            elif ax == axs[1]:
                ax.set_ylabel("$\Delta z^+$")
                # Set y^+ = 1 limit bar
                ax.plot([xmin, xmax], [1, 1], '-r')
                ax.text(xmax*0.9, 1*1.1, "$\Delta z^+ limit$")
                ax.fill_between([xmin, xmax], y1=1*0.9, y2=1*1.1, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
            elif ax == axs[2]:
                ax.set_ylabel("$\Delta y^+$")
                # Set z^+ = 15 limit bar
                ax.plot([xmin, xmax], [40, 40], '-r')
                ax.plot([xmin, xmax], [15, 15], '-r')
                ax.text(xmax*0.9, 15, "$\Delta y^+ limit$")
                ax.fill_between([xmin, xmax], y1=15, y2=40, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
            #ax.set_ylim([1e-4, ymax])
            ax.legend(loc='lower right')
            ax.grid()

    if savename:
        for fig, bname in zip(figs, boundary_names):
            fig.savefig(savename + "-" + bname.replace(".csv","") + ".png", bbox_inches="tight")
            #fig.savefig(savename + ".pdf", bbox_inches="tight")

    #ax.set_xlim([0.33, 0.79])
    #ax.set_ylim([-0.005, 0.045])
    #if savename:
    #    #fig.savefig(savename + "-zoom.png", bbox_inches="tight")
    #    fig.savefig(savename + "-zoom.pdf", bbox_inches="tight")


    plt.show()

