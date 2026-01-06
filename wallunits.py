#!/usr/bin/env  python3

import matplotlib.pyplot as plt
params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params)
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pandas as pd
import os


from config import directory_names, path_to_directories, save_directory, boundary_names, boundary_region_to_mesh_map, \
    ctu_len, kinvis, \
    path_to_mesh, path_to_mesh_boundary, boundary_names_plot, meshname, wallunits_ref_name
from utilities import get_label



global_xlim = [-0.05, 1.7]
savename = f"wallunits"
savename = save_directory + savename


if __name__ == "__main__":

    # Create figures
    # figs = list()
    fig = plt.figure(figsize=(6, 4))
    axs = list()
    for i in range(0,3):
        # figs.append(plt.figure(figsize=(6,2)))
        if i == 0:
            axs.append(fig.add_subplot(3, 1, i+1))
        else:
            axs.append(fig.add_subplot(3, 1, i+1, sharex=axs[0]))

    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        # Note we assume this file is in the same directory as the mesh file
        file_path = path_to_mesh.replace(f"{meshname}.xml", wallunits_ref_name).replace("wss", "wallunits")
        label = get_label(file_path)
        label = label[0] # ignore all other returns

        # Loops all wing elements (boundaries)
        xorigin = 0
        for bname, b_color in zip(boundary_names, TABLEAU_COLORS):
            # Get csv file for wss
            boundary_file_path = file_path + "_" + bname + ".csv"

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
                # Note: divide by polynomial order P = 4 for average (equidistant) spacing of quad points!
                wallunit = np.abs(df[var]) / 4
                ax.plot(x, wallunit, marker='o', linestyle='', markeredgewidth=1.0, color=b_color, label=boundary_names_plot[boundary_names.index(bname)], alpha=1.0)

        for ax in axs:
            ax.set_xlabel("$x/c$")
            ax.set_yscale("log")

            if global_xlim:
                ax.set_xlim(global_xlim)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ymax = ymax
            ymin = ymax / 1000

            # Plot limits based on Georgiadis et al. (2010)
            if ax == axs[0]:
                ax.set_ylabel("$\Delta x^+$")
                # Set x^+ = 50 limit bar
                # ax.plot([xmin, xmax], [150, 150], '-r')
                ax.plot([xmin, xmax], [50, 50], '-r', label="Georgiadis et al. (2010)")
                ax.text(xmax*0.6, 50*1.1, "$\Delta x^+ limit <= 50$", fontweight='bold')
                # ax.fill_between([xmin, xmax], y1=50, y2=150, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
                # ax.set_ylim([1e0, 3e2])
                ax.set_ylim([ymin, ymax*10])

                # Hide xtick labels on all but the bottom-most axis
                ax.tick_params(axis='x', which='both', labelbottom=False)
                ax.set_xlabel("")

            elif ax == axs[1]:
                ax.set_ylabel("$\Delta y^+$")
                # Set y^+ = 1 limit bar
                ax.plot([xmin, xmax], [1, 1], '-r', label="Georgiadis et al. (2010)")
                ax.text(xmax*0.6, 1.1 / 10, "$\Delta y^+ limit <= 1$", fontweight='bold')
                # ax.fill_between([xmin, xmax], y1=1*0.9, y2=1*1.1, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
                # ax.set_ylim([1e-5, 5e0])
                ax.set_ylim([ymin / 10, 2])

                # Hide xtick labels on all but the bottom-most axis
                ax.tick_params(axis='x', which='both', labelbottom=False)
                ax.set_xlabel("")

            elif ax == axs[2]:
                ax.set_ylabel("$\Delta z^+$")
                # Set z^+ = 15 limit bar
                # ax.plot([xmin, xmax], [40, 40], '-r')
                ax.plot([xmin, xmax], [15, 15], '-r', label="Georgiadis et al. (2010)")
                ax.text(xmax*0.6, 20*1.1, "$\Delta z^+ limit <= 15$", fontweight='bold')
                # ax.fill_between([xmin, xmax], y1=15, y2=40, alpha=0.2, label="Georgiadis et al. (2010)", color='red')#, color=color, marker=marker, linestyle='', capsize=5)
                # ax.set_ylim([5e-2, 1e2])
                ax.set_ylim([ymin, ymax * 2])

            #ax.set_ylim([1e-4, ymax])
            # Removed per-axis legend to avoid covering data
            # ax.legend(loc='lower right')
            ax.grid()

    # Build a single, figure-level legend below the axes
    # Collect unique handles/labels from all axes to avoid duplicates
    handles_all, labels_all = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_all:
                labels_all.append(ll)
                handles_all.append(hh)

    # Reserve bottom space and place the legend centered below the subplots
    fig.subplots_adjust(bottom=0.25)
    fig.legend(handles_all, labels_all, loc='lower center', ncol=min(4, len(labels_all)), bbox_to_anchor=(0.5, 0.05))

    if savename:
        fig.savefig(savename + ".png", bbox_inches="tight")
        # for fig, bname in zip(figs, boundary_names):
        #     fig.savefig(savename + "-" + bname.replace(".csv","") + ".png", bbox_inches="tight")
            #fig.savefig(savename + ".pdf", bbox_inches="tight")

    #ax.set_xlim([0.33, 0.79])
    #ax.set_ylim([-0.005, 0.045])
    #if savename:
    #    #fig.savefig(savename + "-zoom.png", bbox_inches="tight")
    #    fig.savefig(savename + "-zoom.pdf", bbox_inches="tight")


    plt.show()