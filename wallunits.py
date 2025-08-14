#!/usr/bin/env  python3

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
import os


CTUlen = 0.25 # main plane chord
spanlenNpp = 0.05 # span-wise domain size

dtref = 1e-5 # Refernce time step for CFL \approx 1


savename = "eifw-wallunits"

dirnames = [
        #"3d/5bl/physics/semidt1e-5/means/",
        #"3d/8bl/physics/semidt1e-5/means/",
        #"3d/refined/physics/semidt1e-5/means/",
        "3d/please-work/physics/semidt1e-5/means/",
        #"3d/please-work/physics/implicitdt1e-4/means/",
        #"quasi3d/james/farringdon_data/Cf/",
        ]

ctuname = "ctu_204_287"
#ctuname = "ctu_15_20"

fnames = [
        "mean_fields_" + ctuname + "_avg_wallunits_b0-n1.csv",
        "mean_fields_" + ctuname + "_avg_wallunits_b1-n1.csv",
        "mean_fields_" + ctuname + "_avg_wallunits_b2-n1.csv",
        ]



def getlabel(casestr, color, dt=0.0):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'
    ls='solid'
    color = color

    ## Add time step size
    #if dt == 0.0:
    #    label += ""
    #elif dt < dtref:
    #    label += "${0:.1f}$".format(
    #            round(dt/dtref,1)
    #            )
    #else:
    #    label += "${0:d}$".format(
    #            int(round(dt/dtref))
    #            )
    #label += "$ \Delta t_{CFL}$"

    if "b0" in casestr:
        label += "Main plane"
    elif "b1" in casestr:
        label += "1st flap"
    elif "b2" in casestr:
        label += "2nd flap"
    #if "5bl" in casestr:
    #    label += " Mesh A"
    #elif "8bl" in casestr:
    #    label += " Mesh B"
    #elif "refined" in casestr:
    #    label += " Mesh C"
    #elif "please-work" in casestr:
    #    label += " Mesh D"


    return label, marker, mfc, ls, color






if __name__ == "__main__":

    figs = list()
    axs = list()
    for fname in fnames:
        figs.append(plt.figure(figsize=(6,2)))
        axs.append(figs[-1].add_subplot(111))

    for dname in dirnames:

        for ax, fname, color in zip(axs, fnames, TABLEAU_COLORS):
            if not os.path.exists(dname + fname):
                print("Did not find {0}".format(dname + fname))
                continue

            # Read yplus
            df = pd.read_csv(dname + fname, sep=' ')

            # Process x-coordinate
            if "yplus" in fname:
                x = df.iloc[:, 1].to_numpy()
            else:
                x = df["x"].to_numpy()
            x = x / CTUlen # scale with cord

            # Build case specific label
            label, marker, mfc, ls, color_unused = getlabel(dname + fname, color)

            # Process yplus
            ## overwrite marker
            #if "xplus" in fname:
            #    var = "xplus"
            #    marker = 'x'
            #elif "yplus" in fname:
            #    var = "yplus"
            #    marker = '^'
            #elif "zplus" in fname:
            #    var = "zplus"
            #    marker = 'o'
            variables = ["xplus", "yplus", "zplus"]
            for ax, var in zip(axs, variables):
                wallunit = np.abs(df[var]) / 4
                ax.plot(x, wallunit, marker='o', linestyle='', markeredgewidth=1.0, color=color, label=label, markerfacecolor=mfc)#, alpha=0.8)

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
        for fig, fname in zip(figs, fnames):
            fig.savefig(savename + "-" + fname.replace(".csv","") + ".png", bbox_inches="tight")
            #fig.savefig(savename + ".pdf", bbox_inches="tight")

    #ax.set_xlim([0.33, 0.79])
    #ax.set_ylim([-0.005, 0.045])
    #if savename:
    #    #fig.savefig(savename + "-zoom.png", bbox_inches="tight")
    #    fig.savefig(savename + "-zoom.pdf", bbox_inches="tight")


    plt.show()

