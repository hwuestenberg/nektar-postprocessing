#!/usr/bin/env  python3

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pandas as pd
import os, sys, glob


CTUlen = 0.25 # main plane chord
spanlenNpp = 0.05 # span-wise domain size
kinvis = 1.448e-6



savename = ""

dirnames = [
        "2d/hx1/physics/semidt5e-6/means/",
        #"quasi3d/james/farringdon_data/Cf/",
        ]

ctuname = "ctu_15_25"

fnames = [
        "mean_fields_" + ctuname + "_avg_wss_b0.csv",
        #"mean_fields_" + ctuname + "_avg_wss_b1.csv",
        #"mean_fields_" + ctuname + "_avg_wss_b2.csv",
        ]






def getlabel(fname, dt=0):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'

    # Add time step size
    if dt:
        label += "$\Delta t = {0:.1e}$".format(dt)
        label += " "

    if "5bl" in fname:
        label += " Mesh A"
    elif "8bl" in fname:
        label += " Mesh B"
    elif "refined" in fname:
        label += " Mesh C"
    elif "please-work" in fname:
        label += " Mesh D"

    return label, marker, mfc



if __name__ == "__main__":

    figx = plt.figure(figsize=(7,4))
    axx = figx.add_subplot(111)

    xorigin = 0
    for dname, color in zip(dirnames, TABLEAU_COLORS):
        for fname in fnames:
            if not os.path.exists(dname + fname):
                print("Did not find {0}".format(dname + fname))
                continue
            print("Processing {0}".format(dname + fname))

            df = pd.read_csv(dname + fname)
            if 'Points:0' in df.columns:
                x_ref_coords = df.iloc[:, 4].to_numpy()
                y_ref_coords = df.iloc[:, 5].to_numpy()
            elif 'x' in df.columns or 'y' in df.columns:
                x_ref_coords = df.iloc[:, 0].to_numpy()
                y_ref_coords = df.iloc[:, 1].to_numpy()

            # Get friction velocity
            shear_mag = df["Shear_mag"].to_numpy()
            ustar = shear_mag / 1.0 # unit density

            xpluss = list()
            npoints = len(x_ref_coords)
            i = 0
            for x, y in zip(x_ref_coords, y_ref_coords):
                # Satisfying progress bar
                j = (i+1)/npoints
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %f%%" % ('=' * int(20*j), 100*j))
                sys.stdout.flush()

                # Exclude points with same x OR z coordinates
                # Build a common mask for overlapping points
                mask = (x_ref_coords != x)
                #mask += (y_ref_coords != y)

                # Compute euclidean distance
                distx = x - x_ref_coords[mask]
                disty = y - y_ref_coords[mask]

                # Find index of minimum distance
                dist = np.sqrt( distx**2 + disty**2 )
                pid = np.argmin(dist)

                xplus = ustar[pid] * distx[pid] / kinvis

                # Save x distance for distance to closest point
                xpluss.append(
                        xplus
                        )
                i += 1

            df["xplus"] = xpluss
            df['x'] = x_ref_coords # for nice output column name
            print("df.columns", df.columns)
            
            df.to_csv(dname + "xplus.csv", 
                    sep=',', 
                    columns=["x", "y", "xplus"], 
                    index=False)

            # Visualise
            label, marker, mfc = getlabel(dname + fname)
            axx.scatter(x_ref_coords * CTUlen, xpluss, marker='o', label=label + ' xplus')


    axx.set_yscale("log")
    axx.legend()
    axx.grid()

    plt.show()

