#!/usr/bin/env  python3

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pandas as pd
import os, sys, glob


CTUlen = 0.25 # main plane chord
spanlenNpp = 0.05 # span-wise domain size
kinvis = 1.448e-6


showGeometry=False


savename = "vertex-distance-"

dirnames = [
        #"3d/5bl/mesh/",
        #"3d/8bl/mesh/",
        #"3d/refined/mesh/",
        #"3d/please-work/mesh/",
        "3d/please-work/physics/implicitdt1e-4/means/",
        #"quasi3d/james/farringdon_data/Cf/",
        ]

#ctuname = "ctu_20_30"
ctuname = "ctu_20_30"
#ctuname = "ctu_172_223"

fnames = [
        #"*-surfdistance_b0.csv",
        "mean_fields_" + ctuname + "_avg_wss_b0.csv",
        "mean_fields_" + ctuname + "_avg_wss_b1.csv",
        "mean_fields_" + ctuname + "_avg_wss_b2.csv",
        ]

dtref = 1e-5




def getlabel(casestr, color, dt=0.0):
    # Build case specific label for plots
    label = ""
    marker = "."
    mfc='None'
    ls='solid'
    color = color

    # Add time step size
    if dt == 0.0:
        label += ""
    elif dt < dtref:
        label += "${0:.1f}$".format(
                round(dt/dtref,1)
                )
        label += "$ \Delta t_{CFL}$"
    else:
        label += "${0:d}$".format(
                int(round(dt/dtref))
                )
        label += "$ \Delta t_{CFL}$"

    if "5bl" in casestr:
        label += " Mesh A"
    elif "8bl" in casestr:
        label += " Mesh B"
    elif "refined" in casestr:
        label += " Mesh C"
    elif "please-work" in casestr:
        label += " Mesh D"


    return label, marker, mfc, ls, color



# Define a function to calculate the closest point based on the distance
# Note that we exclude the trivial case of same x AND z value using the mask
def compute_closest_point(row, df_coords):
    mask = (df_coords['x'] != row['x']) & (df_coords['z'] != row['z'])

    # Compute distances for valid points based on mask
    distx = row['x'] - df_coords.loc[mask, 'x']
    disty = row['y'] - df_coords.loc[mask, 'y']
    distz = row['z'] - df_coords.loc[mask, 'z']
    surfd = row['dist'] - df_coords.loc[mask, 'dist']

    # Euclidean distance
    dist = np.sqrt(distx**2 + disty**2 + distz**2)

    # Index of the minimum distance
    pid = dist.idxmin()

    ## Get friction velocity
    shear_mag = df_coords["Shear_mag"]
    ustar = np.sqrt(shear_mag / 1.0) # unit density
    xplus = ustar[pid] * distx[pid] / kinvis
    yplus = ustar[pid] * surfd[pid] / kinvis
    zplus = ustar[pid] * distz[pid] / kinvis

    ## Extract pure distances
    #xplus = distx.loc[pid]
    #yplus = surfd.loc[pid]
    #yplus2 = disty.loc[pid]

    return pd.Series([xplus, yplus, zplus])


if __name__ == "__main__":

    figx = plt.figure(figsize=(7,4))
    axx = figx.add_subplot(111)
    figy = plt.figure(figsize=(7,4))
    axy = figy.add_subplot(111)
    figz = plt.figure(figsize=(7,4))
    axz = figz.add_subplot(111)

    for dname, color in zip(dirnames, TABLEAU_COLORS):
        xorigin = 0
        for fname in fnames:
            f = glob.glob(dname + fname)
            print("f", f)
            if not len(f) == 1:
                print("Did not find file in {0}".format(dname + fname))
                continue
            else:
                f = f[0]

            label, marker, mfc, ls, color = getlabel(f, color)
            print("Processing {0}".format(label))

            # Read dataframes
            df_wss = pd.read_csv(f, sep=',', engine='python', skiprows=0)
            surfdist_file = "mesh-surfdistance.csv"
            if "b0" in f:
                surfdist_file = surfdist_file.replace(".csv", "_b0.csv")
            elif "b1" in f:
                surfdist_file = surfdist_file.replace(".csv", "_b1.csv")
            elif "b2" in f:
                surfdist_file = surfdist_file.replace(".csv", "_b2.csv")

            df_surf = pd.read_csv(dname + f"../../../mesh/{surfdist_file}", sep=',', engine='python', skiprows=0)

            df = pd.concat([df_wss, df_surf['dist']], axis=1)

            # Rename columns
            if 'Points:0' in df.columns:
                df.rename(columns={"Points:0": "x", "Points:1": "y", "Points:2": "z"}, inplace=True)
            if 'x' in df.columns or 'y' in df.columns:
                df.rename(columns={"# x": "x", "y": "y", "z": "z"}, inplace=True)

            # Set origin to zero
            if xorigin == 0:
                xorigin = df['x'].min()
            df['x'] = df['x'] - xorigin # Set origin to zero
            print("xorigin:", xorigin)


            if showGeometry:
                axx.scatter(df['x'] / CTUlen, df['y'] / CTUlen, marker='o', label=label + ' x-y plane')
                axy.scatter(df['y'] / CTUlen, df['z'] / CTUlen, marker='o', label=label + ' y-z plane')
                axz.scatter(df['x'] / CTUlen, df['z'] / CTUlen, marker='o', label=label + ' x-z plane')
            else:
                # Create an empty DataFrame to store the results
                df_result = pd.DataFrame(index=df.index)

                # Apply the function row-wise to compute xplus and zplus
                df_result[['xplus', 'yplus', 'zplus']] = df.apply(compute_closest_point, axis=1, df_coords=df)

                # Visualise
                axx.scatter(df['x'] / CTUlen, df_result['xplus'], marker='o', label=label + ' xplus')
                axy.scatter(df['x'] / CTUlen, df_result['yplus'], marker='o', label=label + ' yplus')
                axz.scatter(df['x'] / CTUlen, df_result['zplus'], marker='o', label=label + ' zplus')

                df_out = pd.concat([df, df_result], axis=1)

                df_out.to_csv(f.replace("wss", "wallunits"), sep=' ', index=False)

    if not showGeometry:
        axx.set_yscale("log")
        axy.set_yscale("log")
        axz.set_yscale("log")
    axx.legend()
    axy.legend()
    axz.legend()
    axx.grid()
    axy.grid()
    axz.grid()

    figx.savefig(savename + "-xplus.png", bbox_inches="tight")
    figy.savefig(savename + "-yplus.png", bbox_inches="tight")
    figz.savefig(savename + "-zplus.png", bbox_inches="tight")

    plt.show()

