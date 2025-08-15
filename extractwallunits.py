# NekPy path
import sys
from os.path import basename

from utilities import get_label

sys.path.insert(0, "/home/hwustenb/code/nektar-legacy/build-master/python")
from NekPy.FieldUtils import *

import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from config import directory_names, path_to_directories, save_directory, boundary_names, boundary_map, ctu_len, kinvis, path_to_mesh, path_to_mesh_boundary


showGeometry = False
ctuname = "ctu_20_30"
filenames = [
        "mean_fields_" + ctuname + "_avg_wss",
        ]

savename = f"wallunits"
savename = save_directory + savename

path_to_session = path_to_directories + directory_names[0] + "means/session.xml"

def extract_boundary_id(f):
    filename = basename(f).split('.')[0]
    boundary_name = filename.split('_')[-1]
    bid = boundary_names.index(boundary_name)
    return bid


# FieldConvert calls
def convert_fld_to_csv(f):
    bid = extract_boundary_id(f)

    field = Field(sys.argv, output_points=1, force_output=True)
    InputModule.Create("xml", field, path_to_mesh_boundary[bid]).Run()
    InputModule.Create("xml", field, path_to_session).Run()
    InputModule.Create("fld", field, f).Run()
    OutputModule.Create("csv", field, f.replace(".fld", ".csv")).Run()


def get_csv_file(path_to_file):

    csvfile = ""

    # Glob for files
    path_to_all_files = glob.glob(path_to_file + "*")

    # Remove slicer outputs
    path_to_all_files = [p for p in path_to_all_files if not "slicey" in p]

    if len(path_to_all_files) == 0:
        print("Did not find csv file but found {0}".format(path_to_file))
        return ""
    elif len(path_to_all_files) > 0:
        print("Found files {0}".format(path_to_all_files))
        # First look for csv file
        for filepath in path_to_all_files:
            if filepath.endswith(".csv"):
                return filepath
        # If no csv file, try converting a fld file
        for filepath in path_to_all_files:
            if filepath.endswith(".fld"):
                print("Converting {0} to csv".format(basename(filepath)))
                convert_fld_to_csv(filepath)
                return filepath.replace(".fld", ".csv")
            else:
                print("Found file {0}. Nothing to do. Continue looking..".format(basename(filepath)))
    else:
        return path_to_all_files[0]



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

    # Create figures
    figx = plt.figure(figsize=(7,4))
    axx = figx.add_subplot(111)
    figy = plt.figure(figsize=(7,4))
    axy = figy.add_subplot(111)
    figz = plt.figure(figsize=(7,4))
    axz = figz.add_subplot(111)

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
                wss_file = get_csv_file(boundary_file_path)

                # Skip if no file found
                if wss_file == "":
                    continue

                # Read dataframe
                df_wss = pd.read_csv(wss_file, sep=',', engine='python', skiprows=0)

                # Get csv file for surfdist
                surfdist_file = "mesh-surfdistance_{0}".format(bname)
                surfdist_file_path = path_to_mesh.replace("mesh.xml", surfdist_file)
                surfdist_file = get_csv_file(surfdist_file_path)

                # Skip if no file found
                if surfdist_file_path == "":
                    continue

                # Read dataframe
                df_surf = pd.read_csv(surfdist_file, sep=',', engine='python', skiprows=0)

                # Concatenate wss and surfdistance
                df = pd.concat([df_wss, df_surf['dist']], axis=1)

                # Rename columns
                if 'Points:0' in df.columns:
                    df.rename(columns={"Points:0": "x", "Points:1": "y", "Points:2": "z"}, inplace=True)
                if 'x' in df.columns or 'y' in df.columns:
                    df.rename(columns={"# x": "x", "y": "y", "z": "z"}, inplace=True)

                # Set origin to zero (only for first boundary)
                if xorigin == 0:
                    xorigin = df['x'].min()
                df['x'] = df['x'] - xorigin # Set origin to zero
                print("xorigin:", xorigin)

                if showGeometry:
                    axx.scatter(df['x'] / ctu_len, df['y'] / ctu_len, marker='o', label=label + ' x-y plane')
                    axy.scatter(df['y'] / ctu_len, df['z'] / ctu_len, marker='o', label=label + ' y-z plane')
                    axz.scatter(df['x'] / ctu_len, df['z'] / ctu_len, marker='o', label=label + ' x-z plane')
                else:
                    # Create an empty DataFrame to store the results
                    df_result = pd.DataFrame(index=df.index)

                    # Apply the function row-wise to compute xplus and zplus
                    df_result[['xplus', 'yplus', 'zplus']] = df.apply(compute_closest_point, axis=1, df_coords=df)

                    # Visualise
                    axx.scatter(df['x'] / ctu_len, df_result['xplus'], marker='o', label=label + ' xplus')
                    axy.scatter(df['x'] / ctu_len, df_result['yplus'], marker='o', label=label + ' yplus')
                    axz.scatter(df['x'] / ctu_len, df_result['zplus'], marker='o', label=label + ' zplus')

                    df_out = pd.concat([df, df_result], axis=1)

                    df_out.to_csv(boundary_file_path.replace("wss", "wallunits"), sep=' ', index=False)
                    print("Written {0}".format(boundary_file_path.replace("wss", "wallunits")))

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


