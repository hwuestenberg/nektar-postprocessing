#!/bin/python3.12
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import os

from config import path_to_directories, directory_names, save_directory

fig = plt.figure(figsize=(12,8))
axk = fig.add_subplot(2, 1, 1)
axzeta = fig.add_subplot(2, 1, 2)

axk.set_xlabel("Time [sec]")
axk.set_ylabel("Integral kinetic energy")

axzeta.set_xlabel("Time [sec]")
axzeta.set_ylabel("Enstrophy")



savename = save_directory + "dissipation-highre"



for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
    print(f"\nProcessing {dirname}..")

    # Find local energy file and assume there is only one
    full_directory_path = path_to_directories + dirname
    full_path_to_file = glob(full_directory_path + "*.eny")[0]

    if not os.path.exists(f"{full_path_to_file}"):
        print(f"ERROR. Could not find file {full_path_to_file}. Exiting.")
        exit()

    # Read data
    df = pd.read_csv(f"{full_path_to_file}", sep='\s+', header=2, names=['time', 'k', 'zeta'])
    df['zeta']  = 2 * df['zeta'] / 160000 # compute dissipation rate eps = dk/dt = 2 \nu \zeta
    # print(df)

    # Plot data
    axk.plot(df['time'], df['k'], label=f"{dirname}", color=dir_color)
    axzeta.plot(df['time'], df['zeta'], label=f"{dirname}", color=dir_color)


# # # Read and plot reference for TGV
# df = pd.read_csv(f"{path_to_directories}physics/reference-driakis-re1600.eny", sep=' ', decimal=',', names=['time', 'eps'])
# axzeta.plot(df['time'], df['eps'], '--k', label=f"Driakis Re1600")


axk.legend()
axzeta.legend()

if savename:
    fig.savefig(savename, bbox_inches="tight")

plt.show()
