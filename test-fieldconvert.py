import sys
sys.path.insert(0, "/home/henrik/code/nektar-animation/build-master/python")
from NekPy.FieldUtils import *

import glob

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from config import directory_names, path_to_directories, save_directory



ctuname = "ctu_20_30"

filenames = [
        "mean_fields_" + ctuname + "_avg_wss_b0",
        "mean_fields_" + ctuname + "_avg_wss_b1",
        "mean_fields_" + ctuname + "_avg_wss_b2",
        ]

savename = f"wallunits"
savename = save_directory + savename

path_to_mesh = path_to_directories + "3d/please-work/mesh/mesh.xml"
path_to_session = path_to_directories + directory_names[0] + "means/session.xml"


# FieldConvert calls
def convert_fld_to_csv(f):
    field = Field(sys.argv)
    InputModule.Create("xml", field, path_to_mesh).Run()
    InputModule.Create("xml", field, path_to_session).Run()
    InputModule.Create("fld", field, f).Run()
    OutputModule.Create("csv", field, f.replace(".fld", ".csv")).Run()


def get_csv_file(path_to_file):
    csvfile = ""
    path_to_all_files = glob.glob(path_to_file + "*")
    path_to_all_files = [p for p in path_to_all_files if not "slicey" in p]
    print("filepaths", path_to_all_files)
    if len(path_to_all_files) == 0:
        print("Did not find csv file but found {0}".format(path_to_file))
        return ""
    elif len(path_to_all_files) > 0:
        print("Found files {0}".format(path_to_all_files))
        for filepath in path_to_all_files:
            if filepath.endswith(".csv"):
                return filepath
            elif filepath.endswith(".fld"):
                print("Converting {0} to csv".format(filepath))
                convert_fld_to_csv(filepath)
                return filepath.replace(".fld", ".csv")
            else:
                print("No csv or fld file in {0}".format(path_to_all_files, path_to_file))
                return ""
    else:
        return path_to_all_files[0]



if __name__ == "__main__":
    for dirname, dir_color in zip(directory_names, TABLEAU_COLORS):
        # Setup paths
        full_directory_path = path_to_directories + dirname + "means/"
        xorigin = 0
        for filename in filenames:
            full_file_path = full_directory_path + filename

            # Search for csv file
            csvfile = get_csv_file(full_file_path)

            # Skip if no file found
            if csvfile == "":
                continue


