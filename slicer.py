#!/home/henrik/sw/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython

import argparse
import json
import sys
from paraview.simple import *

from config import boundary_names


# To execute this script:
# module load paraview/5.10.1
# pvbatch generate_slices.py --in_file fields.vtu


def get_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', action='store',
                        default=None, help='Input .vtu file', required=True)
    args = parser.parse_args()
    opts = vars(args)

    if not opts['in_file'].endswith('vtu'):
        print(
            f"(EE) The input file: {opts['in_file']} does not have .vtu file extension")
        sys.exit(1)

    return opts['in_file']


def set_defaults():
    config = {
        "x": {
            "active": False,
            "variables": [],
            "slices": 0,
            "equispaced": True,
            "distance": 0.0,  # in m,
            "origin": [],
            "normal": [1.0, 0.0, 0.0],
            "length": 0.0,  # in m,
            "max": 0.0,  # in m
            "min": 0.0  # in m
        },
        "y": {
            "active": False,
            "variables": [],
            "slices": 0,
            "equispaced": True,
            "distance": 0.0,  # in m
            "origin": [],
            "normal": [0.0, 1.0, 0.0],
            "length": 0.0,  # in m
            "max": 0.0,  # in m
            "min": 0.0  # in m
        },
        "z": {
            "active": False,
            "variables": [],
            "slices": 0,
            "equispaced": True,
            "distance": 0.0,  # in m
            "origin": [],
            "normal": [0.0, 0.0, 1.0],
            "length": 0.0,  # in m
            "max": 0.0,  # in m
            "min": 0.0  # in m
        },
    }

    return config



def set_midplane():
    config = {
        "x": {
            "active": False,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.5,  # in m,
            "origin": [],
        },
        "y": {
            "active": True,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.025,  # in m,
            "origin": [],
        },
        "z": {
            "active": False,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.1,  # in m,
            "origin": [],
        },
    }

    return config


def set_spanwiseAverage():
    config = {
        "x": {
            "active": False,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.5,  # in m,
            "origin": [],
        },
        "y": {
            "active": True,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.0025,  # in m,
            "origin": [],
        },
        "z": {
            "active": False,
            "variables": ["u", "v", "w", "p"],
            "equispaced": True,
            "distance": 0.1,  # in m,
            "origin": [],
        },
    }

    return config



def set_domain_info(input_vtu, options: dict):
    # Collect geometry information
    x_min = input_vtu.GetDataInformation().GetBounds()[0]
    x_max = input_vtu.GetDataInformation().GetBounds()[1]
    y_min = input_vtu.GetDataInformation().GetBounds()[2]
    y_max = input_vtu.GetDataInformation().GetBounds()[3]
    z_min = input_vtu.GetDataInformation().GetBounds()[4]
    z_max = input_vtu.GetDataInformation().GetBounds()[5]
    delta_x = x_max - x_min
    options["x"]["length"] = delta_x
    options["x"]["max"] = x_max
    options["x"]["min"] = x_min
    delta_y = y_max - y_min
    options["y"]["length"] = delta_y
    options["y"]["max"] = y_max
    options["y"]["min"] = y_min
    delta_z = z_max - z_min
    options["z"]["length"] = delta_z
    options["z"]["max"] = z_max
    options["z"]["min"] = z_min
    print(f'(II) x_min: {x_min} m')
    print(f'(II) x_max: {x_max} m')
    print(f'(II) y_min: {y_min} m')
    print(f'(II) y_max: {y_max} m')
    print(f'(II) z_min: {z_min} m')
    print(f'(II) z_max: {z_max} m')
    print(f'(II) Length: {delta_x} m')
    print(f'(II) Width: {delta_y} m')
    print(f'(II) Height: {delta_z} m')


def do_equispaced_slices(input_vtu, max_n_sect: int, options: dict, axis: str):
    slice1 = Slice(Input=input_vtu)
    # Initialise slice position
    position = []
    # Loop the part group with sections on all direction
    for running_section in range(1, max_n_sect):
        pos = options["max"] - float(running_section) * options["distance"]
        print('Slicing at position:', '{0:.6g}'.format(pos), 'm')
        slice1.SliceType.Normal = options["normal"]
        if axis == "x":
            slice1.SliceType.Origin = [pos, 0.0, 0.0]
        if axis == "y":
            slice1.SliceType.Origin = [0.0, pos, 0.0]
        if axis == "z":
            slice1.SliceType.Origin = [0.0, 0.0, pos]
        # Append slice position
        position.append(pos)
        # Save data
        slice_name = in_file.replace(".vtu","").replace(".pvtu","")
        slice_name += "-slice" + axis + str('{0:.6g}'.format(pos)) 
        data_to_save = slice1
        #if options["calculate_cp"]:
        #    data_to_save = calculate_cp(slice1)
        #if options["calculate_cp0"]:
        #    data_to_save = calculate_total_pressure(data_to_save)
        print(options["variables"])

        if any(b in in_file for b in boundary_names):
            slice_name += ".csv"
            SaveData(slice_name, proxy=data_to_save, PointDataArrays=options["variables"],Precision=8)
        else:
            slice_name += ".vtp"
            SaveData(slice_name, proxy=data_to_save,# ChooseArraysToWrite=['u','v','w','p','CFL'],
                PointDataArrays=options["variables"],
                DataMode='Binary')


def do_slices(input_vtu, options: dict, axis: str):
    slice1 = Slice(Input=input_vtu)
    slice1.SliceType.Normal = options["normal"]
    slice_name = in_file.replace(".vtu","").replace(".pvtu","")
    # Loop the part group with sections on all direction
    for pos in options["origin"]:
        print('Slicing at position:', '{0:.6g}'.format(
            pos[0]), '{0:.6g}'.format(pos[1]), '{0:.6g}'.format(pos[2]), 'm')
        if axis == "x":
            slice1.SliceType.Origin = [pos[0], pos[1], pos[2]]
            slice_name += "-slice" + axis + str('{0:.6g}'.format(pos[0])) + ".vtp"
        if axis == "y":
            slice1.SliceType.Origin = [pos[0], pos[1], pos[2]]
            slice_name += "-slice" + axis + str('{0:.6g}'.format(pos[1])) + ".vtp"
        if axis == "z":
            slice1.SliceType.Origin = [pos[0], pos[1], pos[2]]
            slice_name += "-slice" + axis + str('{0:.6g}'.format(pos[2])) + ".vtp"

        # Save data
        data_to_save = slice1
        SaveData(slice_name, proxy=data_to_save,# ChooseArraysToWrite=1,
                 PointDataArrays=options["variables"],
                 DataMode='Binary')


if __name__ == '__main__':
    # Retrieve in_file from commandline
    in_file = ""
    in_file = get_file()
    if in_file:
        dirnames = [""]
        filenames = [in_file]


    print("dirnames:",dirnames)
    print("filenames:",filenames)

    for dirname in dirnames:
        for filename in filenames:
            if not in_file:
                in_file = dirname + "means/" + filename
            print("Prcocessing {0}".format(in_file))

            # Read the settings from the input json file
            #f_settings = open('/home/henrik/Documents/00_phd/03_project/simulations/codeVerification/f1-ifw/eifw/3d/input.json')
            #settings = json.load(f_settings)
            if any(b in in_file for b in boundary_names):
                settings = set_spanwiseAverage()
            else:
                settings = set_midplane()
            options = set_defaults()
            for entry in settings:  # loop through the entries of the json file
                # for the specific direction store settings inside the options dictionary which is used to perform the slices
                options[entry]["active"] = settings[entry]["active"]
                # if the slice is active in this direction store the following information
                if settings[entry]["active"]:
                    # flag for equispaced slices or not
                    options[entry]["equispaced"] = settings[entry]["equispaced"]
                    # check if the slices entry exists in the examined dictionary
                    if "slices" in settings[entry]:
                        if settings[entry]["slices"] != 0:  # fail-safe check
                            options[entry]["slices"] = settings[entry]["slices"]
                    # check if the equispaced entry exists in the examined dictionary
                    if "equispaced" in settings[entry]:
                        if settings[entry]["equispaced"]:  # If equispaced is True
                            if settings[entry]["distance"] != 0:  # fail-safe check
                                options[entry]["distance"] = settings[entry]["distance"]
                                print("Distance:", settings[entry]["distance"])
                            else:
                                raise Exception(
                                    "Distance need to be non-zero for equispaced slices.")
                        else:
                            if settings[entry]["origin"] != []:
                                for point in settings[entry]["origin"]:
                                    options[entry]["origin"].append(point)
                                    print("Point:", point)
                            else:
                                raise Exception(
                                    "Origin empty and equispaced is false: Undefined slices!")


            print('(II) in_file: ', in_file)
            from paraview.simple import *

            num_procs = paraview.servermanager.ActiveConnection.GetNumberOfDataPartitions()
            print(f"(II) Number of ranks: {num_procs}")

            # Create new 'XML Unstructured Grid Reader'
            if ".vtu" in in_file:
                input_vtu = XMLUnstructuredGridReader(FileName=[in_file])
            elif ".pvtu" in in_file:
                input_vtu = XMLPartitionedUnstructuredGridReader(FileName=[in_file])
            else:
                print("Cannot read file via paraview: {0}".format(in_file))

            if "wss" in in_file:
                input_vtu.PointArrayStatus = ['Shear_x', 'Shear_y', 'Shear_z', 'Shear_mag']
            else:
                input_vtu.PointArrayStatus = ['u', 'v', 'w', 'p', 'CFL']

            input_vtu.UpdatePipeline()

            # Collect data
            point_array_status = input_vtu.PointArrayStatus
            print(f'(II) Available data: {point_array_status}')
            set_domain_info(input_vtu, options)
            for direction in options:
                if options[direction]["active"]:
                    if options[direction]["equispaced"]:
                        max_n_sect = int(
                            options[direction]["length"] / options[direction]["distance"])
                        print('(II) Maximum Number of Slices towards',
                              direction, ':', max_n_sect)
                        do_equispaced_slices(input_vtu, max_n_sect,
                                             options[direction], direction)
                    else:
                        print('Slicing towards direction',
                              direction, 'using the input origins..')
                        do_slices(input_vtu, options[direction], direction)

            in_file = "" # Reset to none; this is pretty bad code

