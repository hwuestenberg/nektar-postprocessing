#!/bin/python3
import os
import re
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    path_to_directories,
    directory_names,
    force_file_skip_start,
    force_file_skip_end,
    force_file_glob_strs,
    history_file_glob_strs,
    log_file_glob_strs,
    use_iterations,
    use_cfl,
    DEBUG,
    customMetrics,
)
from utilities import get_data_frame



def adjust_info_length(step_info, cfl_info = [[]], iter_info = [[]]):
    # Check length for each info to find IO_INFOSTEPS and IO_CFLSTEPS
    # Note that len(iter) (iteration info) is always equal to number of time steps
    len_iter = len(iter_info[0])
    len_step = len(step_info[0])
    len_cfl = len(cfl_info[0])
    len_max = max(len_iter, len_step, len_cfl)

    # Determine verbose output rate of step and cfl info
    io_infosteps = step_info[0][0]
    io_cflsteps = 0
    if not len_cfl == 0:
        ratio_cfl_step = len_cfl / len_step
        io_cflsteps =  int(ratio_cfl_step / io_infosteps) if len_cfl > len_step else int(io_infosteps / ratio_cfl_step)
    print(f"Longest list: {len_max}, io_infosteps = {io_infosteps}, io_cflsteps = {io_cflsteps}")

    # Adjust step_info
    # Fill unknown steps and phys_time correctly, fill unknown CPU time with nan
    new_step_info = []
    if len_step != len_max:
        print("Adjusting size of step_info")
        # Set step to continuous number
        # and phys_time and cpu_time to nan
        if len_step == 0:
            new_step_info.append(
                [i for i in range(len_max)]
            )
            new_step_info.append(
                [np.nan for i in range(len_max)]
            )
            new_step_info.append(
                [np.nan for i in range(len_max)]
            )
        # Set steps to continuous number
        # Derive phystime from available step info
        # Set CPU time to nan everywhere except for the known info
        # Another option would be to set an average of the CPU time for n_steps, but I prefer to keep "raw" data
        else:
            # Updated steps
            new_step_info.append(
                [i for i in range(len_max)]
            )

            # Updated phys_time
            dt = (step_info[1][1] - step_info[1][0]) / (io_infosteps)
            initial_phys_time = step_info[1][0] - io_infosteps * dt
            final_phys_time = step_info[1][-1]
            print(f"dt = {dt}, initial_phys_time = {initial_phys_time}, final_phys_time = {final_phys_time}, durations = {final_phys_time - initial_phys_time}")
            new_step_info.append(
                [initial_phys_time + i * dt for i in range(len_max)]
            )

            # Updated cpu time
            cpu_time = np.full(len_max, np.nan)
            cpu_time[::io_infosteps] = step_info[2]
            new_step_info.append(cpu_time)

    # If nothing to do, return the input
    else:
        new_step_info = step_info

    # Adjust len_cfl
    new_cfl_info = []
    if not len_cfl == 0 and not len_cfl == len_max:
        print("Adjusting size of cfl_info")
        # Set CFL and CFL element to nan
        if len_cfl == 0:
            new_cfl_info = [
                [np.nan for i in range(len_max)],
                [np.nan for i in range(len_max)]
            ]
        # Set unknown CFL and CFL element to nan
        else:
            # Updated cfl
            cfl = np.full(len_max, np.nan)
            cfl[::io_cflsteps] = cfl_info[0]
            new_cfl_info.append(cfl)

            # Updated cfl element
            cfl_element = np.full(len_max, np.nan)
            cfl_element[::io_cflsteps] = cfl_info[1]
            new_cfl_info.append(cfl_element)

    # If nothing to do, return the input
    else:
        new_cfl_info = cfl_info

    return new_step_info, new_cfl_info


def parse_meta_data(logfile):
    meta_dict = {}

    variable_pattern = r"Variable: (.*)"
    variables = []

    dimension_pattern = r"Spatial Dim.: (\d)"
    nsteps_pattern = r"No. of Steps: (\d+)"

    step_pattern = r"Steps:\s*(\d+)\s+Time:\s*([\d.e+-]+)\s+CPU\sTime:\s*([\d.e+-]+)s"
    steps_found = 0
    nsteps_prev = 0
    step_break = False

    cfl_info_pattern = r"IO_CFLSTEPS = (\d+)"
    # cfl_pattern = r"CFL:\s*([\d.e+-]+)\s*\(in\s*elmt\s*(\d+)\)"

    with open(logfile, 'r') as f:
        # Loop over each line in the file
        for i, line in enumerate(f):

            # Get variables
            if "Variable:" in line:
                fields = re.search(variable_pattern, line.strip())
                variables.append(fields.group(1))

            # Get number of dimensions
            if "Spatial Dim.:" in line:
                fields = re.search(dimension_pattern, line.strip())
                meta_dict['num_dimensions'] = fields.group(1)

            # Get number of time steps
            if "No. of Steps:" in line:
                fields = re.search(nsteps_pattern, line.strip())
                meta_dict['num_timesteps'] = fields.group(1)

            # Get number of steps between step info
            if 'Steps:' in line and 'Time:' in line:
                steps_found += 1
                fields = re.search(step_pattern, line.strip())
                nsteps = fields.group(1)

                # First time save number of steps
                if steps_found == 1:
                    nsteps_prev = nsteps
                # Second time: determine number steps between
                else:
                    meta_dict['step_info_steps'] = int(nsteps) - int(nsteps_prev)
                    step_break = True

            # Get number of steps between CFL info
            # CFL does not tell step in line (either check previous/next line or do not check at all)
            if 'IO_CFLSTEPS' in line:
                fields = re.search(cfl_info_pattern, line.strip())
                meta_dict['cfl_info_steps'] = fields.group(1)

            # Break meta data scan
            if step_break:
                # print("Breaking meta scan)")
                break

    # Post process lists
    meta_dict['variables'] = np.unique(variables).tolist() # Remove any duplicates

    return meta_dict


def parse_log_file(logfile):
    meta_dict = parse_meta_data(logfile)

    # num_dimensions = meta_dict['num_dimensions']
    num_variables = len(meta_dict['variables'])

    # List of list for iteration counts
    iter_info = [[] for i in range(num_variables)]
    i_variable = 0

    # Define the refined regex pattern to match numbers including those in scientific notation
    # iter_pattern = r"[\d.e+-]+"
    # iter_pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?|\d+"

    # List for step info
    step_pattern = r"Steps:\s*(\d+)\s+Time:\s*([\d.e+-]+)\s+CPU\sTime:\s*([\d.e+-]+)s"
    n_step_info = 3 # always prints step count, physical time and computational time per step(s)
    step_info = [[] for i in range(n_step_info)]

    # List for CFL number and element
    cfl_pattern = r"CFL:\s*([\d.e+-]+)\s*\(in\s*elmt\s*(\d+)\)"
    n_cfl_info = 2 # always prints max cfl estimate and corresponding element ID
    cfl_info =[[] for i in range(n_cfl_info)]

    with open(logfile, 'r') as f:
        # Loop over each line in the file
        for i, line in enumerate(f):
            ## Process iteration counts
            # Only process lines that contain 'iterations made'
            if use_iterations:
                if 'iterations made' in line:
                    # Extract the fifth field from the space-separated line
                    fields = re.split(r'\s+', line.strip())
                    if len(fields) >= 5:
                        iter_info[i_variable].append(fields[4])  # 5th field is index 4
                    # TODO use regex for all/most quantities here
                    # matches = re.findall(iter_pattern, line)

                    i_variable += 1

                    # Reset variable counter after all variables have been processed once
                    # identifies next time step for iteration counts
                    if i_variable == num_variables:
                        i_variable = 0

            ## Process: "Steps: 1        Time: 7.00050e+00  CPU Time: 98.902s"
            if 'Steps:' in line and 'Time:' in line:
                # Extract the fifth field from the space-separated line
                fields = re.search(step_pattern, line.strip())
                step_info[0].append(int(fields.group(1)))
                step_info[1].append(float(fields.group(2)))
                step_info[2].append(float(fields.group(3)))


            ## Process: "CFL: 8.30743e+01 (in elmt 26837)"
            if use_cfl:
                if 'CFL:' in line and 'in elmt' in line:
                    # Extract the fifth field from the space-separated line
                    fields = re.search(cfl_pattern, line.strip())
                    cfl_info[0].append(float(fields.group(1)))
                    cfl_info[1].append(int(fields.group(2)))

    # Adjust length for equal length lists
    if use_iterations and len(step_info[0]) != len(iter_info[0]):
        step_info, cfl_info = adjust_info_length(step_info, cfl_info, iter_info)
    elif use_cfl and len(step_info[0]) != len(cfl_info[0]):
        step_info, cfl_info = adjust_info_length(step_info, cfl_info)
    else:
        step_info, _ = adjust_info_length(step_info)


    # Create data frame from lists
    data_dict = {
        'steps' : step_info[0],
        'phys_time' : step_info[1],
        'cpu_time' : step_info[2],
    }

    # Add cfl info
    if use_cfl:
        data_dict['cfl'] = cfl_info[0]
        data_dict['cfl_element'] = cfl_info[1]

    # Add iterations info
    if use_iterations:
        data_dict['iterations_p'] = iter_info[0]
        data_dict['iterations_u'] = iter_info[1]
        data_dict['iterations_v'] = iter_info[2]

        # Add w only for 3D problems
        if num_variables == 4:
            data_dict['iterations_w'] = iter_info[3]

    # Convert to dataframe and return
    df_log = pd.DataFrame(data_dict)

    return df_log


def test_parse_log_file():
    # Read all given filenames in directory into DataFrame
    logfile = glob(path_to_directories + directory_names[0] + "log.eifw*")[0]
    df_log = parse_log_file(logfile, 3)

    # Loop through columns
    for i in df_log:
        print('Meta info:', i, type(i), len(df_log[i]))


# Find and sort all available subdirs following the naming convention ctu_start_end where start and end are integers
def find_subdirs(full_directory_path):
    subdirs = [f.path for f in os.scandir(full_directory_path) if f.is_dir() and 'ctu' in f.name]
    if len(subdirs) == 0:
        subdirs.append(full_directory_path + ".")  # scan root dirname as well
        print(f"\tNo subdirectories to merge. Directly parsing this directory: {Path(subdirs[0]).parts[-1]}")
    else:
        subdirs = sorted(subdirs, key=lambda x: int(re.search(r'ctu_(\d+)_', x).group(1)))
        print(f"\tFound subdirectories for merging: {[Path(subdir).parts[-1] for subdir in subdirs]}")

    return subdirs


def find_process_file(subdir, file_glob_str):
    # Add / for safety
    cdpath = subdir + "/"

    # Glob all files with glob_str
    files = glob(cdpath + file_glob_str)
    files = [l for l in files if not "log_info.pkl" in l]

    assert(len(files) == 0, f"Error. Could not find any files using glob string: {file_glob_str} on path: {cdpath}")

    # Choose process file as first or only file
    if len(files) > 1:
        print(f"WARNING. Could not identify unique file in list: {files}. Using the first file: {files[0]} on path: {cdpath}")
        process_file = files[0]
    elif len(files) == 0:
        process_file = None
    else:
        process_file = files[0]

    return process_file


def split_history_points(df_all_points):
    # Get number of history points
    npoints = int(len(df_all_points['Time']) / len(df_all_points['Time'].unique()))

    idx = np.arange(len(df_all_points))
    df_all_points = df_all_points.assign(
        point=idx % npoints,  # cycles 0..N-1
        time=idx // npoints  # increases every N rows
    )

    # Put keys into the index (optional but handy)
    df_all_points = df_all_points.set_index(['time', 'point']).sort_index()

    # Separate history points with keys
    return df_all_points



def read_file(file_glob_str, process_file):
    if file_glob_str in ["log*", "*.l*"]:
        df = parse_log_file(process_file)
    elif file_glob_str.endswith('.fce'):
        df = get_data_frame(
            process_file,
            skip_start=force_file_skip_start,
            skip_end=force_file_skip_end,
        )
    elif file_glob_str.endswith('.his'):
        df = get_data_frame(
            process_file,
            skip_start=force_file_skip_start,
            skip_end=force_file_skip_end,
        )
        df = split_history_points(df)
    else:
        df = get_data_frame(
            process_file,
            skip_start=force_file_skip_start,
            skip_end=force_file_skip_end,
        )
    return df




# TODO extend this for Energy (3D) files
if __name__ == "__main__":
    # Combine log, force and history file patterns
    all_file_glob_strs = []
    output_names = []
    if len(log_file_glob_strs) > 0:
        all_file_glob_strs += [log_file_glob_strs]
        output_names += ["log_info"]
    if len(force_file_glob_strs) > 0:
        all_file_glob_strs += [force_file_glob_strs]
        output_names += ["forces"]
    if len(history_file_glob_strs) > 0:
        all_file_glob_strs += [history_file_glob_strs]
        output_names += ["historypoints"]

    # Loop all directories (cases)
    for directory_name in directory_names:
        print(f"Processing directory {directory_name}")
        full_directory_path = path_to_directories + directory_name

        # Find all subdirectories in directory (concatenate multiple simulations)
        subdirs = find_subdirs(full_directory_path)

        # Loop all groups of files to be processed (log, force, history, ...)
        for output_name, file_glob_strs in zip(output_names, all_file_glob_strs):
            print(f"\tglob string {output_name}")

            # Create empty dict for all files from one group
            parts = {}  # level_name -> df_full

            # Loop all files for this group
            for file_glob_str in file_glob_strs:
                print(f"\t\tprocessing {file_glob_str}")

                # Create empty dataframe for this file(-type)
                df_full = pd.DataFrame()

                # Loop through all subdirectories, read files and merge all data
                for subdir in subdirs:
                    # Verbose print
                    # print(f"\tProcessing sub-directory {subdir}")

                    # Find log file(s) in sub-directory
                    process_file = find_process_file(subdir, file_glob_str)

                    # Parse log, force or history file
                    df_subdir = read_file(file_glob_str, process_file)


                    # Copy initial dataframe or concatenate parsed logs
                    if df_full.empty:
                        df_full = df_subdir
                    else:
                        df_full = pd.concat([df_full, df_subdir], axis=0, ignore_index=True)

                if df_full.empty:
                    print(f"WARNING. Could not find any files to process. Skipping files for {file_glob_str}.")
                    continue

                # df_full.to_csv(full_directory_path + output_name, index=False)

                # Debug print/plot
                if DEBUG:
                    # print(df_full)
                    if "TOTAL" in file_glob_str:
                        df_full.plot(x='Time', y=customMetrics[-1], label=directory_name, kind='scatter', title=output_name)
                    if "log" in file_glob_str:
                        df_full.plot(y='phys_time', label=directory_name, title=output_name)

                # Get level_name for this file and add to parts dict
                level_name = file_glob_str.split(".")[0]
                parts[level_name] = df_full

                # Clear memory
                del df_subdir
                del df_full

            # Concatenate all dataframes with keys = level_name(s)
            df_all = pd.concat(parts, axis=1)  # keys come from dict keys
            print(f"\tWriting {output_name} to directory {full_directory_path} ...")
            df_all.to_pickle(full_directory_path + output_name + ".pkl")

            # Ce
            del df_all



    # Debug print/plot
    if DEBUG:
        plt.legend()
        plt.show()
