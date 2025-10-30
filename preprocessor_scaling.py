#!/bin/python3
import os
import re
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib

from preprocessor import find_subdirs, find_process_file, read_file

matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from config import (
    path_to_directories,
    directory_names,
    force_file_skip_start,
    force_file_skip_end,
    log_file_glob_strs,
    use_iterations,
    DEBUG,
    scaling,
    customMetrics,
)


# Do not use CFL for scaling
use_cfl = False


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
    print(f"\t\tLongest list: {len_max}, io_infosteps = {io_infosteps}, io_cflsteps = {io_cflsteps}")

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



# TODO extend this for Energy (3D) files
if __name__ == "__main__":
    # Combine log, force and history file patterns
    all_file_glob_strs = []
    output_names = []
    if len(log_file_glob_strs) > 0:
        all_file_glob_strs += [log_file_glob_strs]
        output_names += ["log_info"]

    # Loop all directories (cases)
    for directory_name in directory_names:
        print(f"Processing directory {directory_name}")
        full_directory_path = path_to_directories + directory_name

        # Loop all groups of files to be processed (log, force, history, ...)
        for output_name, file_glob_strs in zip(output_names, all_file_glob_strs):
            print(f"\tglob string {output_name}")

            # Process only a single glob string for log file
            file_glob_str = file_glob_strs[0]

            # Create empty dict for all files from one group
            parts = {}  # level_name -> df_full

            # Verbose print
            full_directory_path = path_to_directories + directory_name
            full_directory_path = full_directory_path.replace("physics", "scaling")
            node_directories = [x for x in os.walk(full_directory_path)][0][1]  # get all Yx64 directories
            nodes = sorted([int(x.split("x")[0]) for x in node_directories])
            n_cpus = max([int(x.split("x")[1]) for x in node_directories])
            print(f"Processing {full_directory_path}")

            for n_nodes in nodes:
                # Create node directory path
                node_directory_path = full_directory_path + f"{n_nodes}x{n_cpus}/"
                print(f"\tProcessing node-directory {node_directory_path}")

                # Find log file(s) in sub-directory
                process_file = find_process_file(node_directory_path, file_glob_str)
                if process_file is None:
                    print(f"\t\tNo {file_glob_str} file found in {node_directory_path}. Skipping.")
                    continue

                # Parse log or forces file
                df_file = read_file(file_glob_str, process_file)

                # Save similar to preprocessor for convenience
                level_name = file_glob_str
                parts[level_name] = df_file
                df_all = pd.concat(parts, axis=1)  # keys come from dict keys
                df_all.to_pickle(node_directory_path + output_name + ".pkl")

                # Debug print/plot
                if DEBUG:
                    df_file.plot(y='phys_time', label=node_directory_path)

                # Clear memory
                del df_file

    # Debug print/plot
    if DEBUG:
        plt.legend()
        plt.show()
