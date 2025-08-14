# Matplotlib setup with latex
import matplotlib.pyplot as plt
params = {'text.usetex': True,
 'font.size' : 10,
}
plt.rcParams.update(params) 
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd

from utilities import get_time_step_size, mser, get_label, get_scheme
from config import directory_names, path_to_directories, dtref, \
    customMetrics, ref_area, ctu_len, divtol, force_file_skip_start, save_directory, file_glob_strs


# Choose lift [1] or drag [0]
metric = customMetrics[0]
forces_file = file_glob_strs[1]

averaging_len = 30 # [CTU] redundant due to MSER, just use large number
n_downsample = 2

savename = f"mean-{metric}-{forces_file.split('.')[0]}"
savename = save_directory + savename



xlim = [8e-6 / dtref, 8e-4 / dtref]
ylim = []


# Verbose prints
print("Using forces_file:", forces_file)
# print("Averaging over {0} CTUs".format(averaging_len))




if __name__ == "__main__":

    # Create figure and axis
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ylabel = r"$\overline{C}_l$"
    if metric == customMetrics[0]:
        ylabel = r"$\overline{C}_d$"
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(r"Time step increase $\Delta t_{CFL}$")
    ax.set_xlabel(r"Time step increase $\times \Delta t$")
    # ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.grid(which='both', axis='both')


    # Dataframe for gathering statistics for each case
    df_stat = pd.DataFrame(columns=[
        "scheme",
        "dt",
        f"{metric}-mean",
        f"{metric}-std",
    ])


    # Loop all files
    for dirname in directory_names:
        # Setup paths
        full_directory_path = path_to_directories + dirname

        # Add processed file name
        filename = forces_file.replace(".fce", f"-process-overlap-{force_file_skip_start}.fce")

        full_file_path = full_directory_path + filename

        # Create dictionary for gathering data
        case_dict = {'scheme': get_scheme(full_file_path)}

        # Get time step size
        dt = get_time_step_size(full_directory_path)
        case_dict['dt'] = dt

        # Get plot styling
        label, marker, mfc, ls, color = get_label(full_file_path, dt)
        print("\nProcessing {0}...".format(label))

        # Read file
        df = pd.read_csv(full_file_path, sep=',')

        # Extract time and data
        physTime = df["Time"]
        physTime = physTime / ctu_len # Normalise to CTUs

        # Build mask based on time interval
        tmax = physTime.max()
        lowerMask = physTime >= tmax - averaging_len
        upperMask = physTime <= tmax
        mask = (lowerMask == 1) & (upperMask == 1)
        if not np.any(mask):
            print("No data for interval = [{0}, {1}]".format(tmax - averaging_len, tmax))
            continue

        # Extract signal
        metricData = df[metric]

        # Reduce using time-interval mask
        metricData = metricData[mask]

        # Correct data (coeff = 2 * Force)
        metricData = 2 * metricData

        # Normalise by area
        # Note quasi-3d is averaged along spanwise
        if "quasi3d" in full_file_path:
            metricData = metricData / ctu_len
        else:
            metricData = metricData / ref_area

        # Downsample
        # Note: do this before MSER
        if n_downsample > 1:
            metricData = metricData[::n_downsample]
            physTime = physTime[::n_downsample]

        # Determine end of transient via mser
        mser_stride_length = 10 if dt < 5e-5 else 1
        intTransient = mser(metricData, physTime, stride_length=mser_stride_length)
        timeTransient = physTime.iloc[intTransient]
        print("End of transient at time {0} CTU and index {1}".format(timeTransient, intTransient))

        # Remove end of transient from signal
        metricData = metricData.iloc[intTransient:]

        # Do statistics
        mean = metricData.mean()
        std = metricData.std()
        cv = std/mean

        # Ignore if diverged
        if np.abs(metricData.iloc[-1]) > divtol or np.abs(mean) > divtol:
            print("Last datapoint = {0:.1e} or mean = {1:.1e} is larger than {2:.1e}. Assuming divergence and skipping.".format(np.abs(metricData.iloc[-1]), np.abs(mean), divtol))
            break

        # # Verbose statistics
        # print("Mean = {0}".format(mean))
        # print("Std  = {0}".format(std))
        # print("CV   = {0}\n".format(cv))

        # Add statistics to dict
        case_dict[f'{metric}-mean'] = metricData.mean()
        case_dict[f'{metric}-std'] = metricData.std()

        # Transform to DataFrame and concatenate
        df_case = pd.DataFrame([case_dict])
        df_stat = pd.concat([df_stat, df_case], axis=0, ignore_index=True)

    # Verbose check concatenation
    print(df_stat)

    # Plot by scheme: dt vs mean force
    for scheme, scheme_color in zip(df_stat['scheme'].unique(), TABLEAU_COLORS):
        df_plot = df_stat.loc[df_stat['scheme'] == scheme]
        ax.plot(df_plot['dt'] / dtref, df_plot[f'{metric}-mean'], marker='o', label=scheme)
        ax.errorbar(df_plot['dt'] / dtref, df_plot[f'{metric}-mean'], df_plot[f'{metric}-std'], color=scheme_color, capsize=4)

    # # Add +/- 1% error of semi-implicit
    # ref_scheme = 'semi-implicit'
    # if ref_scheme in df_stat['scheme'].unique():
    #     df_plot = df_stat.loc[df_stat['scheme'] == ref_scheme]
    #     dt_max_min = [df_stat['dt'].max() / dtref, df_stat['dt'].min() / dtref]
    #     ax.fill_between(dt_max_min, y1=df_plot[f'{metric}-mean']*1.01, y2=df_plot[f'{metric}-mean']*0.99,
    #                     alpha=0.2, label=rf"+/- 1\% error", color=list(TABLEAU_COLORS)[0])

    ax.legend(loc='best')

    ## Aesthetics
    # Set x/y-limits
    if not xlim:
        xlim = ax.get_xlim()
    if not ylim:
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Save data
    plt.savefig(savename + ".pdf", bbox_inches='tight')
    df_stat.to_csv(savename + ".csv", sep=',')
    print(f"Wrote files {savename} as pdf and csv")

    plt.show()
