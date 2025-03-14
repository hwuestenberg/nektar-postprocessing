#!/bin/python3

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



# Parse command line arguments                          
parser = argparse.ArgumentParser()                      
parser.add_argument(                                    
    "forces_file", help="File that contains the force data", type=str, default='LFW_element_2_forces.fce', nargs='?')
parser.add_argument(
    "avgLen", help="Length of time interval from latest time for sampling average, in CTU", type=float, default=10.0, nargs='?')
args = parser.parse_args()


# Verbose prints
print("Using forces_file:", args.forces_file)              
print("Averaging over {0} CTUs".format(
    args.avgLen))


forces_files = [
        "FWING_TOTAL_forces.fce",
        "LFW_fia_mp_forces.fce",
        "LFW_element_1_forces.fce",
        "LFW_element_2_forces.fce",
        ]


ylabels = ["$C_d$", "$C_l$"]#, "$C_S$", "$C_L$"]
ynames = ["Drag", "Lift"]#, "Sheer", "Lift"]
customMetrics = ["F1-total", "F3-total"]#, "F2-total", "F3-total"]
CTUlen = 0.25
spanlenNpp = 0.05

dtref = 1e-5 # Refernce time step for CFL \approx 1
dtol = 1e+1 # Divergence tolerance

xlim  = []#[8e-7/dtref,7e-4/dtref]
ylim  = []
yliml = []#[-8.8,-8.2]


savename = "eifw-stability"
dirnames = [
        #"3d/5bl/physics/semidt1e-5/",
        #"3d/8bl/physics/semidt1e-5/",
        #"3d/refined/physics/semidt1e-5/",
        "3d/please-work/physics/semidt1e-5/",
        "3d/please-work/physics/implicitdt1e-5/",
        "3d/please-work/physics/implicitdt5e-5/",
        "3d/please-work/physics/implicitdt1e-4/",
        "3d/please-work/physics/implicitdt5e-4/",
        "3d/please-work/physics/implicitdt1e-3/",
        ]


"""
    Minor utility functions
"""
def getForceLabel(metric):
    yIndex = customMetrics.index(metric)
    return ynames[yIndex]



if __name__ == "__main__":
    #logcrawler(dirnames, args.forces_file, 13)

    # Loop all files (including ref)
    for forces_file in forces_files:

        # Setup plot
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        figl = plt.figure(figsize=(4,3))
        axl = figl.add_subplot(111)

        # Set drag plot
        ax.set_xscale("log")
        ax.set_ylabel(r"$\overline{C}_d$")
        ax.set_xlabel(r"Time step increase $\Delta t_{CFL}$")
        ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.grid(which='both', axis='both')

        # Setup lift plot
        axl.set_xscale("log")
        axl.set_ylabel(r"$\overline{C}_l$")
        axl.set_xlabel(r"Time step increase $\Delta t_{CFL}$")
        axl.ticklabel_format(style='sci',axis='y', scilimits=(0,0), useMathText=True)
        axl.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axl.grid(which='both', axis='both')

        # Lists to save time step size and average iteration counts
        schemestrs = ["Implicit", "Semi-implicit"]#, "Sub-stepping"]
        meshstrs = ["5bl", "8bl", "refined", "please-work"]
        advstrs = ["Extrapolated", "Updated"]
        expstrs = ["equal-order", "taylor-hood"]
        imexstrs = ["IMEXOrder1", "IMEXOrder2"]
        forcestrs = ynames
        allstrs = [s + " " + m + " " + f for s in schemestrs for m in meshstrs for f in forcestrs]

        nlist = len(allstrs)
        dts = [list() for i in range(0,nlist)]
        data = [list() for i in range(0,nlist)]
        stds = [list() for i in range(0,nlist)]
        colors = list(TABLEAU_COLORS.values())

        filenames = [dname + forces_file for dname in dirnames]
        for filename in filenames:
            if not os.path.exists(filename):
                print("Did not find {0}".format(filename))
                continue
            else:
                print("\nProcessing file:",filename)

            df = get_data_frame(filename, customMetrics[0])

            # Get time/step info for x-axis
            dftime = pd.read_csv(filename.replace(forces_file,"cputime.dat"), sep=" ")
            physTime = dftime["phystime"].to_numpy()
            steps = dftime["step"].to_numpy()
            dt = (physTime[1] - physTime[0]) / (steps[1] - steps[0]) # Get time step size

            # Extract time from force reading, overwrite above
            time = df["Time"].to_numpy()
            time = time / CTUlen # Normalise as CTUs

            # Build mask based on time interval
            tmax = np.max(time)
            lowerMask = time >= tmax - args.avgLen
            upperMask = time <= tmax
            # Build mask based on time interval
            #lowerMask = time > args.beginAverage
            #upperMask = time < args.endAverage
            mask = (lowerMask == 1) & (upperMask == 1)
            if not np.any(mask):
                print("No data for interval = [{0}, {1}]".format(tmax - args.avgLen, tmax))
                continue

            # process each metric
            for metric in customMetrics:
                # Extract data
                metricData = df[metric]

                # Reduce data set based on mask
                metricData = metricData[mask]

                # Correct data (coeff = 2 * Force)
                metricData = 2 * metricData

                # Normalise by area
                metricData = metricData / CTUlen

                if not "quasi3d" in filename:
                    metricData = metricData / spanlenNpp

                # Do statistics
                if customMetrics[0] in metric:
                    mean = np.mean(metricData)
                elif customMetrics[1] in metric:
                    mean = np.mean(metricData)
                std = np.std(metricData)
                cv = std/mean

                # Ignore if diverged
                if np.abs(metricData.iloc[-1]) > dtol or np.abs(mean) > dtol:
                    print("Last datapoint = {0:.1e} or mean = {1:.1e} is larger than {2:.1e}. Assuming divergence and skipping.".format(np.abs(metricData.iloc[-1]), np.abs(mean), dtol))
                    break

                # Verbose statistics
                print("Metric: {0} on interval [{1}, {2}] using {3} points".format(
                    getForceLabel(metric),
                    time[mask][0], 
                    time[mask][-1], 
                    len(metricData)))
                print("Mean = {0}".format(mean))
                print("Std  = {0}".format(std))
                print("CV   = {0}\n".format(cv))

                indx = -1
                if "implicit" in filename:
                    indx = 0
                elif "semi" in filename:
                    indx = int(len(allstrs) / len(schemestrs))

                if "5bl" in filename:
                    indx = indx
                elif "8bl" in filename:
                    indx += 2
                elif "refined" in filename:
                    indx += 4
                elif "please-work" in filename:
                    indx += 6

                if customMetrics[0] in metric:
                    indx = indx # do nothing
                elif customMetrics[1] in metric:
                    indx = indx + 1 # iterate to higher indexes

                dts[indx].append(dt)
                data[indx].append(mean)
                stds[indx].append(std)

        mindt = min([min(dt) for dt in dts if len(dt) != 0])
        maxdt = max([max(dt) for dt in dts if len(dt) != 0])
        mindt = mindt / dtref
        maxdt = maxdt / dtref
        print("mindt:", mindt, "maxdt:", maxdt)
        dtspan = [mindt, maxdt]

        # Plot data
        for dt, mean, std, astr in zip(dts, data, stds, allstrs):
            # Skip not available configs
            if not len(dt):
                #print("No data for {0}".format(astr))
                continue

            # Get label by scheme
            marker = 'o'
            color = 'purple'
            ls = 'solid'
            if "Implicit" in astr:
                color = colors[1]
                label = "Implicit"
            elif "Semi-implicit" in astr:
                color = colors[0]
                label = "Semi-implicit"

            #if "5bl" in astr:
            #    label += " Mesh A"
            #    marker = 'x'
            #elif "8bl" in astr:
            #    label += " Mesh B"
            #    marker = '^'
            #elif "refined" in astr:
            #    label += " Mesh C"
            #    marker = 's'
            #elif "please-work" in astr:
            #    label += " Mesh D"
            #    marker = 'o'

            # Get marker by precon
            dtcfl = np.array(dt) / dtref
            if "Drag" in astr:
                ax.plot(dtcfl, mean, color=color, label=label, marker=marker, linestyle=ls)#, edgecolor='white')
                if "Semi-implicit" in astr:
                    ax.fill_between(dtspan, y1=mean[0]*1.01, y2=mean[0]*0.99, alpha=0.2, label="Semi-implicit +/- 1\% error", color=color)#, color=color, marker=marker, linestyle='', capsize=5)
                #ax.errorbar(dtcfl, mean, yerr = 2*np.array(std), color=color, marker=marker, linestyle='', capsize=5)
            elif "Lift" in astr:
                axl.plot(dtcfl, mean, color=color, label=label, marker=marker, linestyle=ls)#, edgecolor='white')
                if "Semi-implicit" in astr:
                    axl.fill_between(dtspan, y1=mean[0]*1.01, y2=mean[0]*0.99, alpha=0.2, label="Semi-implicit +/- 1\% error", color=color)#, color=color, marker=marker, linestyle='', capsize=5)
                #axl.errorbar(dtcfl, mean, yerr = 2*np.array(std), color=color, marker=marker, linestyle='', capsize=5)

        # Setup legend
        # Shrink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles, labels, loc="best")#,bbox_to_anchor=(1, 0.5))

        #handles, labels = axl.get_legend_handles_labels()
        #axl.legend(handles, labels, loc="best")#,bbox_to_anchor=(1, 0.5))


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="best")#,bbox_to_anchor=(1, 0.5))

        handles, labels = axl.get_legend_handles_labels()
        axl.legend(handles, labels, loc="best")#,bbox_to_anchor=(1, 0.5))

        if xlim:
            ax.set_xlim(xlim)
            axl.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if yliml:
            axl.set_ylim(yliml)

        if savename:
            print("Saving plot as",savename, "+ \"-cd\" or \"-cl\"")
            fig.savefig( savename + f"-{forces_file.replace('.fce','')}" + "-cd.pdf", bbox_inches="tight")
            figl.savefig(savename + f"-{forces_file.replace('.fce','')}" + "-cl.pdf", bbox_inches="tight")

        # Clear axes
        ax.cla()
        axl.cla()
        fig.clf()
        figl.clf()
        plt.close()
        plt.close()

    plt.show()
