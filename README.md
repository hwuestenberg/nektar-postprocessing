This a collection of python3 scripts to post-process Nektar++ cases.
The tools include:

Parsing log files for information such as iteration counts, physical time, CPU time, CFL number
Post-processing force files to

combine individual files for a longer simulation with restarts
define an overlap for these files (to remove pressure-kick at restart)
also applicable to historyPoint.his and EnergyFile.mdl file types
automatic evaluation of statistical convergence


Further tools allow

plot PSDs via Welch's method or direct FFT
plot skin friction and pressure coefficient data with or without spanwise averaging



The configuration file config.py must be filled with appropriate path definitions, reference length/velocity/time scales, overlap for force files and others.