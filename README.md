# Nektar++ post-processing

This repository contains a collection of Python 3 scripts for post-processing Nektar++ CFD simulations. The scripts share a common workflow centered around `preprocessor.py`, which extracts information from raw simulation output and prepares it for analysis.

## Configuration

Before running the scripts, edit `config.py` to set the paths to your simulation directories, reference scales (length, velocity, time), and any options such as file overlaps or custom metric names. Most scripts read these settings.

## Preprocessing

`preprocessor.py` is the main entry point. It:

- parses Nektar++ log files to retrieve iteration counts, time step, physical time, CPU time and optional CFL information;
- combines force, history (`.his`) and energy (`.mdl`) files from simulations with restarts, applying user‑defined overlaps to remove restart artefacts;
- optionally evaluates statistical convergence of force and history signals;
- writes the processed data and summary `log_info.csv` files used by the analysis scripts.

For scaling runs the companion script `preprocessor_scaling.py` performs the same tasks on a directory structure with multiple MPI node counts.

## Analysis scripts

The repository includes a variety of stand‑alone scripts that operate on the files produced by the preprocessor:

- `cfl_bar_plot.py` – create bar charts of CFL statistics extracted from `log_info.csv`.
- `draglift.py` – plot drag or lift histories, apply filtering and evaluate convergence.
- `energy.py` – plot integral kinetic energy and enstrophy from `.eny` files.
- `history-psd.py` – compute power spectral densities of history point data or spatial FFTs of field data.
- `log_info_plot.py` – plot variables such as CFL or iteration counts versus time from `log_info.csv`.
- `psd.py` – compute PSDs of drag and lift coefficients using Welch’s method or a direct FFT.
- `scaling.py` – strong-scaling analysis with speed‑up and parallel efficiency from `log_info.csv`.
- `scaling-by-function.py` – timing‑by‑function scaling analysis using raw log output.
- `slicer.py` – ParaView batch script to extract slices from `.vtu`/`.pvtu` files and write CSV/VTP.
- `surface.py` – plot surface pressure or skin‑friction distributions for different averaging intervals.
- `surface-span-avg.py` – compute spanwise‑averaged surface quantities using alphashape interpolation.
- `stabilityaccuracy.py` – assess stability and accuracy by plotting mean force vs. time‑step size.
- `test-fieldconvert.py` – helper script to convert Nektar++ `.fld` files to CSV using FieldConvert.
- `utilities.py` – shared helper functions used throughout the scripts.

Each script relies on the processed data produced by `preprocessor.py` and uses the paths defined in `config.py`. Figures are written to the directory specified by `save_directory`.

## Requirements

The scripts require Python 3 with common scientific packages such as `numpy`, `pandas`, `matplotlib`, `scipy`, and `alphashape`. Some utilities call external tools like ParaView’s `pvbatch` or Nektar++’s `FieldConvert` when needed.

## Usage

1. Adjust `config.py` to point to your simulation data.
2. Run `preprocessor.py` (or `preprocessor_scaling.py` for scaling cases) to generate `log_info.csv` and processed force/history/energy files.
3. Execute any of the analysis scripts on the preprocessed data, for example:

   ```bash
   python draglift.py
   python psd.py
   python surface.py
   ```

The scripts will create plots in the directory given by `save_directory`.
