from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Sequence
import os

import pandas as pd
from matplotlib.colors import TABLEAU_COLORS

from utilities import get_time_step_size, get_label, filter_time_interval


@dataclass
class CaseMetadata:
    """Metadata describing a single simulation case."""

    dirname: str
    full_directory_path: str
    file_path: str
    dt: float
    label: str
    marker: str
    marker_face_color: str
    linestyle: str
    color: str


@dataclass
class ForceTimeseries:
    """Processed force time series for a case."""

    metadata: CaseMetadata
    phys_time: pd.Series
    signal: pd.Series


def iter_case_metadata(
    file_name: str,
    directory_names: Iterable[str],
    base_path: str,
    colors: Sequence[str] = TABLEAU_COLORS,
) -> Generator[CaseMetadata, None, None]:
    """Yield metadata for all available cases.

    The helper verifies that the requested file exists for each case and
    enriches it with time-step information and plotting style returned by
    :func:`utilities.get_label`.
    """

    for dirname, dir_color in zip(directory_names, colors):
        full_directory_path = base_path + dirname
        file_path = full_directory_path + file_name

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue

        dt = get_time_step_size(full_directory_path)
        label, marker, mfc, ls, color = get_label(file_path, dt, color=dir_color)

        yield CaseMetadata(
            dirname=dirname,
            full_directory_path=full_directory_path,
            file_path=file_path,
            dt=dt,
            label=label,
            marker=marker,
            marker_face_color=mfc,
            linestyle=ls,
            color=color,
        )


def load_force_series(
    metadata: CaseMetadata,
    forces_file_noext: str,
    metric: str,
    ctu_skip: float,
    n_downsample: int,
    ref_area: float,
    ctu_len: float,
) -> ForceTimeseries:
    """Read and preprocess a force time series for a single case."""

    df = pd.read_pickle(metadata.file_path)
    df = df[forces_file_noext]

    phys_time = df["Time"] / ctu_len
    signal = df[metric]

    phys_time, signal = filter_time_interval(phys_time, signal, ctu_skip)

    signal = 2 * signal
    if "quasi3d" in metadata.file_path:
        signal = signal / ctu_len
    else:
        signal = signal / ref_area

    if n_downsample > 1:
        signal = signal[::n_downsample]
        phys_time = phys_time[::n_downsample]

    return ForceTimeseries(metadata=metadata, phys_time=phys_time, signal=signal)


def iter_force_cases(
    directory_names: Iterable[str],
    path_to_directories: str,
    forces_file_noext: str,
    metric: str,
    ctu_skip: float,
    n_downsample: int,
    ref_area: float,
    ctu_len: float,
    colors: Sequence[str] = TABLEAU_COLORS,
) -> Generator[ForceTimeseries, None, None]:
    """Yield processed force signals across all configured cases."""

    for metadata in iter_case_metadata("forces.pkl", directory_names, path_to_directories, colors):
        yield load_force_series(
            metadata,
            forces_file_noext=forces_file_noext,
            metric=metric,
            ctu_skip=ctu_skip,
            n_downsample=n_downsample,
            ref_area=ref_area,
            ctu_len=ctu_len,
        )
