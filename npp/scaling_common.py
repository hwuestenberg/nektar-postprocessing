from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd

from utilities import get_dof, get_label, get_scheme, get_time_step_size


NodeCase = Dict[str, object]


def _scaling_root(path_to_directories: str, dirname: str) -> Path:
    """Return the directory that stores scaling results for a given case."""
    base = Path(path_to_directories) / dirname
    return Path(str(base).replace("physics", "scaling"))


def _parse_node_directory_name(name: str) -> tuple[int, int]:
    nodes, cpus = name.split("x")
    return int(nodes), int(cpus)


def iter_scaling_nodes(
    directory_names: Iterable[str],
    path_to_directories: str,
    required_pattern: str,
) -> Iterator[tuple[Path, NodeCase]]:
    """Yield node directories and associated metadata for scaling runs.

    Directories are expected to follow the ``<nodes>x<cpus>`` naming pattern and
    contain files matching ``required_pattern``.  The metadata dictionary
    includes scheme, time step, CPU counts, degrees of freedom per rank and a
    display label.
    """

    for dirname in directory_names:
        scaling_dir = _scaling_root(path_to_directories, dirname)
        if not scaling_dir.exists():
            continue

        for node_dir in sorted(p for p in scaling_dir.iterdir() if p.is_dir()):
            if not list(node_dir.glob(required_pattern)):
                continue

            nodes, cpus = _parse_node_directory_name(node_dir.name)
            case: NodeCase = {
                "scheme": get_scheme(str(scaling_dir)),
                "ncpu": cpus,
                "nodes": nodes,
                "ncpus": nodes * cpus,
            }

            node_path_str = str(node_dir) + "/"
            get_dof(case, node_path_str)
            case["global_dof_per_rank"] = case["global_dof"] / case["ncpus"]
            case["local_dof_per_rank"] = case["local_dof"] / case["ncpus"]

            dt = get_time_step_size(node_path_str)
            case["dt"] = dt
            label, marker, mfc, ls, color = get_label(node_path_str, dt)
            case.update({"label": label, "color": color, "marker": marker, "ls": ls})

            yield node_dir, case


def read_timer_output(
    node_dir: Path,
    log_glob: str,
    timer_columns: List[str],
    replacements: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Extract timer statistics from a raw log file into a DataFrame."""

    log_files = [
        lf
        for lf in node_dir.glob(log_glob)
        if "log_info.csv" not in lf.name and not lf.name.endswith(".pkl")
    ]
    if not log_files:
        raise FileNotFoundError(f"No log file matching {log_glob} in {node_dir}")

    buffer = io.StringIO()
    reached_timers = False
    with log_files[0].open(encoding="utf-8") as f_in:
        for line in f_in:
            if "Execute" in line or reached_timers:
                reached_timers = True
            else:
                continue

            if "Victory!" in line:
                break

            for old, new in replacements:
                line = line.replace(old, new)
            buffer.write(line)

    df_func = pd.read_csv(buffer, names=timer_columns, sep="\s+|\t+|\s+\t+|\t+\s+", engine="python")
    df_func = df_func.dropna(axis=1)
    df_func.columns = timer_columns
    return df_func

