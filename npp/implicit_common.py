"""Shared utilities for implicit scaling analyses."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from npp.scaling_common import iter_scaling_nodes

MetricDict = Dict[str, object]


def _extract_metric(metric: str, df: pd.DataFrame, dt: float, ctu_len: float) -> pd.Series:
    """Return the series for a given metric with implicit-specific handling."""

    ref_metric = metric
    if metric == "ctu_time":
        ref_metric = "cpu_time"
    elif metric == "iterations_uvw":
        ref_metric = "iterations_u"

    metric_data = df[ref_metric]
    if metric == "ctu_time":
        num_time_steps = ctu_len / dt
        metric_data = metric_data * num_time_steps
    elif metric == "iterations_uvw":
        metric_data = df["iterations_u"] + df["iterations_v"] + df["iterations_w"]

    return metric_data


def _load_processed_log(node_dir, process_file: str, log_str: str) -> pd.DataFrame:
    df = pd.read_pickle(node_dir / process_file)[log_str]
    df = df.apply(pd.to_numeric)

    npoints_remove = int(0.1 * len(df))
    return df.iloc[npoints_remove:]


def iter_implicit_cases(
    directory_names: Iterable[str],
    path_to_directories: str,
    process_file: str,
    log_str: str,
    additional_metrics: Sequence[str] | None,
    ctu_len: float,
    compute_ci: bool = False,
) -> Iterator[MetricDict]:
    """Yield statistics for implicit scaling runs.

    The iterator walks the scaling directories for the provided ``directory_names``
    and returns dictionaries containing metadata (scheme, CPU counts, degrees of
    freedom, time-step) along with aggregated statistics for the requested
    metrics.
    """

    for node_dir, case in iter_scaling_nodes(directory_names, path_to_directories, process_file):
        df = _load_processed_log(node_dir, process_file, log_str)

        metrics: List[str] = df.columns.to_list()
        if additional_metrics:
            metrics.extend(additional_metrics)

        case_data: MetricDict = dict(case)
        for metric in metrics:
            metric_values = _extract_metric(metric, df, case_data["dt"], ctu_len)
            case_data[f"{metric}-mean"] = metric_values.mean()
            case_data[f"{metric}-std"] = metric_values.std()

            if compute_ci:
                confidence = 0.95
                n = len(metric_values)
                sem = metric_values.std() / np.sqrt(n)
                t = stats.t.ppf((1 + confidence) / 2, df=n - 1)
                case_data[f"{metric}-ci95"] = sem * t

        yield case_data
