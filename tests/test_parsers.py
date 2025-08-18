import os
from pathlib import Path
import random
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import preprocessor
import utilities

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_parse_log_file_and_plot(tmp_path):
    logfile = EXAMPLES / "log-eifw-linear-dt1e-5.l10521283"
    preprocessor.use_iterations = True
    preprocessor.use_cfl = False
    df = preprocessor.parse_log_file(str(logfile))

    expected_cols = {
        "steps",
        "phys_time",
        "cpu_time",
        "iterations_p",
        "iterations_u",
        "iterations_v",
        "iterations_w",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 110

    fig, ax = plt.subplots()
    ax.plot(df["phys_time"], df["cpu_time"])
    fig_path = tmp_path / "log_info.png"
    fig.savefig(fig_path)
    assert fig_path.exists()
    plt.close(fig)


def test_get_data_frame_force_file_and_plot(tmp_path):
    forcefile = EXAMPLES / "FWING_TOTAL_forces.fce"
    df = utilities.get_data_frame(str(forcefile))

    expected = [
        "Time",
        "F1-press",
        "F1-visc",
        "F1-total",
        "F2-press",
        "F2-visc",
        "F2-total",
        "F3-press",
        "F3-visc",
        "F3-total",
        "M1-press",
        "M1-visc",
        "M1-total",
        "M2-press",
        "M2-visc",
        "M2-total",
        "M3-press",
        "M3-visc",
        "M3-total",
    ]
    assert df.columns.tolist() == expected
    assert len(df) >= 100

    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["F1-total"])
    fig_path = tmp_path / "forces.png"
    fig.savefig(fig_path)
    assert fig_path.exists()
    plt.close(fig)


def test_get_data_frame_history_file_and_plot(tmp_path):
    hisfile = EXAMPLES / "mainplane-suction-midplane.his"
    df = utilities.get_data_frame(str(hisfile))

    assert df.columns.tolist() == ["Time", "u", "v", "w", "p"]
    assert len(df) >= 100

    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["p"])
    fig_path = tmp_path / "history.png"
    fig.savefig(fig_path)
    assert fig_path.exists()
    plt.close(fig)


if __name__ == "__main__":
    size = 10
    chars = string.ascii_uppercase + string.digits
    tmp_name = ''.join(random.choice(chars) for _ in range(size))
    tmp_path = Path("./tests/" + tmp_name)

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    test_parse_log_file_and_plot(tmp_path)
    test_get_data_frame_force_file_and_plot(tmp_path)
    test_get_data_frame_history_file_and_plot(tmp_path)

