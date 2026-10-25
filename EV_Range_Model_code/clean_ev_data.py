"""Clean and combine BMW i3 trip CSV files for the first RFR model.

The source files use semicolon delimiters, Windows-1252 text encoding and a
0.1-second sample interval. This script accepts either one merged CSV or many
individual trip CSVs. It keeps only the common modelling columns, creates trip
identifiers, derives measured battery power and road grade, and writes a short
quality report.

Example:
    python clean_ev_data.py TripA01.csv TripA02.csv \
        --output cleaned_ev_trips.csv \
        --report cleaning_report.txt

Use --sample-period-s 0.1 to retain the original 10 Hz data, or 0 to disable
time-bin aggregation completely. The default 1.0 second output is intended for
a quick first Random Forest implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_TO_CLEAN = {
    "Time [s]": "time_s",
    "Velocity [km/h]": "velocity_kmh",
    "Elevation [m]": "elevation_m",
    "Throttle [%]": "throttle_percent",
    "Longitudinal Acceleration [m/s^2]": "acceleration_mps2",
    "Regenerative Braking Signal": "regen_braking_signal",
    "Battery Voltage [V]": "battery_voltage_v",
    "Battery Current [A]": "battery_current_a",
    "SoC [%]": "soc_percent",
}

REQUIRED_CLEAN_COLUMNS = tuple(RAW_TO_CLEAN.values())

MISSING_MARKERS = [
    "",
    " ",
    "NA",
    "N/A",
    "n/a",
    "NaN",
    "nan",
    "null",
    "None",
    "-",
    "--",
]

# Values outside these broad physical bounds are changed to NaN and reported.
# They are not clipped to an apparently valid value.
VALID_RANGES = {
    "time_s": (0.0, None),
    "velocity_kmh": (0.0, 250.0),
    "elevation_m": (-500.0, 9_000.0),
    "throttle_percent": (0.0, 100.0),
    "acceleration_mps2": (-15.0, 15.0),
    "regen_braking_signal": (0.0, 1.0),
    "battery_voltage_v": (100.0, 500.0),
    "battery_current_a": (-1_000.0, 1_000.0),
    "soc_percent": (0.0, 100.0),
}

OUTPUT_COLUMNS = [
    "trip_id",
    "source_file",
    "time_s",
    "distance_m",
    "velocity_mps",
    "acceleration_mps2",
    "elevation_m",
    "gradient_percent",
    "throttle_percent",
    "regen_braking_signal",
    "battery_voltage_v",
    "battery_current_a",
    "measured_battery_power_w",
    "soc_percent",
]


def normalise_header(header: str) -> str:
    """Remove BOMs and accidental leading/trailing spaces from a heading."""

    return header.lstrip("\ufeff").strip()


def inspect_headers(path: Path) -> list[str]:
    header_frame = pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        nrows=0,
    )
    return list(header_frame.columns)


def read_required_columns(path: Path) -> pd.DataFrame:
    """Read only the columns common to every supplied trip schema."""

    headers = inspect_headers(path)
    selected_headers = [
        header
        for header in headers
        if normalise_header(header) in RAW_TO_CLEAN
    ]
    available = {normalise_header(header) for header in selected_headers}
    missing = sorted(set(RAW_TO_CLEAN) - available)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    frame = pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        usecols=selected_headers,
        na_values=MISSING_MARKERS,
        keep_default_na=True,
        low_memory=False,
    )
    frame.rename(
        columns={
            header: RAW_TO_CLEAN[normalise_header(header)]
            for header in selected_headers
        },
        inplace=True,
    )
    return frame


def load_inputs(paths: Iterable[Path], report: list[str]) -> pd.DataFrame:
    """Load inputs and assign trip IDs from files and internal time resets."""

    frames: list[pd.DataFrame] = []
    trip_offset = 0

    for path in paths:
        frame = read_required_columns(path)
        report.append(
            f"{path.name}: {len(frame):,} rows, "
            f"{len(inspect_headers(path))} source columns"
        )

        conversion_failures: dict[str, int] = {}
        for column in REQUIRED_CLEAN_COLUMNS:
            missing_before = int(frame[column].isna().sum())
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            missing_after = int(frame[column].isna().sum())
            conversion_failures[column] = missing_after - missing_before

        failures = {
            key: value
            for key, value in conversion_failures.items()
            if value > 0
        }
        if failures:
            report.append(f"  non-numeric cells converted to NaN: {failures}")

        # A negative time jump marks the first row of a new trip. Each input
        # file also starts a new trip, even if it contains no internal reset.
        local_trip = frame["time_s"].diff().lt(0.0).cumsum().astype(int)
        frame["trip_id"] = local_trip + trip_offset + 1
        trip_offset = int(frame["trip_id"].max())
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def invalidate_impossible_values(
    frame: pd.DataFrame,
    report: list[str],
) -> None:
    report.append("")
    report.append("Values changed to NaN because they were outside bounds:")
    any_invalid = False
    for column, (minimum, maximum) in VALID_RANGES.items():
        invalid = pd.Series(False, index=frame.index)
        if minimum is not None:
            invalid |= frame[column] < minimum
        if maximum is not None:
            invalid |= frame[column] > maximum
        count = int(invalid.sum())
        if count:
            any_invalid = True
            frame.loc[invalid, column] = np.nan
            report.append(f"  {column}: {count:,}")
    if not any_invalid:
        report.append("  none")


def combine_duplicate_timestamps(
    frame: pd.DataFrame,
    report: list[str],
) -> pd.DataFrame:
    """Average duplicate timestamps, while retaining legitimate stopped rows."""

    exact_duplicates = int(
        frame.duplicated(
            subset=["trip_id", *REQUIRED_CLEAN_COLUMNS],
            keep="first",
        ).sum()
    )
    report.append(f"Exact duplicate rows removed: {exact_duplicates:,}")
    if exact_duplicates:
        frame = frame.drop_duplicates(
            subset=["trip_id", *REQUIRED_CLEAN_COLUMNS],
            keep="first",
        )

    duplicate_time_rows = frame.duplicated(
        subset=["trip_id", "time_s"],
        keep=False,
    )
    duplicate_time_count = int(duplicate_time_rows.sum())
    report.append(
        f"Rows participating in duplicate timestamps: {duplicate_time_count:,}"
    )
    if not duplicate_time_count:
        return frame

    numeric_columns = list(REQUIRED_CLEAN_COLUMNS)
    return (
        frame.groupby(["trip_id", "time_s"], as_index=False, sort=False)
        .agg(
            {
                **{
                    column: "mean"
                    for column in numeric_columns
                    if column != "time_s"
                },
                "source_file": "first",
            }
        )
    )


def aggregate_time_bins(
    frame: pd.DataFrame,
    sample_period_s: float,
    report: list[str],
) -> pd.DataFrame:
    if sample_period_s <= 0.0:
        report.append("Time-bin aggregation: disabled")
        return frame

    original_rows = len(frame)
    frame = frame.copy()
    frame["sample_bin"] = np.floor(
        frame["time_s"] / sample_period_s + 1e-9
    ).astype("Int64")

    numeric_columns = [
        column
        for column in REQUIRED_CLEAN_COLUMNS
        if column != "time_s"
    ]
    frame = (
        frame.groupby(["trip_id", "sample_bin"], as_index=False, sort=False)
        .agg(
            {
                **{column: "mean" for column in numeric_columns},
                "source_file": "first",
            }
        )
    )
    frame["time_s"] = frame["sample_bin"] * sample_period_s
    frame.drop(columns="sample_bin", inplace=True)
    report.append(
        f"Time-bin aggregation: {sample_period_s:g} s "
        f"({original_rows:,} -> {len(frame):,} rows)"
    )
    return frame


def add_derived_columns(
    frame: pd.DataFrame,
    sample_period_s: float,
    gradient_half_window_s: float,
) -> pd.DataFrame:
    """Add velocity, distance, measured battery power and smoothed grade."""

    frame = frame.copy()
    frame["velocity_mps"] = frame["velocity_kmh"] / 3.6

    # The sample trips show negative current during propulsion/discharge and
    # positive current during regenerative charging. Positive power below
    # therefore means energy is being removed from the battery.
    frame["measured_battery_power_w"] = -(
        frame["battery_voltage_v"] * frame["battery_current_a"]
    )

    if sample_period_s > 0.0:
        half_window_rows = max(
            1,
            int(round(gradient_half_window_s / sample_period_s)),
        )
    else:
        half_window_rows = 20

    completed_groups: list[pd.DataFrame] = []
    for _, trip in frame.groupby("trip_id", sort=False):
        trip = trip.sort_values("time_s", kind="stable").copy()
        dt_s = trip["time_s"].diff()
        average_speed_mps = 0.5 * (
            trip["velocity_mps"] + trip["velocity_mps"].shift(1)
        )
        distance_increment_m = average_speed_mps * dt_s
        distance_increment_m = distance_increment_m.where(
            (dt_s > 0.0) & np.isfinite(distance_increment_m),
            0.0,
        )
        trip["distance_m"] = distance_increment_m.cumsum()

        elevation_before = trip["elevation_m"].shift(half_window_rows)
        elevation_after = trip["elevation_m"].shift(-half_window_rows)
        distance_before = trip["distance_m"].shift(half_window_rows)
        distance_after = trip["distance_m"].shift(-half_window_rows)
        distance_span_m = distance_after - distance_before
        elevation_change_m = elevation_after - elevation_before

        gradient_percent = 100.0 * elevation_change_m / distance_span_m
        gradient_percent = gradient_percent.where(distance_span_m >= 5.0)

        if trip["elevation_m"].nunique(dropna=True) <= 1:
            gradient_percent = gradient_percent.fillna(0.0)
        else:
            gradient_percent = gradient_percent.interpolate(
                limit_direction="both"
            )
        trip["gradient_percent"] = gradient_percent
        completed_groups.append(trip)

    return pd.concat(completed_groups, ignore_index=True, sort=False)


def build_report(
    frame: pd.DataFrame,
    initial_rows: int,
    report: list[str],
) -> str:
    report.extend(
        [
            "",
            f"Initial rows: {initial_rows:,}",
            f"Cleaned rows: {len(frame):,}",
            f"Trips detected: {frame['trip_id'].nunique():,}",
            "",
            "Missing values in final output:",
        ]
    )
    for column in OUTPUT_COLUMNS:
        report.append(f"  {column}: {int(frame[column].isna().sum()):,}")

    report.extend(["", "Final numeric ranges:"])
    for column in OUTPUT_COLUMNS:
        if column in {"trip_id", "source_file"}:
            continue
        report.append(
            f"  {column}: min={frame[column].min():.6g}, "
            f"max={frame[column].max():.6g}"
        )

    report.extend(["", "Rows per trip:"])
    for trip_id, count in frame.groupby("trip_id").size().items():
        report.append(f"  trip {trip_id}: {count:,}")
    return "\n".join(report) + "\n"


def clean_data(
    input_paths: list[Path],
    output_path: Path,
    report_path: Path,
    sample_period_s: float,
    gradient_half_window_s: float,
) -> pd.DataFrame:
    report = ["EV trip cleaning report", "=======================", ""]
    frame = load_inputs(input_paths, report)
    initial_rows = len(frame)

    missing_time = int(frame["time_s"].isna().sum())
    frame = frame.dropna(subset=["time_s"]).copy()
    report.append(f"Rows removed because time was missing: {missing_time:,}")

    invalidate_impossible_values(frame, report)
    frame = frame.dropna(subset=["time_s"]).copy()
    frame = combine_duplicate_timestamps(frame, report)
    frame = aggregate_time_bins(frame, sample_period_s, report)
    frame = add_derived_columns(
        frame,
        sample_period_s,
        gradient_half_window_s,
    )
    frame = frame[OUTPUT_COLUMNS].sort_values(
        ["trip_id", "time_s"],
        kind="stable",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report_text = build_report(frame, initial_rows, report)
    report_path.write_text(report_text, encoding="utf-8")
    return frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean one merged EV CSV or multiple individual trip CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Merged CSV or one or more individual trip CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cleaned_ev_trips.csv"),
        help="Cleaned CSV output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("cleaning_report.txt"),
        help="Text quality-report output path.",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=1.0,
        help=(
            "Output time-bin size in seconds. Use 0.1 for original 10 Hz "
            "resolution or 0 to disable aggregation."
        ),
    )
    parser.add_argument(
        "--gradient-half-window-s",
        type=float,
        default=2.0,
        help="Half-width of the centred elevation/grade calculation window.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    missing_files = [path for path in arguments.inputs if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Input files not found: {missing_files}")
    if arguments.sample_period_s < 0.0:
        raise ValueError("sample period cannot be negative")
    if arguments.gradient_half_window_s <= 0.0:
        raise ValueError("gradient half-window must be positive")

    cleaned = clean_data(
        input_paths=arguments.inputs,
        output_path=arguments.output,
        report_path=arguments.report,
        sample_period_s=arguments.sample_period_s,
        gradient_half_window_s=arguments.gradient_half_window_s,
    )
    print(f"Cleaned {cleaned['trip_id'].nunique()} trips")
    print(f"Wrote {len(cleaned):,} rows to {arguments.output}")
    print(f"Quality report: {arguments.report}")


if __name__ == "__main__":
    main()
