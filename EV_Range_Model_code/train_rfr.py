"""Train and evaluate the Random Forest residual-correction model.

The physics-enriched input must already contain a trip-based ``dataset_split``
column.  Rows labelled ``train`` fit the model and rows labelled ``test`` are
reserved for evaluation.  The target is measured battery power minus the
physics-model battery-power estimate.

Example
-------
    python train_rfr.py physics_enriched_ev_trips.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "velocity_mps",
    "acceleration_mps2",
    "gradient_percent",
    "estimated_wheel_power_w",
    "throttle_percent",
    "regen_braking_signal",
]

TARGET_COLUMN = "battery_power_residual_w"

INPUT_COLUMNS = [
    "trip_id",
    "dataset_split",
    "time_s",
    "dt_s",
    "measured_battery_power_w",
    "estimated_battery_power_w",
    TARGET_COLUMN,
    *FEATURE_COLUMNS,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RFR to correct physics-model battery power."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("physics_enriched_ev_trips.csv"),
        help="physics-enriched CSV (default: physics_enriched_ev_trips.csv)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("rfr_results"),
        help="directory for the trained model and results",
    )
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument(
        "--max-samples",
        type=float,
        default=0.25,
        help="fraction of training rows sampled for each tree",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel workers; -1 uses all available CPU cores",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.input.is_file():
        raise FileNotFoundError(f"input file not found: {arguments.input}")
    if arguments.n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if arguments.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if arguments.min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive")
    if not 0.0 < arguments.max_samples <= 1.0:
        raise ValueError("max_samples must be in the interval (0, 1]")


def load_data(path: Path) -> pd.DataFrame:
    """Load only columns required for training and evaluation."""

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(INPUT_COLUMNS) - set(available_columns))
    if missing:
        raise ValueError(f"input dataset is missing columns: {missing}")

    data = pd.read_csv(path, usecols=INPUT_COLUMNS, low_memory=False)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    invalid_split = ~data["dataset_split"].isin(["train", "test"])
    if invalid_split.any():
        values = sorted(data.loc[invalid_split, "dataset_split"].unique())
        raise ValueError(f"unexpected dataset_split values: {values}")

    # These quantities cannot be imputed because they define the target and
    # the energy-accounting result. Feature NaNs are handled by the pipeline.
    essential = [
        "trip_id",
        "time_s",
        "dt_s",
        "measured_battery_power_w",
        "estimated_battery_power_w",
        TARGET_COLUMN,
    ]
    invalid_essential = data[essential].isna().any(axis=1)
    if invalid_essential.any():
        counts = data.loc[invalid_essential].groupby("trip_id").size().to_dict()
        raise ValueError(
            "rows with missing target/physics data remain; clean or exclude "
            f"the affected trips first: {counts}"
        )

    train_trip_ids = set(
        data.loc[data["dataset_split"] == "train", "trip_id"].unique()
    )
    test_trip_ids = set(
        data.loc[data["dataset_split"] == "test", "trip_id"].unique()
    )
    overlap = sorted(train_trip_ids & test_trip_ids)
    if overlap:
        raise ValueError(f"trips occur in both train and test sets: {overlap}")
    if not train_trip_ids or not test_trip_ids:
        raise ValueError("both training and testing trips are required")

    return data


def build_model(arguments: argparse.Namespace) -> Pipeline:
    """Create an imputation and Random Forest pipeline."""

    forest = RandomForestRegressor(
        n_estimators=arguments.n_estimators,
        max_depth=arguments.max_depth,
        min_samples_leaf=arguments.min_samples_leaf,
        max_samples=arguments.max_samples,
        bootstrap=True,
        random_state=arguments.random_seed,
        n_jobs=arguments.n_jobs,
        verbose=1,
    )
    return Pipeline(
        steps=[
            # The median is learned from training trips only when fit() runs.
            ("imputer", SimpleImputer(strategy="median")),
            ("forest", forest),
        ]
    )


def regression_metrics(
    measured_power: np.ndarray,
    predicted_power: np.ndarray,
) -> dict[str, float]:
    error = predicted_power - measured_power
    return {
        "mae_w": float(mean_absolute_error(measured_power, predicted_power)),
        "rmse_w": float(np.sqrt(mean_squared_error(measured_power, predicted_power))),
        "r2": float(r2_score(measured_power, predicted_power)),
        "mean_error_w": float(np.mean(error)),
    }


def make_trip_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Integrate power over time and compare complete test trips."""

    frame = predictions.copy()
    frame["measured_energy_j"] = (
        frame["measured_battery_power_w"] * frame["dt_s"]
    )
    frame["physics_energy_j"] = (
        frame["estimated_battery_power_w"] * frame["dt_s"]
    )
    frame["hybrid_energy_j"] = (
        frame["hybrid_battery_power_w"] * frame["dt_s"]
    )

    trip_metrics = (
        frame.groupby("trip_id", as_index=False)
        .agg(
            rows=("time_s", "size"),
            duration_s=("time_s", "max"),
            measured_energy_j=("measured_energy_j", "sum"),
            physics_energy_j=("physics_energy_j", "sum"),
            hybrid_energy_j=("hybrid_energy_j", "sum"),
        )
    )
    for prefix in ["measured", "physics", "hybrid"]:
        trip_metrics[f"{prefix}_energy_kwh"] = (
            trip_metrics.pop(f"{prefix}_energy_j") / 3.6e6
        )

    denominator = trip_metrics["measured_energy_kwh"].abs()
    valid_denominator = denominator > 1e-9

    trip_metrics["physics_error_kwh"] = (
        trip_metrics["physics_energy_kwh"]
        - trip_metrics["measured_energy_kwh"]
    )
    trip_metrics["hybrid_error_kwh"] = (
        trip_metrics["hybrid_energy_kwh"]
        - trip_metrics["measured_energy_kwh"]
    )
    trip_metrics["physics_absolute_error_percent"] = np.where(
        valid_denominator,
        100.0 * trip_metrics["physics_error_kwh"].abs() / denominator,
        np.nan,
    )
    trip_metrics["hybrid_absolute_error_percent"] = np.where(
        valid_denominator,
        100.0 * trip_metrics["hybrid_error_kwh"].abs() / denominator,
        np.nan,
    )
    return trip_metrics


def write_metrics_report(
    path: Path,
    data: pd.DataFrame,
    physics_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
    trip_metrics: pd.DataFrame,
    arguments: argparse.Namespace,
) -> None:
    train_rows = int((data["dataset_split"] == "train").sum())
    test_rows = int((data["dataset_split"] == "test").sum())
    train_trips = data.loc[
        data["dataset_split"] == "train", "trip_id"
    ].nunique()
    test_trips = data.loc[
        data["dataset_split"] == "test", "trip_id"
    ].nunique()

    lines = [
        "Random Forest residual-correction results",
        "=========================================",
        "",
        f"Training rows: {train_rows:,}",
        f"Testing rows: {test_rows:,}",
        f"Training trips: {train_trips}",
        f"Testing trips: {test_trips}",
        f"Features: {', '.join(FEATURE_COLUMNS)}",
        f"Target: {TARGET_COLUMN}",
        "",
        "Model settings",
        f"  Trees: {arguments.n_estimators}",
        f"  Maximum depth: {arguments.max_depth}",
        f"  Minimum samples per leaf: {arguments.min_samples_leaf}",
        f"  Sample fraction per tree: {arguments.max_samples}",
        f"  Random seed: {arguments.random_seed}",
        "",
        "Instantaneous test-power metrics",
        f"  Physics MAE: {physics_metrics['mae_w'] / 1000.0:.4f} kW",
        f"  Hybrid MAE: {hybrid_metrics['mae_w'] / 1000.0:.4f} kW",
        f"  Physics RMSE: {physics_metrics['rmse_w'] / 1000.0:.4f} kW",
        f"  Hybrid RMSE: {hybrid_metrics['rmse_w'] / 1000.0:.4f} kW",
        f"  Physics R2: {physics_metrics['r2']:.5f}",
        f"  Hybrid R2: {hybrid_metrics['r2']:.5f}",
        f"  Physics mean error: {physics_metrics['mean_error_w'] / 1000.0:.4f} kW",
        f"  Hybrid mean error: {hybrid_metrics['mean_error_w'] / 1000.0:.4f} kW",
        "",
        "Complete-test-trip metrics",
        "  Physics mean absolute percentage error: "
        f"{trip_metrics['physics_absolute_error_percent'].mean():.3f}%",
        "  Hybrid mean absolute percentage error: "
        f"{trip_metrics['hybrid_absolute_error_percent'].mean():.3f}%",
        "  Physics median absolute percentage error: "
        f"{trip_metrics['physics_absolute_error_percent'].median():.3f}%",
        "  Hybrid median absolute percentage error: "
        f"{trip_metrics['hybrid_absolute_error_percent'].median():.3f}%",
        "",
        "Residual convention",
        "  target = measured battery power - physics battery power",
        "  hybrid = physics battery power + predicted target",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    data = load_data(arguments.input)

    train_mask = data["dataset_split"] == "train"
    test_mask = data["dataset_split"] == "test"

    # float32 reduces memory usage for the large 10 Hz dataset.
    x_train = data.loc[train_mask, FEATURE_COLUMNS].astype(np.float32)
    y_train = data.loc[train_mask, TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_test = data.loc[test_mask, FEATURE_COLUMNS].astype(np.float32)

    print("\nTraining Random Forest residual model")
    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows:  {len(x_test):,}")
    print(f"Features:      {FEATURE_COLUMNS}")

    model = build_model(arguments)
    model.fit(x_train, y_train)
    predicted_residual = model.predict(x_test)

    test_results = data.loc[
        test_mask,
        [
            "trip_id",
            "time_s",
            "dt_s",
            "measured_battery_power_w",
            "estimated_battery_power_w",
            TARGET_COLUMN,
        ],
    ].copy()
    test_results["predicted_battery_power_residual_w"] = predicted_residual
    test_results["hybrid_battery_power_w"] = (
        test_results["estimated_battery_power_w"]
        + test_results["predicted_battery_power_residual_w"]
    )

    measured = test_results["measured_battery_power_w"].to_numpy()
    physics = test_results["estimated_battery_power_w"].to_numpy()
    hybrid = test_results["hybrid_battery_power_w"].to_numpy()
    physics_metrics = regression_metrics(measured, physics)
    hybrid_metrics = regression_metrics(measured, hybrid)
    trip_metrics = make_trip_metrics(test_results)

    forest = model.named_steps["forest"]
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = output_directory / "trained_rfr.joblib"
    predictions_path = output_directory / "rfr_test_predictions.csv"
    trip_metrics_path = output_directory / "rfr_trip_metrics.csv"
    importance_path = output_directory / "rfr_feature_importance.csv"
    report_path = output_directory / "rfr_metrics.txt"

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "residual_definition": (
            "measured_battery_power_w - estimated_battery_power_w"
        ),
        "random_seed": arguments.random_seed,
    }
    joblib.dump(model_bundle, model_path, compress=3)
    test_results.to_csv(predictions_path, index=False)
    trip_metrics.to_csv(trip_metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    write_metrics_report(
        report_path,
        data,
        physics_metrics,
        hybrid_metrics,
        trip_metrics,
        arguments,
    )

    print("\nTest-set results")
    print(
        f"Physics power RMSE: {physics_metrics['rmse_w'] / 1000.0:.3f} kW"
    )
    print(
        f"Hybrid power RMSE:  {hybrid_metrics['rmse_w'] / 1000.0:.3f} kW"
    )
    print(
        "Physics trip MAPE: "
        f"{trip_metrics['physics_absolute_error_percent'].mean():.3f}%"
    )
    print(
        "Hybrid trip MAPE:  "
        f"{trip_metrics['hybrid_absolute_error_percent'].mean():.3f}%"
    )
    print(f"\nResults saved in: {output_directory}")


if __name__ == "__main__":
    main()
