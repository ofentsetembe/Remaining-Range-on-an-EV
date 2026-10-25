"""Add physics-model estimates and train/test labels to cleaned EV trips.

The split is made by complete trip IDs, never by individual rows.  This avoids
placing neighbouring 10 Hz samples from the same journey in both datasets.

Examples
--------
Reproducible random 80/20 trip split::

    python build_physics_dataset.py cleaned_ev_trips.csv

Choose the test trips explicitly::

    python build_physics_dataset.py cleaned_ev_trips.csv \
        --test-trip-ids 3 8 14 21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ev_range_project.physical_model import (
    G,
    Environment,
    Powertrain,
    VehicleParameters,
)


REQUIRED_COLUMNS = {
    "trip_id",
    "time_s",
    "velocity_mps",
    "acceleration_mps2",
    "gradient_percent",
    "measured_battery_power_w",
}


def assign_trip_splits(
    trip_ids: np.ndarray,
    test_fraction: float,
    random_seed: int,
    requested_test_trip_ids: list[int] | None,
) -> tuple[set[int], set[int]]:
    """Return disjoint training and testing trip-ID sets."""

    unique_trip_ids = np.array(sorted({int(value) for value in trip_ids}))
    if len(unique_trip_ids) < 2:
        raise ValueError("at least two trips are required for a train/test split")

    available = set(unique_trip_ids.tolist())
    if requested_test_trip_ids:
        test_ids = set(requested_test_trip_ids)
        unknown = sorted(test_ids - available)
        if unknown:
            raise ValueError(f"requested test trip IDs were not found: {unknown}")
        if test_ids == available:
            raise ValueError("at least one trip must remain in the training set")
    else:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")

        generator = np.random.default_rng(random_seed)
        shuffled_ids = generator.permutation(unique_trip_ids)
        number_of_test_trips = int(round(len(unique_trip_ids) * test_fraction))
        number_of_test_trips = int(
            np.clip(number_of_test_trips, 1, len(unique_trip_ids) - 1)
        )
        test_ids = set(shuffled_ids[:number_of_test_trips].tolist())

    train_ids = available - test_ids
    return train_ids, test_ids


def calculate_physics_columns(
    data: pd.DataFrame,
    vehicle: VehicleParameters,
    powertrain: Powertrain,
    environment: Environment,
) -> pd.DataFrame:
    """Apply the longitudinal model to every row using NumPy arrays."""

    vehicle.validate()
    powertrain.validate()

    speed = data["velocity_mps"].to_numpy(dtype=float)
    acceleration = data["acceleration_mps2"].to_numpy(dtype=float)
    grade_fraction = data["gradient_percent"].to_numpy(dtype=float) / 100.0
    road_angle = np.arctan(grade_fraction)

    rolling_force = np.where(
        speed > 0.0,
        vehicle.rolling_resistance_coefficient
        * vehicle.mass_kg
        * G
        * np.cos(road_angle),
        0.0,
    )

    relative_air_speed = speed + environment.wind_speed_mps
    aerodynamic_force = (
        0.5
        * environment.air_density_kg_m3
        * vehicle.drag_coefficient
        * vehicle.frontal_area_m2
        * relative_air_speed
        * np.abs(relative_air_speed)
    )
    gradient_force = vehicle.mass_kg * G * np.sin(road_angle)
    inertial_force = vehicle.equivalent_mass_kg * acceleration
    required_wheel_force = (
        rolling_force
        + aerodynamic_force
        + gradient_force
        + inertial_force
    )
    estimated_wheel_power = required_wheel_force * speed

    propulsion = estimated_wheel_power >= 0.0
    requested_traction_power = np.where(
        propulsion,
        estimated_wheel_power / powertrain.drivetrain_efficiency,
        0.0,
    )
    traction_battery_power = np.minimum(
        requested_traction_power,
        powertrain.max_motor_power_w,
    )

    required_braking_power = np.where(
        propulsion,
        0.0,
        -estimated_wheel_power,
    )
    regenerative_battery_power = np.minimum(
        required_braking_power * powertrain.regen_efficiency,
        powertrain.max_regen_power_w,
    )

    if powertrain.regen_efficiency > 0.0:
        regenerative_wheel_power = (
            regenerative_battery_power / powertrain.regen_efficiency
        )
    else:
        regenerative_wheel_power = np.zeros_like(regenerative_battery_power)

    friction_brake_power = np.maximum(
        required_braking_power - regenerative_wheel_power,
        0.0,
    )
    available_wheel_power = (
        traction_battery_power * powertrain.drivetrain_efficiency
    )
    unmet_wheel_power = np.where(
        propulsion,
        np.maximum(estimated_wheel_power - available_wheel_power, 0.0),
        0.0,
    )

    estimated_battery_power = (
        traction_battery_power
        + powertrain.auxiliary_power_w
        - regenerative_battery_power
    )
    measured_battery_power = data["measured_battery_power_w"].to_numpy(
        dtype=float
    )
    power_residual = measured_battery_power - estimated_battery_power

    # The first row of each trip has no preceding interval, so its dt is zero.
    dt_s = data.groupby("trip_id", sort=False)["time_s"].diff().fillna(0.0)
    if (dt_s < 0.0).any():
        raise ValueError("time must increase within each trip")

    output = data.copy()
    output["default_air_density_kg_m3"] = environment.air_density_kg_m3
    output["default_wind_speed_mps"] = environment.wind_speed_mps
    output["road_angle_rad"] = road_angle
    output["rolling_force_n"] = rolling_force
    output["aerodynamic_force_n"] = aerodynamic_force
    output["gradient_force_n"] = gradient_force
    output["inertial_force_n"] = inertial_force
    output["required_wheel_force_n"] = required_wheel_force
    output["estimated_wheel_power_w"] = estimated_wheel_power
    output["traction_battery_power_w"] = traction_battery_power
    output["regenerative_battery_power_w"] = regenerative_battery_power
    output["friction_brake_power_w"] = friction_brake_power
    output["unmet_wheel_power_w"] = unmet_wheel_power
    output["estimated_battery_power_w"] = estimated_battery_power
    output["battery_power_residual_w"] = power_residual
    output["dt_s"] = dt_s.to_numpy(dtype=float)
    output["measured_battery_energy_j"] = measured_battery_power * output["dt_s"]
    output["estimated_battery_energy_j"] = estimated_battery_power * output["dt_s"]
    output["battery_energy_residual_j"] = power_residual * output["dt_s"]
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add physics estimates and trip-based train/test labels."
    )
    parser.add_argument("input", type=Path, help="cleaned 10 Hz trip CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("physics_enriched_ev_trips.csv"),
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--test-trip-ids",
        type=int,
        nargs="+",
        help="explicit test IDs; overrides --test-fraction",
    )
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument("--wind-speed-mps", type=float, default=0.0)
    parser.add_argument("--pressure-pa", type=float, default=101_325.0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.input.is_file():
        raise FileNotFoundError(f"input file not found: {arguments.input}")

    data = pd.read_csv(arguments.input, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"cleaned dataset is missing columns: {missing}")

    # Excluded from the first Random Forest baseline:
    # Trips 36 and 41 include plugged-in charging sessions.
    # Trip 70 has no valid velocity data for the physics calculations.
    excluded_trip_ids = {36, 41, 70}
    excluded_rows = data["trip_id"].isin(excluded_trip_ids)
    print(
        f"Excluding {int(excluded_rows.sum()):,} rows "
        f"from trips {sorted(excluded_trip_ids)}"
    )
    data = data.loc[~excluded_rows].copy()

    data.sort_values(["trip_id", "time_s"], kind="stable", inplace=True)
    data.reset_index(drop=True, inplace=True)

    train_ids, test_ids = assign_trip_splits(
        trip_ids=data["trip_id"].to_numpy(),
        test_fraction=arguments.test_fraction,
        random_seed=arguments.random_seed,
        requested_test_trip_ids=arguments.test_trip_ids,
    )
    data["dataset_split"] = np.where(
        data["trip_id"].isin(test_ids),
        "test",
        "train",
    )

    environment = Environment.from_temperature(
        temperature_c=arguments.temperature_c,
        wind_speed_mps=arguments.wind_speed_mps,
        pressure_pa=arguments.pressure_pa,
    )

    # Baseline parameters for the 2014 BMW i3 60 Ah dataset vehicle.
    vehicle = VehicleParameters(
        mass_kg=1_270.0,
        rolling_resistance_coefficient=0.009,
        drag_coefficient=0.29,
        frontal_area_m2=2.38,
        wheel_radius_m=0.350,
        gear_ratio=9.7,
        rotor_inertia_kg_m2=0.0,
    )
    powertrain = Powertrain(
        drivetrain_efficiency=0.90,
        regen_efficiency=0.70,
        max_motor_power_w=140_000,
        max_regen_power_w=50_000,
        auxiliary_power_w=500,
    )

    enriched = calculate_physics_columns(
        data=data,
        vehicle=vehicle,
        powertrain=powertrain,
        environment=environment,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(arguments.output, index=False)

    print("\nPhysics dataset created")
    print(f"Input rows:       {len(data):,}")
    print(f"Output rows:      {len(enriched):,}")
    print(f"Training trips:   {len(train_ids)}")
    print(f"Testing trips:    {len(test_ids)}")
    print(f"Training IDs:     {sorted(train_ids)}")
    print(f"Testing IDs:      {sorted(test_ids)}")
    print(f"Default air density: {environment.air_density_kg_m3:.4f} kg/m^3")
    print(f"Default wind speed:  {environment.wind_speed_mps:.2f} m/s")
    print(f"Output file:      {arguments.output}")
    print(
        "Residual definition: measured battery power - "
        "estimated battery power"
    )


if __name__ == "__main__":
    main()
